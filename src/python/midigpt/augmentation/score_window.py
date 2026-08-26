"""Score windowing: pick a contiguous bar range and a random track subset."""

from __future__ import annotations

import copy
import math
import random
from typing import Literal, Sequence

from midigpt._types import Score, Track

WindowSelectionMode = Literal["fixed", "uniform", "weighted"]


def select_window(
    score: Score,
    n_bars: int,
    n_tracks: int,
    min_fill_ratio: float = 0.75,
    rng: random.Random | None = None,
    return_metadata: bool = False,
) -> Score | tuple[Score, int, list[int]] | None:
    """Return a Score slice of exactly `n_bars` bars on exactly `n_tracks` tracks.

    A track is "valid" for a window starting at `start` if it has at least
    ceil(min_fill_ratio * n_bars) non-empty bars in [start, start + n_bars).

    Returns None when no valid (start, track_subset) combination exists for
    the requested parameters — the caller should fall back to smaller values.
    With ``return_metadata=True``, also returns the selected bar offset and
    original track indices. This keeps paired modalities, such as audio stems,
    aligned with the resulting score window.
    """
    _rng = rng or random
    min_nonempty = max(1, math.ceil(min_fill_ratio * n_bars))

    # Build list of valid (start, [track_indices]) pairs
    total_bars = min(
        (len(t.bars) for t in score.tracks),
        default=0,
    )
    if total_bars < n_bars:
        return None

    valid_starts: list[tuple[int, list[int]]] = []
    for start in range(total_bars - n_bars + 1):
        good_tracks = []
        for t_idx, track in enumerate(score.tracks):
            if len(track.bars) < start + n_bars:
                continue
            window_bars = track.bars[start : start + n_bars]
            nonempty = sum(1 for b in window_bars if b.notes)
            if nonempty >= min_nonempty:
                good_tracks.append(t_idx)
        if len(good_tracks) >= n_tracks:
            valid_starts.append((start, good_tracks))

    if not valid_starts:
        return None

    start, good_tracks = _rng.choice(valid_starts)
    _rng.shuffle(good_tracks)
    selected_indices = good_tracks[:n_tracks]

    new_tracks = [
        Track(
            bars=copy.deepcopy(score.tracks[t_idx].bars[start : start + n_bars]),
            instrument=score.tracks[t_idx].instrument,
            track_type=score.tracks[t_idx].track_type,
            attributes=dict(score.tracks[t_idx].attributes),
        )
        for t_idx in selected_indices
    ]
    window = Score(tracks=new_tracks, resolution=score.resolution, tempo=score.tempo)
    if return_metadata:
        return window, start, selected_indices
    return window


# ---------------------------------------------------------------------------
# Content-aware window selection
# ---------------------------------------------------------------------------


def _build_candidates(
    score: Score,
    num_bars_choices: Sequence[int],
    max_tracks: int,
    min_tracks: int = 1,
    min_fill_ratio: float = 0.75,
) -> list[tuple[int, int, float, list[int]]]:
    """Enumerate all viable (start, n_bars) windows and score them.

    Returns a list of (start, n_bars, quality, active_track_indices) tuples.
    Quality = sum of note counts for the top min(max_tracks, active_tracks)
    tracks in the window.  Windows with fewer than ``min_tracks`` active
    tracks are excluded.
    """
    total_bars = min((len(t.bars) for t in score.tracks), default=0)
    if total_bars == 0:
        return []

    # Precompute per-track note counts for every bar
    track_notes: list[list[int]] = []
    for track in score.tracks:
        track_notes.append([len(b.notes) for b in track.bars])

    candidates: list[tuple[int, int, float, list[int]]] = []
    for n_bars in num_bars_choices:
        if n_bars > total_bars:
            continue
        min_nonempty = max(1, math.ceil(min_fill_ratio * n_bars))
        for start in range(total_bars - n_bars + 1):
            active_tracks: list[tuple[int, int]] = []  # (note_count, t_idx)
            for t_idx, notes_per_bar in enumerate(track_notes):
                if len(notes_per_bar) < start + n_bars:
                    continue
                nonempty = sum(1 for n in notes_per_bar[start : start + n_bars] if n > 0)
                if nonempty >= min_nonempty:
                    total_notes = sum(notes_per_bar[start : start + n_bars])
                    active_tracks.append((total_notes, t_idx))
            if len(active_tracks) < min_tracks:
                continue
            # Quality: sum of note counts for top-k tracks
            active_tracks.sort(reverse=True)
            k = min(max_tracks, len(active_tracks))
            quality = float(sum(count for count, _ in active_tracks[:k]))
            indices = [t_idx for _, t_idx in active_tracks[:k]]
            candidates.append((start, n_bars, quality, indices))
    return candidates


def select_window_uniform(
    score: Score,
    num_bars_choices: Sequence[int],
    max_tracks: int,
    min_tracks: int = 1,
    min_fill_ratio: float = 0.75,
    rng: random.Random | None = None,
    return_metadata: bool = False,
) -> Score | tuple[Score, int, list[int]] | None:
    """Content-aware window selection with uniform sampling across candidates.

    Enumerates all viable (start, n_bars) windows and picks one uniformly at
    random.  The number of tracks is determined by the window content (how
    many tracks are active), capped at ``max_tracks``.
    """
    _rng = rng or random
    candidates = _build_candidates(score, num_bars_choices, max_tracks, min_tracks, min_fill_ratio)
    if not candidates:
        return None
    start, n_bars, _quality, indices = _rng.choice(candidates)
    _rng.shuffle(indices)
    return _slice_window(score, start, n_bars, indices, return_metadata)


def select_window_weighted(
    score: Score,
    num_bars_choices: Sequence[int],
    max_tracks: int,
    min_tracks: int = 1,
    min_fill_ratio: float = 0.75,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    return_metadata: bool = False,
) -> Score | tuple[Score, int, list[int]] | None:
    """Content-aware window selection with quality-weighted sampling.

    Like ``select_window_uniform`` but samples proportionally to each
    window's quality score (total note density of top-k active tracks).
    Lower temperature concentrates probability on denser windows; higher
    temperature approaches uniform.
    """
    _rng = rng or random
    candidates = _build_candidates(score, num_bars_choices, max_tracks, min_tracks, min_fill_ratio)
    if not candidates:
        return None

    qualities = [c[2] for c in candidates]
    if temperature <= 0 or len(candidates) == 1:
        idx = qualities.index(max(qualities))
    else:
        scaled = [q / temperature for q in qualities]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]
        idx = _rng.choices(range(len(candidates)), weights=probs, k=1)[0]

    start, n_bars, _quality, indices = candidates[idx]
    _rng.shuffle(indices)
    return _slice_window(score, start, n_bars, indices, return_metadata)


def _slice_window(
    score: Score,
    start: int,
    n_bars: int,
    track_indices: list[int],
    return_metadata: bool,
) -> Score | tuple[Score, int, list[int]]:
    """Slice a score window from the given start/bar/track parameters."""
    new_tracks = [
        Track(
            bars=copy.deepcopy(score.tracks[t_idx].bars[start : start + n_bars]),
            instrument=score.tracks[t_idx].instrument,
            track_type=score.tracks[t_idx].track_type,
            attributes=dict(score.tracks[t_idx].attributes),
        )
        for t_idx in track_indices
    ]
    window = Score(tracks=new_tracks, resolution=score.resolution, tempo=score.tempo)
    if return_metadata:
        return window, start, track_indices
    return window


def select_window_dispatch(
    score: Score,
    mode: WindowSelectionMode = "fixed",
    *,
    # Fixed-mode params (used when mode="fixed")
    n_bars: int | None = None,
    n_tracks: int | None = None,
    # Content-aware params (used when mode="uniform" or "weighted")
    num_bars_choices: Sequence[int] | None = None,
    max_tracks: int | None = None,
    min_tracks: int = 1,
    # Common params
    min_fill_ratio: float = 0.75,
    temperature: float = 1.0,
    rng: random.Random | None = None,
    return_metadata: bool = False,
) -> Score | tuple[Score, int, list[int]] | None:
    """Dispatch to the appropriate window selection strategy.

    Modes:
      "fixed"    — pick exactly n_bars bars on exactly n_tracks tracks
                   (original algorithm, caller specifies dimensions).
      "uniform"  — enumerate all viable (start, n_bars) windows, pick one
                   uniformly at random.  n_tracks determined by content.
      "weighted" — like uniform but sample proportionally to note density
                   (total notes in top-k active tracks).
    """
    if mode == "fixed":
        if n_bars is None or n_tracks is None:
            raise ValueError("mode='fixed' requires n_bars and n_tracks")
        return select_window(
            score, n_bars, n_tracks,
            min_fill_ratio=min_fill_ratio, rng=rng,
            return_metadata=return_metadata,
        )
    elif mode == "uniform":
        if num_bars_choices is None or max_tracks is None:
            raise ValueError("mode='uniform' requires num_bars_choices and max_tracks")
        return select_window_uniform(
            score, num_bars_choices, max_tracks,
            min_tracks=min_tracks, min_fill_ratio=min_fill_ratio,
            rng=rng, return_metadata=return_metadata,
        )
    elif mode == "weighted":
        if num_bars_choices is None or max_tracks is None:
            raise ValueError("mode='weighted' requires num_bars_choices and max_tracks")
        return select_window_weighted(
            score, num_bars_choices, max_tracks,
            min_tracks=min_tracks, min_fill_ratio=min_fill_ratio,
            temperature=temperature, rng=rng,
            return_metadata=return_metadata,
        )
    else:
        raise ValueError(f"unknown window selection mode {mode!r} (choose 'fixed', 'uniform', 'weighted')")
