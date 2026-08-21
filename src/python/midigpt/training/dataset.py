from __future__ import annotations

import copy
import glob as _glob_mod
import hashlib
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq

import midigpt._core as _core
from midigpt._types import Score
from midigpt.augmentation.mask_bar import MaskBar, MaskBarConfig
from midigpt.augmentation.score_window import select_window
from midigpt.augmentation.transpose import Transpose
from midigpt.augmentation.velocity import VelocityScale
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.utils import cached_indices, file_cache_key

# ── dataset filter ────────────────────────────────────────────────────────────


def _metadata_valid_mask(parquet_path: str, min_bars: int, min_tracks: int) -> list[bool]:
    """Metadata-only filter — no MIDI parsing (avoids C++ crashes on corrupt rows).

    GigaMIDI parquet columns used:
      num_tracks   — track count
      total_notes  — excludes silent/empty files

    Bar-count filtering requires parsing the score and is done in Phase 2.
    """
    import pyarrow.compute as pc

    meta = pq.read_table(
        parquet_path,
        columns=["num_tracks", "total_notes"],
    )

    m_tracks = pc.greater_equal(meta["num_tracks"], min_tracks).to_pylist()
    m_notes = pc.greater(meta["total_notes"], 0).to_pylist()

    return [t and n for t, n in zip(m_tracks, m_notes, strict=False)]


def _probe_chunk(
    parquet_path: str,
    chunk: list[int],
    valid_ts: frozenset[str] | None,
    min_bars: int = 1,
) -> list[int]:
    """Parse and validate a chunk in a subprocess; bisect on segfault.

    Checks:
      - Score.from_bytes() succeeds (Python exceptions are tolerated — only
        SIGSEGV kills the subprocess and triggers bisection).
      - Every track has at least min_bars bars.
      - If valid_ts is provided, every (ts_num/ts_den) in the parsed score is
        in the allowed set.
    """
    import json
    import os
    import subprocess
    import sys

    valid_ts_list = sorted(valid_ts) if valid_ts is not None else None
    sys_path_repr = repr(sys.path)
    parquet_repr = repr(parquet_path)
    chunk_repr = repr(chunk)
    ts_repr = repr(valid_ts_list)
    min_bars_repr = repr(min_bars)

    script = "\n".join(
        [
            "import sys, json",
            f"sys.path[:0] = {sys_path_repr}",
            "import datasets as hf",
            "from midigpt._types import Score",
            f"_vts = {ts_repr}",
            f"_min_bars = {min_bars_repr}",
            "valid_ts = set(_vts) if _vts is not None else None",
            f"data = hf.load_dataset('parquet', data_files={parquet_repr}, split='train')",
            f"chunk = {chunk_repr}",
            "ok = []",
            "for i in chunk:",
            "    try:",
            "        score = Score.from_bytes(data[i]['music'])",
            "        if not all(len(t.bars) >= _min_bars for t in score.tracks):",
            "            continue",
            "        if valid_ts is not None:",
            "            ts_in = {str(b.ts_numerator)+'/'+str(b.ts_denominator) for t in score.tracks for b in t.bars}",
            "            if not ts_in.issubset(valid_ts):",
            "                continue",
            "        ok.append(i)",
            "    except Exception:",
            "        ok.append(i)",
            "print(json.dumps(ok))",
        ]
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout.strip())
    if len(chunk) == 1:
        return []  # single index caused segfault — drop it
    mid = len(chunk) // 2
    return _probe_chunk(parquet_path, chunk[:mid], valid_ts, min_bars) + _probe_chunk(
        parquet_path, chunk[mid:], valid_ts, min_bars
    )


def _load_or_build_valid_indices(
    parquet_path: str,
    min_bars: int,
    min_tracks: int,
    valid_time_sigs: frozenset[str] | None = None,
) -> list[int]:
    """Two-phase filter: fast metadata check then subprocess parse+TS validation.

    Cache key includes encoder time-signature hash so it invalidates when the
    encoder changes.
    """
    import numpy as np

    ts_key = (
        hashlib.md5("|".join(sorted(valid_time_sigs)).encode()).hexdigest()[:8]
        if valid_time_sigs
        else "nots"
    )
    cache_file = cached_indices(
        file_cache_key(parquet_path, min_bars=min_bars, min_tracks=min_tracks, ts=ts_key)
    )

    if cache_file.exists():
        valid = np.load(cache_file).tolist()
        print(f"  dataset filter: {len(valid)} valid rows loaded from cache")
        return valid

    # Phase 1: fast metadata filter (pure pyarrow, no MIDI parsing)
    print("  dataset filter: phase 1 — metadata check…", flush=True)
    mask = _metadata_valid_mask(parquet_path, min_bars=min_bars, min_tracks=min_tracks)
    candidates = [i for i, v in enumerate(mask) if v]
    print(f"  dataset filter: {len(candidates)}/{len(mask)} pass metadata filter")

    # Phase 2: parse + time-sig validation (subprocess-isolated to catch segfaults)
    print(f"  dataset filter: phase 2 — parse+TS validation ({len(candidates)} rows)…", flush=True)
    chunk_size = 500
    valid: list[int] = []
    try:
        from tqdm.auto import tqdm as _tqdm

        it = _tqdm(range(0, len(candidates), chunk_size), desc="  parse+TS scan")
    except ImportError:
        it = range(0, len(candidates), chunk_size)
    for start in it:
        chunk = candidates[start : start + chunk_size]
        valid.extend(_probe_chunk(parquet_path, chunk, valid_time_sigs, min_bars))

    dropped = len(candidates) - len(valid)
    print(f"  dataset filter: dropped {dropped} rows (parse errors / bad TS) — {len(valid)} kept")
    np.save(cache_file, np.array(valid, dtype=np.int64))
    print("  dataset filter: saved to cache")
    return valid


# ── schema for DatasetBuilder (preprocessed format) ───────────────────────────

_SCORE_SCHEMA = pa.schema(
    [
        pa.field("resolution", pa.int32()),
        pa.field("tempo", pa.int32()),
        pa.field(
            "tracks",
            pa.list_(
                pa.struct(
                    [
                        pa.field("instrument", pa.int32()),
                        pa.field("track_type", pa.string()),
                        pa.field(
                            "bars",
                            pa.list_(
                                pa.struct(
                                    [
                                        pa.field("ts_numerator", pa.int32()),
                                        pa.field("ts_denominator", pa.int32()),
                                        pa.field("future", pa.bool_()),
                                        pa.field(
                                            "notes",
                                            pa.list_(
                                                pa.struct(
                                                    [
                                                        pa.field("pitch", pa.int32()),
                                                        pa.field("velocity", pa.int32()),
                                                        pa.field("onset_ticks", pa.int32()),
                                                        pa.field("duration_ticks", pa.int32()),
                                                        pa.field("delta", pa.int32()),
                                                    ]
                                                )
                                            ),
                                        ),
                                    ]
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
    ]
)


class DatasetBuilder:
    def build(
        self, midi_paths: list[str], output_path: str, splits: dict[str, float] | None = None
    ):
        if splits is None:
            splits = {"train": 0.9, "valid": 0.05, "test": 0.05}
        scores = []
        for path in midi_paths:
            try:
                score = Score.from_midi(path)
                scores.append(score.to_dict())
            except Exception:
                pass
        table = pa.Table.from_pylist(scores, schema=_SCORE_SCHEMA)
        pq.write_table(table, output_path)


def _resolve_paths(parquet_path: str | list[str]) -> list[str]:
    """Expand a single path, glob pattern, or list to a sorted list of parquet paths."""
    if isinstance(parquet_path, list):
        return sorted(parquet_path)
    expanded = sorted(_glob_mod.glob(parquet_path))
    return expanded if expanded else [parquet_path]


# ── static augments applied once per score before windowing ───────────────────

_TRANSPOSE = Transpose(range(-6, 7))
_VELOCITY = VelocityScale((0.8, 1.2))

# ── expressive-token masking (style-conditioning prototype support) ──────────

_EXPRESSIVE_TYPES = (
    _core.TokenType.VelocityLevel,
    _core.TokenType.DeltaDirection,
    _core.TokenType.Delta,
)


def _expressive_mask(
    tokens: list[int],
    vocab,
    restrict_bars: set[tuple[int, int]] | None = None,
) -> list[bool]:
    """Boolean mask over `tokens`, True at VelocityLevel/DeltaDirection/Delta
    positions — i.e. exactly the "how it was played" tokens a style encoder
    should see, never pitch/onset/duration.

    If `restrict_bars` is given (a set of local (track_idx, bar_idx) pairs,
    same indexing as EncodeOptions.multi_fill/multi_humanize), only tokens
    belonging to one of those bars' in-place content are marked. Anything at
    or after a TrackEnd token (FillIn/Humanize appendix content) is always
    excluded regardless of restrict_bars — an appendix block is never a bar's
    in-place skeleton, and (track_idx, bar_idx) tracking would otherwise stay
    frozen at the last bar seen before TrackEnd and could spuriously match.
    """
    mask = [False] * len(tokens)
    track_idx = -1
    bar_idx = -1
    in_appendix = False
    for i, tok in enumerate(tokens):
        tt = vocab.get_type(tok)
        if tt == _core.TokenType.Track:
            track_idx += 1
            bar_idx = -1
            in_appendix = False
        elif tt == _core.TokenType.TrackEnd:
            in_appendix = True
        elif tt == _core.TokenType.Bar:
            bar_idx += 1
        elif not in_appendix and tt in _EXPRESSIVE_TYPES:
            if restrict_bars is None or (track_idx, bar_idx) in restrict_bars:
                mask[i] = True
    return mask


class MidiGPTDataset:
    """Training dataset with encoder-aware windowing, infill, and bar masking.

    Per-sample pipeline:
      1. Pitch/velocity augmentation on the full score (once).
      2. Independently roll three decisions:
           - is_infill    (infill_probability): encode with FillIn tokens.
           - is_humanize  (humanize_probability): encode with Humanize tokens
             (skeleton in place, Velocity/Delta deferred to an appendix).
           - do_mask      (mask_bar_config.apply_probability): apply MASK_BAR.
         All three are independent, but their *bar sets* are mutually
         exclusive: humanize_bars is sampled disjoint from infill_bars, and
         MaskBar is given the union of both so it never marks either kind of
         target as MASK_BAR.
      3. Retry loop over (n_bars, n_tracks) pairs:
           - Infill overflow → step n_tracks down, then n_bars down; never
             clip FillIn tokens.
           - AR overflow → random-window clip (legacy behaviour).
      4. Fallback: smallest window, 1 track, AR clip.

    Every sample also carries a "style_ref_mask" key (bool list, same length
    as input_ids): True at VelocityLevel/DeltaDirection/Delta positions of a
    humanize-eligible bar subset disjoint from whatever was actually withheld
    as the humanize target (all-False when unavailable or is_humanize=False).
    This is scaffolding for scripts/style_prototype/'s style-conditioning
    work — inert for ordinary training, since nothing consumes the key today.

    When style_pretrain_mode=True, __getitem__ instead returns two
    independent plain-encoded windows from the same piece (no infill/
    humanize/AR shape at all) for contrastive style-encoder pretraining — see
    _encode_style_pair.

    max_seq_len acts as a dataset-side cap and must not exceed the model's
    positional budget (model.max_context()). Pass 0 to use the encoder's
    largest num_bars_map entry as an implicit limit.
    """

    def __init__(
        self,
        parquet_path: str | list[str],
        tokenizer: Tokenizer,
        # Infill training
        infill_probability: float = 0.75,
        infill_bar_fraction: float = 0.5,
        # Humanize training (independent of infill; a bar can be an infill
        # target or a humanize target but never both — see _encode_one).
        humanize_probability: float = 0.0,
        humanize_bar_fraction: float = 0.5,
        # Bar masking (independent of infill)
        mask_bar_config: MaskBarConfig | None = None,
        # Sequence budget
        max_seq_len: int = 2048,
        # Window / track sampling
        max_tracks: int = 12,
        min_tracks: int = 1,
        min_fill_ratio: float = 0.75,
        # Switchable mode dropout probabilities (only used when the encoder
        # config has switchable_velocity / switchable_microtiming = true).
        # Each value is the probability that the feature is turned OFF for a
        # given training sample, so the model sees both modes during training.
        # 0.0 = always on (no dropout); 0.5 = equal mix; 1.0 = always off.
        velocity_off_probability: float = 0.0,
        microtiming_off_probability: float = 0.0,
        # Genre injection probability (only used when encoder config has
        # genre_groups configured). When the sample has a recognized genre,
        # include the Genre token with this probability, omit with 1-p.
        # 1.0 = always include when known; 0.0 = never include.
        genre_probability: float = 1.0,
        # Style-conditioning prototype (scripts/style_prototype/): when True,
        # __getitem__ returns two independent, plain-encoded windows from the
        # same piece (anchor/positive) for contrastive style-encoder
        # pretraining, instead of the normal infill/humanize/AR sample shape.
        # See _encode_style_pair.
        style_pretrain_mode: bool = False,
        # ── Mechanical context (independent of everything above; only takes
        # effect when is_humanize) ─────────────────────────────────────────
        # Simulates a user's real input: some tracks in the window are
        # "mechanically entered" (quantized, no expressiveness) rather than
        # real performances. Maximum per-track density. Each sample draws
        # p ~ Uniform(0, this), then each track is independently marked
        # mechanical with probability p (a track is EITHER fully mechanical
        # for every bar in the window OR fully real -- never mixed within a
        # track; see mechanize.py). 0.0 = disabled (current behaviour: every
        # context bar is always real).
        context_mechanical_fraction: float = 0.0,
        # Probability that humanize-target selection draws a structured
        # pattern (whole track / vertical bar-slice / small multi-track
        # window / whole window) instead of the independent per-cell
        # Bernoulli scheme. 0.0 = always per-cell (current behaviour).
        structured_target_probability: float = 0.0,
        # When True (and context_mechanical_fraction > 0), structured target
        # selection respects mechanical/expressive coherence: a mechanical
        # track can only be targeted as a whole (never a single bar within
        # it, since its untouched neighbours would still be mechanical --
        # musically incoherent), and vertical/small-window patterns only
        # draw from the currently-expressive tracks. False = naive (shape
        # selection ignores mechanical status entirely).
        mechanical_coherent_targets: bool = False,
        # When mechanical_coherent_targets=True, the probability of instead
        # using the naive (incoherent) rule for a given sample anyway -- a
        # small residual exposure so the model isn't purely zero-shot OOD if
        # an incoherent configuration is ever requested at inference. 0.0 =
        # always coherent.
        mechanical_coherent_residual: float = 0.0,
    ):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]") from None
        paths = _resolve_paths(parquet_path)
        self._data = hf.load_dataset("parquet", data_files=paths, split="train")
        self._encoder_config_json: str = tokenizer._vocab.config().to_json()
        self._tokenizer = tokenizer
        self._infill_prob = infill_probability
        self._infill_frac = infill_bar_fraction
        self._humanize_prob = humanize_probability
        self._humanize_frac = humanize_bar_fraction
        self._style_pretrain_mode = style_pretrain_mode
        self._mask_cfg = mask_bar_config
        self._max_seq_len = max_seq_len
        self._max_tracks = max_tracks
        self._min_tracks = min_tracks
        self._min_fill_ratio = min_fill_ratio
        self._velocity_off_prob = velocity_off_probability
        self._microtiming_off_prob = microtiming_off_probability
        self._genre_probability = genre_probability
        self._context_mech_frac = context_mechanical_fraction
        self._structured_target_prob = structured_target_probability
        self._mech_coherent = mechanical_coherent_targets
        self._mech_coherent_residual = mechanical_coherent_residual

        cfg_dict = json.loads(tokenizer._vocab.config().to_json())
        raw_map = cfg_dict.get("num_bars_map") or [4]
        self._num_bars_choices: list[int] = sorted(set(int(x) for x in raw_map), reverse=True)

        if infill_probability > 0 and not cfg_dict.get("supports_infill", False):
            raise ValueError(
                f"infill_probability={infill_probability} > 0 but the encoder config "
                f"has supports_infill=false. Set infill_probability=0.0 or use an "
                f"infill-capable checkpoint."
            )
        if humanize_probability > 0 and not cfg_dict.get("supports_humanize", False):
            raise ValueError(
                f"humanize_probability={humanize_probability} > 0 but the encoder config "
                f"has supports_humanize=false. Set humanize_probability=0.0 or use a "
                f"humanize-capable checkpoint."
            )
        if mask_bar_config is not None:
            if not tokenizer._vocab.has(_core.TokenType.MaskBar):
                raise ValueError(
                    "mask_bar_config is set but the encoder vocab does not include "
                    "the MaskBar token. Set mask_bar_config=None or use a "
                    "masking-capable checkpoint."
                )

        self._switchable_velocity    = bool(cfg_dict.get("switchable_velocity",    False))
        self._switchable_microtiming = bool(cfg_dict.get("switchable_microtiming", False))
        if velocity_off_probability > 0.0 and not self._switchable_velocity:
            raise ValueError(
                "velocity_off_probability > 0 but encoder config has switchable_velocity=false. "
                "Only set this for models trained with switchable velocity."
            )
        if microtiming_off_probability > 0.0 and not self._switchable_microtiming:
            raise ValueError(
                "microtiming_off_probability > 0 but encoder config has switchable_microtiming=false. "
                "Only set this for models trained with switchable microtiming."
            )
        if not (0.0 <= genre_probability <= 1.0):
            raise ValueError(f"genre_probability must be in [0, 1] (got {genre_probability})")
        self._genre_grouping = tokenizer._vocab.config().genre_grouping  # None or GenreGrouping

        ts_list = cfg_dict.get("time_signatures")
        valid_ts: frozenset[str] | None = frozenset(ts_list) if ts_list else None

        # Per-shard filtering (each shard gets its own cache entry).
        # Global indices are reconstructed from per-shard row counts.
        all_valid: list[int] = []
        offset = 0
        for path in paths:
            n = pq.read_metadata(path).num_rows
            local_valid = _load_or_build_valid_indices(
                path,
                min_bars=self._num_bars_choices[-1],
                min_tracks=min_tracks,
                valid_time_sigs=valid_ts,
            )
            all_valid.extend(i + offset for i in local_valid)
            offset += n
        self._data = self._data.select(all_valid)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # C++ pybind11 objects are not picklable; strip them so spawn workers
        # can receive the dataset. _ensure_tokenizer() rebuilds them lazily.
        state["_tokenizer"] = None
        state["_genre_grouping"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            cfg = _core.EncoderConfig.from_json(self._encoder_config_json)
            self._tokenizer = Tokenizer(cfg)
            self._genre_grouping = self._tokenizer._vocab.config().genre_grouping

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int, _depth: int = 0) -> dict:
        if _depth >= 10:
            raise RuntimeError("MidiGPTDataset: failed to load a valid sample after 10 attempts")
        try:
            if self._style_pretrain_mode:
                return self._encode_style_pair(idx)
            return self._encode_one(idx)
        except Exception:
            return self.__getitem__(random.randrange(len(self._data)), _depth + 1)

    def _genre_token(self, idx: int) -> int:
        """Return genre token id for sample idx, or -1 if unavailable/omitted."""
        from urllib.parse import unquote

        self._ensure_tokenizer()
        if self._genre_grouping is None or random.random() >= self._genre_probability:
            return -1
        raw = self._data[idx].get("music_style_scraped") or ""
        if not raw:
            return -1
        genres = [g.strip() for g in unquote(raw).split(",") if g.strip()]
        known = [g for g in genres if self._genre_grouping.contains(g)]
        if not known:
            return -1
        return self._genre_grouping.encode(random.choice(known))

    def _stash_nomml(self, score: Score, idx: int) -> None:
        """Seed track.attributes["_nomml_raw"] from GigaMIDI's precomputed
        per-track NOMML column so Nomml.compute() uses ground truth instead
        of recomputing from an already bar-quantized Score (which would lose
        the fine onset alignment the metric depends on). -1 entries (no valid
        value for that track) are left unset, falling back to Nomml.compute's
        own (lower-fidelity) recomputation.
        """
        values = self._data[idx].get("NOMML")
        if not values:
            return
        for t_idx, track in enumerate(score.tracks):
            if t_idx >= len(values):
                break
            v = values[t_idx]
            if v is not None and v != -1:
                track.attributes["_nomml_raw"] = int(v)

    @staticmethod
    def _draw_structured_targets(
        window, infill_bars: set[tuple[int, int]], mechanical_tracks: set[int], coherent: bool,
    ) -> set[tuple[int, int]]:
        """One of: whole track, vertical bar-slice across tracks, a small
        multi-track/multi-bar window, or the whole window -- chosen at
        random among the patterns that are geometrically possible.

        If coherent=True: a mechanical track is only eligible for the
        whole-track pattern (targeting a single bar within it would leave
        its untouched neighbours mechanical -- incoherent, see mechanize.py
        docstring / project discussion). Vertical and small-window patterns
        only draw from tracks NOT in mechanical_tracks. If coherent=False,
        mechanical status is ignored entirely (naive shape selection).

        Returns an empty set if the drawn pattern has no eligible notes
        (caller falls back to the per-cell scheme in that case).
        """
        n_tracks = len(window.tracks)
        if n_tracks == 0:
            return set()
        n_bars = len(window.tracks[0].bars)
        if n_bars == 0:
            return set()

        def has_notes(t: int, b: int) -> bool:
            return (t, b) not in infill_bars and window.tracks[t].bars[b].notes

        expressive_tracks = [t for t in range(n_tracks) if t not in mechanical_tracks] if coherent else list(range(n_tracks))

        patterns = ["whole_track", "whole_window"]
        if len(expressive_tracks) >= 2:
            patterns.append("vertical")
        if len(expressive_tracks) >= 2 and n_bars >= 2:
            patterns.append("small_window")

        pattern = random.choice(patterns)

        if pattern == "whole_track":
            t = random.randrange(n_tracks)
            return {(t, b) for b in range(n_bars) if has_notes(t, b)}

        if pattern == "vertical":
            b = random.randrange(n_bars)
            return {(t, b) for t in expressive_tracks if has_notes(t, b)}

        if pattern == "small_window":
            k = random.randint(2, min(4, len(expressive_tracks)))
            m = random.randint(2, min(4, n_bars))
            tracks_sel = random.sample(expressive_tracks, k)
            bar_start = random.randrange(0, n_bars - m + 1)
            return {
                (t, b) for t in tracks_sel for b in range(bar_start, bar_start + m) if has_notes(t, b)
            }

        # whole_window
        return {(t, b) for t in range(n_tracks) for b in range(n_bars) if has_notes(t, b)}

    def _encode_one(self, idx: int) -> dict:
        self._ensure_tokenizer()
        # Score.from_bytes() already returns a fresh, owned object; the
        # subsequent augmentations mutate copies, so an extra deepcopy here is
        # pure overhead (~1.5 ms/sample).
        score = Score.from_bytes(self._data[idx]["music"])
        self._stash_nomml(score, idx)

        score = _TRANSPOSE(score)
        score = _VELOCITY(score)

        # Roll decisions independently.
        is_infill = random.random() < self._infill_prob
        is_humanize = random.random() < self._humanize_prob
        do_mask = self._mask_cfg is not None  # gate is inside MaskBar itself

        initial = random.choice(self._num_bars_choices)
        fallbacks = sorted([n for n in self._num_bars_choices if n < initial], reverse=True)

        for n_bars in [initial, *fallbacks]:
            for n_tracks in range(self._max_tracks, self._min_tracks - 1, -1):
                window = select_window(score, n_bars, n_tracks, self._min_fill_ratio)
                if window is None:
                    continue

                # Sample a random per-cell probability uniformly in
                # [0, infill_bar_fraction], then independently apply it to
                # every (track, bar) cell. This varies density across samples
                # so the model trains on the full spectrum (few to many infill
                # targets). If no cell is selected, force one at random.
                infill_bars: set[tuple[int, int]] = set()
                if is_infill:
                    p = random.uniform(0.0, self._infill_frac)
                    infill_bars = {
                        (t_idx, b_idx)
                        for t_idx in range(len(window.tracks))
                        for b_idx in range(len(window.tracks[t_idx].bars))
                        if random.random() < p
                    }
                    if not infill_bars:
                        t = random.randrange(len(window.tracks))
                        b = random.randrange(len(window.tracks[t].bars))
                        infill_bars = {(t, b)}

                # Humanize: bars whose skeleton (pitch/onset/duration) stays
                # fixed in place and only Velocity/Delta are regenerated via
                # the appendix. Candidates must actually contain notes (an
                # empty bar has nothing to humanize) and must be disjoint
                # from infill_bars (a bar can't be both a hidden infill
                # target and a visible humanize skeleton).
                humanize_bars: set[tuple[int, int]] = set()
                # Style-conditioning prototype (scripts/style_prototype/): a
                # subset of humanize-eligible bars disjoint from humanize_bars,
                # used as the style encoder's input (see style_ref_mask
                # below). Never drawn from humanize_bars itself — the encoder
                # must never see the exact content it's meant to help predict,
                # or it degenerates into memorizing the answer instead of
                # learning reusable style. If there aren't enough remaining
                # candidates, style conditioning is simply unavailable for
                # this sample (style_ref_mask stays all-False).
                reference_bars: set[tuple[int, int]] = set()
                # Mechanical context: a subset of tracks in the window are
                # marked as "mechanically entered" (quantized, no real
                # expressiveness) rather than real performances -- simulates
                # the from-scratch/full-piece deployment case, not just
                # touch-up of an already-expressive piece. Drawn once per
                # sample (not per n_bars/n_tracks retry) would be cleaner,
                # but the window itself (and therefore track count) isn't
                # known until here, so it's redrawn per attempt like
                # infill_bars/humanize_bars already are.
                mechanical_tracks: set[int] = set()
                if is_humanize and self._context_mech_frac > 0.0:
                    mech_p = random.uniform(0.0, self._context_mech_frac)
                    mechanical_tracks = {
                        t_idx for t_idx in range(len(window.tracks)) if random.random() < mech_p
                    }
                if is_humanize:
                    candidates = [
                        (t_idx, b_idx)
                        for t_idx in range(len(window.tracks))
                        for b_idx in range(len(window.tracks[t_idx].bars))
                        if (t_idx, b_idx) not in infill_bars
                        and window.tracks[t_idx].bars[b_idx].notes
                    ]
                    if candidates:
                        if self._structured_target_prob > 0.0 and random.random() < self._structured_target_prob:
                            use_coherent = self._mech_coherent and (
                                self._mech_coherent_residual <= 0.0
                                or random.random() >= self._mech_coherent_residual
                            )
                            humanize_bars = self._draw_structured_targets(
                                window, infill_bars, mechanical_tracks, use_coherent
                            )
                        if not humanize_bars:
                            p = random.uniform(0.0, self._humanize_frac)
                            humanize_bars = {c for c in candidates if random.random() < p}
                            if not humanize_bars:
                                humanize_bars = {random.choice(candidates)}

                        remaining = [c for c in candidates if c not in humanize_bars]
                        if remaining:
                            rp = random.uniform(0.0, self._humanize_frac)
                            reference_bars = {c for c in remaining if random.random() < rp}
                            if not reference_bars:
                                reference_bars = {random.choice(remaining)}

                # Apply mechanization to every non-target bar of a mechanical
                # track. Target bars keep their real values -- those become
                # the appendix's prediction label regardless of the track's
                # mechanical/real status; only what the model gets to SEE as
                # context is affected. Must happen after humanize_bars is
                # finalized (needs to know which bars are targets) and before
                # encoding.
                if mechanical_tracks:
                    from midigpt.augmentation.mechanize import mechanize_bar

                    for t_idx in mechanical_tracks:
                        for b_idx, bar in enumerate(window.tracks[t_idx].bars):
                            if (t_idx, b_idx) not in humanize_bars:
                                mechanize_bar(bar, window.resolution)

                if do_mask:
                    window = MaskBar(
                        self._mask_cfg, infill_bars=infill_bars | humanize_bars
                    )(window)

                encode_opts = _core.EncodeOptions()
                if is_infill and infill_bars:
                    encode_opts.multi_fill = infill_bars
                if is_humanize and humanize_bars:
                    encode_opts.multi_humanize = humanize_bars
                # Switchable mode dropout: randomly disable velocity/microtiming
                # so the model sees both modes during training.
                if self._switchable_velocity and self._velocity_off_prob > 0.0:
                    encode_opts.use_velocity = 0 if random.random() < self._velocity_off_prob else 1
                if self._switchable_microtiming and self._microtiming_off_prob > 0.0:
                    encode_opts.use_microtiming = 0 if random.random() < self._microtiming_off_prob else 1
                encode_opts.genre = self._genre_token(idx)
                try:
                    tokens = self._tokenizer.encode(window, opts=encode_opts)
                except Exception:
                    continue

                if (is_infill or is_humanize) and len(tokens) > self._max_seq_len:
                    # Never clip FillIn/Humanize tokens — a mid-appendix cut
                    # would break the exact per-block note count the grammar
                    # enforces. Retry with fewer resources instead.
                    continue

                if len(tokens) > self._max_seq_len:
                    offset = random.randint(0, len(tokens) - self._max_seq_len)
                    tokens = tokens[offset : offset + self._max_seq_len]

                # Unreachable when is_humanize (the max_seq_len check above
                # already `continue`s in that case), so `tokens` here is
                # always the full, unclipped stream whenever reference_bars
                # was computed — style_ref_mask stays index-aligned with it.
                style_ref_mask = (
                    _expressive_mask(tokens, self._tokenizer._vocab, reference_bars)
                    if reference_bars
                    else [False] * len(tokens)
                )
                return {
                    "input_ids": tokens,
                    "labels": tokens,
                    "style_ref_mask": style_ref_mask,
                    "humanize_bars": humanize_bars,
                    "reference_bars": reference_bars,
                }

        # Fallback: smallest window, 1 track, no infill, hard clip.
        window = select_window(
            score, self._num_bars_choices[-1], self._min_tracks, self._min_fill_ratio
        )
        try:
            fallback_opts = _core.EncodeOptions()
            if self._switchable_velocity and self._velocity_off_prob > 0.0:
                fallback_opts.use_velocity = 0 if random.random() < self._velocity_off_prob else 1
            if self._switchable_microtiming and self._microtiming_off_prob > 0.0:
                fallback_opts.use_microtiming = 0 if random.random() < self._microtiming_off_prob else 1
            fallback_opts.genre = self._genre_token(idx)
            tokens = self._tokenizer.encode(
                window if window else score, opts=fallback_opts
            )[: self._max_seq_len]
        except Exception as exc:
            raise ValueError(f"Failed to encode sample {idx} after all retries") from exc
        return {
            "input_ids": tokens,
            "labels": tokens,
            "style_ref_mask": [False] * len(tokens),
            "humanize_bars": set(),
            "reference_bars": set(),
        }

    def _encode_style_pair(self, idx: int) -> dict:
        """Two independent, plain-encoded windows from the same piece, for
        contrastive style-encoder pretraining (scripts/style_prototype/).

        No infill/humanize/AR shaping — each window is encoded exactly as a
        plain AR sample would be, and the returned mask flags the full
        expressive-token stream (VelocityLevel/DeltaDirection/Delta) a style
        encoder should read. The two windows get independent VelocityScale
        draws (`_VELOCITY` redraws its random factor on every call — see
        augmentation/velocity.py), not one shared per-piece scale, so a
        contrastive objective trained on (anchor, positive) can't shortcut on
        matching absolute loudness — it has to learn dynamic *shape*.

        Always single-track (n_tracks=1), not up to self._max_tracks: a
        multi-track window's encoded stream would interleave several tracks'
        expressive tokens into one sequence, mixing their distinct playing
        styles together — the opposite of what a per-performance style
        vector should represent. Retries across self._num_bars_choices
        (largest first) if the chosen bar count has no valid single-track
        window, mirroring _encode_one's fallback pattern.
        """
        self._ensure_tokenizer()
        score = Score.from_bytes(self._data[idx]["music"])
        score = _TRANSPOSE(score)

        views = []
        for _ in range(2):
            window = None
            for n_bars in self._num_bars_choices:
                window = select_window(score, n_bars, 1, self._min_fill_ratio)
                if window is not None:
                    break
            if window is None:
                raise ValueError(
                    f"No valid single-track window for sample {idx} (style_pretrain_mode)"
                )
            window = _VELOCITY(window)
            tokens = self._tokenizer.encode(window)
            if len(tokens) > self._max_seq_len:
                tokens = tokens[: self._max_seq_len]
            mask = _expressive_mask(tokens, self._tokenizer._vocab)
            if not any(mask):
                # Degenerate window: zero VelocityLevel/DeltaDirection/Delta
                # tokens (plausible with velocity_sticky=True on an
                # already-quantized, perfectly flat-velocity clip — GigaMIDI
                # has plenty of these). Nothing for the style encoder to
                # read; let __getitem__'s existing retry/resample handle it
                # rather than feeding a zero-length sequence downstream.
                raise ValueError(
                    f"No expressive tokens in window for sample {idx} (style_pretrain_mode)"
                )
            views.append((tokens, mask))

        (anchor_ids, anchor_mask), (positive_ids, positive_mask) = views
        return {
            "anchor_ids": anchor_ids,
            "anchor_mask": anchor_mask,
            "positive_ids": positive_ids,
            "positive_mask": positive_mask,
        }
