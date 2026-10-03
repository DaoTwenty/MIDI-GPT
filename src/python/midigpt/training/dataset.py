from __future__ import annotations

import copy
import glob as _glob_mod
import hashlib
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

import midigpt._core as _core
from midigpt._types import Score
from midigpt.augmentation.mask_bar import MaskBar, MaskBarConfig
from midigpt.augmentation.score_window import select_window, select_window_dispatch
from midigpt.augmentation.transpose import Transpose
from midigpt.augmentation.velocity import VelocityScale
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.utils import cached_indices, file_cache_key


# ---------------------------------------------------------------------------
# Dataset filter — validates parquet rows against encoder constraints
# ---------------------------------------------------------------------------


def _metadata_valid_mask(parquet_path: str, min_bars: int, min_tracks: int) -> list[bool]:
    """Metadata-only filter — no MIDI parsing (avoids C++ crashes on corrupt rows)."""
    import pyarrow.compute as pc

    meta = pq.read_table(parquet_path, columns=["num_tracks", "total_notes"])
    m_tracks = pc.greater_equal(meta["num_tracks"], min_tracks).to_pylist()
    m_notes = pc.greater(meta["total_notes"], 0).to_pylist()
    return [t and n for t, n in zip(m_tracks, m_notes, strict=False)]


def _probe_chunk(
    parquet_path: str,
    chunk: list[int],
    valid_ts: frozenset[str] | None,
    min_bars: int = 1,
) -> list[int]:
    """Parse and validate a chunk in a subprocess; bisect on segfault."""
    import os
    import subprocess
    import sys

    valid_ts_list = sorted(valid_ts) if valid_ts is not None else None
    sys_path_repr = repr(sys.path)
    parquet_repr = repr(parquet_path)
    chunk_repr = repr(chunk)
    ts_repr = repr(valid_ts_list)
    min_bars_repr = repr(min_bars)

    script = "\n".join([
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
        "        pass",  # unparseable row (e.g. malformed velocity bytes) -- exclude, don't keep
        "print(json.dumps(ok))",
    ])

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120, env=env,
        )
    except subprocess.TimeoutExpired:
        if len(chunk) == 1:
            # A single row that still hangs at 120s is pathologically slow
            # (or stuck) to parse -- drop it rather than let it block the
            # whole shard indefinitely.
            return []
        mid = len(chunk) // 2
        return _probe_chunk(parquet_path, chunk[:mid], valid_ts, min_bars) + _probe_chunk(
            parquet_path, chunk[mid:], valid_ts, min_bars
        )

    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout.strip())
    if len(chunk) == 1:
        return []
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
    """Two-phase filter: fast metadata check then subprocess parse+TS validation."""
    import numpy as np

    ts_key = (
        hashlib.md5("|".join(sorted(valid_time_sigs)).encode()).hexdigest()[:8]
        if valid_time_sigs else "nots"
    )
    cache_file = cached_indices(
        file_cache_key(parquet_path, min_bars=min_bars, min_tracks=min_tracks, ts=ts_key)
    )

    if cache_file.exists():
        valid = np.load(cache_file).tolist()
        print(f"  dataset filter: {len(valid)} valid rows loaded from cache")
        return valid

    print("  dataset filter: phase 1 — metadata check…", flush=True)
    mask = _metadata_valid_mask(parquet_path, min_bars=min_bars, min_tracks=min_tracks)
    candidates = [i for i, v in enumerate(mask) if v]
    print(f"  dataset filter: {len(candidates)}/{len(mask)} pass metadata filter")

    print(f"  dataset filter: phase 2 — parse+TS validation ({len(candidates)} rows)…", flush=True)
    chunk_size = 500
    chunks = [candidates[s : s + chunk_size] for s in range(0, len(candidates), chunk_size)]
    valid: list[int] = []
    # Each _probe_chunk spawns a subprocess that reloads the shard, so run them
    # concurrently rather than one-at-a-time (the old sequential loop was the
    # main cost -- a 170k-row shard filtered ~112 chunks serially = hours).
    # ThreadPool is fine: the work is subprocess-bound (GIL released while each
    # child runs). Cap workers so we don't hold too many full-shard loads in
    # RAM at once (CHUNK_WORKERS env overrides; default min(8, cpus)).
    import concurrent.futures as _cf
    workers = int(os.environ.get("CHUNK_WORKERS", "0")) or min(8, os.cpu_count() or 4)
    workers = max(1, min(workers, len(chunks)))
    if workers == 1 or len(chunks) <= 1:
        for chunk in chunks:
            valid.extend(_probe_chunk(parquet_path, chunk, valid_time_sigs, min_bars))
    else:
        with _cf.ThreadPoolExecutor(max_workers=workers) as _ex:
            for res in _ex.map(
                lambda ch: _probe_chunk(parquet_path, ch, valid_time_sigs, min_bars), chunks
            ):
                valid.extend(res)
    valid.sort()

    dropped = len(candidates) - len(valid)
    print(f"  dataset filter: dropped {dropped} rows (parse errors / bad TS) — {len(valid)} kept")
    np.save(cache_file, np.array(valid, dtype=np.int64))
    return valid


def _resolve_paths(parquet_path: str | list[str]) -> list[str]:
    """Expand globs and return a flat list of parquet file paths."""
    if isinstance(parquet_path, str):
        parquet_path = [parquet_path]
    paths: list[str] = []
    for p in parquet_path:
        expanded = sorted(_glob_mod.glob(p)) if "*" in p else [p]
        paths.extend(expanded)
    return paths


# ---------------------------------------------------------------------------
# Base class — shared pipeline for all MIDI-GPT training datasets
# ---------------------------------------------------------------------------


class MidiGPTDatasetBase:
    """Base dataset with the shared encode/label pipeline.

    Subclasses override:
      - ``_load_score(idx)``         → Score (from parquet, in-memory, etc.)
      - ``prepare_score(score, …)``  → (score, is_infill, infill_bars, **extra)
        Called after window selection.  Adds audio-track assignment, feature
        extraction, pitch augmentation, etc.  Must return the (possibly
        mutated) score plus any extra keyword args to pass to ``_encode``.
      - ``_encode(…)``              → dict  (token_ids, labels, extra fields)
        Called after encoding.  Adds audio_bar_positions, context frames, etc.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_bars_choices: list[int],
        infill_prob: float,
        infill_bar_fraction: float,
        max_seq_len: int,
        max_tracks: int,
        min_tracks: int,
        min_fill_ratio: float,
        window_selection: str,
        temperature: float,
        mask_bar_config: MaskBarConfig | None = None,
        velocity_off_prob: float = 0.0,
        microtiming_off_prob: float = 0.0,
        genre_probability: float = 1.0,
        transpose_range: tuple[int, int] = (-6, 6),
    ):
        self._tokenizer = tokenizer
        self._num_bars_choices = num_bars_choices
        self._infill_prob = infill_prob
        self._infill_frac = infill_bar_fraction
        self._max_seq_len = max_seq_len
        self._max_tracks = max_tracks
        self._min_tracks = min_tracks
        self._min_fill_ratio = min_fill_ratio
        self._window_selection = window_selection
        self._temperature = temperature
        self._mask_cfg = mask_bar_config
        self._velocity_off_prob = velocity_off_prob
        self._microtyping_off_prob = microtiming_off_prob
        self._genre_probability = genre_probability
        self._transpose_range = transpose_range
        self._velocity_range = (0.8, 1.2)

        cfg_dict = json.loads(tokenizer._vocab.config().to_json())
        raw_map = cfg_dict.get("num_bars_map") or [4]
        self._num_bars_choices = sorted(set(int(x) for x in raw_map), reverse=True)

        self._switchable_velocity = bool(cfg_dict.get("switchable_velocity", False))
        self._switchable_microtiming = bool(cfg_dict.get("switchable_microtiming", False))
        self._genre_grouping = tokenizer._vocab.config().genre_grouping

    # --- Abstract methods (subclasses must implement) ---

    def _load_score(self, idx: int) -> Score:
        raise NotImplementedError

    def prepare_score(
        self, score: Score, is_infill: bool, infill_bar_indices: set[int], idx: int,
        *, semitones: int = 0, start_bar: int = 0, original_track_indices: list[int] | None = None,
    ) -> tuple[Score, set[int], dict]:
        """Hook for subclasses to augment/mutate the score before encoding.

        ``semitones`` is the transpose shift already applied to the full score
        before windowing; ``start_bar``/``original_track_indices`` locate this
        window within the pre-transpose, pre-windowing original score (the
        bar offset and the original score's track indices, in the order they
        appear in ``score``). Subclasses that need to align a paired modality
        (audio stems, pre-synthesized renders) with the resulting window use
        these to crop/pitch-shift a full-track render instead of re-deriving
        it from scratch.

        Returns (score, infill_bar_indices, extra) where extra is merged into
        the output dict.  Default: no-op.
        """
        return score, infill_bar_indices, {}

    def _encode_hook(
        self, token_ids: list[int], is_infill: bool, idx: int, **extra,
    ) -> dict:
        """Hook for subclasses to add audio_bar_positions, context frames, etc.

        Called after encoding and label creation.  Return a dict of extra
        fields to merge into the output.  Default: no-op.
        """
        return {}

    # --- Shared pipeline ---

    def _getitem_inner(self, idx: int) -> dict:
        score = self._load_score(idx)
        # _load_score returns a reference into an in-memory list shared with
        # sibling DataLoader workers via fork()'s copy-on-write. Transpose/
        # VelocityScale/prepare_score all mutate notes/tracks in place, so
        # without this copy every access permanently dirties that object's
        # pages in the touching worker, defeating CoW sharing. With
        # shuffle=True and many workers, that converts progressively more of
        # the shared dataset into private per-worker copies over the course
        # of training -- steady, unbounded RSS growth independent of any
        # single sample's size (see amber OOMs 2026-08-28, ~2.85GB/min,
        # identical across all training modes and mem limits).
        score = copy.deepcopy(score)

        semitones = random.choice(range(self._transpose_range[0], self._transpose_range[1] + 1))
        score = Transpose(semitones)(score)
        score = VelocityScale(self._velocity_range)(score)

        is_infill = random.random() < self._infill_prob
        do_mask = self._mask_cfg is not None

        # --- Window selection ---
        # start_bar/original_track_indices locate the window within the
        # pre-transpose, pre-windowing score, so paired modalities (audio
        # renders) can be cropped from a cached full-track render.
        window = None
        start_bar = 0
        original_track_indices: list[int] | None = None
        if self._window_selection in ("uniform", "weighted"):
            result = select_window_dispatch(
                score,
                mode=self._window_selection,
                num_bars_choices=self._num_bars_choices,
                max_tracks=self._max_tracks,
                min_tracks=self._min_tracks,
                min_fill_ratio=self._min_fill_ratio,
                temperature=self._temperature,
                return_metadata=True,
            )
            if result is not None:
                window, start_bar, original_track_indices = result
        if window is None:
            initial = random.choice(self._num_bars_choices)
            fallbacks = sorted([n for n in self._num_bars_choices if n < initial], reverse=True)
            for n_bars in [initial, *fallbacks]:
                for n_tracks in range(self._max_tracks, self._min_tracks - 1, -1):
                    result = select_window(
                        score, n_bars, n_tracks, self._min_fill_ratio, return_metadata=True
                    )
                    if result is not None:
                        window, start_bar, original_track_indices = result
                        break
                if window is not None:
                    break
        if window is None:
            result = select_window(
                score, self._num_bars_choices[-1], self._min_tracks, self._min_fill_ratio,
                return_metadata=True,
            )
            if result is not None:
                window, start_bar, original_track_indices = result
        if window is None:
            raise ValueError(f"Sample {idx}: no valid window found")

        num_bars = max((len(t.bars) for t in window.tracks), default=0)

        # --- Infill bar selection ---
        infill_bar_indices: set[int] = set()
        if is_infill:
            p = random.uniform(0.0, self._infill_frac)
            infill_bar_indices = {
                b_idx
                for b_idx in range(num_bars)
                if random.random() < p
            }
            if not infill_bar_indices:
                infill_bar_indices = {random.randrange(num_bars)}

        # --- Subclass hook: prepare score (audio assignment, features, etc.) ---
        window, infill_bar_indices, extra = self.prepare_score(
            window, is_infill, infill_bar_indices, idx,
            semitones=semitones, start_bar=start_bar, original_track_indices=original_track_indices,
        )

        if do_mask:
            window = MaskBar(self._mask_cfg, infill_bars=infill_bar_indices)(window)

        # --- Encode ---
        encode_opts = _core.EncodeOptions()
        if is_infill and infill_bar_indices:
            encode_opts.multi_fill = {
                (t_idx, b_idx)
                for t_idx in range(len(window.tracks))
                for b_idx in infill_bar_indices
            }
        if self._switchable_velocity and self._velocity_off_prob > 0.0:
            encode_opts.use_velocity = 0 if random.random() < self._velocity_off_prob else 1
        if self._switchable_microtiming and self._microtyping_off_prob > 0.0:
            encode_opts.use_microtiming = 0 if random.random() < self._microtyping_off_prob else 1
        encode_opts.genre = self._genre_token(idx)
        token_ids = self._tokenizer.encode(window, opts=encode_opts)

        if is_infill and len(token_ids) > self._max_seq_len:
            raise ValueError(
                f"Sample {idx}: infill tokens exceed max_seq_len "
                f"({len(token_ids)} > {self._max_seq_len})"
            )

        # --- Labels ---
        labels = self._create_labels(token_ids, is_infill, extra.get("has_audio_tracks", False))

        # --- Subclass hook: audio positions, context frames, etc. ---
        extra_fields = self._encode_hook(token_ids, is_infill, idx, **extra)

        token_ids = token_ids[: self._max_seq_len]
        labels = labels[: self._max_seq_len]

        result = {
            "input_ids": token_ids,
            "labels": labels,
            "is_infill": is_infill,
        }
        result.update(extra_fields)
        return result

    def _create_labels(
        self, token_ids: list[int], is_infill: bool, has_audio_tracks: bool,
    ) -> list[int]:
        """Create labels: -100 for context/soft tokens, active for loss targets.

        When has_audio_tracks is True, audio track tokens are masked to -100
        since audio tracks are context-only and never generation targets.
        """
        if not has_audio_tracks:
            return list(token_ids)

        labels = [-100] * len(token_ids)
        vocab = self._tokenizer._vocab
        # A track is AUDIO (context) iff its token span contains an
        # [AudioContext] placeholder. The encoder marks audio tracks by EMITTING
        # AudioContext tokens, NOT by a distinct Track value — in this config the
        # Track domain is only {melodic, drum}, there is no "audio" Track value.
        # (Verified against a real amber training example.) So identify audio
        # spans by AudioContext presence and mask them entirely to -100 (pure
        # context, never a generation target); everything else — the MIDI target
        # tracks and the piece header — is a generation target and gets loss.
        track_start_ids = {
            vocab.encode_val(_core.TokenType.Track, v)
            for v in range(vocab.domain_size(_core.TokenType.Track))
        }
        track_end_token = (
            vocab.encode_val(_core.TokenType.TrackEnd, 0)
            if vocab.has(_core.TokenType.TrackEnd) else None
        )
        audio_ctx_id = (
            vocab.encode_val(_core.TokenType.AudioContext, 0)
            if vocab.has(_core.TokenType.AudioContext) else None
        )

        if is_infill:
            # NOTE: infill is DISABLED for the extractor sweep (infill_prob=0),
            # so this branch is unreachable during the sweep. It still has the
            # known audio-leak bug (audio tracks' fill-bar notes get labeled +
            # emitted); fixing infill is deferred until after the sweep. Do NOT
            # re-enable infill without excluding audio tracks from multi_fill and
            # masking audio fill blocks here.
            if vocab.has(_core.TokenType.FillInStart):
                fill_start_token = vocab.encode_val(_core.TokenType.FillInStart, 0)
                try:
                    fill_start_idx = token_ids.index(fill_start_token)
                    for i in range(fill_start_idx + 1, len(token_ids)):
                        labels[i] = token_ids[i]
                except ValueError:
                    pass
            return labels

        # Autoregressive: delimit track spans [Track ... TrackEnd], then mask any
        # span that contains an AudioContext token.
        spans: list[tuple[int, int]] = []
        start = None
        for i, tok in enumerate(token_ids):
            if tok in track_start_ids:
                start = i
            elif tok == track_end_token and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:  # final track span without a trailing TrackEnd
            spans.append((start, len(token_ids) - 1))

        audio_span = [False] * len(token_ids)
        if audio_ctx_id is not None:
            for s, e in spans:
                if any(token_ids[k] == audio_ctx_id for k in range(s, e + 1)):
                    for k in range(s, e + 1):
                        audio_span[k] = True

        for i, tok in enumerate(token_ids):
            if not audio_span[i]:
                labels[i] = tok
        return labels

    def _genre_token(self, idx: int) -> int:
        """Return genre token id, or -1 if none/disabled."""
        if self._genre_grouping is None or self._genre_probability <= 0.0:
            return -1
        if random.random() > self._genre_probability:
            return -1
        try:
            genre = self._data[idx].get("music_style_scraped", "")
            if genre and self._genre_grouping.contains(genre):
                return self._genre_grouping.encode(genre)
        except Exception:
            pass
        return -1


# ---------------------------------------------------------------------------
# Parquet dataset (GigaMIDI / similar)
# ---------------------------------------------------------------------------


class MidiGPTDataset(MidiGPTDatasetBase):
    """MIDI-only training dataset loaded from parquet."""

    def __init__(
        self,
        parquet_path: str | list[str],
        tokenizer: Tokenizer,
        infill_probability: float = 0.75,
        infill_bar_fraction: float = 0.5,
        mask_bar_config: MaskBarConfig | None = None,
        max_seq_len: int = 2048,
        max_tracks: int = 12,
        min_tracks: int = 1,
        min_fill_ratio: float = 0.75,
        window_selection: str = "fixed",
        temperature: float = 1.0,
        velocity_off_probability: float = 0.0,
        microtiming_off_probability: float = 0.0,
        genre_probability: float = 1.0,
    ):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]") from None

        paths = _resolve_paths(parquet_path)
        self._data = hf.load_dataset("parquet", data_files=paths, split="train")
        self._encoder_config_json: str = tokenizer._vocab.config().to_json()

        if infill_probability > 0 and not json.loads(self._encoder_config_json).get("supports_infill", False):
            raise ValueError(
                f"infill_probability={infill_probability} > 0 but the encoder config "
                f"has supports_infill=false"
            )
        if mask_bar_config is not None and not tokenizer._vocab.has(_core.TokenType.MaskBar):
            raise ValueError("mask_bar_config is set but encoder vocab has no MaskBar token")
        if velocity_off_probability > 0.0 and not json.loads(self._encoder_config_json).get("switchable_velocity", False):
            raise ValueError("velocity_off_probability > 0 but encoder has switchable_velocity=false")
        if microtiming_off_probability > 0.0 and not json.loads(self._encoder_config_json).get("switchable_microtiming", False):
            raise ValueError("microtiming_off_probability > 0 but encoder has switchable_microtiming=false")

        cfg_dict = json.loads(tokenizer._vocab.config().to_json())
        num_bars_choices = sorted(set(int(x) for x in (cfg_dict.get("num_bars_map") or [4])), reverse=True)

        ts_list = cfg_dict.get("time_signatures")
        valid_ts: frozenset[str] | None = frozenset(ts_list) if ts_list else None

        all_valid: list[int] = []
        offset = 0
        for path in paths:
            n = pq.read_metadata(path).num_rows
            local_valid = _load_or_build_valid_indices(
                path, min_bars=num_bars_choices[-1], min_tracks=min_tracks, valid_time_sigs=valid_ts,
            )
            all_valid.extend(i + offset for i in local_valid)
            offset += n
        self._data = self._data.select(all_valid)

        super().__init__(
            tokenizer=tokenizer,
            num_bars_choices=num_bars_choices,
            infill_prob=infill_probability,
            infill_bar_fraction=infill_bar_fraction,
            max_seq_len=max_seq_len,
            max_tracks=max_tracks,
            min_tracks=min_tracks,
            min_fill_ratio=min_fill_ratio,
            window_selection=window_selection,
            temperature=temperature,
            mask_bar_config=mask_bar_config,
            velocity_off_prob=velocity_off_probability,
            microtiming_off_prob=microtiming_off_probability,
            genre_probability=genre_probability,
        )

        self._stash_cache: dict[int, dict] = {}
        self._ensure_tokenizer = lambda: None  # no-op; tokenizer is set in __init__

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict | None:
        try:
            return self._getitem_inner(idx)
        except Exception:
            return None

    def _load_score(self, idx: int) -> Score:
        score = Score.from_bytes(self._data[idx]["music"])
        self._stash_nomml(score, idx)
        return score

    def _stash_nomml(self, score: Score, idx: int) -> None:
        if idx in self._stash_cache:
            values = self._stash_cache[idx]
        else:
            row = self._data[idx]
            values = {}
            for key in ("NOMML", "nomml", "nomml_raw"):
                if key in row and row[key] is not None:
                    values = row[key] if isinstance(row[key], dict) else {}
                    break
            self._stash_cache[idx] = values
        for t_idx, track in enumerate(score.tracks):
            v = values.get(t_idx, values.get(str(t_idx)))
            if v is not None and v != -1:
                track.attributes["_nomml_raw"] = int(v)
