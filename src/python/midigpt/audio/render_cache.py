"""On-disk cache for pre-synthesized SoundFont renders.

Keyed by (score content hash, original track index, soundfont name), storing
the *full*, untransposed track render — not a specific training window. This
is what makes the cache reusable across the effectively-infinite random
(window, transpose) draws ``AudioTransitionDataset`` makes at training time:
each draw crops the cached full-track waveform to its bar range (using the
score's bar_intervals, saved alongside) and pitch-shifts it by whatever
semitone amount was drawn, mirroring how ``SlakhPairedStemDataset`` already
handles pre-rendered stems in ``_extract_slakh_audio_features``.

Populated offline by ``scripts/presynthesize_audio_cache.py`` and consulted
at training time by ``midigpt.training.audio_dataset`` — this is the same
cache in both cases, not two separate mechanisms.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


class AudioRenderCache:
    """Stores/retrieves pre-synthesized mono waveforms as float16 tensors on disk.

    Layout::

        cache_dir/{score_id}/bar_intervals.json         # list[[start_sec, end_sec], ...]
        cache_dir/{score_id}/track{track_idx}__{soundfont_stem}.pt
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def _score_dir(self, score_id: str) -> Path:
        return self.cache_dir / str(score_id)

    def _waveform_path(self, score_id: str, track_idx: int, soundfont_name: str) -> Path:
        stem = Path(soundfont_name).stem
        return self._score_dir(score_id) / f"track{track_idx}__{stem}.pt"

    def _bar_intervals_path(self, score_id: str) -> Path:
        return self._score_dir(score_id) / "bar_intervals.json"

    # --- waveforms ---

    def has(self, score_id: str, track_idx: int, soundfont_name: str) -> bool:
        return self._waveform_path(score_id, track_idx, soundfont_name).exists()

    def load(self, score_id: str, track_idx: int, soundfont_name: str) -> torch.Tensor | None:
        """Load a cached full-track waveform, or None on a cache miss/read failure."""
        p = self._waveform_path(score_id, track_idx, soundfont_name)
        if not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu").float()
        except Exception:
            return None

    def save(self, score_id: str, track_idx: int, soundfont_name: str, waveform: torch.Tensor) -> None:
        """Write a waveform atomically (write to a temp file, then rename)."""
        p = self._waveform_path(score_id, track_idx, soundfont_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
        torch.save(waveform.detach().to(torch.float16).contiguous(), tmp)
        tmp.replace(p)

    # --- bar intervals (one per score, shared by every track/soundfont) ---

    def has_bar_intervals(self, score_id: str) -> bool:
        return self._bar_intervals_path(score_id).exists()

    def load_bar_intervals(self, score_id: str) -> list[tuple[float, float]] | None:
        p = self._bar_intervals_path(score_id)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                data = json.load(f)
            return [(float(s), float(e)) for s, e in data]
        except Exception:
            return None

    def save_bar_intervals(self, score_id: str, intervals: list[tuple[float, float]]) -> None:
        p = self._bar_intervals_path(score_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump([[s, e] for s, e in intervals], f)
        tmp.replace(p)
