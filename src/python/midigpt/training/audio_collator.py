"""Batch collation for variable-length audio-conditioned MIDI examples."""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)


class AudioConditionedCollator:
    """Pad token sequences while retaining per-example variable audio data.

    Handles both MIDI-only examples (no audio fields) and audio-conditioned
    examples (with audio_waveforms, audio_bar_positions, etc.). Audio feature
    extraction itself happens later, batched, on GPU inside
    AudioConditionedLightningModule._compute_loss — this collator only ever
    handles raw waveform tensors and metadata, never extracted features.
    """

    def __init__(self, pad_token_id: int = 0, max_seq_len: int = 2048):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, examples: list[dict[str, Any] | None]) -> dict[str, Any] | None:
        valid = [x for x in examples if x is not None]
        if not valid:
            # Every sample in this draw failed dataset-side (e.g. no valid window found)
            # — a real possibility whenever per-sample rejection rate is high, not a
            # crash-worthy condition. Signal "skip this batch" up to the training loop
            # instead of killing the whole run; AudioConditionedLightningModule's
            # training_step/validation_step check for None and return None themselves,
            # which Lightning treats as "skip this batch" (no backward/optimizer step).
            log.warning("Skipping a fully-empty batch (%d/%d samples failed).", len(examples), len(examples))
            return None
        length = min(self.max_seq_len, max(len(x["input_ids"]) for x in valid))
        input_ids = torch.full((len(valid), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(valid), length), -100, dtype=torch.long)
        for index, example in enumerate(valid):
            ids = example["input_ids"]
            lab = example["labels"]
            # Convert lists to tensors if needed
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab, dtype=torch.long)
            n = min(length, len(ids))
            input_ids[index, :n] = ids[:n]
            labels[index, :n] = lab[:n]

            # Audio fields — absent for MIDI-only examples. audio_bar_positions and
            # audio_bar_refs are parallel, one entry per (track, bar); truncating the
            # token sequence to max_seq_len can drop trailing audio bars, so both get
            # filtered together. audio_waveforms/audio_track_indices/bar_intervals are
            # per-track (not per-bar) and pass through whole — a track whose every bar
            # got truncated away just goes unused downstream, which is harmless.
            audio_positions = example.get("audio_bar_positions", [])
            audio_refs = example.get("audio_bar_refs", [])

            n_audio = min(len(audio_positions), len(audio_refs))
            keep = [i for i in range(n_audio) if audio_positions[i] < length]
            audio_positions = [audio_positions[i] for i in keep]
            audio_refs = [audio_refs[i] for i in keep]

            example["audio_bar_positions"] = audio_positions
            example["audio_bar_refs"] = audio_refs

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_waveforms": [x.get("audio_waveforms", []) for x in valid],
            "audio_track_indices": [x.get("audio_track_indices", []) for x in valid],
            "bar_intervals": [x.get("bar_intervals", []) for x in valid],
            "audio_bar_positions": [x.get("audio_bar_positions", []) for x in valid],
            "audio_bar_refs": [x.get("audio_bar_refs", []) for x in valid],
            "is_infill": [x.get("is_infill", False) for x in valid],
        }
