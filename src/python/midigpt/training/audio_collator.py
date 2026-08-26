"""Batch collation for variable-length audio-conditioned MIDI examples."""

from __future__ import annotations

from typing import Any

import torch


class AudioConditionedCollator:
    """Pad token sequences while retaining per-example variable audio frames.

    Handles both MIDI-only examples (no audio fields) and audio-conditioned
    examples (with context_bar_frames, audio_bar_positions, etc.).
    """

    def __init__(self, pad_token_id: int = 0, max_seq_len: int = 2048):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, examples: list[dict[str, Any] | None]) -> dict[str, Any]:
        valid = [x for x in examples if x is not None]
        if not valid:
            raise ValueError("Cannot collate an empty batch (all samples failed).")
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

            # Audio fields — absent for MIDI-only examples
            audio_positions = example.get("audio_bar_positions", [])
            ctx_bar = example.get("context_bar_frames", [])
            ctx_track = example.get("context_track_frames", [])
            audio_refs = example.get("audio_bar_refs", [])

            n_audio = min(len(audio_positions), len(ctx_bar), len(ctx_track), len(audio_refs))
            keep = [i for i in range(n_audio) if audio_positions[i] < length]
            ctx_bar = [ctx_bar[i] for i in keep]
            ctx_track = [ctx_track[i] for i in keep]
            audio_positions = [audio_positions[i] for i in keep]
            audio_refs = [audio_refs[i] for i in keep]

            example["context_bar_frames"] = ctx_bar
            example["context_track_frames"] = ctx_track
            example["audio_bar_positions"] = audio_positions
            example["audio_bar_refs"] = audio_refs

        return {
            "input_ids": input_ids,
            "labels": labels,
            "context_bar_frames": [x.get("context_bar_frames", []) for x in valid],
            "context_track_frames": [x.get("context_track_frames", []) for x in valid],
            "audio_bar_positions": [x.get("audio_bar_positions", []) for x in valid],
            "audio_bar_refs": [x.get("audio_bar_refs", []) for x in valid],
            "is_infill": [x.get("is_infill", False) for x in valid],
        }
