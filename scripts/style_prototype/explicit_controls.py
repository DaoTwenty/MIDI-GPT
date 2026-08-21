"""Variant C: explicit, hand-designed control statistics as a proxy for
performance style -- contrasted against Variant A/B's learned StyleEncoder.

Operates on the exact same reference-segment raw token ids MidiGPTDataset's
`style_ref_mask` already isolates for A/B (see dataset.py's `_encode_one` /
`_expressive_mask`) -- no new data-pipeline plumbing needed. Instead of a
small transformer compressing that sequence into a black-box z, this computes
a handful of interpretable summary statistics directly from the raw
VelocityLevel/DeltaDirection/Delta token ids:

  mean_velocity   -- overall loudness level (mean VelocityLevel, normalized)
  velocity_std    -- dynamic variability (how much loudness varies)
  mean_delta_mag  -- mean magnitude of timing deviation ("looseness")
  timing_bias     -- mean DeltaDirection, recoded to [-1, +1]: ahead-of-beat
                     vs behind-the-beat tendency (independent of magnitude)

Each is quantized into N_BINS levels; ExplicitControlsGPT2 embeds each
control's bin via its own small lookup table -- literally interpretable
scalars, not a learned high-capacity encoder, which is the whole point of
this variant.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import midigpt._core as _core

CONTROL_NAMES = ("mean_velocity", "velocity_std", "mean_delta_mag", "timing_bias")
N_CONTROLS = len(CONTROL_NAMES)
N_BINS = 8


@dataclass
class _TypeRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class ControlTypeRanges:
    """Raw-token-id ranges for VelocityLevel/DeltaDirection/Delta, read once
    from a Vocabulary -- lets us decode a raw token id back to its
    within-type index (0..domain_size-1) directly, unlike StyleVocab's
    combined local re-indexing (which discards which type a local id came
    from, by design -- fine for a learned encoder's embedding table, not for
    computing typed statistics)."""

    def __init__(self, vocab: "_core.Vocabulary"):
        self.velocity = self._range(vocab, _core.TokenType.VelocityLevel)
        self.delta_direction = self._range(vocab, _core.TokenType.DeltaDirection)
        self.delta = self._range(vocab, _core.TokenType.Delta)

    @staticmethod
    def _range(vocab, tt) -> "_TypeRange | None":
        if not vocab.has(tt):
            return None
        s, e = vocab.range(tt)
        return _TypeRange(s, e)


def _local_index(raw_id: int, r: "_TypeRange | None") -> int | None:
    if r is None or not (r.start <= raw_id < r.end):
        return None
    return raw_id - r.start


def compute_controls(raw_token_ids: list[int], ranges: ControlTypeRanges) -> list[float] | None:
    """Returns 4 floats (mean_velocity/velocity_std/mean_delta_mag in
    [0, 1], timing_bias in [-1, 1]), or None if the segment carries no
    velocity/delta tokens at all (caller falls back to "no reference
    available", same as A/B's has_style=False path)."""
    vel_vals: list[float] = []
    delta_mag_vals: list[float] = []
    direction_vals: list[float] = []

    for raw in raw_token_ids:
        idx = _local_index(raw, ranges.velocity)
        if idx is not None:
            vel_vals.append(idx / max(1, ranges.velocity.size - 1))
            continue
        idx = _local_index(raw, ranges.delta)
        if idx is not None:
            delta_mag_vals.append(idx / max(1, ranges.delta.size - 1))
            continue
        idx = _local_index(raw, ranges.delta_direction)
        if idx is not None:
            # size-2 domain -> {0, 1}; recode onto {-1.0, +1.0}.
            direction_vals.append(-1.0 if idx == 0 else 1.0)

    if not vel_vals and not delta_mag_vals:
        return None

    mean_velocity = statistics.mean(vel_vals) if vel_vals else 0.5
    velocity_std = statistics.pstdev(vel_vals) if len(vel_vals) > 1 else 0.0
    mean_delta_mag = statistics.mean(delta_mag_vals) if delta_mag_vals else 0.0
    timing_bias = statistics.mean(direction_vals) if direction_vals else 0.0

    return [mean_velocity, velocity_std, mean_delta_mag, timing_bias]


def quantize_controls(values: list[float], n_bins: int = N_BINS) -> list[int]:
    """values must be in the order CONTROL_NAMES produces them (see
    compute_controls) -- timing_bias (index 3) is the only one in [-1, 1],
    the rest are in [0, 1]."""
    bins = []
    for i, v in enumerate(values):
        lo, hi = (-1.0, 1.0) if i == 3 else (0.0, 1.0)
        frac = (v - lo) / (hi - lo)
        bins.append(max(0, min(n_bins - 1, int(frac * n_bins))))
    return bins
