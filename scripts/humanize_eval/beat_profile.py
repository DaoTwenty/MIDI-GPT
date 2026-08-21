"""Per-beat-position velocity/microtiming profile -- shared between E0/E4
here and the style-conditioning plan's metric 2
(scripts/style_prototype/eval/style_transfer_faithfulness.py), per
EXPERIMENT_PLAN.md section E's "share the per-beat-position profile
implementation" note.

Buckets notes by their bar-relative onset position into `n_bins` sub-beat
bins and reports, per bin: mean velocity, and mean |microtiming offset| --
the decoded note's own `delta` residual (see tokenizer.py::resample_delta's
docstring: delta is a signed microtiming residual in units of 1/resolution
of one grid cell), normalized by the score's resolution. `profile_points`'s
(position, velocity) semantics are unchanged from e4_context_probes.py's
original inline version -- this is a superset (adds the delta field), not a
behavior change, so e4 can adopt this module as a drop-in later without
altering its already-logged results.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

N_PROFILE_BINS = 12  # matches e4_context_probes.py's existing default


def profile_points(score, bars_of_interest: set[int]) -> list[tuple[float, float, float]]:
    """(bar-relative position in [0, 1), velocity, |delta| / resolution)
    per note in the given bars."""
    pts = []
    res = max(1, score.resolution)
    for b, bar in enumerate(score.tracks[0].bars):
        if b not in bars_of_interest:
            continue
        bar_len = bar.beat_length * score.resolution
        for n in bar.notes:
            pos = (n.onset_ticks / bar_len) if bar_len > 0 else 0.0
            pts.append((pos, n.velocity, abs(n.delta) / res))
    return pts


def _bucket(pts, value_idx: int, n_bins: int) -> list[float | None]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for pt in pts:
        idx = min(n_bins - 1, int(pt[0] * n_bins))
        buckets[idx].append(pt[value_idx])
    return [statistics.mean(buckets[i]) if buckets[i] else None for i in range(n_bins)]


def velocity_profile_vector(pts, n_bins: int = N_PROFILE_BINS) -> list[float | None]:
    return _bucket(pts, 1, n_bins)


def timing_profile_vector(pts, n_bins: int = N_PROFILE_BINS) -> list[float | None]:
    return _bucket(pts, 2, n_bins)


def cosine_similarity(a: list[float | None], b: list[float | None]) -> float | None:
    """None-skipping cosine similarity over paired bins; None if fewer than
    2 bins have real values in both vectors, or either sub-vector is all
    zeros."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    dot = sum(x * y for x, y in pairs)
    norm_a = sum(x * x for x, _ in pairs) ** 0.5
    norm_b = sum(y * y for _, y in pairs) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)
