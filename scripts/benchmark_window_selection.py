#!/usr/bin/env python3
"""Benchmark and compare the three window selection algorithms.

Usage:
    python scripts/benchmark_window_selection.py [--rows N] [--tracks N] [--iter N]

Loads GigaMIDI parquet (or generates synthetic scores), runs each algorithm
many times, and reports:
  - Wall-clock time per call
  - Distribution of selected n_bars, n_tracks, start position
  - Distribution of quality (total notes in window)
  - Distribution of note density (notes per bar per track)
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import midigpt._core as _core
from midigpt._types import Bar, Note, Score, Track
from midigpt.augmentation.score_window import (
    select_window,
    select_window_uniform,
    select_window_weighted,
)

NUM_BARS_CHOICES = list(range(4, 68, 4))  # [4, 8, ..., 64]


def _make_note(pitch: int = 60, onset: int = 0, duration: int = 120) -> Note:
    return Note(pitch=pitch, onset_ticks=onset, duration_ticks=duration, velocity=80)


def _make_bar(n_notes: int = 5, pitch_start: int = 60) -> Bar:
    notes = [_make_note(pitch=pitch_start + i * 2, onset=i * 24) for i in range(n_notes)]
    return Bar(notes=notes, ts_numerator=4, ts_denominator=4)


def _empty_bar() -> Bar:
    return Bar(notes=[], ts_numerator=4, ts_denominator=4)


def _synthetic_score(n_tracks: int = 4, n_bars: int = 32, density: float = 0.7) -> Score:
    """Generate a synthetic score where `density` controls the fraction of
    non-empty bars per track.  Active bars have 3-8 notes with varying pitch."""
    import random
    rng = random.Random(42)
    tracks = []
    for t in range(n_tracks):
        bars = []
        for b in range(n_bars):
            if rng.random() < density:
                n_notes = rng.randint(3, 8)
                pitch_base = 48 + t * 12 + rng.randint(0, 6)
                bars.append(_make_bar(n_notes, pitch_base))
            else:
                bars.append(_empty_bar())
        tracks.append(Track(bars=bars, track_type="melodic"))
    return Score(tracks=tracks, resolution=12)


def _load_gigamidi_score(parquet_path: str, idx: int = 0) -> Score | None:
    """Load a single score from GigaMIDI parquet."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=["music"])
        row = table.slice(idx, 1).to_pydict()
        return Score.from_bytes(row["music"][0])
    except Exception as e:
        print(f"  Could not load parquet: {e}")
        return None


def _window_stats(window: Score) -> dict:
    """Compute statistics for a selected window."""
    n_bars = max((len(t.bars) for t in window.tracks), default=0)
    n_tracks = len(window.tracks)
    total_notes = sum(len(b.notes) for t in window.tracks for b in t.bars)
    notes_per_bar = total_notes / max(n_bars, 1)
    notes_per_track = total_notes / max(n_tracks, 1)
    return {
        "n_bars": n_bars,
        "n_tracks": n_tracks,
        "total_notes": total_notes,
        "notes_per_bar": notes_per_bar,
        "notes_per_track": notes_per_track,
    }


def _print_dist(label: str, values: list[int | float], fmt: str = ".1f"):
    """Print distribution stats for a list of values."""
    if not values:
        print(f"  {label}: (no data)")
        return
    print(
        f"  {label}: min={min(values)}, "
        f"median={statistics.median(values):{fmt}}, "
        f"mean={statistics.mean(values):{fmt}}, "
        f"max={max(values)}, "
        f"stdev={statistics.stdev(values):{fmt}}" if len(values) > 1
        else f"  {label}: {values[0]:{fmt}}"
    )


def benchmark(
    name: str,
    func,
    scores: list[Score],
    n_iters: int,
    kwargs: dict,
) -> dict:
    """Run a window selection function many times and collect stats."""
    all_stats: list[dict] = []
    nulls = 0
    times: list[float] = []

    for _ in range(n_iters):
        for score in scores:
            t0 = time.perf_counter()
            result = func(score, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if result is None:
                nulls += 1
            else:
                window = result[0] if isinstance(result, tuple) else result
                all_stats.append(_window_stats(window))

    n_total = n_iters * len(scores)
    n_bars_list = [s["n_bars"] for s in all_stats]
    n_tracks_list = [s["n_tracks"] for s in all_stats]
    total_notes_list = [s["total_notes"] for s in all_stats]
    npb_list = [s["notes_per_bar"] for s in all_stats]

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Calls: {n_total}, Windows: {len(all_stats)}, Nulls: {nulls} ({100*nulls/max(n_total,1):.1f}%)")
    print(f"  Time:  mean={statistics.mean(times)*1000:.2f}ms, "
          f"median={statistics.median(times)*1000:.2f}ms, "
          f"total={sum(times):.3f}s")
    _print_dist("n_bars", n_bars_list, fmt=".0f")
    _print_dist("n_tracks", n_tracks_list, fmt=".0f")
    _print_dist("total_notes", total_notes_list, fmt=".0f")
    _print_dist("notes/bar", npb_list)
    return {
        "name": name,
        "n_total": n_total,
        "n_nulls": nulls,
        "times_ms": [t * 1000 for t in times],
        "n_bars": n_bars_list,
        "n_tracks": n_tracks_list,
        "total_notes": total_notes_list,
        "notes_per_bar": npb_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark window selection algorithms")
    parser.add_argument("--rows", type=int, default=20, help="Number of synthetic scores")
    parser.add_argument("--tracks", type=int, default=6, help="Tracks per score")
    parser.add_argument("--bars", type=int, default=64, help="Bars per score")
    parser.add_argument("--iter", type=int, default=50, help="Iterations per score")
    parser.add_argument("--gigamidi", type=str, default=None, help="Path to GigaMIDI parquet")
    parser.add_argument("--gigamidi-idx", type=int, default=0, help="Starting row index in parquet")
    parser.add_argument("--gigamidi-count", type=int, default=10, help="How many GigaMIDI scores to load")
    args = parser.parse_args()

    # Build score pool
    scores: list[Score] = []
    if args.gigamidi:
        path = args.gigamidi
        print(f"Loading {args.gigamidi_count} scores from {path}...")
        for i in range(args.gigamidi_count):
            s = _load_gigamidi_score(path, idx=args.gigamidi_idx + i)
            if s is not None:
                scores.append(s)
                print(f"  [{i}] {len(s.tracks)} tracks, {min(len(t.bars) for t in s.tracks)} bars")
        if not scores:
            print("  No scores loaded, falling back to synthetic")
    
    if not scores:
        print(f"Generating {args.rows} synthetic scores ({args.tracks} tracks, {args.bars} bars)...")
        for i in range(args.rows):
            # Vary density across scores
            density = 0.3 + (i % 5) * 0.15  # 0.3, 0.45, 0.6, 0.75, 0.9
            scores.append(_synthetic_score(args.tracks, args.bars, density))
            print(f"  [{i}] density={density:.2f}")

    max_tracks = max(len(s.tracks) for s in scores)
    n_iters = args.iter

    common_kwargs = dict(
        num_bars_choices=NUM_BARS_CHOICES,
        max_tracks=max_tracks,
        min_tracks=1,
        min_fill_ratio=0.75,
    )

    # --- Original: iterate over n_bars/n_tracks (simulates current caller) ---
    def original(score, **_kw):
        initial = random.choice(NUM_BARS_CHOICES)
        fallbacks = sorted([n for n in NUM_BARS_CHOICES if n < initial], reverse=True)
        for n_bars in [initial, *fallbacks]:
            for n_tracks in range(max_tracks, 0, -1):
                w = select_window(score, n_bars, n_tracks, min_fill_ratio=0.75)
                if w is not None:
                    return w
        return None

    import random
    rng_orig = random.Random(0)
    rng_uni = random.Random(0)
    rng_wgt = random.Random(0)

    print(f"\nBenchmarking {n_iters} iterations x {len(scores)} scores...")
    print(f"Common: max_tracks={max_tracks}, min_fill_ratio=0.75")
    print(f"Weighted: temperature=1.0")

    r1 = benchmark("Original (fixed n_bars/n_tracks)", original, scores, n_iters, {})
    r2 = benchmark("Uniform (content-aware, uniform)", select_window_uniform, scores, n_iters, {**common_kwargs, "rng": rng_uni})
    r3 = benchmark("Weighted (content-aware, softmax)", select_window_weighted, scores, n_iters, {**common_kwargs, "temperature": 1.0, "rng": rng_wgt})

    # --- Comparison table ---
    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Original':>12} {'Uniform':>12} {'Weighted':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    
    def row(label, data, fmt=".1f"):
        vals = [statistics.mean(d) if d else 0 for d in data]
        print(f"  {label:<25} {vals[0]:>12{fmt}} {vals[1]:>12{fmt}} {vals[2]:>12{fmt}}")

    row("mean time (ms)", [r1["times_ms"], r2["times_ms"], r3["times_ms"]], ".2f")
    row("null rate (%)", [100*r1["n_nulls"]/max(r1["n_total"],1), 
                          100*r2["n_nulls"]/max(r2["n_total"],1),
                          100*r3["n_nulls"]/max(r3["n_total"],1)], ".1f")
    row("mean n_bars", [r1["n_bars"], r2["n_bars"], r3["n_bars"]], ".0f")
    row("mean n_tracks", [r1["n_tracks"], r2["n_tracks"], r3["n_tracks"]], ".0f")
    row("mean total_notes", [r1["total_notes"], r2["total_notes"], r3["total_notes"]], ".0f")
    row("mean notes/bar", [r1["notes_per_bar"], r2["notes_per_bar"], r3["notes_per_bar"]], ".2f")
    row("stddev notes/bar", [
        [statistics.stdev(r1["notes_per_bar"]) if len(r1["notes_per_bar"]) > 1 else 0],
        [statistics.stdev(r2["notes_per_bar"]) if len(r2["notes_per_bar"]) > 1 else 0],
        [statistics.stdev(r3["notes_per_bar"]) if len(r3["notes_per_bar"]) > 1 else 0],
    ], ".2f")


if __name__ == "__main__":
    main()
