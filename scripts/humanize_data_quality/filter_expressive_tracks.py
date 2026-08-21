"""Full-dataset scan: apply the "genuinely expressive, not drift/artifact"
filter to every no-drums row/track in GigaMIDI, and report how many pieces
would be eligible for humanize training.

Filter (all required, computed per track from RAW/native-resolution ticks,
no resampling needed):

  1. median NOMML == 12 (already GigaMIDI-provided; "freely timed", not
     already-quantized)
  2. concentration score >= SCORE_THRESHOLD: 1 - entropy(onset_ticks mod 12
     equivalent) / log(12) -- rejects drift/misaligned tracks whose quantized
     onset-phase histogram is closer to uniform than to a real metrical grid.
  3. timing_mode_frac < TIMING_MODE_FRAC_THRESHOLD: within the dominant
     phase bin, the single most common *exact* fine-tick residual value must
     not cover almost all the notes -- rejects tracks that are actually
     quantized-plus-constant-tick-offset artifacts (hundreds of notes landing
     on the literal same tick), which trivially ace check #2 with zero real
     expressiveness.
  4. velocity_mode_frac < VELOCITY_MODE_FRAC_THRESHOLD: the single most
     common exact velocity value must not cover nearly all notes -- rejects
     tracks where dynamics were never authored (every note the same
     velocity), separate from and orthogonal to the timing checks.

This replaces an earlier version of check #3 that used a std-ratio
threshold (dominant-bin residual stdev / half the coarse unit >= 0.15).
That metric can be fooled by a mechanical process alternating between a
small number of discrete offsets (nonzero std, zero real variability), and
its threshold was picked without a principled justification.

Both #3 and #4 were derived empirically, not guessed: corpus-wide histograms
of several candidate timing-organicness metrics (std_ratio, entropy-of-
residual, mode-concentration) were all smooth and unimodal -- no natural
valley exists to split "real" from "fake" timing at some middle value, so
timing only gets a hard reject at the extreme, individually-verified-
degenerate tail (mode_frac >= 0.90; example: 159/163 notes at the literal
same one-of-40 possible tick offsets). Velocity mode-concentration, in
contrast, showed a genuine bimodal signature: a smooth distribution below
~0.7, then an almost-empty gap, then a distinct cluster pinned at ~1.0
(perfectly flat velocity, ~4-5% of the corpus) -- a real, separable
population, because "were dynamics authored at all" is a much more binary
authorial choice than continuous timing variability ever is. See
scripts/humanize_eval/e0h_entropy_metric.py / e0i_tail_examples.py /
e0j_mode_frac.py / e0k_velocity_shape.py (ad hoc analysis, not checked in)
for the corpus-wide distributions this was derived from.

Computes BOTH the old std_ratio metric and the new mode_frac metrics in the
same pass (over raw ticks, no extra file I/O) so the old vs. new filter's
pass/fail decisions can be directly compared without re-scanning the corpus
twice.

Parallelized across shards (each shard is independent, matches
training/preprocess.py's --workers pattern). Writes one result file per
shard (row/track indices + scores for every candidate track, pass/fail under
both the old and new filter) plus a combined summary, so the output can be
reused directly as preprocess.py's valid-index cache once wired in.

Usage:
    python3 filter_expressive_tracks.py \\
        --parquet /scratch/triana24/MIDI-GPT/data/v2.0.0/train/*.parquet \\
                  /scratch/triana24/MIDI-GPT/data/v2.0.0/validation/*.parquet \\
        --output-dir /scratch/triana24/MIDI-GPT/humanize_filter_results \\
        --workers 12
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

TARGET_TPQ = 12
SCORE_THRESHOLD = 0.6
STD_RATIO_THRESHOLD = 0.15          # old criterion, kept for comparison only
TIMING_MODE_FRAC_THRESHOLD = 0.90   # new: reject if >=90% of dominant-bin notes share one exact tick
VELOCITY_MODE_FRAC_THRESHOLD = 0.95  # new: reject if >=95% of notes share one exact velocity
MIN_NOTES = 40
BATCH_SIZE = 2000
TYPE_FILTER = "no-drums"


def analyze_track(bars, raw_resolution: int):
    """bars: list of midigpt._types.Bar. Returns a dict of scores, or None
    if too few notes. Single pass over raw (unresampled) ticks."""
    coarse_unit = raw_resolution / TARGET_TPQ
    by_bin: dict[int, list[float]] = {}
    velocities: list[int] = []
    for bar in bars:
        for note in bar.notes:
            nearest_cell = round(note.onset_ticks / coarse_unit)
            residual = note.onset_ticks - nearest_cell * coarse_unit
            coarse_bin = nearest_cell % TARGET_TPQ
            by_bin.setdefault(coarse_bin, []).append(round(residual))
            velocities.append(note.velocity)

    total = sum(len(v) for v in by_bin.values())
    if total < MIN_NOTES:
        return None

    h = 0.0
    for vals in by_bin.values():
        p = len(vals) / total
        h -= p * math.log(p)
    conc_score = 1.0 - h / math.log(TARGET_TPQ)

    _, dominant_vals = max(by_bin.items(), key=lambda kv: len(kv[1]))
    std = statistics.pstdev(dominant_vals) if len(dominant_vals) > 1 else 0.0
    std_ratio = std / (coarse_unit / 2) if coarse_unit else 0.0

    dom_counts = Counter(dominant_vals)
    timing_mode_frac = max(dom_counts.values()) / len(dominant_vals)

    vel_counts = Counter(velocities)
    velocity_mode_frac = max(vel_counts.values()) / len(velocities)

    return {
        "conc_score": conc_score,
        "n_notes": total,
        "std_ratio": std_ratio,
        "timing_mode_frac": timing_mode_frac,
        "velocity_mode_frac": velocity_mode_frac,
    }


def passes_old(r: dict) -> bool:
    return r["conc_score"] >= SCORE_THRESHOLD and r["std_ratio"] >= STD_RATIO_THRESHOLD


def passes_new(r: dict) -> bool:
    return (
        r["conc_score"] >= SCORE_THRESHOLD
        and r["timing_mode_frac"] < TIMING_MODE_FRAC_THRESHOLD
        and r["velocity_mode_frac"] < VELOCITY_MODE_FRAC_THRESHOLD
    )


def process_shard(path: str) -> dict:
    from midigpt._types import Score  # imported in-worker: not picklable across spawn otherwise

    t0 = time.time()
    stats = {
        "path": path,
        "rows_total": 0,
        "rows_no_drums": 0,
        "rows_nomml12_candidate": 0,
        "rows_passing_old": 0,
        "rows_passing_new": 0,
        "tracks_nomml12_candidate": 0,
        "tracks_passing_old": 0,
        "tracks_passing_new": 0,
        # (row_i, track_idx, conc_score, n_notes, std_ratio, timing_mode_frac,
        #  velocity_mode_frac, pass_old, pass_new)
        "scored": [],
    }

    pf = pq.ParquetFile(path)
    offset = 0
    for batch in pf.iter_batches(columns=["music", "NOMML", "Type"], batch_size=BATCH_SIZE):
        types = batch.column("Type").to_pylist()
        noms = batch.column("NOMML").to_pylist()
        music = batch.column("music")
        stats["rows_total"] += len(types)

        for local_i, (ty, lst) in enumerate(zip(types, noms)):
            if ty != TYPE_FILTER:
                continue
            stats["rows_no_drums"] += 1
            if not lst:
                continue
            candidate_tracks = [ti for ti, v in enumerate(lst) if v == 12]
            if not candidate_tracks:
                continue
            stats["rows_nomml12_candidate"] += 1
            stats["tracks_nomml12_candidate"] += len(candidate_tracks)

            raw = bytes(music[local_i].as_py())
            try:
                score_obj = Score.from_bytes(raw)
            except Exception:
                continue

            row_passed_old = False
            row_passed_new = False
            for ti in candidate_tracks:
                if ti >= len(score_obj.tracks):
                    continue
                result = analyze_track(score_obj.tracks[ti].bars, score_obj.resolution)
                if result is None:
                    continue
                p_old = passes_old(result)
                p_new = passes_new(result)
                if p_old:
                    stats["tracks_passing_old"] += 1
                    row_passed_old = True
                if p_new:
                    stats["tracks_passing_new"] += 1
                    row_passed_new = True
                stats["scored"].append((
                    offset + local_i, ti,
                    round(result["conc_score"], 4), result["n_notes"],
                    round(result["std_ratio"], 4),
                    round(result["timing_mode_frac"], 4),
                    round(result["velocity_mode_frac"], 4),
                    p_old, p_new,
                ))
            if row_passed_old:
                stats["rows_passing_old"] += 1
            if row_passed_new:
                stats["rows_passing_new"] += 1

        offset += len(types)
        del batch, types, noms, music

    stats["elapsed_sec"] = time.time() - t0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.parquet:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")

    print(f"Scanning {len(paths)} shard(s), conc_score>={SCORE_THRESHOLD}, "
          f"OLD: std_ratio>={STD_RATIO_THRESHOLD} | "
          f"NEW: timing_mode_frac<{TIMING_MODE_FRAC_THRESHOLD}, "
          f"velocity_mode_frac<{VELOCITY_MODE_FRAC_THRESHOLD}, "
          f"min_notes={MIN_NOTES}, type={TYPE_FILTER}")
    print(f"Workers: {args.workers}\n", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def report(stats):
        shard_name = Path(stats["path"]).name
        print(f"[{shard_name}] rows={stats['rows_total']} no_drums={stats['rows_no_drums']} "
              f"nomml12_rows={stats['rows_nomml12_candidate']} "
              f"OLD_rows={stats['rows_passing_old']} tracks={stats['tracks_passing_old']} | "
              f"NEW_rows={stats['rows_passing_new']} tracks={stats['tracks_passing_new']} "
              f"({stats['elapsed_sec']:.0f}s)", flush=True)
        shard_out = out_dir / f"{Path(stats['path']).parent.name}_{Path(stats['path']).stem}.json"
        with open(shard_out, "w") as f:
            json.dump(stats, f)

    t0 = time.time()
    results = []
    if args.workers > 1:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for stats in pool.imap_unordered(process_shard, paths):
                results.append(stats)
                report(stats)
    else:
        for path in paths:
            stats = process_shard(path)
            results.append(stats)
            report(stats)

    total_elapsed = time.time() - t0
    summary = {
        "n_shards": len(paths),
        "rows_total": sum(r["rows_total"] for r in results),
        "rows_no_drums": sum(r["rows_no_drums"] for r in results),
        "rows_nomml12_candidate": sum(r["rows_nomml12_candidate"] for r in results),
        "rows_passing_old": sum(r["rows_passing_old"] for r in results),
        "rows_passing_new": sum(r["rows_passing_new"] for r in results),
        "tracks_nomml12_candidate": sum(r["tracks_nomml12_candidate"] for r in results),
        "tracks_passing_old": sum(r["tracks_passing_old"] for r in results),
        "tracks_passing_new": sum(r["tracks_passing_new"] for r in results),
        "wall_time_sec": total_elapsed,
        "score_threshold": SCORE_THRESHOLD,
        "std_ratio_threshold": STD_RATIO_THRESHOLD,
        "timing_mode_frac_threshold": TIMING_MODE_FRAC_THRESHOLD,
        "velocity_mode_frac_threshold": VELOCITY_MODE_FRAC_THRESHOLD,
        "min_notes": MIN_NOTES,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n--> OLD filter: {summary['rows_passing_old']} rows / {summary['tracks_passing_old']} tracks eligible")
    print(f"--> NEW filter: {summary['rows_passing_new']} rows / {summary['tracks_passing_new']} tracks eligible")
    print(f"    out of {summary['rows_no_drums']} no-drums rows scanned "
          f"({summary['rows_total']} total rows across all shards).")
    print(f"Wall time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
