"""E2 -- sampling, calibration, and tau* (EXPERIMENT_PLAN.md section E2).

Generates Humanize output for held-out pieces across a temperature/top_p
grid and both bars_per_step settings, at ~50% bar coverage, and compares
generated Velocity/Delta distributions against round-tripped ground truth
(the oracle upper bound -- same encode/decode lossy floor the model itself
is subject to, established in E0) and two baselines: flat/mechanical
(constant velocity, zero delta) and shuffled (a donor piece's values on the
same skeleton).

Metrics per piece, per (tau, top_p, bars_per_step) condition:
  - Wasserstein distance between generated and GT velocity-level marginals
  - Wasserstein distance between generated and GT delta marginals
  - per-piece dispersion (population stdev) and entropy, generated vs GT
  - degeneracy: generated velocity-level stdev < DEGENERACY_STDEV_THRESHOLD

Gate G2: at some tau, (i) degeneracy rate ~ 0, and (ii) per-piece dispersion
distribution overlaps GT's (compared via bootstrap CI on the mean).

Scope/caveats (see report): single track, ~50% of a 4-bar window's non-empty
bars humanized per piece (not the full multi-track plan); one checkpoint.

Usage:
    python3 e2_sampling_calibration.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny-20260807-035822/model_final.safetensors \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e2 \\
        [--limit N] [--temperatures 0.7,0.85,1.0,1.15] [--top-ps 1.0,0.95] \\
        [--bars-per-step-modes single,full] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

VEL_LEVELS = 128
DELTA_LO, DELTA_HI = -6, 6
DEGENERACY_STDEV_THRESHOLD = 0.5


def velocity_encode(v: int, num_levels: int = VEL_LEVELS) -> int:
    if v <= 0:
        return 0
    if v >= 127:
        return num_levels - 1
    return min(1 + v * (num_levels - 1) // 128, num_levels - 1)


def canonical_note_order(notes):
    by_onset: dict[int, list] = defaultdict(list)
    for n in notes:
        by_onset[n.onset_ticks].append(n)
    out = []
    for onset in sorted(by_onset):
        out.extend(sorted(by_onset[onset], key=lambda n: n.pitch))
    return out


def model_res_values(score_like, target_bars: set[int]):
    """[(vel_level, signed_delta)] for notes in target_bars, resampled to
    model resolution (12) via the real resample_delta."""
    from midigpt.tokenizer.tokenizer import resample_delta

    resampled = copy.deepcopy(score_like)
    resample_delta(resampled, resampled.resolution, 12, use_delta=True)
    out = []
    for b, bar in enumerate(resampled.tracks[0].bars):
        if b not in target_bars:
            continue
        for note in canonical_note_order(bar.notes):
            vel = velocity_encode(note.velocity)
            d = max(DELTA_LO, min(DELTA_HI, note.delta))
            out.append((vel, d))
    return out


def pick_window_and_targets(score, n_bars: int, coverage: float, rng: random.Random):
    from midigpt.augmentation.score_window import select_window

    window = select_window(score, n_bars, 1, min_fill_ratio=0.0)
    if window is None or not window.tracks:
        return None
    track = window.tracks[0]
    eligible = [b for b, bar in enumerate(track.bars) if bar.notes]
    if not eligible:
        return None
    k = max(1, round(len(eligible) * coverage))
    target_bars = set(rng.sample(eligible, min(k, len(eligible))))
    return window, target_bars


def flat_baseline(window, target_bars: set[int], const_velocity: int = 80):
    out = copy.deepcopy(window)
    for b, bar in enumerate(out.tracks[0].bars):
        if b in target_bars:
            for note in bar.notes:
                note.velocity = const_velocity
                note.delta = 0
    return out


def shuffled_baseline(window, target_bars: set[int], donor_notes: list, rng: random.Random):
    out = copy.deepcopy(window)
    if not donor_notes:
        return flat_baseline(window, target_bars)
    i = rng.randrange(len(donor_notes))
    for b, bar in enumerate(out.tracks[0].bars):
        if b not in target_bars:
            continue
        for note in canonical_note_order(bar.notes):
            donor = donor_notes[i % len(donor_notes)]
            note.velocity = donor.velocity
            note.delta = donor.delta
            i += 1
    return out


def wasserstein_1d(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except ImportError:
        a_s, b_s = sorted(a), sorted(b)
        n = max(len(a_s), len(b_s))

        def q(xs, p):
            idx = min(len(xs) - 1, int(p * len(xs)))
            return xs[idx]

        return sum(abs(q(a_s, i / n) - q(b_s, i / n)) for i in range(n)) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--temperatures", default="0.7,0.85,1.0,1.15")
    parser.add_argument("--top-ps", default="1.0,0.95")
    parser.add_argument("--bars-per-step-modes", default="single,full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
    from midigpt.inference.engine import InferenceEngine

    temperatures = [float(x) for x in args.temperatures.split(",")]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    bps_modes = args.bars_per_step_modes.split(",")

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...", flush=True)
    engine = InferenceEngine.from_checkpoint(args.checkpoint)

    print("Loading validation pieces...", flush=True)
    pf = pq.ParquetFile(args.val_parquet)
    raw_pieces: list[bytes] = []
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            raw_pieces.append(bytes(music.as_py()))
            if len(raw_pieces) >= max(args.limit * 2, 50):
                break
        if len(raw_pieces) >= max(args.limit * 2, 50):
            break
    rng.shuffle(raw_pieces)

    conditions = []
    for bps_mode in bps_modes:
        for tau in temperatures:
            for top_p in top_ps:
                conditions.append((bps_mode, tau, top_p))
    print(f"{len(conditions)} conditions x up to {args.limit} pieces", flush=True)

    # records[condition_key][piece_i] = per-piece metrics
    records: dict[str, list[dict]] = defaultdict(list)
    n_used = 0

    for raw in raw_pieces:
        if n_used >= args.limit:
            break
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        if not score.tracks:
            continue
        picked = pick_window_and_targets(score, args.n_bars, args.coverage, rng)
        if picked is None:
            continue
        window, target_bars = picked

        try:
            tok = engine._tokenizer
            gt_tokens = tok.encode(copy.deepcopy(window), compute_attributes=False)
            gt_decoded = tok.decode(gt_tokens)
        except Exception:
            continue
        gt_values = model_res_values(gt_decoded, target_bars)
        if not gt_values:
            continue
        gt_vel = [v for v, _ in gt_values]
        gt_delta = [d for _, d in gt_values]

        donor_raw = raw_pieces[rng.randrange(len(raw_pieces))]
        try:
            donor_score = Score.from_bytes(donor_raw)
            donor_notes = [n for t in donor_score.tracks for b in t.bars for n in b.notes]
        except Exception:
            donor_notes = []

        flat_values = model_res_values(flat_baseline(window, target_bars), target_bars)
        shuf_values = model_res_values(shuffled_baseline(window, target_bars, donor_notes, rng), target_bars)

        any_ok = False
        for bps_mode, tau, top_p in conditions:
            bars_per_step = 1 if bps_mode == "single" else max(1, len(target_bars))
            req = GenerationRequest(
                tracks=[TrackPrompt(id=0, bars=sorted(target_bars), humanize=True)],
                config=InferenceConfig(
                    temperature=tau, top_p=top_p, bars_per_step=bars_per_step,
                    tracks_per_step=1, model_dim=args.n_bars, mask_mode="remove",
                    seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
                ),
            )
            try:
                gen_score = engine.session(copy.deepcopy(window), req).run()
            except Exception as exc:
                print(f"  [skip cond {bps_mode},{tau},{top_p}] {exc}", flush=True)
                continue
            gen_values = model_res_values(gen_score, target_bars)
            if not gen_values:
                continue
            gen_vel = [v for v, _ in gen_values]
            gen_delta = [d for _, d in gen_values]

            rec = {
                "gen_vel_stdev": statistics.pstdev(gen_vel) if len(gen_vel) >= 2 else 0.0,
                "gt_vel_stdev": statistics.pstdev(gt_vel) if len(gt_vel) >= 2 else 0.0,
                "vel_wasserstein": wasserstein_1d(gen_vel, gt_vel),
                "delta_wasserstein": wasserstein_1d(gen_delta, gt_delta),
                "flat_vel_wasserstein": wasserstein_1d([v for v, _ in flat_values], gt_vel),
                "shuf_vel_wasserstein": wasserstein_1d([v for v, _ in shuf_values], gt_vel),
                "degenerate": (statistics.pstdev(gen_vel) if len(gen_vel) >= 2 else 0.0) < DEGENERACY_STDEV_THRESHOLD,
                # GT itself is sometimes "degenerate" under the same absolute
                # threshold -- real short/naturally-low-variance passages, not
                # a model failure. An absolute gen degeneracy rate conflates
                # the two; excess_degeneracy (gen rate minus gt rate, per
                # condition) isolates the model's real contribution. See
                # e2_degeneracy_diagnostic.py investigation: at n_notes<4, GT
                # itself is "degenerate" ~41% of the time under this
                # threshold, purely from small-sample stdev instability.
                "gt_degenerate": (statistics.pstdev(gt_vel) if len(gt_vel) >= 2 else 0.0) < DEGENERACY_STDEV_THRESHOLD,
                "n_notes": len(gen_values),
            }
            key = f"bps={bps_mode}|tau={tau}|top_p={top_p}"
            records[key].append(rec)
            any_ok = True

        if any_ok:
            n_used += 1
            if n_used % 10 == 0:
                print(f"  generated {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done: {n_used} pieces used.", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    def bootstrap_ci(xs, n_boot=2000, seed=0):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        rb = random.Random(seed)
        n = len(xs)
        means = sorted(sum(xs[rb.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
        return {"mean": sum(xs) / n, "ci95_lo": means[int(0.025 * n_boot)], "ci95_hi": means[int(0.975 * n_boot)], "n": n}

    gt_dispersion_ci = bootstrap_ci([r["gt_vel_stdev"] for recs in records.values() for r in recs])

    summary = {}
    for key, recs in records.items():
        degeneracy_rate = sum(1 for r in recs if r["degenerate"]) / len(recs) if recs else None
        gt_degeneracy_rate = sum(1 for r in recs if r["gt_degenerate"]) / len(recs) if recs else None
        excess_degeneracy_rate = (
            degeneracy_rate - gt_degeneracy_rate if degeneracy_rate is not None and gt_degeneracy_rate is not None else None
        )
        gen_disp_ci = bootstrap_ci([r["gen_vel_stdev"] for r in recs])
        overlaps_gt = (
            gen_disp_ci is not None and gt_dispersion_ci is not None
            and gen_disp_ci["ci95_lo"] <= gt_dispersion_ci["ci95_hi"]
            and gt_dispersion_ci["ci95_lo"] <= gen_disp_ci["ci95_hi"]
        )
        summary[key] = {
            "n_pieces": len(recs),
            "degeneracy_rate": degeneracy_rate,
            "gt_degeneracy_rate": gt_degeneracy_rate,
            "excess_degeneracy_rate": excess_degeneracy_rate,
            "mean_vel_wasserstein": mean([r["vel_wasserstein"] for r in recs]),
            "mean_delta_wasserstein": mean([r["delta_wasserstein"] for r in recs]),
            "mean_flat_vel_wasserstein": mean([r["flat_vel_wasserstein"] for r in recs]),
            "mean_shuf_vel_wasserstein": mean([r["shuf_vel_wasserstein"] for r in recs]),
            "gen_dispersion_ci": gen_disp_ci,
            "overlaps_gt_dispersion": overlaps_gt,
            # Gated on EXCESS degeneracy (gen rate minus GT's own rate under
            # the same threshold), not the raw absolute rate -- the raw rate
            # is never near zero even for the oracle (GT), so gating on it
            # directly rejects every condition regardless of model quality.
            "gate_G2_pass": (excess_degeneracy_rate is not None and excess_degeneracy_rate < 0.05 and overlaps_gt),
        }

    best_tau_key = None
    for key, s in summary.items():
        if s["gate_G2_pass"]:
            best_tau_key = key
            break

    result = {
        "checkpoint": args.checkpoint,
        "n_pieces": n_used,
        "gt_dispersion_ci": gt_dispersion_ci,
        "conditions": summary,
        "gate_G2_any_pass": best_tau_key is not None,
        "gate_G2_first_passing_condition": best_tau_key,
    }

    with open(out_dir / "e2_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    lines = ["# E2 -- sampling, calibration, and tau*\n",
             f"- Checkpoint: `{args.checkpoint}`",
             f"- Pieces: {n_used}",
             f"- GT (round-tripped) velocity-level dispersion: mean={gt_dispersion_ci['mean']:.3f} "
             f"CI=[{gt_dispersion_ci['ci95_lo']:.3f}, {gt_dispersion_ci['ci95_hi']:.3f}]\n" if gt_dispersion_ci else "",
             "| condition | n | degeneracy rate (gen) | degeneracy rate (GT) | excess degeneracy | "
             "vel Wasserstein (gen) | vel Wasserstein (flat) | "
             "vel Wasserstein (shuffled) | delta Wasserstein | gen dispersion (mean [CI]) | overlaps GT | G2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, s in sorted(summary.items()):
        disp = s["gen_dispersion_ci"]
        disp_str = f"{disp['mean']:.3f} [{disp['ci95_lo']:.3f},{disp['ci95_hi']:.3f}]" if disp else "-"
        lines.append(
            f"| {key} | {s['n_pieces']} | {s['degeneracy_rate']:.3f} | {s['gt_degeneracy_rate']:.3f} | "
            f"{s['excess_degeneracy_rate']:.3f} | {s['mean_vel_wasserstein']:.3f} | "
            f"{s['mean_flat_vel_wasserstein']:.3f} | {s['mean_shuf_vel_wasserstein']:.3f} | "
            f"{s['mean_delta_wasserstein']:.3f} | {disp_str} | "
            f"{'YES' if s['overlaps_gt_dispersion'] else 'no'} | {'PASS' if s['gate_G2_pass'] else 'fail'} |"
        )
    lines.append(f"\n**G2 gate**: {'at least one condition passes' if best_tau_key else 'NO condition passes'} "
                 f"(first passing: `{best_tau_key}`)\n" if best_tau_key else
                 "\n**G2 gate FAILS for every condition tested** -- see conversation for what that would mean.\n")
    with open(out_dir / "e2_report.md", "w") as f:
        f.write("\n".join(lines))

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
