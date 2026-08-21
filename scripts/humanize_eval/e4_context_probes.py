"""E4 -- context probes (EXPERIMENT_PLAN.md section E4). Only meaningful
because E1's G1b already passed (true context beats swapped context on
held-out teacher-forced NLL, paired, CI excluding zero) -- this is the
sampling-level version of that same question, now with actual generation
instead of scoring, plus two things E1 alone can't answer:

Pass 1 -- coverage sweep {100%, 50%, 25%} (sec:C-C): does distributional
match to GT hold up (or improve) as coverage shrinks and more real context
remains available to condition on?

Pass 2 -- context ablation probe, fixed coverage=0.5:
  (a) true context, (b) context bars flattened to constant velocity / zero
  delta, (c) context bars swapped in from a donor piece.
  - Degeneracy rate and dispersion under each, vs GT and vs each other --
    does flat context make output collapse toward flat, or go unpredictable
    (elevated dispersion/degeneracy vs true-context, not matching flat OR
    GT)? These are different failure modes and training never saw
    non-NOMML==12 context, so this is a genuine OOD probe.
  - Expressive-context style match: per-beat-position profile (12
    model-resolution bins) of the GENERATED target bars vs THIS PIECE'S OWN
    true context bars' profile, under true context vs under swapped context.
    Paired per piece; bootstrap 95% CI of (sim_true - sim_swap) across
    pieces. A CI excluding zero (positive) is direct evidence the model
    performs real context-driven style transfer during actual sampling, not
    just something E1 measured indirectly via NLL.

No formal gate is defined for E4 in the plan (informative, not a stop/go
gate) -- results are reported descriptively.

Usage:
    python3 e4_context_probes.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny-20260807-035822/model_final.safetensors \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e4 \\
        [--limit N] [--temperature 0.7] [--top-p 1.0] [--bars-per-step-mode full] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from e2_sampling_calibration import (  # noqa: E402
    DEGENERACY_STDEV_THRESHOLD,
    canonical_note_order,
    model_res_values,
    pick_window_and_targets,
    wasserstein_1d,
)
from e3_structural_correctness import pearson  # noqa: E402

N_PROFILE_BINS = 12
N_BOOTSTRAP = 1000


def flatten_context_bars(window, target_bars: set[int], const_velocity: int = 80):
    from midigpt.augmentation.mechanize import mechanize_bar

    out = copy.deepcopy(window)
    for b, bar in enumerate(out.tracks[0].bars):
        if b in target_bars:
            continue
        mechanize_bar(bar, out.resolution, const_velocity)
    return out


def swap_context_bars(window, target_bars: set[int], donor_notes: list, rng: random.Random):
    out = copy.deepcopy(window)
    if not donor_notes:
        return flatten_context_bars(window, target_bars)
    i = rng.randrange(len(donor_notes))
    for b, bar in enumerate(out.tracks[0].bars):
        if b in target_bars:
            continue
        for note in canonical_note_order(bar.notes):
            donor = donor_notes[i % len(donor_notes)]
            note.velocity = donor.velocity
            note.delta = donor.delta
            i += 1
    return out


def profile_points(score, bars_of_interest: set[int]):
    pts = []
    for b, bar in enumerate(score.tracks[0].bars):
        if b not in bars_of_interest:
            continue
        bar_len = bar.beat_length * score.resolution
        for n in bar.notes:
            pos = (n.onset_ticks / bar_len) if bar_len > 0 else 0.0
            pts.append((pos, n.velocity))
    return pts


def profile_vector(pts) -> list[float | None]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for pos, v in pts:
        idx = min(N_PROFILE_BINS - 1, int(pos * N_PROFILE_BINS))
        buckets[idx].append(v)
    return [statistics.mean(buckets[i]) if buckets[i] else None for i in range(N_PROFILE_BINS)]


def generate(engine, window, target_bars, temperature, top_p, bars_per_step, n_bars, rng):
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

    req = GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=sorted(target_bars), humanize=True)],
        config=InferenceConfig(
            temperature=temperature, top_p=top_p, bars_per_step=bars_per_step,
            tracks_per_step=1, model_dim=n_bars, mask_mode="remove",
            seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
        ),
    )
    return engine.session(copy.deepcopy(window), req).run()


def bootstrap_diff_ci(a: list[float], b: list[float], rng: random.Random, n_boot: int = N_BOOTSTRAP):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    boot_means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    return {
        "mean": statistics.mean(diffs),
        "ci_lo": boot_means[int(0.025 * n_boot)],
        "ci_hi": boot_means[int(0.975 * n_boot)],
        "n_pairs": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--bars-per-step-mode", choices=["single", "full"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.engine import InferenceEngine

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...", flush=True)
    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    tok = engine._tokenizer
    bars_per_step = 1 if args.bars_per_step_mode == "single" else max(1, args.n_bars)

    print("Loading validation pieces...", flush=True)
    pf = pq.ParquetFile(args.val_parquet)
    raw_pieces: list[bytes] = []
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            raw_pieces.append(bytes(music.as_py()))
            if len(raw_pieces) >= max(args.limit * 3, 50):
                break
        if len(raw_pieces) >= max(args.limit * 3, 50):
            break
    rng.shuffle(raw_pieces)

    # ---------------- Pass 1: coverage sweep (true context only) -------
    print("Pass 1: coverage sweep...", flush=True)
    coverage_results: dict[str, list[dict]] = defaultdict(list)
    for coverage in (1.0, 0.5, 0.25):
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
            try:
                picked = pick_window_and_targets(score, args.n_bars, coverage, rng)
            except Exception:
                continue
            if picked is None:
                continue
            window, target_bars = picked
            try:
                gt_tokens = tok.encode(copy.deepcopy(window), compute_attributes=False)
                gt_decoded = tok.decode(gt_tokens)
            except Exception:
                continue
            gt_values = model_res_values(gt_decoded, target_bars)
            if not gt_values:
                continue
            try:
                gen_decoded = generate(engine, window, target_bars, args.temperature, args.top_p,
                                        bars_per_step, args.n_bars, rng)
            except Exception:
                continue
            gen_values = model_res_values(gen_decoded, target_bars)
            if not gen_values:
                continue
            gt_vel = [v for v, _ in gt_values]
            gen_vel = [v for v, _ in gen_values]
            coverage_results[str(coverage)].append({
                "vel_wasserstein": wasserstein_1d(gen_vel, gt_vel),
                "gen_vel_stdev": statistics.pstdev(gen_vel) if len(gen_vel) >= 2 else 0.0,
                "gt_vel_stdev": statistics.pstdev(gt_vel) if len(gt_vel) >= 2 else 0.0,
                "degenerate": (statistics.pstdev(gen_vel) if len(gen_vel) >= 2 else 0.0) < DEGENERACY_STDEV_THRESHOLD,
            })
            n_used += 1
        print(f"  coverage={coverage}: {n_used} pieces", flush=True)

    coverage_summary = {}
    for cov, recs in coverage_results.items():
        if not recs:
            continue
        coverage_summary[cov] = {
            "n_pieces": len(recs),
            "mean_vel_wasserstein": statistics.mean(r["vel_wasserstein"] for r in recs if r["vel_wasserstein"] is not None),
            "mean_gen_vel_stdev": statistics.mean(r["gen_vel_stdev"] for r in recs),
            "mean_gt_vel_stdev": statistics.mean(r["gt_vel_stdev"] for r in recs),
            "degeneracy_rate": sum(r["degenerate"] for r in recs) / len(recs),
        }

    # ---------------- Pass 2: context ablation probe, coverage=0.5 -----
    print("Pass 2: context ablation (true/flat/swap)...", flush=True)
    ablation_records = []
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
        try:
            picked = pick_window_and_targets(score, args.n_bars, 0.5, rng)
        except Exception:
            continue
        if picked is None:
            continue
        window, target_bars = picked
        context_bars = set(range(args.n_bars)) - target_bars
        if not context_bars:
            continue

        try:
            gt_tokens = tok.encode(copy.deepcopy(window), compute_attributes=False)
            gt_decoded = tok.decode(gt_tokens)
        except Exception:
            continue
        gt_values = model_res_values(gt_decoded, target_bars)
        if not gt_values:
            continue
        gt_vel = [v for v, _ in gt_values]
        own_context_profile = profile_vector(profile_points(gt_decoded, context_bars))

        donor_raw = raw_pieces[rng.randrange(len(raw_pieces))]
        try:
            donor_score = Score.from_bytes(donor_raw)
            donor_notes = [n for t in donor_score.tracks for b in t.bars for n in b.notes]
        except Exception:
            donor_notes = []
        if not donor_notes:
            continue

        flat_window = flatten_context_bars(window, target_bars)
        swap_window = swap_context_bars(window, target_bars, donor_notes, rng)

        try:
            gen_true = generate(engine, window, target_bars, args.temperature, args.top_p, bars_per_step, args.n_bars, rng)
            gen_flat = generate(engine, flat_window, target_bars, args.temperature, args.top_p, bars_per_step, args.n_bars, rng)
            gen_swap = generate(engine, swap_window, target_bars, args.temperature, args.top_p, bars_per_step, args.n_bars, rng)
        except Exception as exc:
            print(f"  [skip] generation failed: {exc}", flush=True)
            continue

        vals_true = model_res_values(gen_true, target_bars)
        vals_flat = model_res_values(gen_flat, target_bars)
        vals_swap = model_res_values(gen_swap, target_bars)
        if not (vals_true and vals_flat and vals_swap):
            continue

        vel_true = [v for v, _ in vals_true]
        vel_flat = [v for v, _ in vals_flat]
        vel_swap = [v for v, _ in vals_swap]

        profile_true = profile_vector(profile_points(gen_true, target_bars))
        profile_swap = profile_vector(profile_points(gen_swap, target_bars))
        sim_true = pearson(profile_true, own_context_profile)
        sim_swap = pearson(profile_swap, own_context_profile)

        ablation_records.append({
            "true_vel_wasserstein": wasserstein_1d(vel_true, gt_vel),
            "flat_vel_wasserstein": wasserstein_1d(vel_flat, gt_vel),
            "swap_vel_wasserstein": wasserstein_1d(vel_swap, gt_vel),
            "true_vel_stdev": statistics.pstdev(vel_true) if len(vel_true) >= 2 else 0.0,
            "flat_vel_stdev": statistics.pstdev(vel_flat) if len(vel_flat) >= 2 else 0.0,
            "swap_vel_stdev": statistics.pstdev(vel_swap) if len(vel_swap) >= 2 else 0.0,
            "true_degenerate": (statistics.pstdev(vel_true) if len(vel_true) >= 2 else 0.0) < DEGENERACY_STDEV_THRESHOLD,
            "flat_degenerate": (statistics.pstdev(vel_flat) if len(vel_flat) >= 2 else 0.0) < DEGENERACY_STDEV_THRESHOLD,
            "swap_degenerate": (statistics.pstdev(vel_swap) if len(vel_swap) >= 2 else 0.0) < DEGENERACY_STDEV_THRESHOLD,
            "sim_true": sim_true,
            "sim_swap": sim_swap,
        })
        n_used += 1
        if n_used % 25 == 0:
            print(f"  ablation scored {n_used}/{args.limit} pieces...", flush=True)

    print(f"Pass 2 done: {n_used} pieces.", flush=True)

    def agg(key):
        return statistics.mean(r[key] for r in ablation_records if r[key] is not None)

    ablation_summary = {}
    if ablation_records:
        ablation_summary = {
            "n_pieces": len(ablation_records),
            "mean_vel_wasserstein": {
                "true": agg("true_vel_wasserstein"), "flat": agg("flat_vel_wasserstein"), "swap": agg("swap_vel_wasserstein"),
            },
            "mean_vel_stdev": {
                "true": agg("true_vel_stdev"), "flat": agg("flat_vel_stdev"), "swap": agg("swap_vel_stdev"),
            },
            "degeneracy_rate": {
                "true": sum(r["true_degenerate"] for r in ablation_records) / len(ablation_records),
                "flat": sum(r["flat_degenerate"] for r in ablation_records) / len(ablation_records),
                "swap": sum(r["swap_degenerate"] for r in ablation_records) / len(ablation_records),
            },
            "style_match_effect": bootstrap_diff_ci(
                [r["sim_true"] for r in ablation_records], [r["sim_swap"] for r in ablation_records], rng,
            ),
        }

    summary = {
        "condition": {"temperature": args.temperature, "top_p": args.top_p, "bars_per_step_mode": args.bars_per_step_mode},
        "coverage_sweep": coverage_summary,
        "context_ablation": ablation_summary,
    }
    (out_dir / "e4_summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# E4 -- Context probes\n"]
    lines.append(f"Condition: temperature={args.temperature}, top_p={args.top_p}, "
                 f"bars_per_step_mode={args.bars_per_step_mode} (provisional, see E3 for tau* status)\n")

    lines.append("## Pass 1: coverage sweep\n")
    lines.append("| coverage | n pieces | mean vel Wasserstein (vs GT) | gen vel stdev | GT vel stdev | degeneracy rate |")
    lines.append("|---|---|---|---|---|---|")
    for cov in ("1.0", "0.5", "0.25"):
        s = coverage_summary.get(cov)
        if not s:
            lines.append(f"| {cov} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(f"| {cov} | {s['n_pieces']} | {s['mean_vel_wasserstein']:.3f} | "
                     f"{s['mean_gen_vel_stdev']:.2f} | {s['mean_gt_vel_stdev']:.2f} | {s['degeneracy_rate']:.3f} |")
    lines.append("")

    lines.append("## Pass 2: context ablation (true / flat / swap), coverage=0.5\n")
    if ablation_summary:
        lines.append(f"Pieces: {ablation_summary['n_pieces']}\n")
        lines.append("| condition | mean vel Wasserstein (vs GT) | mean vel stdev | degeneracy rate |")
        lines.append("|---|---|---|---|")
        for cond in ("true", "flat", "swap"):
            lines.append(f"| {cond} | {ablation_summary['mean_vel_wasserstein'][cond]:.3f} | "
                         f"{ablation_summary['mean_vel_stdev'][cond]:.2f} | "
                         f"{ablation_summary['degeneracy_rate'][cond]:.3f} |")
        lines.append("")
        eff = ablation_summary["style_match_effect"]
        lines.append("### Expressive-context style match (sampling-level G1b analog)\n")
        if eff:
            lines.append(f"sim_true - sim_swap: mean={eff['mean']:.4f}, 95% CI=[{eff['ci_lo']:.4f}, {eff['ci_hi']:.4f}], "
                         f"n={eff['n_pairs']} paired pieces")
            excludes_zero = eff["ci_lo"] > 0 or eff["ci_hi"] < 0
            lines.append(f"CI excludes zero: {'YES' if excludes_zero else 'NO'} "
                         f"({'real context-driven style transfer during sampling' if excludes_zero and eff['mean'] > 0 else 'inconclusive'})")
        else:
            lines.append("insufficient paired data for CI")
    else:
        lines.append("no usable pieces")
    lines.append("")

    (out_dir / "e4_report.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'e4_summary.json'} and {out_dir / 'e4_report.md'}", flush=True)
    print(f"Coverage summary: {coverage_summary}", flush=True)
    print(f"Ablation summary: {ablation_summary}", flush=True)


if __name__ == "__main__":
    main()
