"""E6 -- per-instrument breakdown (EXPERIMENT_PLAN.md section E6, reframed).

No drums (track_type == 'drum' pieces are skipped -- Humanize's velocity/
timing semantics don't apply the same way to drum tracks and the plan
explicitly excludes them, sec:A6). Groups melodic GM programs by
`instrument_merge_groups` from the encoder config (same grouping the model's
own tokenizer uses) and checks whether generated per-group velocity/timing
profiles track GT's real per-group differences, or whether the model
homogenizes across instruments.

Single condition (cheap, per plan): one temperature/top_p/bars_per_step,
generated once per piece -- this is not a grid sweep like E2.

Metrics, per instrument group (pooled across pieces whose track's GM program
falls in that group):
  - mean velocity, velocity stdev, mean |delta| -- GT (round-tripped) vs
    generated.
  - Cross-group homogenization check: Pearson correlation, across groups,
    between GT's per-group mean velocity and generated's per-group mean
    velocity. Also the ratio of cross-group *variance* in per-group mean
    velocity (generated / GT) -- near 0 means the model washes out real
    per-instrument differences (homogenization); near 1 means it preserves
    them.

Usage:
    python3 e6_per_instrument.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny-20260807-035822/model_final.safetensors \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config $SCRATCH/MIDI-GPT/worktree-humanize/models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/e6 \\
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
from e2_sampling_calibration import canonical_note_order, pick_window_and_targets  # noqa: E402

MIN_PIECES_PER_GROUP = 3


def build_instrument_grouping(merge_groups: list[list[int]], total_instruments: int = 128) -> dict[int, int]:
    """Replicates domain_transforms.h::InstrumentGrouping in Python: returns
    {midi_instrument: representative_instrument}. Un-mentioned instruments
    map to themselves."""
    representative = {i: i for i in range(total_instruments)}
    for group in merge_groups:
        if not group:
            continue
        rep = group[0]
        for inst in group:
            representative[inst] = rep
    return representative


def instrument_group_label(rep: int, merge_groups: list[list[int]]) -> str:
    for group in merge_groups:
        if group and group[0] == rep:
            return f"group[{','.join(str(g) for g in group)}]"
    return f"program {rep}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--bars-per-step-mode", choices=["single", "full"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
    from midigpt.inference.engine import InferenceEngine

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_cfg = json.loads(Path(args.encoder_config).read_text())
    merge_groups = encoder_cfg["instrument_merge_groups"]
    forward_map = build_instrument_grouping(merge_groups)

    print("Loading checkpoint...", flush=True)
    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    tok = engine._tokenizer

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

    bars_per_step = 1 if args.bars_per_step_mode == "single" else max(1, args.n_bars)

    # group_id -> list of per-piece dicts
    by_group: dict[int, list[dict]] = defaultdict(list)
    n_used = 0
    n_attempted = 0
    n_skipped_drum = 0

    for raw in raw_pieces:
        if n_used >= args.limit:
            break
        n_attempted += 1
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        if not score.tracks:
            continue

        try:
            picked = pick_window_and_targets(score, args.n_bars, args.coverage, rng)
        except Exception:
            continue
        if picked is None:
            continue
        window, target_bars = picked

        track = window.tracks[0]
        if getattr(track, "track_type", "melodic") == "drum":
            n_skipped_drum += 1
            continue
        group_id = forward_map.get(track.instrument, track.instrument)

        try:
            gt_tokens = tok.encode(copy.deepcopy(window), compute_attributes=False)
            gt_decoded = tok.decode(gt_tokens)
        except Exception:
            continue

        req = GenerationRequest(
            tracks=[TrackPrompt(id=0, bars=sorted(target_bars), humanize=True)],
            config=InferenceConfig(
                temperature=args.temperature, top_p=args.top_p, bars_per_step=bars_per_step,
                tracks_per_step=1, model_dim=args.n_bars, mask_mode="remove",
                seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
            ),
        )
        try:
            gen_decoded = engine.session(copy.deepcopy(window), req).run()
        except Exception as exc:
            print(f"  [skip] generation failed: {exc}", flush=True)
            continue

        gt_notes = [n for b, bar in enumerate(gt_decoded.tracks[0].bars) if b in target_bars
                    for n in canonical_note_order(bar.notes)]
        gen_notes = [n for b, bar in enumerate(gen_decoded.tracks[0].bars) if b in target_bars
                     for n in canonical_note_order(bar.notes)]
        if not gt_notes or not gen_notes:
            continue

        gt_vel = [n.velocity for n in gt_notes]
        gen_vel = [n.velocity for n in gen_notes]
        gt_delta_abs = [abs(n.delta) for n in gt_notes]
        gen_delta_abs = [abs(n.delta) for n in gen_notes]

        by_group[group_id].append({
            "instrument": track.instrument,
            "gt_vel_mean": statistics.mean(gt_vel), "gen_vel_mean": statistics.mean(gen_vel),
            "gt_vel_stdev": statistics.pstdev(gt_vel) if len(gt_vel) >= 2 else 0.0,
            "gen_vel_stdev": statistics.pstdev(gen_vel) if len(gen_vel) >= 2 else 0.0,
            "gt_delta_abs_mean": statistics.mean(gt_delta_abs), "gen_delta_abs_mean": statistics.mean(gen_delta_abs),
        })
        n_used += 1
        if n_used % 25 == 0:
            print(f"  scored {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done: {n_used} pieces used, {n_skipped_drum} drum pieces skipped, {n_attempted} attempted.", flush=True)

    group_summary = []
    for group_id, records in sorted(by_group.items()):
        if len(records) < MIN_PIECES_PER_GROUP:
            continue
        group_summary.append({
            "group_id": group_id,
            "label": instrument_group_label(group_id, merge_groups),
            "n_pieces": len(records),
            "gt_vel_mean": statistics.mean(r["gt_vel_mean"] for r in records),
            "gen_vel_mean": statistics.mean(r["gen_vel_mean"] for r in records),
            "gt_vel_stdev_mean": statistics.mean(r["gt_vel_stdev"] for r in records),
            "gen_vel_stdev_mean": statistics.mean(r["gen_vel_stdev"] for r in records),
            "gt_delta_abs_mean": statistics.mean(r["gt_delta_abs_mean"] for r in records),
            "gen_delta_abs_mean": statistics.mean(r["gen_delta_abs_mean"] for r in records),
        })

    # Robust-pearson threshold: MIN_PIECES_PER_GROUP=3 is generous enough to keep
    # rare-instrument groups in the reported table, but a 3-piece group's mean
    # velocity is noisy enough to single-handedly swing an unweighted pearson
    # across ~40 groups (confirmed on humanize_tiny_e: dropping one n=3 outlier
    # group moved r from 0.770 to 0.925). Report both: the raw pearson (all
    # reported groups, historical metric) and a size-filtered one restricted to
    # groups with enough pieces to trust their mean.
    ROBUST_MIN_PIECES = 10
    homogenization = {
        "pearson_vel_mean": None, "gt_cross_group_var": None, "gen_cross_group_var": None, "variance_ratio": None,
        "pearson_vel_mean_robust": None, "pearson_vel_mean_robust_min_pieces": ROBUST_MIN_PIECES, "pearson_vel_mean_robust_n_groups": None,
    }
    if len(group_summary) >= 3:
        gt_means = np.array([g["gt_vel_mean"] for g in group_summary])
        gen_means = np.array([g["gen_vel_mean"] for g in group_summary])
        if gt_means.std() > 0 and gen_means.std() > 0:
            homogenization["pearson_vel_mean"] = float(np.corrcoef(gt_means, gen_means)[0, 1])
        gt_var = float(np.var(gt_means))
        gen_var = float(np.var(gen_means))
        homogenization["gt_cross_group_var"] = gt_var
        homogenization["gen_cross_group_var"] = gen_var
        homogenization["variance_ratio"] = (gen_var / gt_var) if gt_var > 0 else None

        robust = [g for g in group_summary if g["n_pieces"] >= ROBUST_MIN_PIECES]
        homogenization["pearson_vel_mean_robust_n_groups"] = len(robust)
        if len(robust) >= 3:
            gt_r = np.array([g["gt_vel_mean"] for g in robust])
            gen_r = np.array([g["gen_vel_mean"] for g in robust])
            if gt_r.std() > 0 and gen_r.std() > 0:
                homogenization["pearson_vel_mean_robust"] = float(np.corrcoef(gt_r, gen_r)[0, 1])

    summary = {
        "condition": {"temperature": args.temperature, "top_p": args.top_p, "bars_per_step_mode": args.bars_per_step_mode},
        "n_pieces_used": n_used, "n_pieces_skipped_drum": n_skipped_drum, "n_groups_reported": len(group_summary),
        "groups": group_summary, "homogenization": homogenization,
    }
    (out_dir / "e6_summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# E6 -- Per-instrument breakdown\n",
             f"Pieces used: {n_used}  |  drum pieces skipped: {n_skipped_drum}  |  groups reported (>= {MIN_PIECES_PER_GROUP} pieces): {len(group_summary)}\n",
             "| group | n pieces | gt vel mean | gen vel mean | gt vel stdev | gen vel stdev | gt |delta| mean | gen |delta| mean |",
             "|---|---|---|---|---|---|---|---|"]
    for g in group_summary:
        lines.append(f"| {g['label']} | {g['n_pieces']} | {g['gt_vel_mean']:.1f} | {g['gen_vel_mean']:.1f} | "
                     f"{g['gt_vel_stdev_mean']:.2f} | {g['gen_vel_stdev_mean']:.2f} | "
                     f"{g['gt_delta_abs_mean']:.2f} | {g['gen_delta_abs_mean']:.2f} |")
    lines.append("")
    lines.append("## Homogenization check\n")
    h = homogenization
    lines.append(f"- Pearson r (GT vs generated per-group mean velocity, across groups): {h['pearson_vel_mean']}")
    lines.append(f"- Pearson r, robust (groups with >= {h['pearson_vel_mean_robust_min_pieces']} pieces only, "
                 f"n={h['pearson_vel_mean_robust_n_groups']} groups): {h['pearson_vel_mean_robust']}")
    lines.append(f"- Cross-group variance of per-group mean velocity: GT={h['gt_cross_group_var']}, generated={h['gen_cross_group_var']}")
    lines.append(f"- Variance ratio (generated / GT): {h['variance_ratio']} "
                 "(near 0 = model homogenizes across instruments; near 1 = preserves real per-instrument differences)")
    (out_dir / "e6_report.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'e6_summary.json'} and {out_dir / 'e6_report.md'}", flush=True)
    print(f"Homogenization: {homogenization}", flush=True)


if __name__ == "__main__":
    main()
