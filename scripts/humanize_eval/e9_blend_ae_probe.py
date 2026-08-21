"""E9 -- does an A/E logit-space mixture beat either checkpoint alone?

Motivation (see chat discussion, RESULTS_LOG.md's final verdict): A is the
best checkpoint under trustworthy context, E is the most robust to
mechanical/unreliable context, and E8's `mixed_ctx` scenario -- some but not
all other tracks mechanized -- is exactly the realistic case a hard
checkpoint choice can't represent well. `mixed_ctx` alone is the real test;
whole_track/mechanical_ctx are included only as sanity bookends confirming
the blend behaves sensibly at the alpha=1/0 extremes (BlendedModel's exact
reduction to a single model at those extremes was already verified by a
local unit-level smoke test -- these bookends are an end-to-end scenario
check, not the interesting result).

For each piece, THE SAME window/target/mechanization pattern is scored under
three conditions -- A alone, E alone, and BlendedModel with
alpha = 1 - mechanized_fraction (the fraction of *other* tracks mechanized in
that piece's mixed_ctx draw) -- so the comparison is apples-to-apples, not
across independently-sampled pieces.

No gate: descriptive, three-way comparison is the point.

Usage:
    python3 e9_blend_ae_probe.py \\
        --checkpoint-a <path> --checkpoint-e <path> \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir <dir> \\
        [--limit 200] [--n-bars 4] [--min-tracks 2] [--max-tracks 6] [--temperature 0.85] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from blend_model import BlendedModel  # noqa: E402
from e8_deployment_scenarios import (  # noqa: E402
    eligible_cells,
    mechanize_tracks,
    metrics_for,
    pick_multitrack_window,
    score_scenario,
)


def _build_engine(model, encoder_config, analyzer):
    from midigpt.attributes.base import AttributeAnalyzer
    from midigpt.inference.engine import InferenceEngine
    from midigpt.tokenizer.tokenizer import Tokenizer

    tokenizer = Tokenizer(encoder_config, analyzer)
    engine = InferenceEngine(model, tokenizer, analyzer or AttributeAnalyzer.from_config(encoder_config))
    engine.warmup()
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-e", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--min-tracks", type=int, default=2)
    parser.add_argument("--max-tracks", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from midigpt._types import Score
    from midigpt.tokenizer.checkpoint import load_checkpoint

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...", flush=True)
    bundle_a = load_checkpoint(args.checkpoint_a)
    bundle_e = load_checkpoint(args.checkpoint_e)
    model_a, model_e = bundle_a.model.eval(), bundle_e.model.eval()
    if model_a.cfg.vocab_size != model_e.cfg.vocab_size:
        raise ValueError(
            f"checkpoint-a/checkpoint-e must share a vocab (got "
            f"{model_a.cfg.vocab_size} vs {model_e.cfg.vocab_size})."
        )

    enc_cfg = bundle_a.encoder_config
    engine_a = _build_engine(model_a, enc_cfg, None)
    engine_e = _build_engine(model_e, enc_cfg, None)
    blended = BlendedModel(model_a, model_e, alpha=0.5)
    engine_blend = _build_engine(blended, enc_cfg, None)
    tok = engine_a._tokenizer

    print("Loading validation pieces...", flush=True)
    pf = pq.ParquetFile(args.val_parquet)
    raw_pieces: list[bytes] = []
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            raw_pieces.append(bytes(music.as_py()))
            if len(raw_pieces) >= max(args.limit * 3, 60):
                break
        if len(raw_pieces) >= max(args.limit * 3, 60):
            break
    rng.shuffle(raw_pieces)

    conditions = ["A_alone", "E_alone", "Blend"]
    scenario_records: dict[str, dict[str, list[dict]]] = {
        scen: {c: [] for c in conditions} for scen in ["whole_track", "mechanical_ctx", "mixed_ctx"]
    }
    mixed_ctx_frac_records: list[float] = []

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
        window = pick_multitrack_window(score, args.n_bars, args.min_tracks, args.max_tracks, rng)
        if window is None:
            continue
        n_tracks = len(window.tracks)
        cells = eligible_cells(window)
        if not cells:
            continue
        tracks_with_notes = sorted({t for t, _ in cells})
        if len(tracks_with_notes) < 2:
            continue

        target_track = rng.choice(tracks_with_notes)
        target_cells = {(target_track, b) for b in range(len(window.tracks[target_track].bars)) if window.tracks[target_track].bars[b].notes}
        other_tracks = [t for t in range(n_tracks) if t != target_track]

        def run_all(w, alpha: float, scenario: str):
            import copy as _copy

            for cond, engine in [("A_alone", engine_a), ("E_alone", engine_e), ("Blend", engine_blend)]:
                if cond == "Blend":
                    blended.alpha = alpha
                res = score_scenario(engine, tok, _copy.deepcopy(w), target_cells, args.temperature, rng)
                if res:
                    scenario_records[scenario][cond].append(metrics_for(*res))

        # ---- whole_track bookend: alpha=1 (pure A), all-real context -----
        run_all(window, 1.0, "whole_track")

        # ---- mechanical_ctx bookend: alpha=0 (pure E), all-mechanical ----
        import copy as _copy

        w_mech = _copy.deepcopy(window)
        mechanize_tracks(w_mech, other_tracks)
        run_all(w_mech, 0.0, "mechanical_ctx")

        # ---- mixed_ctx: THE test. alpha = 1 - mechanized_fraction --------
        w_mixed = _copy.deepcopy(window)
        if other_tracks:
            k = rng.randint(1, len(other_tracks))
            mixed_mech = rng.sample(other_tracks, k)
            mechanize_tracks(w_mixed, mixed_mech)
            frac_mech = k / len(other_tracks)
        else:
            frac_mech = 0.0
        mixed_ctx_frac_records.append(frac_mech)
        run_all(w_mixed, 1.0 - frac_mech, "mixed_ctx")

        n_used += 1
        if n_used % 20 == 0:
            print(f"  {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done: {n_used} pieces used.", flush=True)

    def agg(records: list[dict]) -> dict:
        if not records:
            return {"n": 0}
        return {
            "n": len(records),
            "mean_vel_wasserstein": statistics.mean(r["vel_wasserstein"] for r in records if r["vel_wasserstein"] is not None),
            "degeneracy_rate": sum(r["degenerate"] for r in records) / len(records),
        }

    summary = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_e": args.checkpoint_e,
        "n_pieces": n_used,
        "mean_mixed_ctx_mechanized_fraction": statistics.mean(mixed_ctx_frac_records) if mixed_ctx_frac_records else None,
        "scenarios": {
            scen: {cond: agg(recs) for cond, recs in by_cond.items()}
            for scen, by_cond in scenario_records.items()
        },
    }
    (out_dir / "e9_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_dir / 'e9_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
