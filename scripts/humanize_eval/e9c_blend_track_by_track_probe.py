"""E9c -- does blending help on `track_by_track`, the scenario that actually
drove the final verdict (see RESULTS_LOG.md's 2026-08-09 finding: A/B
collapse 65-90% of the time under this exact iterative pattern, which
Wasserstein alone was hiding)?

E9/E9b established the blend mechanism works and the auto-detector is
reasonably accurate in the single-shot mixed_ctx probe. Neither touched
track_by_track, which is structurally different: context composition
EVOLVES over the sequence (steps builds up real, generated context; early
steps are almost entirely mechanical, late steps are almost entirely real)
-- exactly the shape a continuously-varying alpha should be suited for, and
the highest-value remaining test of whether blending is actually useful for
the realistic iterative-build workflow, not just single-shot snapshots.

Runs THREE independent full iterative trajectories per piece (A_alone,
E_alone, Blend), same starting point (fully mechanized window, same random
track order) but diverging as soon as each model generates different
content -- exactly mirroring E8's track_by_track, just three parallel model
conditions instead of one.

Blend's alpha at step i (0-indexed, track order[i] being generated) is set
from ORACLE knowledge (this harness controls the sequence, so it knows
exactly how many other tracks are already-real vs still-mechanical at every
step -- same "validate the mechanism with ground truth first" staging as
E9): alpha = i / (len(order) - 1). Step 0 (nothing generated yet) -> alpha=0
(pure E). Last step (everything else already real) -> alpha=1 (pure A).

No gate: descriptive, three-way comparison of degeneracy_rate_by_step and
mean_wasserstein_by_step is the point.

Usage:
    python3 e9c_blend_track_by_track_probe.py \\
        --checkpoint-a <path> --checkpoint-e <path> \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --output-dir <dir> \\
        [--limit 60] [--n-bars 4] [--min-tracks 2] [--max-tracks 6] [--temperature 0.85] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
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


def run_track_by_track(engine, tok, window, order, temperature, rng, blended_model=None):
    """One full iterative trajectory. If `blended_model` is given, its
    `.alpha` is set per-step from oracle context composition before each
    generation call -- otherwise `engine` is used as-is (A_alone/E_alone)."""
    from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

    w = copy.deepcopy(window)
    n_tracks = len(w.tracks)
    mechanize_tracks(w, list(range(n_tracks)))
    seq = []
    denom = max(1, len(order) - 1)
    for step, t in enumerate(order):
        t_cells = {(t, b) for b in range(len(w.tracks[t].bars)) if w.tracks[t].bars[b].notes}
        if not t_cells:
            continue
        if blended_model is not None:
            blended_model.alpha = step / denom
        res = score_scenario(engine, tok, w, t_cells, temperature, rng, gt_window=window)
        if res is None:
            continue
        gt_vel, gen_vel = res
        m = metrics_for(gt_vel, gen_vel)
        m["step"] = step
        if blended_model is not None:
            m["alpha"] = blended_model.alpha
        seq.append(m)

        t_bars = sorted(b for _, b in t_cells)
        req = GenerationRequest(
            tracks=[
                TrackPrompt(id=tt, bars=t_bars, humanize=True) if tt == t
                else TrackPrompt(id=tt, bars=[], ignore=True)
                for tt in range(len(w.tracks))
            ],
            config=InferenceConfig(
                temperature=temperature, top_p=1.0, bars_per_step=len(t_cells),
                tracks_per_step=1, model_dim=len(w.tracks[0].bars), mask_mode="remove",
                seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
            ),
        )
        try:
            w = engine.session(copy.deepcopy(w), req).run()
        except Exception:
            break
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-e", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=60)
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
        raise ValueError("checkpoint-a/checkpoint-e must share a vocab to blend.")

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
    sequences: dict[str, list[list[dict]]] = {c: [] for c in conditions}

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
        cells = eligible_cells(window)
        if not cells:
            continue
        tracks_with_notes = sorted({t for t, _ in cells})
        if len(tracks_with_notes) < 2:
            continue

        order = list(tracks_with_notes)
        rng.shuffle(order)

        seq_a = run_track_by_track(engine_a, tok, window, order, args.temperature, rng)
        seq_e = run_track_by_track(engine_e, tok, window, order, args.temperature, rng)
        seq_blend = run_track_by_track(engine_blend, tok, window, order, args.temperature, rng, blended_model=blended)

        if seq_a:
            sequences["A_alone"].append(seq_a)
        if seq_e:
            sequences["E_alone"].append(seq_e)
        if seq_blend:
            sequences["Blend"].append(seq_blend)

        n_used += 1
        if n_used % 10 == 0:
            print(f"  {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done: {n_used} pieces used.", flush=True)

    def agg_by_step(seqs: list[list[dict]]) -> dict:
        by_step_wass: dict[int, list[float]] = {}
        by_step_deg: dict[int, list[bool]] = {}
        for seq in seqs:
            for m in seq:
                if m["vel_wasserstein"] is not None:
                    by_step_wass.setdefault(m["step"], []).append(m["vel_wasserstein"])
                by_step_deg.setdefault(m["step"], []).append(m["degenerate"])
        return {
            "n_sequences": len(seqs),
            "mean_wasserstein_by_step": {str(k): statistics.mean(v) for k, v in sorted(by_step_wass.items())},
            "degeneracy_rate_by_step": {str(k): sum(v) / len(v) for k, v in sorted(by_step_deg.items())},
        }

    summary = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_e": args.checkpoint_e,
        "n_pieces": n_used,
        "track_by_track": {cond: agg_by_step(seqs) for cond, seqs in sequences.items()},
    }
    (out_dir / "e9c_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_dir / 'e9c_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
