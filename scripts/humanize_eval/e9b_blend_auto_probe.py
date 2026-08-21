"""E9b -- does the DEPLOYABLE mode=auto (heuristic-estimated alpha) hold up,
not just the oracle-alpha blend E9 already validated?

E9 set alpha from ground truth -- the test harness itself mechanizes a known
fraction of context tracks, so `alpha = 1 - frac_mech` is exact, not
estimated. That validated the blending MECHANISM (given the right alpha,
does interpolation behave sensibly) but said nothing about whether
`humanize_server.py`'s real detector (`_track_mechanicalness`: fraction of a
context track's notes that land exactly on mechanize.py's own quantization
grid) can actually ESTIMATE that alpha from a real, unlabeled piece.

This script reuses E9's exact mixed_ctx construction (same pieces, same
random mechanization draws would require the same seed -- not attempted;
this is a fresh, comparably-sized sample, not a paired rerun) and adds a
FOURTH condition, Auto_Blend, using the real heuristic instead of oracle
frac_mech -- plus logs both alphas per piece so the detector's accuracy
(oracle vs. estimated) is a direct byproduct, not a separate run.

No gate: descriptive, four-way comparison (A_alone / E_alone / Oracle_Blend
/ Auto_Blend) is the point.

Usage:
    python3 e9b_blend_auto_probe.py \\
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
    from midigpt.http.humanize_server import _track_mechanicalness
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

    conditions = ["A_alone", "E_alone", "Oracle_Blend", "Auto_Blend"]
    records: dict[str, list[dict]] = {c: [] for c in conditions}
    alpha_pairs: list[tuple[float, float]] = []  # (oracle_alpha, estimated_alpha) per piece

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
        if not other_tracks:
            continue

        import copy as _copy

        w_mixed = _copy.deepcopy(window)
        k = rng.randint(1, len(other_tracks))
        mixed_mech = rng.sample(other_tracks, k)
        mechanize_tracks(w_mixed, mixed_mech)
        frac_mech_oracle = k / len(other_tracks)
        oracle_alpha = 1.0 - frac_mech_oracle

        # Real, deployable detector -- exactly what humanize_server.py's
        # mode=auto would compute from this same window, no oracle
        # knowledge of which tracks were actually mechanized.
        resolution = w_mixed.resolution
        fracs = [
            f for t in other_tracks
            if (f := _track_mechanicalness(w_mixed.tracks[t], resolution)) is not None
        ]
        estimated_mechanicalness = sum(fracs) / len(fracs) if fracs else 0.0
        auto_alpha = 1.0 - estimated_mechanicalness
        alpha_pairs.append((oracle_alpha, auto_alpha))

        for cond, alpha in [("A_alone", None), ("E_alone", None), ("Oracle_Blend", oracle_alpha), ("Auto_Blend", auto_alpha)]:
            if cond == "A_alone":
                engine = engine_a
            elif cond == "E_alone":
                engine = engine_e
            else:
                blended.alpha = alpha
                engine = engine_blend
            res = score_scenario(engine, tok, _copy.deepcopy(w_mixed), target_cells, args.temperature, rng)
            if res:
                records[cond].append(metrics_for(*res))

        n_used += 1
        if n_used % 20 == 0:
            print(f"  {n_used}/{args.limit} pieces...", flush=True)

    print(f"Done: {n_used} pieces used.", flush=True)

    def agg(recs: list[dict]) -> dict:
        if not recs:
            return {"n": 0}
        return {
            "n": len(recs),
            "mean_vel_wasserstein": statistics.mean(r["vel_wasserstein"] for r in recs if r["vel_wasserstein"] is not None),
            "degeneracy_rate": sum(r["degenerate"] for r in recs) / len(recs),
        }

    oracle_alphas = [p[0] for p in alpha_pairs]
    auto_alphas = [p[1] for p in alpha_pairs]
    mean_abs_err = statistics.mean(abs(o - a) for o, a in alpha_pairs) if alpha_pairs else None
    corr = None
    if len(alpha_pairs) >= 3 and statistics.pstdev(oracle_alphas) > 0 and statistics.pstdev(auto_alphas) > 0:
        n = len(alpha_pairs)
        mo, ma = statistics.mean(oracle_alphas), statistics.mean(auto_alphas)
        cov = sum((o - mo) * (a - ma) for o, a in alpha_pairs) / n
        corr = cov / (statistics.pstdev(oracle_alphas) * statistics.pstdev(auto_alphas))

    summary = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_e": args.checkpoint_e,
        "n_pieces": n_used,
        "detector_accuracy": {
            "mean_abs_alpha_error": mean_abs_err,
            "pearson_r_oracle_vs_estimated_alpha": corr,
            "mean_oracle_alpha": statistics.mean(oracle_alphas) if oracle_alphas else None,
            "mean_auto_alpha": statistics.mean(auto_alphas) if auto_alphas else None,
        },
        "mixed_ctx": {cond: agg(recs) for cond, recs in records.items()},
    }
    (out_dir / "e9b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_dir / 'e9b_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
