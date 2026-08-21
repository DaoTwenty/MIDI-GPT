"""Eval metric 4-AR: real, KV-cached, grammar-constrained autoregressive
Humanize generation under activation steering -- the mechanism metric 4
(steering_strength_sweep.py) approximates with a single static forward pass
plus one-shot resampling (resample_expressive_from_logits). That approach
cannot exercise the real dependency structure the model was actually
trained on: humanize_probability=1.0 training data is teacher-forced
causally end-to-end, so note N's velocity/delta genuinely conditions what
note N+1's distribution should be, and real generation (session.py's
_sample_step, driving _KVRunner) reproduces exactly that -- each sampled
token is fed back through the KV cache before the next token is drawn.
This script routes activation steering through THAT real loop instead,
via InferenceEngine, for a proper apples-to-apples check against metric 4's
one-shot numbers.

Mechanism: SteeredGPT2LMHeadModel (below) wraps a plain, unconditioned
GPT2LMHeadModel and reuses steered_forward.py's steered_forward() function
unchanged as its forward() body -- same per-block alpha*steer_vec residual
injection, same (input_ids, past_kv, key_mask, position_ids) -> (logits,
presents) signature _KVRunner already expects (session.py:252-283 calls
self._model(ctx_t, pkv, **kwargs) purely by duck typing, no isinstance
checks). InferenceEngine.__init__ accepts an already-built model object
directly (engine.py:60-64), so the checkpoint is loaded once via
tokenizer.checkpoint.load_checkpoint (same call from_checkpoint makes
internally) to get the base model + encoder_config, the model is wrapped,
and InferenceEngine is constructed by hand around the wrapped instance --
production engine.py/session.py/gpt2.py are never modified, matching this
whole directory's existing "never touch SamplingSession/InferenceEngine
internals" convention (steered_forward.py's own docstring).

Only the STEERING mechanism (metric 4) is covered here, not soft-prefix
(metric 2): ConditionedGPT2/ExplicitControlsGPT2 have no KV-cache path (see
conditioned_gpt2.py / metric 2's docstring) and aren't drop-in compatible
with SamplingSession's incremental forward() contract without much larger
changes -- out of scope for this eval-only script.

Two different pipelines feed one comparison, deliberately: the REFERENCE
side (for computing steer_true/steer_shuf, and for the profile we compare
against) reuses steered_forward.py's existing MidiGPTDataset-based
_collect_usable_pieces/compute_steer_vector machinery (token-level,
already validated by metric 2/4). The TARGET side needs a real Score
object with real notes for InferenceEngine.session() to humanize, which
MidiGPTDataset doesn't expose -- so target pieces are picked directly from
the validation parquet via e2_sampling_calibration.py's
pick_window_and_targets (the same real-generation pattern e6_per_instrument.py
already uses), independent of the reference pool. A single-track window,
matching e6's/e2's own scope.

Usage:
    python3 eval/steering_autoregressive.py \\
        --base-checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny_e-20260807-223945/model_final.safetensors \\
        --conditioned-checkpoint $SCRATCH/MIDI-GPT/runs/style_conditioned_joint_random_10gb/conditioned_gpt2_random.pt \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/style_metric4ar_A_random \\
        [--limit 100] [--n-bars 4] [--coverage 0.5] [--alphas 0,1,4] [--layers all]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "humanize_eval"))

import torch

import midigpt._core as _core
from beat_profile import cosine_similarity, profile_points, timing_profile_vector, velocity_profile_vector  # noqa: E402
from e2_sampling_calibration import pick_window_and_targets  # noqa: E402
from explicit_controls import ControlTypeRanges  # noqa: E402
from midigpt.attributes.base import AttributeAnalyzer  # noqa: E402
from midigpt.tokenizer.tokenizer import Tokenizer  # noqa: E402
from midigpt.training.dataset import MidiGPTDataset  # noqa: E402
from steered_forward import (  # noqa: E402
    SteeredGPT2LMHeadModel,
    _collect_usable_pieces,
    build_steered_engine,
    compute_steer_vector,
    load_steering_source,
    steered_forward,
)
from style_vocab import StyleVocab  # noqa: E402

# SteeredGPT2LMHeadModel / build_steered_engine moved to steered_forward.py
# this session so this eval script and humanize_style_server.py share one
# definition instead of two -- see steered_forward.py's module docstring.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--conditioned-checkpoint", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--n-bars", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--layers", default="all", help="Comma-separated block indices, or 'all'")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    vocab = tokenizer._vocab
    style_vocab = StyleVocab(vocab)

    print(f"Loading base checkpoint (once): {args.base_checkpoint}", flush=True)
    from midigpt.tokenizer.checkpoint import load_checkpoint

    base_bundle = load_checkpoint(args.base_checkpoint, device=str(device))
    base_model = base_bundle.model
    base_model.eval()
    base_enc_cfg = base_bundle.encoder_config
    layers = set(range(base_model.cfg.n_layer)) if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    if not layers.issubset(set(range(base_model.cfg.n_layer))):
        raise ValueError(f"--layers {sorted(layers)} outside base model's {base_model.cfg.n_layer} blocks")

    print(f"Loading steering source: {args.conditioned_checkpoint}", flush=True)
    encoder, z_proj, style_cfg = load_steering_source(args.conditioned_checkpoint, device)
    if style_cfg.vocab_size != style_vocab.size:
        raise ValueError(
            f"Conditioned checkpoint's StyleVocab size={style_cfg.vocab_size} does not match "
            f"this encoder config's StyleVocab size={style_vocab.size}."
        )

    print("Building reference pool (MidiGPTDataset, token-level)...", flush=True)
    ref_dataset = MidiGPTDataset(
        args.val_parquet, tokenizer,
        infill_probability=0.0, humanize_probability=1.0, humanize_bar_fraction=0.5,
        mask_bar_config=None, max_seq_len=2048, context_mechanical_fraction=1.0,
        structured_target_probability=0.4, mechanical_coherent_targets=True,
    )
    ref_pieces = _collect_usable_pieces(ref_dataset, style_vocab, args.limit)
    n_ref = len(ref_pieces)
    print(f"Reference pool: {n_ref} usable pieces", flush=True)
    if n_ref < 10:
        raise RuntimeError(f"Too few reference pieces ({n_ref}).")
    ref_profiles = [
        (lambda score: (
            (velocity_profile_vector(profile_points(score, p["reference_bars"])), timing_profile_vector(profile_points(score, p["reference_bars"])))
            if score.tracks and score.tracks[0].bars else ([None] * 12, [None] * 12)
        ))(tokenizer.decode(p["input_ids"]))
        for p in ref_pieces
    ]

    print("Picking target pieces directly from validation parquet (real Score, real notes)...", flush=True)
    import pyarrow.parquet as pq
    from midigpt._types import Score

    pf = pq.ParquetFile(args.val_parquet)
    raw_pieces: list[bytes] = []
    for batch in pf.iter_batches(columns=["music"], batch_size=200):
        for music in batch.column("music"):
            raw_pieces.append(bytes(music.as_py()))
        if len(raw_pieces) >= max(args.limit * 3, 50):
            break
    rng.shuffle(raw_pieces)

    targets = []
    for raw in raw_pieces:
        if len(targets) >= args.limit:
            break
        try:
            score = Score.from_bytes(raw)
        except Exception:
            continue
        picked = pick_window_and_targets(score, args.n_bars, args.coverage, rng)
        if picked is None:
            continue
        window, target_bars = picked
        if getattr(window.tracks[0], "track_type", "melodic") == "drum":
            continue
        targets.append((window, target_bars))
    n_tgt = len(targets)
    print(f"Target pieces: {n_tgt}", flush=True)
    if n_tgt < 5:
        raise RuntimeError(f"Too few target pieces ({n_tgt}).")

    def diff_ci(a, b):
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        n_d = len(diffs)
        if n_d < 5:
            return None
        boot_means = []
        for _ in range(args.n_bootstrap):
            sample = [diffs[rng.randrange(n_d)] for _ in range(n_d)]
            boot_means.append(statistics.mean(sample))
        boot_means.sort()
        return {
            "mean": statistics.mean(diffs),
            "ci_lo": boot_means[int(0.025 * args.n_bootstrap)],
            "ci_hi": boot_means[int(0.975 * args.n_bootstrap) - 1],
            "win_rate": sum(1 for d in diffs if d > 0) / n_d,
            "n_pairs": n_d,
        }

    sweep_results = []
    for alpha in alphas:
        sims_vel_true, sims_vel_shuf = [], []
        sims_time_true, sims_time_shuf = [], []

        for ti, (window, target_bars) in enumerate(targets):
            ref_i = rng.randrange(n_ref)
            shuf_i = rng.choice([j for j in range(n_ref) if j != ref_i])
            ref_vel_profile, ref_time_profile = ref_profiles[ref_i]

            steer_true = compute_steer_vector(encoder, z_proj, ref_pieces[ref_i]["style_ids"], device)
            steer_shuf = compute_steer_vector(encoder, z_proj, ref_pieces[shuf_i]["style_ids"], device)

            req_kwargs = dict(
                temperature=args.temperature, top_p=args.top_p, bars_per_step=max(1, args.n_bars),
                tracks_per_step=1, model_dim=args.n_bars, mask_mode="remove",
                seed=rng.randrange(1 << 30), novelty_check=False, silence_check=False,
            )

            from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt

            def _generate(steer_vec):
                engine = build_steered_engine(base_model, base_enc_cfg, steer_vec, layers, alpha, device)
                req = GenerationRequest(
                    tracks=[TrackPrompt(id=0, bars=sorted(target_bars), humanize=True)],
                    config=InferenceConfig(**req_kwargs),
                )
                return engine.session(copy.deepcopy(window), req).run()

            try:
                gen_true = _generate(steer_true)
                gen_shuf = _generate(steer_shuf)
            except Exception as exc:
                print(f"  [skip target {ti}] generation failed: {exc}", flush=True)
                continue

            pts_true = profile_points(gen_true, target_bars)
            pts_shuf = profile_points(gen_shuf, target_bars)
            gen_true_vel, gen_true_time = velocity_profile_vector(pts_true), timing_profile_vector(pts_true)
            gen_shuf_vel, gen_shuf_time = velocity_profile_vector(pts_shuf), timing_profile_vector(pts_shuf)

            sim_vt = cosine_similarity(gen_true_vel, ref_vel_profile)
            sim_vs = cosine_similarity(gen_shuf_vel, ref_vel_profile)
            sim_tt = cosine_similarity(gen_true_time, ref_time_profile)
            sim_ts = cosine_similarity(gen_shuf_time, ref_time_profile)
            if None not in (sim_vt, sim_vs):
                sims_vel_true.append(sim_vt)
                sims_vel_shuf.append(sim_vs)
            if None not in (sim_tt, sim_ts):
                sims_time_true.append(sim_tt)
                sims_time_shuf.append(sim_ts)

        entry = {
            "alpha": alpha,
            "velocity_faithfulness": diff_ci(sims_vel_true, sims_vel_shuf),
            "timing_faithfulness": diff_ci(sims_time_true, sims_time_shuf),
        }
        sweep_results.append(entry)
        print(f"  alpha={alpha:>5}  {json.dumps(entry)}", flush=True)

    summary = {
        "base_checkpoint": args.base_checkpoint,
        "conditioned_checkpoint": args.conditioned_checkpoint,
        "layers": sorted(layers),
        "n_ref_pieces": n_ref,
        "n_target_pieces": n_tgt,
        "sweep": sweep_results,
    }
    (out_dir / "metric4ar_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'metric4ar_summary.json'}")


if __name__ == "__main__":
    main()
