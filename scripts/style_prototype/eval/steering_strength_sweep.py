"""Eval metric 4: sweep steering strength (alpha) and report, per alpha,
metric 2's faithfulness score (per-beat-profile cosine similarity of the
generated continuation vs. the reference's own profile, true-z vs.
shuffled-z control) together with a fluency/degeneracy proxy (generated
velocity_std / mean_delta_mag, reused from Variant C's
explicit_controls.compute_controls -- a value collapsing toward 0 as alpha
grows is the "distribution saturates" failure mode the plan calls out).

Same one-shot, appendix-scoped resampling protocol as metric 2
(resample_expressive_from_logits, shared via steered_forward.py --
restricted to HumanizeStart...HumanizeEnd, the only region the model was
ever trained to vary based on z; see that function's docstring), with the
injection mechanism swapped from soft-prefix (ConditionedGPT2) to
activation steering (steered_forward on the base, unconditioned
checkpoint) -- so these faithfulness numbers are directly comparable to
metric 2's soft-prefix result. Profile comparison is likewise scoped to
each piece's own humanize_bars/reference_bars (MidiGPTDataset), not the
whole piece.

No checkpoint paths are hardcoded here or in the slurm wrapper
(slurm/style_eval_all_metrics_cpu.sh) that bundles this with metrics 1-3 --
pass --base-checkpoint/--conditioned-checkpoint (or the wrapper's
BASE_CKPT/RANDOM_CKPT/PRETRAINED_CKPT env vars) for whichever training run
you want evaluated. --variant {A,C} (default A) selects Variant A's
z_proj/StyleEncoder or Variant C's combine/control_embeds as the steering
source -- see steered_forward.py's load_steering_source_c.

Usage:
    python3 eval/steering_strength_sweep.py \\
        --base-checkpoint $SCRATCH/MIDI-GPT/runs/humanize_tiny_e-20260807-223945/model_final.safetensors \\
        --conditioned-checkpoint $SCRATCH/MIDI-GPT/runs/style_conditioned_joint_random_10gb/conditioned_gpt2_random.pt \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/style_metric4_A_random \\
        [--limit 100] [--alphas 0,0.25,0.5,1,2,4,8] [--layers all]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "humanize_eval"))

import torch

import midigpt._core as _core
from beat_profile import cosine_similarity  # noqa: E402
from explicit_controls import ControlTypeRanges, compute_controls  # noqa: E402
from midigpt.attributes.base import AttributeAnalyzer  # noqa: E402
from midigpt.inference.model.gpt2 import GPT2LMHeadModel  # noqa: E402
from midigpt.tokenizer.tokenizer import Tokenizer  # noqa: E402
from midigpt.training.dataset import MidiGPTDataset  # noqa: E402
from steered_forward import (  # noqa: E402
    _collect_usable_pieces,
    _collect_usable_pieces_c,
    compute_steer_vector,
    compute_steer_vector_c,
    load_steering_source,
    load_steering_source_c,
    resample_expressive_from_logits,
    steered_forward,
)
from style_transfer_faithfulness import piece_profiles  # noqa: E402
from style_vocab import StyleVocab  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--conditioned-checkpoint", required=True)
    parser.add_argument("--variant", default="A", choices=["A", "C"])
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--alphas", default="0,0.25,0.5,1,2,4,8")
    parser.add_argument("--layers", default="all", help="Comma-separated block indices, or 'all'")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    ranges = ControlTypeRanges(vocab)

    print(f"Loading base checkpoint: {args.base_checkpoint}", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.base_checkpoint, device=device)
    model.eval()
    layers = (
        set(range(model.cfg.n_layer)) if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    )
    if not layers.issubset(set(range(model.cfg.n_layer))):
        raise ValueError(f"--layers {sorted(layers)} outside base model's {model.cfg.n_layer} blocks")

    print(f"Loading steering source (variant {args.variant}): {args.conditioned_checkpoint}", flush=True)
    if args.variant == "A":
        style_vocab = StyleVocab(vocab)
        encoder, z_proj, style_cfg = load_steering_source(args.conditioned_checkpoint, device)
        if style_cfg.vocab_size != style_vocab.size:
            raise ValueError(
                f"Conditioned checkpoint's StyleVocab size={style_cfg.vocab_size} does not match "
                f"this encoder config's StyleVocab size={style_vocab.size}."
            )
    else:
        control_embeds, combine = load_steering_source_c(args.conditioned_checkpoint, device)

    print("Building held-out dataset...", flush=True)
    dataset = MidiGPTDataset(
        args.val_parquet,
        tokenizer,
        infill_probability=0.0,
        humanize_probability=1.0,
        humanize_bar_fraction=0.5,
        mask_bar_config=None,
        max_seq_len=args.max_seq_len,
        context_mechanical_fraction=1.0,
        structured_target_probability=0.4,
        mechanical_coherent_targets=True,
    )
    pieces = (
        _collect_usable_pieces(dataset, style_vocab, args.limit)
        if args.variant == "A"
        else _collect_usable_pieces_c(dataset, ranges, args.limit)
    )
    n = len(pieces)
    print(f"Held-out pieces with a usable reference: {n}", flush=True)
    if n < 10:
        raise RuntimeError(f"Too few usable samples ({n}) -- check dataset/checkpoint compatibility.")

    print("Decoding each piece's own (reference-segment) profile...", flush=True)
    ref_profiles = [
        piece_profiles(tokenizer, p["input_ids"][: args.max_seq_len], p["reference_bars"]) for p in pieces
    ]

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    idxs = list(range(n))
    pairs = []
    for target_i in range(n):
        others = [j for j in idxs if j != target_i]
        ref_i = rng.choice(others)
        shuf_i = rng.choice([j for j in others if j != ref_i])
        pairs.append((target_i, ref_i, shuf_i))

    def diff_ci(a: list[float], b: list[float]) -> dict | None:
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
        vel_stds, delta_mags = [], []

        for target_i, ref_i, shuf_i in pairs:
            target_ids = pieces[target_i]["input_ids"][: args.max_seq_len]
            target_bars = pieces[target_i]["humanize_bars"]
            ref_vel_profile, ref_time_profile = ref_profiles[ref_i]

            if args.variant == "A":
                steer_true = compute_steer_vector(encoder, z_proj, pieces[ref_i]["style_ids"], device)
                steer_shuf = compute_steer_vector(encoder, z_proj, pieces[shuf_i]["style_ids"], device)
            else:
                steer_true = compute_steer_vector_c(control_embeds, combine, pieces[ref_i]["control_bins"], device)
                steer_shuf = compute_steer_vector_c(control_embeds, combine, pieces[shuf_i]["control_bins"], device)

            ids_t = torch.tensor([target_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits_true, _ = steered_forward(model, ids_t, steer_true, layers, alpha)
                logits_shuf, _ = steered_forward(model, ids_t, steer_shuf, layers, alpha)
            gen_true_ids = resample_expressive_from_logits(
                logits_true, target_ids, vocab, ranges, args.temperature, gen
            )
            gen_shuf_ids = resample_expressive_from_logits(
                logits_shuf, target_ids, vocab, ranges, args.temperature, gen
            )

            gen_true_vel, gen_true_time = piece_profiles(tokenizer, gen_true_ids, target_bars)
            gen_shuf_vel, gen_shuf_time = piece_profiles(tokenizer, gen_shuf_ids, target_bars)

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

            controls_true = compute_controls(gen_true_ids, ranges)
            if controls_true is not None:
                vel_stds.append(controls_true[1])
                delta_mags.append(controls_true[2])

        entry = {
            "alpha": alpha,
            "velocity_faithfulness": diff_ci(sims_vel_true, sims_vel_shuf),
            "timing_faithfulness": diff_ci(sims_time_true, sims_time_shuf),
            "degeneracy": {
                "mean_velocity_std": statistics.mean(vel_stds) if vel_stds else None,
                "mean_delta_mag": statistics.mean(delta_mags) if delta_mags else None,
            },
        }
        sweep_results.append(entry)
        print(f"  alpha={alpha:>5}  {json.dumps(entry)}", flush=True)

    summary = {
        "base_checkpoint": args.base_checkpoint,
        "conditioned_checkpoint": args.conditioned_checkpoint,
        "variant": args.variant,
        "layers": sorted(layers),
        "n_pairs": n,
        "sweep": sweep_results,
    }
    (out_dir / "metric4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'metric4_summary.json'}")


if __name__ == "__main__":
    main()
