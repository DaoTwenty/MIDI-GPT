"""Eval metric 2 (only for combinations that pass metric 1): does
conditioning generation on a reference performance's z actually make the
generated content resemble that reference's own playing style, more than a
random/mismatched z would?

For held-out (reference, target) piece pairs: resample the target's
Velocity/DeltaDirection/Delta token VALUES conditioned on the reference's z
(soft-prefix, via ConditionedGPT2 -- the only variant with a trained z_proj
today; steering's version of this metric would reuse steered_forward.py's
generation loop instead, not built here), decode to a Score, and compute a
per-beat-position profile (mean velocity + mean |microtiming offset| per
sub-beat bin, see beat_profile.py, shared with E0/E4 per
EXPERIMENT_PLAN.md's "share this" note). Compare the generated profile to
the reference piece's OWN full-piece profile via cosine similarity, under
the true reference z vs. a shuffled (randomly mismatched) z as the null
baseline. Gate: bootstrap CI of mean(sim_true - sim_shuffled) should
clearly exclude zero (positive) -- direct evidence of real style transfer,
not just "the prefix exists".

Scoping (fixed -- MidiGPTDataset now returns `humanize_bars`/`reference_bars`,
see dataset.py's `_encode_one`): resampling (resample_expressive_from_logits,
steered_forward.py) is restricted to VelocityLevel/DeltaDirection/Delta
tokens inside the target's HumanizeStart...HumanizeEnd appendix -- the only
region the model was ever trained to vary based on z. In-place expressive
tokens elsewhere in the sequence (real or mechanized context bars) are never
touched, since the model was trained to reproduce those as a fixed value
regardless of z. Profile comparison is scoped to match: the generated
profile is computed only over the target piece's own `humanize_bars`, and
the reference profile only over the reference piece's own `reference_bars`
(the exact segment its z was computed from) -- not whole-piece averages,
which would dilute any real difference under a majority of bars that were
never meant to change.

Real scope simplifications still in effect (documented, not silently
assumed, matching held_out_conditional_loss.py's precedent):
  - Resampling is a single teacher-forced forward pass over the target's
    TRUE sequence (conditioned on the chosen z), with a new value drawn
    independently at each expressive slot (restricted to that slot's own
    known token type's raw-id range -- not general grammar-constrained
    decoding, just "we already know this position must be a VelocityLevel/
    DeltaDirection/Delta token from the true sequence"). This is NOT
    autoregressive left-to-right generation: ConditionedGPT2 has no KV-cache
    path (see conditioned_gpt2.py), so true sequential generation would
    need one full-sequence forward pass PER expressive position -- too
    expensive for this eval at scale. Each resampled position is
    conditioned on the true surrounding context, not on other newly-
    resampled positions; this still exercises the same conditioning
    mechanism metric 1 validated, just without autoregressive drift.

--variant {A,C} (default A) selects Variant A's ConditionedGPT2/style_ids or
Variant C's ExplicitControlsGPT2/control_bins -- both funnel through the
same appendix-scoped resample_expressive_from_logits and bar-scoped
piece_profiles, so metric 2 numbers are directly comparable across variants.

Usage:
    python3 eval/style_transfer_faithfulness.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/style_conditioned_joint_random_10gb/conditioned_gpt2_random.pt \\
        --variant A \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/style_metric2_A_random \\
        [--limit 100] [--temperature 1.0]
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
from beat_profile import cosine_similarity, profile_points, timing_profile_vector, velocity_profile_vector  # noqa: E402
from explicit_controls import ControlTypeRanges  # noqa: E402
from held_out_conditional_loss import _load_variant_a, _load_variant_c  # noqa: E402
from midigpt.attributes.base import AttributeAnalyzer  # noqa: E402
from midigpt.tokenizer.tokenizer import Tokenizer  # noqa: E402
from midigpt.training.dataset import MidiGPTDataset  # noqa: E402
from steered_forward import (  # noqa: E402
    _collect_usable_pieces,
    _collect_usable_pieces_c,
    resample_expressive_from_logits,
)
from style_vocab import StyleVocab  # noqa: E402

def resample_expressive(
    model,
    input_ids: list[int],
    style_ids: list[int],
    vocab: "_core.Vocabulary",
    ranges: ControlTypeRanges,
    temperature: float,
    device: torch.device,
    generator: torch.Generator,
) -> list[int]:
    """One teacher-forced forward pass conditioned on `style_ids`'s z;
    resamples a new value at every expressive slot from the model's own
    predicted distribution there (restricted to that slot's known token
    type's raw-id range). See module docstring for the scope tradeoff vs.
    true autoregressive generation."""
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    if style_ids:
        style_t = torch.tensor([style_ids], dtype=torch.long, device=device)
        style_mask = torch.ones(1, len(style_ids), dtype=torch.bool, device=device)
        has_style = torch.ones(1, dtype=torch.bool, device=device)
    else:
        style_t = torch.zeros(1, 1, dtype=torch.long, device=device)
        style_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        has_style = torch.zeros(1, dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = model(ids_t, style_t, style_mask, has_style)  # (1, T, V)

    return resample_expressive_from_logits(logits, input_ids, vocab, ranges, temperature, generator)


def resample_expressive_c(
    model,
    input_ids: list[int],
    control_bins: list[int],
    vocab: "_core.Vocabulary",
    ranges: ControlTypeRanges,
    temperature: float,
    device: torch.device,
    generator: torch.Generator,
) -> list[int]:
    """Variant C counterpart of resample_expressive: same protocol, but the
    prefix comes from ExplicitControlsGPT2's control_bins/has_controls
    instead of Variant A's style_ids/style_mask/has_style."""
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    bins_t = torch.tensor([control_bins], dtype=torch.long, device=device)
    has_controls = torch.ones(1, dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = model(ids_t, bins_t, has_controls)  # (1, T, V)

    return resample_expressive_from_logits(logits, input_ids, vocab, ranges, temperature, generator)


def piece_profiles(tokenizer: Tokenizer, token_ids: list[int], bars_of_interest: set[int] | None = None) -> tuple[list, list]:
    """Per-beat-position profile over `bars_of_interest` (track 0), or every
    bar if none given/empty (fallback for a piece with no usable target/
    reference bars). Restricting to the actual bars of interest matters:
    most of a piece's notes are untouched by z (see module docstring), so
    including them here would dilute any real profile difference under
    a majority of identical, uninformative bins."""
    score = tokenizer.decode(token_ids)
    if not score.tracks or not score.tracks[0].bars:
        return [None] * 12, [None] * 12
    all_bars = set(range(len(score.tracks[0].bars)))
    bars = (bars_of_interest & all_bars) if bars_of_interest else all_bars
    if not bars:
        bars = all_bars
    pts = profile_points(score, bars)
    return velocity_profile_vector(pts), timing_profile_vector(pts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="A", choices=["A", "C"])
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    vocab = tokenizer._vocab
    ranges = ControlTypeRanges(vocab)

    print(f"Loading checkpoint (variant {args.variant})...", flush=True)
    if args.variant == "A":
        style_vocab = StyleVocab(vocab)
        model, _, style_cfg = _load_variant_a(args.checkpoint, device)
        if style_cfg.vocab_size != style_vocab.size:
            raise ValueError(
                f"Checkpoint's StyleVocab size={style_cfg.vocab_size} does not match "
                f"this encoder config's StyleVocab size={style_vocab.size}."
            )
    else:
        model, _ = _load_variant_c(args.checkpoint, device)

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

    sims_vel_true, sims_vel_shuf = [], []
    sims_time_true, sims_time_shuf = [], []

    idxs = list(range(n))
    for target_i in range(n):
        others = [j for j in idxs if j != target_i]
        ref_i = rng.choice(others)
        shuf_i = rng.choice([j for j in others if j != ref_i])

        target_ids = pieces[target_i]["input_ids"][: args.max_seq_len]
        target_bars = pieces[target_i]["humanize_bars"]
        ref_vel_profile, ref_time_profile = ref_profiles[ref_i]

        if args.variant == "A":
            gen_true_ids = resample_expressive(
                model, target_ids, pieces[ref_i]["style_ids"], vocab, ranges, args.temperature, device, gen
            )
            gen_shuf_ids = resample_expressive(
                model, target_ids, pieces[shuf_i]["style_ids"], vocab, ranges, args.temperature, device, gen
            )
        else:
            gen_true_ids = resample_expressive_c(
                model, target_ids, pieces[ref_i]["control_bins"], vocab, ranges, args.temperature, device, gen
            )
            gen_shuf_ids = resample_expressive_c(
                model, target_ids, pieces[shuf_i]["control_bins"], vocab, ranges, args.temperature, device, gen
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

        if (target_i + 1) % 20 == 0:
            print(f"  {target_i + 1}/{n} pieces scored...", flush=True)

    def bootstrap_diff_ci(a: list[float], b: list[float], n_boot: int):
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        n_d = len(diffs)
        if n_d < 5:
            return None
        boot_means = []
        for _ in range(n_boot):
            sample = [diffs[rng.randrange(n_d)] for _ in range(n_d)]
            boot_means.append(statistics.mean(sample))
        boot_means.sort()
        return {
            "mean": statistics.mean(diffs),
            "ci_lo": boot_means[int(0.025 * n_boot)],
            "ci_hi": boot_means[int(0.975 * n_boot) - 1],
            "win_rate": sum(1 for d in diffs if d > 0) / n_d,
            "n_pairs": n_d,
        }

    vel_result = bootstrap_diff_ci(sims_vel_true, sims_vel_shuf, args.n_bootstrap)
    time_result = bootstrap_diff_ci(sims_time_true, sims_time_shuf, args.n_bootstrap)

    summary = {
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "n_scored": n,
        "velocity_profile": {
            "mean_sim_true": statistics.mean(sims_vel_true) if sims_vel_true else None,
            "mean_sim_shuffled": statistics.mean(sims_vel_shuf) if sims_vel_shuf else None,
            "diff_ci": vel_result,
            "gate_passed": bool(vel_result and vel_result["ci_lo"] > 0),
        },
        "timing_profile": {
            "mean_sim_true": statistics.mean(sims_time_true) if sims_time_true else None,
            "mean_sim_shuffled": statistics.mean(sims_time_shuf) if sims_time_shuf else None,
            "diff_ci": time_result,
            "gate_passed": bool(time_result and time_result["ci_lo"] > 0),
        },
    }
    (out_dir / "metric2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'metric2_summary.json'}")


if __name__ == "__main__":
    main()
