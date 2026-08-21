"""Eval metric 1 (plan's hard gate): does the style/control prefix actually
help predict a held-out piece's own expressive tokens, or is the model
ignoring it?

For each held-out piece with a real reference segment, scores the SAME
input_ids/labels under three prefix conditions:

  true       -- the piece's own reference segment (or explicit controls
                computed from it)
  mismatched -- another, randomly chosen piece's reference/controls
  none       -- the model's learned "no reference available" fallback
                (has_style=False / has_controls=False)

Loss is restricted to expressive-token positions (VelocityLevel/
DeltaDirection/Delta) strictly inside the HumanizeStart...HumanizeEnd
appendix (steered_forward.py's appendix_expressive_mask) -- the only region
the model was ever trained to vary based on z. In-place expressive tokens
elsewhere (real or mechanized context bars, held fixed regardless of z) are
excluded. An earlier version of this script scored in-place expressive
tokens across the WHOLE sequence instead (via dataset.py's
`_expressive_mask`, which itself always excludes appendix content -- built
for a different purpose, flagging reference-segment tokens for the style
encoder) -- that version could never actually score the humanize target the
conditioning mechanism is meant to affect, and diluted the true/mismatched/
none CE gap with loss on positions z was never trained to influence.

Gate: true must clearly beat both mismatched and none (win-rate mean +
bootstrap CI clearly above 0.5) for the conditioning mechanism to be worth
building further eval (metrics 2-5) on top of. Works for any of the three
style-conditioning variants (A/random, A/pretrained, C/explicit) via
--variant.

Usage:
    python3 eval/held_out_conditional_loss.py \\
        --checkpoint $SCRATCH/MIDI-GPT/runs/style_conditioned_joint_random/conditioned_gpt2_random.pt \\
        --variant A \\
        --val-parquet $SCRATCH/MIDI-GPT/data/humanize_filtered_v2/validation.parquet \\
        --encoder-config ../../models/humanize_encoder.json \\
        --output-dir $SCRATCH/MIDI-GPT/humanize_eval/style_metric1_A_random \\
        [--limit 300]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import midigpt._core as _core
from conditioned_gpt2 import ConditionedGPT2
from explicit_controls import ControlTypeRanges, N_CONTROLS, compute_controls, quantize_controls
from explicit_controls_gpt2 import ExplicitControlsGPT2
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.model.gpt2 import GPT2Config
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.dataset import MidiGPTDataset
from steered_forward import appendix_expressive_mask
from style_encoder import StyleEncoderConfig
from style_vocab import StyleVocab


def _load_variant_a(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gpt2_cfg: GPT2Config = ckpt["gpt2_config"]
    style_cfg: StyleEncoderConfig = ckpt["style_encoder_config"]
    model = ConditionedGPT2(gpt2_cfg, style_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["encoder_config_json"], style_cfg


def _pad_style_seq(seq: list[int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.zeros(1, max(1, len(seq)), dtype=torch.long)
    mask = torch.zeros(1, max(1, len(seq)), dtype=torch.bool)
    if seq:
        ids[0, : len(seq)] = torch.tensor(seq[:max_len], dtype=torch.long)
        mask[0, : len(seq)] = True
    return ids, mask


def _variant_a_scoring(model, style_vocab, style_ids_by_piece):
    """Returns fn(input_ids_1xT, ref_idx, mismatch_idx | None) -> per-token
    CE tensor (1, T), for Variant A's (style_ids, style_key_mask,
    has_style)-shaped conditioning."""

    def score(input_ids: torch.Tensor, use_idx: int | None, device: torch.device):
        if use_idx is None:
            style_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            style_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            has_style = torch.zeros(1, dtype=torch.bool, device=device)
        else:
            seq = style_ids_by_piece[use_idx]
            ids, mask = _pad_style_seq(seq, model.style_cfg.max_len)
            style_ids, style_mask = ids.to(device), mask.to(device)
            has_style = torch.ones(1, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = model(input_ids, style_ids, style_mask, has_style)
        return logits

    return score


def _load_variant_c(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gpt2_cfg: GPT2Config = ckpt["gpt2_config"]
    model = ExplicitControlsGPT2(gpt2_cfg, control_dim=ckpt["control_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["encoder_config_json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["A", "C"])
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--encoder-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-seq-len", type=int, default=1024)
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

    print(f"Loading checkpoint ({args.variant})...", flush=True)
    if args.variant == "A":
        model, _, style_cfg = _load_variant_a(args.checkpoint, device)
        style_vocab = StyleVocab(vocab)
    else:
        model, _ = _load_variant_c(args.checkpoint, device)
        ranges = ControlTypeRanges(vocab)

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

    n_avail = len(dataset)
    idxs = list(range(n_avail))
    rng.shuffle(idxs)

    # Pass 1: collect samples with a real reference segment, and precompute
    # each one's style_ids / control_bins so "mismatched" can draw from any
    # other already-collected piece.
    samples = []
    for i in idxs:
        if len(samples) >= args.limit:
            break
        item = dataset[i]
        ids, mask = item["input_ids"], item["style_ref_mask"]
        ref_raw = [t for t, m in zip(ids, mask, strict=True) if m]
        if not ref_raw:
            continue
        entry = {"input_ids": ids}
        if args.variant == "A":
            entry["style_ids"] = style_vocab.to_local_seq(ref_raw)
            if not entry["style_ids"]:
                continue
        else:
            values = compute_controls(ref_raw, ranges)
            if values is None:
                continue
            entry["control_bins"] = quantize_controls(values)
        samples.append(entry)

    n = len(samples)
    print(f"Held-out pieces with a usable reference: {n}", flush=True)
    if n < 10:
        raise RuntimeError(f"Too few usable samples ({n}) -- check dataset/checkpoint compatibility.")

    def expr_ce(input_ids: torch.Tensor, logits: torch.Tensor, mask_bool: list[bool]) -> float | None:
        # logits: (1, T, V) predicting input_ids[:, i] at position i (see
        # ConditionedGPT2/ExplicitControlsGPT2 forward docstrings).
        mask_t = torch.tensor(mask_bool, dtype=torch.bool)
        if not mask_t.any():
            return None
        sel_logits = logits[0][mask_t]
        sel_labels = input_ids[0][mask_t]
        loss = F.cross_entropy(sel_logits, sel_labels, reduction="mean")
        return loss.item()

    results = {"true": [], "mismatched": [], "none": []}
    wins_vs_mismatched = []
    wins_vs_none = []

    if args.variant == "A":
        style_ids_by_piece = [s["style_ids"] for s in samples]
        score_fn = _variant_a_scoring(model, style_vocab, style_ids_by_piece)

    for i, s in enumerate(samples):
        input_ids = torch.tensor([s["input_ids"][: args.max_seq_len]], dtype=torch.long, device=device)
        expr_mask = appendix_expressive_mask(s["input_ids"][: args.max_seq_len], vocab)

        if args.variant == "A":
            mismatch_idx = rng.choice([j for j in range(n) if j != i])
            logits_true = score_fn(input_ids, i, device)
            logits_mismatched = score_fn(input_ids, mismatch_idx, device)
            logits_none = score_fn(input_ids, None, device)
        else:
            def bins_tensor(bins):
                return torch.tensor([bins], dtype=torch.long, device=device)

            mismatch_idx = rng.choice([j for j in range(n) if j != i])
            has_true = torch.ones(1, dtype=torch.bool, device=device)
            has_none = torch.zeros(1, dtype=torch.bool, device=device)
            with torch.no_grad():
                logits_true = model(input_ids, bins_tensor(s["control_bins"]), has_true)
                logits_mismatched = model(
                    input_ids, bins_tensor(samples[mismatch_idx]["control_bins"]), has_true
                )
                logits_none = model(
                    input_ids, bins_tensor([0] * N_CONTROLS), has_none
                )

        ce_true = expr_ce(input_ids, logits_true, expr_mask)
        ce_mismatched = expr_ce(input_ids, logits_mismatched, expr_mask)
        ce_none = expr_ce(input_ids, logits_none, expr_mask)
        if ce_true is None or ce_mismatched is None or ce_none is None:
            continue

        results["true"].append(ce_true)
        results["mismatched"].append(ce_mismatched)
        results["none"].append(ce_none)
        wins_vs_mismatched.append(ce_true < ce_mismatched)
        wins_vs_none.append(ce_true < ce_none)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} pieces scored...", flush=True)

    def bootstrap_ci(wins: list[bool], n_boot: int) -> tuple[float, float, float]:
        mean = sum(wins) / len(wins)
        boot_means = []
        for _ in range(n_boot):
            sample = [rng.choice(wins) for _ in range(len(wins))]
            boot_means.append(sum(sample) / len(sample))
        boot_means.sort()
        lo = boot_means[int(0.025 * n_boot)]
        hi = boot_means[int(0.975 * n_boot) - 1]
        return mean, lo, hi

    win_rate_mismatched, ci_lo_m, ci_hi_m = bootstrap_ci(wins_vs_mismatched, args.n_bootstrap)
    win_rate_none, ci_lo_n, ci_hi_n = bootstrap_ci(wins_vs_none, args.n_bootstrap)

    summary = {
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "n_scored": len(results["true"]),
        "mean_ce": {k: statistics.mean(v) for k, v in results.items()},
        "win_rate_vs_mismatched": {"mean": win_rate_mismatched, "ci95": [ci_lo_m, ci_hi_m]},
        "win_rate_vs_none": {"mean": win_rate_none, "ci95": [ci_lo_n, ci_hi_n]},
        "gate_passed": ci_lo_m > 0.5 and ci_lo_n > 0.5,
    }
    (out_dir / "metric1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nGate {'PASSED' if summary['gate_passed'] else 'NOT PASSED'}.", flush=True)
    print(f"Wrote {out_dir / 'metric1_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
