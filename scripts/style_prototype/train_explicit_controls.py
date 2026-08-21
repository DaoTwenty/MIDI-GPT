"""Variant C: joint training of a GPT2 Humanize backbone conditioned on
explicit, hand-designed control statistics (see explicit_controls.py)
instead of Variant A/B's learned StyleEncoder -- same soft-prefix injection
point (explicit_controls_gpt2.py), same MidiGPTDataset humanize-shaped
samples and reference/target bar split as Variant A, different "encoder":
mean_velocity/velocity_std/mean_delta_mag/timing_bias computed directly from
the reference segment's raw VelocityLevel/DeltaDirection/Delta token ids,
quantized, and embedded via small per-control lookup tables.

Dataset settings mirror humanize_tiny_e.json (checkpoint E), same as Variant
A, for a fair three-way comparison.

Usage:
    python3 train_explicit_controls.py --train-data /path/*.parquet \
        --encoder-config ../../models/humanize_encoder.json \
        --output conditioned_gpt2_explicit_controls.pt

Prototype-scope: plain torch.save, standalone training loop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import midigpt._core as _core
from explicit_controls import N_CONTROLS, ControlTypeRanges, compute_controls, quantize_controls
from explicit_controls_gpt2 import ExplicitControlsGPT2
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.model.gpt2 import GPT2Config
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.dataset import MidiGPTDataset


class ExplicitControlsCollator:
    """Pads MidiGPTDataset's normal humanize-shaped samples (input_ids/
    labels/style_ref_mask) and derives quantized control bins from the
    reference-segment subsequence style_ref_mask flags -- the same raw
    material JointConditioningCollator (style_collator.py) turns into a
    StyleVocab token sequence for Variant A, here reduced to 4 scalars
    instead."""

    def __init__(self, ranges: ControlTypeRanges, max_seq_len: int = 2048, pad_value: int = -100):
        self.ranges = ranges
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = [
            torch.tensor(item["input_ids"][: self.max_seq_len], dtype=torch.long) for item in batch
        ]
        max_len = max(t.size(0) for t in input_ids)

        padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), self.pad_value, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            seq_len = ids.size(0)
            padded_ids[i, :seq_len] = ids
            labels[i, :seq_len] = ids

        control_bins = torch.zeros(len(batch), N_CONTROLS, dtype=torch.long)
        has_controls = torch.zeros(len(batch), dtype=torch.bool)
        for i, item in enumerate(batch):
            ids = item["input_ids"]
            mask = item["style_ref_mask"]
            ref_ids = [t for t, m in zip(ids, mask, strict=True) if m]
            values = compute_controls(ref_ids, self.ranges) if ref_ids else None
            if values is not None:
                control_bins[i] = torch.tensor(quantize_controls(values), dtype=torch.long)
                has_controls[i] = True

        return {
            "input_ids": padded_ids,
            "labels": labels,
            "control_bins": control_bins,
            "has_controls": has_controls,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit-controls conditioning training.")
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--encoder-config", required=True, help="Path to humanize_encoder.json")
    parser.add_argument("--output", default="conditioned_gpt2_explicit_controls.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-positions", type=int, default=2048)
    parser.add_argument("--control-dim", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Dataset settings mirrored from humanize_tiny_e.json (checkpoint E),
    # same as train_joint_conditioning.py, for a fair 3-way comparison.
    parser.add_argument("--humanize-bar-fraction", type=float, default=0.5)
    parser.add_argument("--structured-target-probability", type=float, default=0.4)
    parser.add_argument("--context-mechanical-fraction", type=float, default=1.0)
    parser.add_argument("--mechanical-coherent-targets", action="store_true", default=True)
    parser.add_argument("--max-tracks", type=int, default=12)
    parser.add_argument("--min-tracks", type=int, default=1)
    parser.add_argument("--min-fill-ratio", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=0, help="0 = only save at the end")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    ranges = ControlTypeRanges(tokenizer._vocab)
    print(f"LM vocab size: {tokenizer.vocab_size()}")

    dataset = MidiGPTDataset(
        args.train_data,
        tokenizer,
        infill_probability=0.0,
        humanize_probability=1.0,
        humanize_bar_fraction=args.humanize_bar_fraction,
        mask_bar_config=None,
        max_seq_len=args.max_seq_len,
        max_tracks=args.max_tracks,
        min_tracks=args.min_tracks,
        min_fill_ratio=args.min_fill_ratio,
        structured_target_probability=args.structured_target_probability,
        context_mechanical_fraction=args.context_mechanical_fraction,
        mechanical_coherent_targets=args.mechanical_coherent_targets,
    )
    print(f"Dataset size: {len(dataset)}")

    collator = ExplicitControlsCollator(ranges=ranges, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    device = torch.device(args.device)
    # ExplicitControlsGPT2 shifts every real token to position 1..max_seq_len to
    # make room for the controls prefix at position 0 (see
    # explicit_controls_gpt2.py::forward), so wpe needs max_seq_len+1 rows, not
    # max_seq_len -- an unmaxed n_positions here is an out-of-bounds embedding
    # gather (silent on CPU, a device-side assert crash on GPU) the moment a
    # batch has a maxed-out sequence.
    n_positions = max(args.n_positions, args.max_seq_len + 1)
    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    model = ExplicitControlsGPT2(gpt2_cfg, control_dim=args.control_dim).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_total:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(step: int) -> None:
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "gpt2_config": gpt2_cfg,
                "control_dim": args.control_dim,
                "encoder_config_json": enc_cfg.to_json(),
            },
            out_path,
        )

    step = 0
    t0 = time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            control_bins = batch["control_bins"].to(device)
            has_controls = batch["has_controls"].to(device)

            logits = model(input_ids, control_bins, has_controls)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                pct_with_controls = has_controls.float().mean().item()
                print(
                    f"step {step:5d}  loss {loss.item():.4f}  "
                    f"has_controls={pct_with_controls:.2f}  {elapsed:.1f}s"
                )
            step += 1
            if args.save_every and step % args.save_every == 0:
                save(step)

    save(args.steps)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
