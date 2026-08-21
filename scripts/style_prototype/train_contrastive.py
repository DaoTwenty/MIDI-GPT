"""Variant B: contrastive (InfoNCE) pretraining of the style encoder.

LM-independent -- no GPT2 checkpoint needed. Anchor/positive = two disjoint
segments from the same piece (MidiGPTDataset(style_pretrain_mode=True)); in-
batch negatives = every other piece's segments in the same batch (standard
SimCLR/InfoNCE, no separate negative mining).

Usage:
    python3 train_contrastive.py --train-data /path/*.parquet \
        --encoder-config ../../models/humanize_encoder.json \
        --output style_encoder_contrastive.pt

This is prototype-scope: plain torch.save (not the production packed-
safetensors format), no Lightning dependency required (kept deliberately
simple -- this is the fastest-feedback experiment in the whole plan, meant
to be re-run/iterated on quickly).
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
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.dataset import MidiGPTDataset
from style_collator import ContrastiveCollator
from style_encoder import StyleEncoder, StyleEncoderConfig
from style_vocab import StyleVocab


def info_nce_loss(z_a: torch.Tensor, z_p: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE / NT-Xent over in-batch negatives.

    z_a, z_p: (B, z_dim), already unit-normalized (StyleEncoder does this).
    For anchor i, positive i is the target; the other B-1 positives (j != i)
    are negatives -- no separate mining, this is the standard SimCLR setup.
    """
    logits = z_a @ z_p.T / temperature  # (B, B)
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive style-encoder pretraining.")
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--encoder-config", required=True, help="Path to humanize_encoder.json")
    parser.add_argument("--output", default="style_encoder_contrastive.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--max-style-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Dataset-side window cap")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    enc_cfg = _core.EncoderConfig.from_json(Path(args.encoder_config).read_text())
    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    style_vocab = StyleVocab(tokenizer._vocab)
    print(f"StyleVocab size: {style_vocab.size}")

    dataset = MidiGPTDataset(
        args.train_data,
        tokenizer,
        infill_probability=0.0,
        humanize_probability=0.0,
        mask_bar_config=None,
        max_seq_len=args.max_seq_len,
        style_pretrain_mode=True,
    )
    print(f"Dataset size: {len(dataset)}")

    collator = ContrastiveCollator(style_vocab=style_vocab, max_len=args.max_style_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,  # InfoNCE needs a full, consistent batch of negatives
    )

    device = torch.device(args.device)
    style_cfg = StyleEncoderConfig(
        vocab_size=style_vocab.size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_len=args.max_style_len,
        z_dim=args.z_dim,
    )
    encoder = StyleEncoder(style_cfg).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr)

    chance_level = torch.log(torch.tensor(float(args.batch_size))).item()
    print(f"Chance-level loss (log(batch_size)): {chance_level:.3f}")

    step = 0
    t0 = time.time()
    encoder.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            anchor_ids = batch["anchor_ids"].to(device)
            anchor_mask = batch["anchor_key_mask"].to(device)
            positive_ids = batch["positive_ids"].to(device)
            positive_mask = batch["positive_key_mask"].to(device)

            z_a = encoder(anchor_ids, anchor_mask)
            z_p = encoder(positive_ids, positive_mask)
            loss = info_nce_loss(z_a, z_p, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"step {step:5d}  loss {loss.item():.4f}  "
                    f"(chance={chance_level:.3f})  {elapsed:.1f}s"
                )
            step += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "style_encoder_state_dict": encoder.state_dict(),
            "style_encoder_config": style_cfg,
            "encoder_config_json": enc_cfg.to_json(),
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
