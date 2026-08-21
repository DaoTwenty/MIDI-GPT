"""Variant A: joint-conditioning of a GPT2 Humanize model + StyleEncoder via
a soft-prefix (see conditioned_gpt2.py).

Ordinary next-token teacher-forced training over MidiGPTDataset's normal
humanize-shaped samples (style_pretrain_mode=False) -- mirrors
humanize_tiny_e.json's dataset settings (that's checkpoint E, the flagship
"expressive" checkpoint this conditioning mechanism is meant to sit on top
of). Both the GPT2 backbone and the StyleEncoder receive gradients unless
--pretrained-style-encoder is passed with --freeze-style-encoder, in which
case only the (already contrastively-pretrained) encoder's z_proj/backbone
stay fixed and just the LM + z_proj... note: z_proj lives on ConditionedGPT2,
not StyleEncoder, so it always trains regardless of --freeze-style-encoder.

Usage:
    python3 train_joint_conditioning.py --train-data /path/*.parquet \
        --encoder-config ../../models/humanize_encoder.json \
        --output conditioned_gpt2_random_init.pt

    # Variant A+B combo: start from Variant B's contrastively-pretrained
    # style encoder instead of random init.
    python3 train_joint_conditioning.py --train-data /path/*.parquet \
        --encoder-config ../../models/humanize_encoder.json \
        --pretrained-style-encoder style_encoder_contrastive.pt \
        --freeze-style-encoder \
        --output conditioned_gpt2_pretrained_style.pt

Prototype-scope: plain torch.save (not the production packed-safetensors
format), standalone training loop (no Lightning dependency).
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
from conditioned_gpt2 import ConditionedGPT2
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.model.gpt2 import GPT2Config
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.dataset import MidiGPTDataset
from style_collator import JointConditioningCollator
from style_encoder import StyleEncoderConfig
from style_vocab import StyleVocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint-conditioning (soft-prefix) training.")
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--encoder-config", required=True, help="Path to humanize_encoder.json")
    parser.add_argument("--output", default="conditioned_gpt2.pt")
    parser.add_argument("--pretrained-style-encoder", default=None, help="Variant B checkpoint (.pt)")
    parser.add_argument("--freeze-style-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-positions", type=int, default=2048)
    parser.add_argument("--style-d-model", type=int, default=128)
    parser.add_argument("--style-n-layer", type=int, default=3)
    parser.add_argument("--style-n-head", type=int, default=4)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--max-style-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    # Dataset settings mirrored from humanize_tiny_e.json (checkpoint E).
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
    style_vocab = StyleVocab(tokenizer._vocab)
    print(f"StyleVocab size: {style_vocab.size}, LM vocab size: {tokenizer.vocab_size()}")

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

    collator = JointConditioningCollator(
        style_vocab=style_vocab, max_seq_len=args.max_seq_len, max_style_len=args.max_style_len
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
    )

    device = torch.device(args.device)
    # ConditionedGPT2 shifts every real token to position 1..max_seq_len to make
    # room for the style prefix at position 0 (see conditioned_gpt2.py::forward),
    # so wpe needs max_seq_len+1 rows, not max_seq_len -- an unmaxed n_positions
    # here is an out-of-bounds embedding gather (silent on CPU, a device-side
    # assert crash on GPU) the moment a batch has a maxed-out sequence.
    n_positions = max(args.n_positions, args.max_seq_len + 1)
    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    style_cfg = StyleEncoderConfig(
        vocab_size=style_vocab.size,
        d_model=args.style_d_model,
        n_layer=args.style_n_layer,
        n_head=args.style_n_head,
        max_len=args.max_style_len,
        z_dim=args.z_dim,
    )
    model = ConditionedGPT2(gpt2_cfg, style_cfg).to(device)

    if args.pretrained_style_encoder:
        # weights_only=False: this is our own prototype checkpoint (plain
        # torch.save of a state_dict + dataclass configs, not third-party),
        # produced by train_contrastive.py in this same script directory.
        ckpt = torch.load(args.pretrained_style_encoder, map_location="cpu", weights_only=False)
        saved_cfg = ckpt["style_encoder_config"]
        if saved_cfg.vocab_size != style_cfg.vocab_size:
            raise ValueError(
                f"Pretrained style encoder vocab_size={saved_cfg.vocab_size} does not match "
                f"this run's StyleVocab size={style_cfg.vocab_size} -- encoder configs differ."
            )
        model.load_pretrained_style_encoder(
            ckpt["style_encoder_state_dict"], freeze=args.freeze_style_encoder
        )
        print(
            f"Loaded pretrained style encoder from {args.pretrained_style_encoder} "
            f"(frozen={args.freeze_style_encoder})"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Params: {n_total:,} total, {n_trainable:,} trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(step: int) -> None:
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "gpt2_config": gpt2_cfg,
                "style_encoder_config": style_cfg,
                "encoder_config_json": enc_cfg.to_json(),
                "pretrained_style_encoder": args.pretrained_style_encoder,
                "freeze_style_encoder": args.freeze_style_encoder,
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
            style_ids = batch["style_ids"].to(device)
            style_key_mask = batch["style_key_mask"].to(device)
            has_style = batch["has_style"].to(device)

            logits = model(input_ids, style_ids, style_key_mask, has_style)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                pct_with_style = has_style.float().mean().item()
                print(
                    f"step {step:5d}  loss {loss.item():.4f}  "
                    f"has_style={pct_with_style:.2f}  {elapsed:.1f}s"
                )
            step += 1
            if args.save_every and step % args.save_every == 0:
                save(step)

    save(args.steps)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
