"""CLI training entrypoint for Audio-Conditioned MIDI-GPT.

Example usage:
    # 1. Train adapter only with YourMT3+ weights
    python scripts/train_audio_conditioned.py \
        --audio-extractor yourmt3 \
        --audio-weights mimbres/yourmt3-small \
        --training-mode adapter_only \
        --pooling perceiver \
        --fusion concat \
        --train-data tmp/gigamidi/v2.0.0/train/00000.parquet \
        --output-dir checkpoints/audio_conditioned

    # 2. Train with MuScriptor and LoRA
    python scripts/train_audio_conditioned.py \
        --audio-extractor muscriptor \
        --audio-weights checkpoints/muscriptor.safetensors \
        --training-mode lora \
        --lora-rank 16 \
        --pooling perceiver \
        --fusion film \
        --train-data tmp/gigamidi/v2.0.0/train/00000.parquet \
        --eval-data tmp/gigamidi/v2.0.0/valid/00000.parquet \
        --output-dir checkpoints/audio_conditioned_lora
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.audio import FastAudioSynthesizer, load_audio_extractor
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.training.audio_dataset import AudioTransitionDataset
from midigpt.training.audio_collator import AudioConditionedCollator
from midigpt.training.audio_lightning_module import AudioConditionedLightningModule
from midigpt.tokenizer.tokenizer import Tokenizer
import midigpt._core as _core

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Audio-Conditioned MIDI-GPT for transition infilling."
    )

    # Data
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--eval-data", default=None, help="Optional eval parquet shard")

    # Audio Extractor CLI arguments
    parser.add_argument("--audio-extractor", type=str, default="yourmt3", choices=["yourmt3", "yourmt3+", "muscriptor"])
    parser.add_argument("--audio-weights", type=str, default=None, help="Path to pretrained weights or HuggingFace Hub repo ID")
    parser.add_argument("--feature-dim", type=int, default=768)

    # Adapter & Pooling CLI arguments
    parser.add_argument("--pooling", type=str, default="perceiver", choices=["mean_conv", "perceiver", "beat_grid"])
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "film", "residual", "none"])
    parser.add_argument("--tokens-per-bar", type=int, default=1)
    parser.add_argument("--max-bars", type=int, default=64)
    parser.add_argument("--bar-step", type=int, default=4)

    # Training Regime CLI arguments
    parser.add_argument("--training-mode", type=str, default="adapter_only", choices=["adapter_only", "lora", "full"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)

    # MIDI-GPT Model & Training params
    parser.add_argument("--midi-ckpt", type=str, default=None, help="Optional path to pretrained MIDI-GPT checkpoint (.safetensors)")
    parser.add_argument("--encoder-config", type=str, default=None, help="Path to encoder config JSON (required if --midi-ckpt set)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="If > 0, overrides max-epochs")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="checkpoints/audio_conditioned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--limit-val-batches", type=int, default=50, help="Max validation batches per eval (0=full)")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit dataset to N samples (0=all)")

    return parser.parse_args()


def _load_midi_ckpt(ckpt_path: str, encoder_config: _core.EncoderConfig):
    """Load a MIDI-GPT checkpoint and return (model, encoder_config).

    Builds the model from the checkpoint's own config, then resizes
    embeddings if the target encoder config has a different vocab size.
    New token rows are randomly initialized.
    """
    import json
    from safetensors import safe_open

    ckpt_path = Path(ckpt_path)
    with safe_open(str(ckpt_path), framework="pt") as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
        meta = f.metadata()
        ckpt_cfg = json.loads(meta.get("config", "{}"))

    model = GPT2LMHeadModel(GPT2Config(**ckpt_cfg))
    model.load_state_dict(state_dict, strict=False)

    target_vocab = _core.Vocabulary(encoder_config).size()
    if target_vocab != ckpt_cfg["vocab_size"]:
        model.resize_token_embeddings(target_vocab)

    return model, encoder_config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    L.seed_everything(args.seed, workers=True)

    num_bars_list = list(range(args.bar_step, args.max_bars + args.bar_step, args.bar_step))

    print("=" * 70)
    print("AUDIO-CONDITIONED MIDI-GPT TRAINING")
    print("=" * 70)
    print(f"Audio Extractor:   {args.audio_extractor}")
    print(f"Audio Weights:     {args.audio_weights or '(Native Architecture)'}")
    print(f"Pooling Strategy:  {args.pooling}")
    print(f"Fusion Strategy:   {args.fusion}")
    print(f"Training Mode:     {args.training_mode}")
    print(f"Context Bars Map:  {num_bars_list} (up to {args.max_bars} bars)")
    print(f"Train Data:        {args.train_data}")
    print(f"Eval Data:         {args.eval_data or '(none)'}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Grad Accum:        {args.grad_accum}")
    print(f"Max Epochs:        {args.max_epochs}")
    print(f"Max Steps:         {args.max_steps or 'auto'}")
    print("=" * 70)

    # 1. Build encoder config + tokenizer
    print("\n[1/5] Building tokenizer...")
    if args.midi_ckpt and args.encoder_config:
        enc_json_str = Path(args.encoder_config).read_text()
        encoder_config = _core.EncoderConfig.from_json(enc_json_str)
    else:
        # Use a default small encoder config for smoke testing
        enc_json = {
            "num_bars_map": [4, 8, 12, 16],
            "supports_infill": True,
        }
        enc_json_str = json.dumps(enc_json)
        encoder_config = _core.EncoderConfig.from_json(enc_json_str)
    tokenizer = Tokenizer(encoder_config)
    print(f"  Vocab size: {tokenizer.vocab_size()}")

    # 2. Build GPT-2 model
    print("\n[2/5] Initializing MIDI-GPT backbone...")
    if args.midi_ckpt:
        print(f"  Loading checkpoint: {args.midi_ckpt}")
        gpt2_model, enc_cfg = _load_midi_ckpt(args.midi_ckpt, encoder_config)
        gpt2_model.encoder_config = enc_cfg
    else:
        gpt2_model = GPT2LMHeadModel(GPT2Config(
            vocab_size=tokenizer.vocab_size(),
            n_positions=args.max_seq_len,
            n_embd=256, n_layer=4, n_head=4,
        ))
        gpt2_model.encoder_config = encoder_config
    model_dim = gpt2_model.transformer.wte.weight.shape[1]
    print(f"  Model params: {sum(p.numel() for p in gpt2_model.parameters()):,d}")

    # 3. Build audio extractor + adapter
    print("\n[3/5] Loading audio extractor...")
    extractor = load_audio_extractor(
        extractor_type=args.audio_extractor,
        weights_path=args.audio_weights,
        feature_dim=args.feature_dim,
    )
    print(f"  Loaded {args.audio_extractor} (feature_dim={extractor.feature_dim})")

    print("\n  Building Audio-to-MIDI Adapter...")
    adapter = AudioToMidiAdapter(
        feature_dim=args.feature_dim,
        model_dim=model_dim,
        num_tokens_per_bar=args.tokens_per_bar,
        pooling_strategy=args.pooling,
        fusion_strategy=args.fusion,
    )

    # 4. Build dataset + dataloader
    print("\n[4/5] Building dataset...")
    synth = FastAudioSynthesizer(sample_rate=16000)
    dataset = AudioTransitionDataset(
        scores=[],  # placeholder — we load from parquet below
        tokenizer=tokenizer,
        audio_extractor=extractor,
        num_bars_choices=num_bars_list,
        synthesizer=synth,
        max_seq_len=args.max_seq_len,
    )

    # Actually load scores from parquet
    try:
        import datasets as hf
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pip install midigpt[train]") from None

    paths = []
    for p in args.train_data.split(","):
        p = p.strip()
        expanded = sorted(glob.glob(p)) if "*" in p else [p]
        paths.extend(expanded)
    log.info("Loading training parquets: %s", paths)
    data = hf.load_dataset("parquet", data_files=paths, split="train")
    log.info("Loaded %d rows from parquet", len(data))

    # Parse MIDI bytes into Score objects
    from midigpt._types import Score as PyScore
    scores = []
    max_parse = args.max_samples if args.max_samples > 0 else len(data)
    for i in range(min(max_parse, len(data))):
        try:
            score = PyScore.from_bytes(data[i]["music"])
            scores.append(score)
        except Exception:
            continue
    log.info("Successfully parsed %d / %d scores", len(scores), max_parse)

    if not scores:
        print("Error: No valid scores loaded from parquet")
        sys.exit(1)

    dataset.scores = scores

    collator = AudioConditionedCollator(pad_token_id=0, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optional eval dataloader
    eval_loader = None
    if args.eval_data:
        eval_paths = sorted(glob.glob(args.eval_data)) if "*" in args.eval_data else [args.eval_data]
        eval_data = hf.load_dataset("parquet", data_files=eval_paths, split="train")
        eval_scores = []
        for i in range(len(eval_data)):
            try:
                score = PyScore.from_bytes(eval_data[i]["music"])
                eval_scores.append(score)
            except Exception:
                continue
        eval_dataset = AudioTransitionDataset(
            scores=eval_scores,
            tokenizer=tokenizer,
            audio_extractor=extractor,
            num_bars_choices=num_bars_list,
            synthesizer=synth,
            max_seq_len=args.max_seq_len,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        log.info("Eval set: %d scores", len(eval_scores))

    # 5. Lightning training
    print("\n[5/5] Starting Lightning training...")
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.max_epochs

    lit_module = AudioConditionedLightningModule(
        gpt2_model=gpt2_model,
        audio_adapter=adapter,
        audio_extractor=extractor,
        training_mode=args.training_mode,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    total_params = sum(p.numel() for p in lit_module.parameters())
    trainable_params = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)
    print(f"  Total Parameters:     {total_params:,d}")
    print(f"  Trainable Parameters: {trainable_params:,d} ({(trainable_params/total_params)*100:.2f}%)")
    print(f"  Steps per epoch:      {steps_per_epoch}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Eval every {args.eval_steps} steps" + (f" (limit={args.limit_val_batches})" if args.limit_val_batches else ""))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    precision_str = {"fp16": "16-mixed", "bf16": "bf16-mixed", "fp32": "32"}[args.precision]

    trainer = L.Trainer(
        max_steps=total_steps,
        precision=precision_str,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=1.0,
        val_check_interval=args.eval_steps if eval_loader else None,
        limit_val_batches=args.limit_val_batches or 1.0,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                every_n_train_steps=args.save_steps,
                save_top_k=-1,
                filename="step={step}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=True,
    )

    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=eval_loader)

    # Save final
    final_path = output_dir / "model_final.safetensors"
    enc_cfg_dict = json.loads(encoder_config.to_json()) if hasattr(encoder_config, "to_json") else {}
    gpt2_model.save_pretrained(str(final_path), encoder_config=enc_cfg_dict)
    log.info("Training complete. Final bundle: %s", final_path)


if __name__ == "__main__":
    import glob
    main()
