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
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.audio import AudioRenderCache, FastAudioSynthesizer, SoundFontSynthesizer, load_audio_extractor
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.training.audio_dataset import AudioTransitionDataset, MixedMidiGPTDataset, SlakhPairedStemDataset
from midigpt.training.audio_collator import AudioConditionedCollator
from midigpt.training.audio_lightning_module import AudioConditionedLightningModule
from midigpt.training.dataset import _load_or_build_valid_indices
from midigpt.tokenizer.tokenizer import Tokenizer
import midigpt._core as _core

log = logging.getLogger(__name__)


def _load_filtered_rows(paths, encoder_config, min_tracks, max_samples=0):
    """Filter parquet rows down to those passing the dataset's real constraints.

    AudioTransitionDataset requires min_tracks (one MIDI target + one audio-context
    track); its own window selection always rejects a score short of that, so
    filtering here — instead of via a bare parse-and-hope loop — avoids wasting
    audio-synthesis work on rows that were always going to be discarded, and
    avoids the batches-worth-of-rejects that can leave a whole DataLoader batch
    empty. Reuses the same cached two-phase filter MidiGPTDataset.from_parquet uses.

    Returns the filtered Arrow dataset itself (lazy, memory-mapped) plus the
    parallel score_ids list — NOT parsed Score objects. A filtered row (min_tracks
    >= 2) averages ~2,690 notes / ~1,255 bars, much larger than an unfiltered row;
    parsing all ~573k of them into live Score objects up front measured at ~430GB
    RSS in this one process, which is why every amber training attempt OOM'd here
    before the DataLoader was even built (2026-08-27 through 2026-08-29).
    AudioTransitionDataset(raw_rows=...) parses each row on demand instead.
    """
    import pyarrow.parquet as pq
    import datasets as hf

    cfg_dict = json.loads(encoder_config.to_json())
    num_bars_choices = sorted(set(int(x) for x in (cfg_dict.get("num_bars_map") or [4])), reverse=True)
    ts_list = cfg_dict.get("time_signatures")
    valid_ts = frozenset(ts_list) if ts_list else None

    data = hf.load_dataset("parquet", data_files=paths, split="train")
    log.info("Loaded %d rows from parquet", len(data))

    all_valid = []
    offset = 0
    for path in paths:
        n = pq.read_metadata(path).num_rows
        local_valid = _load_or_build_valid_indices(
            path, min_bars=num_bars_choices[-1], min_tracks=min_tracks, valid_time_sigs=valid_ts,
        )
        all_valid.extend(i + offset for i in local_valid)
        offset += n
    log.info(
        "Dataset filter: %d/%d rows pass (min_tracks=%d, min_bars=%d)",
        len(all_valid), len(data), min_tracks, num_bars_choices[-1],
    )
    if max_samples > 0:
        all_valid = all_valid[:max_samples]
    data = data.select(all_valid)
    score_ids = data["md5"]
    log.info("%d rows selected for lazy per-item parsing", len(data))
    return data, score_ids


def _load_slakh_samples(slakh_root: str, max_samples: int = 0) -> list[dict]:
    """Load {"score", "audio_stems"} samples for SlakhPairedStemDataset.

    Only tracks with a complete, aligned FLAC stem per MIDI track are kept —
    same criterion scripts/test_audio_midi_pipeline.py uses for its Slakh path.
    """
    from midigpt._types import Score as PyScore

    root = Path(slakh_root)
    samples = []
    midi_paths = sorted(root.glob("Track*/all_src.mid"))
    for midi_path in midi_paths:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        track_dir = midi_path.parent
        stems_dir = track_dir / "stems"
        if not stems_dir.is_dir():
            continue
        try:
            score = PyScore.from_midi(str(midi_path))
        except Exception:
            continue
        stem_paths = [stems_dir / f"S{index:02d}.flac" for index in range(len(score.tracks))]
        if not all(p.is_file() for p in stem_paths):
            continue
        samples.append({"score": score, "audio_stems": [str(p) for p in stem_paths]})
    log.info("Loaded %d / %d Slakh2100 tracks with complete aligned stems", len(samples), len(midi_paths))
    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Audio-Conditioned MIDI-GPT for transition infilling."
    )

    # Data
    parser.add_argument("--train-data", required=True, help="Parquet shard(s): path, list, or glob")
    parser.add_argument("--eval-data", default=None, help="Optional eval parquet shard")
    parser.add_argument("--slakh-root", default=None,
                         help="Slakh2100 split directory (e.g. data/slakh2100/.../train), containing "
                              "Track*/all_src.mid + Track*/stems/S##.flac. When set, training draws "
                              "50/50 from GigaMIDI (on-the-fly synthesis) and Slakh2100 (real "
                              "pre-rendered stems) via MixedMidiGPTDataset.")
    parser.add_argument("--slakh-max-samples", type=int, default=0, help="Limit Slakh2100 tracks used (0=all)")

    # Audio Extractor CLI arguments
    parser.add_argument(
        "--audio-extractor", type=str, default="muscriptor",
        choices=["yourmt3", "yourmt3+", "muscriptor", "mert", "mert2", "sheetsage2", "sheetsage",
                 "dac", "muq", "music2latent", "same"],
    )
    parser.add_argument("--audio-weights", type=str, default=None, help="Path to pretrained weights or HuggingFace Hub repo ID")
    parser.add_argument("--model-size", type=str, default=None,
                         help="Extractor variant/size where applicable: muscriptor {small,medium,large}, "
                              "mert/sheetsage2 {fullsong,30s}, dac {16khz,24khz,44khz}, same {same-l,same-s}")
    parser.add_argument("--feature-dim", type=int, default=768)

    # Synthesis CLI arguments
    parser.add_argument("--synthesizer", type=str, default="fast", choices=["fast", "soundfont"],
                         help="'fast': pure-oscillator on-the-fly synth. 'soundfont': FluidSynth GM "
                              "rendering, backed by --audio-cache-dir if given (see "
                              "scripts/presynthesize_audio_cache.py).")
    parser.add_argument("--soundfont-dir", type=str, default="soundfonts",
                         help="Directory of .sf2 files (only used with --synthesizer soundfont)")
    parser.add_argument("--audio-cache-dir", type=str, default=None,
                         help="On-disk AudioRenderCache directory pre-populated by "
                              "scripts/presynthesize_audio_cache.py. Cache misses fall back to live "
                              "FluidSynth synthesis. Only used with --synthesizer soundfont.")

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
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Lightning .ckpt to resume from, or 'auto' to use "
                              "<output-dir>/checkpoints/last.ckpt if it exists")
    parser.add_argument("--wandb", action="store_true", default=False,
                         help="Log to Weights & Biases (entity/API key from .env, see .env.example)")
    parser.add_argument("--wandb-project", type=str, default="midigpt")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                         help="Defaults to the output-dir basename")

    return parser.parse_args()


def _load_dotenv() -> None:
    """Load .env from the current directory if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _build_wandb_logger(args, output_dir: Path):
    """Mirrors midigpt.training.trainer._build_loggers' wandb setup."""
    import wandb as _wandb
    from lightning.pytorch.loggers import WandbLogger

    run_name = args.wandb_run_name or output_dir.name
    init_kwargs: dict = dict(project=args.wandb_project, name=run_name, dir=str(output_dir))
    if os.environ.get("WANDB_ENTITY"):
        init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
    try:
        run = _wandb.init(**init_kwargs)
    except Exception as exc:
        log.warning(
            "wandb online init failed (%s); switching to offline mode"
            " — run `wandb sync %s` after training to upload.",
            exc, output_dir,
        )
        run = _wandb.init(**init_kwargs, mode="offline")
    return WandbLogger(experiment=run)


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
        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    _load_dotenv()
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
        model_size=args.model_size,
        feature_dim=args.feature_dim,
    )
    print(f"  Loaded {args.audio_extractor} (feature_dim={extractor.feature_dim})")

    print("\n  Building Audio-to-MIDI Adapter...")
    adapter = AudioToMidiAdapter(
        feature_dim=extractor.feature_dim,
        model_dim=model_dim,
        num_tokens_per_bar=args.tokens_per_bar,
        pooling_strategy=args.pooling,
        fusion_strategy=args.fusion,
    )

    # 4. Build dataset + dataloader
    print("\n[4/5] Building dataset...")
    if args.synthesizer == "soundfont":
        synth = SoundFontSynthesizer(args.soundfont_dir, sample_rate=16000)
        audio_cache = AudioRenderCache(args.audio_cache_dir) if args.audio_cache_dir else None
        print(f"  Soundfonts:  {synth.soundfont_names}")
        print(f"  Audio cache: {args.audio_cache_dir or '(none — always live-synthesizes)'}")
    else:
        synth = FastAudioSynthesizer(sample_rate=16000)
        audio_cache = None
    # Load + filter parquet rows (lazy — rows are parsed on demand per DataLoader item)
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
    train_rows, score_ids = _load_filtered_rows(
        paths, encoder_config, min_tracks=2, max_samples=args.max_samples,
    )

    if len(train_rows) == 0:
        print("Error: No valid scores loaded from parquet")
        sys.exit(1)

    giga_dataset = AudioTransitionDataset(
        scores=None,
        raw_rows=train_rows,
        tokenizer=tokenizer,
        num_bars_choices=num_bars_list,
        synthesizer=synth,
        max_seq_len=args.max_seq_len,
        score_ids=score_ids,
        audio_cache=audio_cache,
    )

    if args.slakh_root:
        slakh_samples = _load_slakh_samples(args.slakh_root, max_samples=args.slakh_max_samples)
        if not slakh_samples:
            print(f"Error: No usable Slakh2100 tracks found under {args.slakh_root}")
            sys.exit(1)
        slakh_dataset = SlakhPairedStemDataset(
            samples=slakh_samples,
            tokenizer=tokenizer,
            num_bars_choices=num_bars_list,
            max_seq_len=args.max_seq_len,
        )
        print(f"  Mixed dataset: 50% GigaMIDI ({len(giga_dataset)}) / 50% Slakh2100 ({len(slakh_dataset)})")
        train_dataset = MixedMidiGPTDataset(giga_dataset, slakh_dataset)
    else:
        train_dataset = giga_dataset

    collator = AudioConditionedCollator(pad_token_id=0, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
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
        eval_rows, eval_score_ids = _load_filtered_rows(
            eval_paths, encoder_config, min_tracks=2,
        )
        eval_dataset = AudioTransitionDataset(
            scores=None,
            raw_rows=eval_rows,
            tokenizer=tokenizer,
            num_bars_choices=num_bars_list,
            synthesizer=synth,
            max_seq_len=args.max_seq_len,
            score_ids=eval_score_ids,
            audio_cache=audio_cache,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        log.info("Eval set: %d scores", len(eval_rows))

    # 5. Lightning training
    print("\n[5/5] Starting Lightning training...")
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.max_epochs

    # The Dataset only synthesizes and hands back raw waveforms now (CPU-side, in
    # forked DataLoader workers) — audio_extractor is used solely here, as a real
    # submodule of this LightningModule, so Lightning puts it on the accelerator
    # device alongside gpt2_model and audio_adapter and _compute_loss runs the
    # actual feature extraction on GPU. No separate CPU-only copy needed.
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

    trainer_logger = _build_wandb_logger(args, output_dir) if args.wandb else True

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
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=trainer_logger,
    )

    resume_path = args.resume_from
    if resume_path == "auto":
        auto_path = ckpt_dir / "last.ckpt"
        resume_path = str(auto_path) if auto_path.exists() else None
    if resume_path:
        log.info("Resuming from %s", resume_path)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=eval_loader, ckpt_path=resume_path)

    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Peak CUDA memory allocated: {peak_alloc_gb:.2f} GB")
        print(f"  Peak CUDA memory reserved:  {peak_reserved_gb:.2f} GB")
        print(f"  Device total memory:        {total_gb:.2f} GB")
        print(f"  Headroom (reserved):        {total_gb - peak_reserved_gb:.2f} GB")

    # Save final
    final_path = output_dir / "model_final.safetensors"
    enc_cfg_dict = json.loads(encoder_config.to_json()) if hasattr(encoder_config, "to_json") else {}
    gpt2_model.save_pretrained(str(final_path), encoder_config=enc_cfg_dict)
    log.info("Training complete. Final bundle: %s", final_path)


if __name__ == "__main__":
    import glob
    main()
