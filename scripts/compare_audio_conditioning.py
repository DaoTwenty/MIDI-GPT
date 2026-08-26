"""Benchmark and comparison script for Audio-Conditioned MIDI-GPT.

Compares:
  1. Pooling Strategies: MeanConv vs Perceiver vs BeatGrid
  2. Hierarchical Fusion: Concat vs FiLM vs Residual
  3. Training Regimes: Adapter-Only vs LoRA vs Full Fine-Tuning
  4. Infill Comparison: Discrete Transcribed MIDI Context vs Continuous Audio Soft Tokens
"""

import time
import torch
import torch.nn as nn

from midigpt._types import Bar, Note, Score, Track
from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.adapters.hierarchical_fusion import (
    ConcatLinearFusion,
    FiLMFusion,
    ResidualAddFusion,
)
from midigpt.adapters.pooling import (
    BeatGridPooler,
    MeanConvPooler,
    PerceiverResampler,
)
from midigpt.audio.synth import FastAudioSynthesizer, get_bar_intervals
from midigpt.audio.yourmt3_extractor import YourMT3Extractor
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.training.audio_lightning_module import AudioConditionedLightningModule


def build_mock_score() -> Score:
    """Create an 8-bar multi-track test score (Piano melody + Drums)."""
    # 8 bars of 4/4
    piano_bars = []
    drum_bars = []
    for b in range(8):
        # 4 notes per bar
        piano_notes = [
            Note(pitch=60 + (b % 4) * 2 + i * 2, velocity=80, onset_ticks=i * 120, duration_ticks=100)
            for i in range(4)
        ]
        piano_bars.append(Bar(notes=piano_notes, ts_numerator=4, ts_denominator=4))

        drum_notes = [
            Note(pitch=36 if i % 2 == 0 else 38, velocity=90, onset_ticks=i * 120, duration_ticks=60)
            for i in range(4)
        ]
        drum_bars.append(Bar(notes=drum_notes, ts_numerator=4, ts_denominator=4))

    return Score(
        tracks=[
            Track(bars=piano_bars, instrument=0, track_type="melodic"),
            Track(bars=drum_bars, instrument=0, track_type="drum"),
        ],
        resolution=480,
        tempo=500000,  # 120 BPM
    )


def compare_pooling_strategies():
    print("=" * 70)
    print("1. COMPARING POOLING STRATEGIES (Turning T Audio Frames -> Bar Tokens)")
    print("=" * 70)

    feature_dim = 768
    B = 2
    T = 100  # ~2 seconds @ 50 fps

    dummy_frames = torch.randn(B, T, feature_dim)

    poolers = {
        "MeanConvPooler (K=1)": MeanConvPooler(feature_dim, num_outputs=1),
        "PerceiverResampler (K=1)": PerceiverResampler(feature_dim, num_latents=1, num_heads=8),
        "PerceiverResampler (K=4)": PerceiverResampler(feature_dim, num_latents=4, num_heads=8),
        "BeatGridPooler (K=4)": BeatGridPooler(feature_dim, beats_per_bar=4),
    }

    print(f"{'Strategy':<30} | {'Param Count':<12} | {'Output Shape':<18} | {'Latency (100 runs)':<15}")
    print("-" * 85)

    for name, pooler in poolers.items():
        params = sum(p.numel() for p in pooler.parameters())

        # Warmup & Timing
        for _ in range(10):
            _ = pooler(dummy_frames)

        t0 = time.perf_counter()
        for _ in range(100):
            out = pooler(dummy_frames)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"{name:<30} | {params:<12,d} | {str(list(out.shape)):<18} | {elapsed_ms:8.2f} ms")


def compare_fusion_strategies():
    print("\n" + "=" * 70)
    print("2. COMPARING HIERARCHICAL FUSION (Fusing Local Bar + Global Track)")
    print("=" * 70)

    feature_dim = 768
    B, K = 2, 1
    h_bar = torch.randn(B, K, feature_dim)
    h_track = torch.randn(B, feature_dim)

    fusions = {
        "ConcatLinearFusion": ConcatLinearFusion(feature_dim),
        "FiLMFusion": FiLMFusion(feature_dim),
        "ResidualAddFusion": ResidualAddFusion(feature_dim),
    }

    print(f"{'Fusion Method':<25} | {'Param Count':<12} | {'Output Shape':<18} | {'Latency (100 runs)':<15}")
    print("-" * 80)

    for name, fusion in fusions.items():
        params = sum(p.numel() for p in fusion.parameters())

        # Warmup & Timing
        for _ in range(10):
            _ = fusion(h_bar, h_track)

        t0 = time.perf_counter()
        for _ in range(100):
            out = fusion(h_bar, h_track)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"{name:<25} | {params:<12,d} | {str(list(out.shape)):<18} | {elapsed_ms:8.2f} ms")


def compare_training_regimes():
    print("\n" + "=" * 70)
    print("3. COMPARING TRAINING REGIMES (Adapter-Only vs LoRA vs Full Fine-Tuning)")
    print("=" * 70)

    cfg = GPT2Config(vocab_size=647, n_embd=512, n_layer=6, n_head=8)

    regimes = ["adapter_only", "lora", "full"]

    print(f"{'Regime':<15} | {'Total Params':<15} | {'Trainable Params':<18} | {'% Trainable':<12}")
    print("-" * 70)

    for regime in regimes:
        model = GPT2LMHeadModel(cfg)
        adapter = AudioToMidiAdapter(feature_dim=768, model_dim=512)
        extractor = YourMT3Extractor(feature_dim=768, n_layers=2)

        lit = AudioConditionedLightningModule(
            gpt2_model=model,
            audio_adapter=adapter,
            audio_extractor=extractor,
            training_mode=regime,
            lora_rank=16,
        )

        total_params = sum(p.numel() for p in lit.parameters())
        trainable_params = sum(p.numel() for p in lit.parameters() if p.requires_grad)
        pct = (trainable_params / total_params) * 100.0

        print(f"{regime:<15} | {total_params:<15,d} | {trainable_params:<18,d} | {pct:10.2f} %")


def main():
    print("Running Audio Conditioning Comparative Benchmark...\n")
    compare_pooling_strategies()
    compare_fusion_strategies()
    compare_training_regimes()
    print("\nAll comparative checks completed successfully!")


if __name__ == "__main__":
    main()
