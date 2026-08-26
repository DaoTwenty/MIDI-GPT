"""Unit tests for audio conditioning training regimes (Adapter-Only, LoRA, Full)."""

import pytest
import torch

from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.audio.yourmt3_extractor import YourMT3Extractor
from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
from midigpt.training.audio_lightning_module import AudioConditionedLightningModule


def test_adapter_only_training_regime():
    cfg = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4, n_positions=512)
    model = GPT2LMHeadModel(cfg)
    adapter = AudioToMidiAdapter(feature_dim=128, model_dim=64)
    extractor = YourMT3Extractor(feature_dim=128, n_mels=32, n_layers=1, n_heads=2)

    lit = AudioConditionedLightningModule(
        gpt2_model=model,
        audio_adapter=adapter,
        audio_extractor=extractor,
        training_mode="adapter_only",
    )

    # Check that model & extractor are frozen, adapter is trainable
    assert not any(p.requires_grad for p in lit.audio_extractor.parameters())
    assert not any(p.requires_grad for p in lit.gpt2_model.parameters())
    assert all(p.requires_grad for p in lit.audio_adapter.parameters())

    # Forward & loss
    batch = {
        "input_ids": torch.randint(0, 100, (16,)),
        "labels": torch.randint(0, 100, (16,)),
        "context_bar_frames": [torch.randn(20, 128)],
        "track_frames": [torch.randn(50, 128)],
    }
    loss = lit.training_step(batch, 0)
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_lora_training_regime():
    cfg = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4, n_positions=512)
    model = GPT2LMHeadModel(cfg)
    adapter = AudioToMidiAdapter(feature_dim=128, model_dim=64)
    extractor = YourMT3Extractor(feature_dim=128, n_mels=32, n_layers=1, n_heads=2)

    lit = AudioConditionedLightningModule(
        gpt2_model=model,
        audio_adapter=adapter,
        audio_extractor=extractor,
        training_mode="lora",
        lora_rank=4,
    )

    # Base transformer weights should be frozen, LoRA parameters trainable
    assert not any(p.requires_grad for p in lit.audio_extractor.parameters())
    assert all(p.requires_grad for p in lit.audio_adapter.parameters())

    lora_params = [p for n, p in lit.gpt2_model.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0
    assert all(p.requires_grad for p in lora_params)

    # Non-LoRA weights in gpt2 should be frozen
    base_params = [p for n, p in lit.gpt2_model.named_parameters() if "lora_" not in n]
    assert not any(p.requires_grad for p in base_params)


def test_full_training_regime():
    cfg = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4, n_positions=512)
    model = GPT2LMHeadModel(cfg)
    adapter = AudioToMidiAdapter(feature_dim=128, model_dim=64)
    extractor = YourMT3Extractor(feature_dim=128, n_mels=32, n_layers=1, n_heads=2)

    lit = AudioConditionedLightningModule(
        gpt2_model=model,
        audio_adapter=adapter,
        audio_extractor=extractor,
        training_mode="full",
    )

    assert not any(p.requires_grad for p in lit.audio_extractor.parameters())
    assert all(p.requires_grad for p in lit.audio_adapter.parameters())
    assert all(p.requires_grad for p in lit.gpt2_model.parameters())
