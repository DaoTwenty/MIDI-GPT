"""Tests for the training pipeline (trainer, data module, lightning module).

Marker strategy:
  inference — needs torch (and any GPU); skipped in cibuildwheel CI
  slow      — needs parquet data on disk; skipped everywhere without it

The plain (unmarked) tests cover unit-level concerns that have no external deps:
  - TrainConfig round-trip serialisation
  - _validate_train_config rejects incompatible encoder/config pairs
"""
from __future__ import annotations

import json
import math
import pathlib
import tempfile

import pytest
import torch

import midigpt._core as _core


# --------------------------------------------------------------------------- #
#  TrainConfig — unit tests, no parquet, no torch, no markers
# --------------------------------------------------------------------------- #

class TestTrainConfig:
    def test_defaults(self):
        from midigpt.training.trainer import TrainConfig

        cfg = TrainConfig()
        assert cfg.learning_rate == pytest.approx(5e-5)
        assert cfg.precision in ("fp16", "bf16", "fp32")
        assert cfg.num_epochs >= 1
        assert cfg.per_device_batch_size >= 1

    def test_from_json_file(self, tmp_path):
        from midigpt.training.trainer import TrainConfig

        data = {"learning_rate": 1e-4, "n_layer": 2, "n_embd": 64, "n_head": 2}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(data))
        cfg = TrainConfig.from_file(str(p))
        assert cfg.learning_rate == pytest.approx(1e-4)
        assert cfg.n_layer == 2

    def test_unknown_keys_ignored(self, tmp_path):
        from midigpt.training.trainer import TrainConfig

        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"learning_rate": 3e-4, "not_a_field": 999}))
        cfg = TrainConfig.from_file(str(p))
        assert cfg.learning_rate == pytest.approx(3e-4)


class TestValidateTrainConfig:
    def test_infill_rejected_without_support(self, ghost_config_json):
        from midigpt.training.trainer import TrainConfig, _validate_train_config

        cfg = TrainConfig(infill_probability=0.5)
        enc = json.loads(ghost_config_json)
        enc["supports_infill"] = False
        with pytest.raises(ValueError, match="supports_infill"):
            _validate_train_config(cfg, enc)

    def test_no_infill_passes(self, ghost_config_json):
        from midigpt.training.trainer import TrainConfig, _validate_train_config

        cfg = TrainConfig(infill_probability=0.0, mask_apply_probability=0.0)
        _validate_train_config(cfg, json.loads(ghost_config_json))

    def test_humanize_rejected_without_support(self, ghost_config_json):
        from midigpt.training.trainer import TrainConfig, _validate_train_config

        cfg = TrainConfig(infill_probability=0.0, humanize_probability=0.5)
        enc = json.loads(ghost_config_json)
        enc["supports_humanize"] = False
        with pytest.raises(ValueError, match="supports_humanize"):
            _validate_train_config(cfg, enc)

    def test_no_humanize_passes(self, ghost_config_json):
        from midigpt.training.trainer import TrainConfig, _validate_train_config

        cfg = TrainConfig(
            infill_probability=0.0, humanize_probability=0.0, mask_apply_probability=0.0
        )
        _validate_train_config(cfg, json.loads(ghost_config_json))


# --------------------------------------------------------------------------- #
#  LightningModule — forward/backward, no parquet needed
# --------------------------------------------------------------------------- #

@pytest.mark.inference
class TestLightningModule:
    def test_training_step_loss_finite(self, tiny_gpt2, ghost_tokenizer):
        """One synthetic training step should produce a finite loss."""
        import dataclasses
        from midigpt.training.lightning_module import MidiGPTLightningModule

        @dataclasses.dataclass
        class _Cfg:
            learning_rate: float = 1e-3
            weight_decay: float = 0.01
            warmup_steps: int = 0
            lr_scheduler_type: str = "constant"

        lit = MidiGPTLightningModule(tiny_gpt2, _Cfg())
        lit.total_steps = 10
        lit.train()

        B, T = 2, 32
        V = ghost_tokenizer.vocab_size()
        ids = torch.randint(0, V, (B, T))
        labels = ids.clone()

        batch = {"input_ids": ids, "labels": labels}
        loss = lit.training_step(batch, 0)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        assert loss > 0


# --------------------------------------------------------------------------- #
#  MidiGPTDataset + DataModule — need parquet on disk
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.inference
class TestMidiGPTDataset:
    def test_dataset_loads_and_filters(self, ghost_tokenizer, training_parquet):
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            ghost_tokenizer,
            infill_probability=0.0,
            mask_bar_config=None,
            max_seq_len=128,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
        )
        assert len(ds) > 0, "Dataset is empty after filtering"

    def test_dataset_item_shape(self, ghost_tokenizer, training_parquet):
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            ghost_tokenizer,
            infill_probability=0.0,
            mask_bar_config=None,
            max_seq_len=64,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
        )
        # Find a non-None sample.
        sample = None
        for i in range(min(50, len(ds))):
            s = ds[i]
            if s is not None:
                sample = s
                break
        assert sample is not None, "No valid sample in first 50 rows"
        assert "input_ids" in sample
        assert len(sample["input_ids"]) <= 64

    def test_data_module_setup(self, ghost_tokenizer, training_parquet):
        from midigpt.training.data_module import MidiGPTDataModule

        dm = MidiGPTDataModule(
            train_path=str(training_parquet),
            tokenizer=ghost_tokenizer,
            infill_probability=0.0,
            mask_bar_config=None,
            max_seq_len=64,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
            per_device_batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        dm.setup()
        assert dm.train_dataset_size > 0
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "input_ids" in batch


@pytest.mark.slow
@pytest.mark.inference
class TestHumanizeDataset:
    """MidiGPTDataset's Humanize path, exercised against the real
    models/humanize_encoder.json (supports_infill=false, supports_humanize=true)
    and real MIDI-derived training data (training_parquet fixture)."""

    @pytest.fixture
    def humanize_tokenizer(self):
        from midigpt.attributes.base import AttributeAnalyzer
        from midigpt.tokenizer.tokenizer import Tokenizer

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        enc_json = (repo_root / "models" / "humanize_encoder.json").read_text()
        cfg = _core.EncoderConfig.from_json(enc_json)
        analyzer = AttributeAnalyzer.from_config(cfg)
        return Tokenizer(cfg, analyzer)

    def test_humanize_dataset_emits_humanize_tokens(self, humanize_tokenizer, training_parquet):
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            humanize_tokenizer,
            infill_probability=0.0,
            humanize_probability=1.0,
            humanize_bar_fraction=0.5,
            mask_bar_config=None,
            max_seq_len=1024,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
        )
        assert len(ds) > 0

        vocab = humanize_tokenizer._vocab
        hs = vocab.encode_val(_core.TokenType.HumanizeStart, 0)
        sks = vocab.encode_val(_core.TokenType.HumanizeSkeletonStart, 0)
        found = False
        for i in range(min(30, len(ds))):
            sample = ds[i]
            if sample is None:
                continue
            ids = sample["input_ids"]
            if hs in ids and sks in ids:
                found = True
                break
        assert found, "No sample among the first 30 contained Humanize tokens"

    def test_style_ref_mask_only_flags_expressive_tokens_before_appendix(
        self, humanize_tokenizer, training_parquet
    ):
        """style_ref_mask (style-conditioning prototype scaffolding) must only
        ever flag VelocityLevel/DeltaDirection/Delta positions, and only
        within a track's in-place skeleton region (never a Humanize appendix
        block — reference bars are, by construction, disjoint from whatever
        was withheld as the humanize target, so their expressive tokens are
        never deferred). Each track has its own TrackEnd, so "appendix zone"
        means "after this track's TrackEnd, before the next Track token" —
        not a single global cutoff."""
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            humanize_tokenizer,
            infill_probability=0.0,
            humanize_probability=1.0,
            humanize_bar_fraction=0.5,
            mask_bar_config=None,
            max_seq_len=1024,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
        )
        vocab = humanize_tokenizer._vocab
        expressive = (
            _core.TokenType.VelocityLevel,
            _core.TokenType.DeltaDirection,
            _core.TokenType.Delta,
        )

        saw_any_mask = False
        for i in range(min(30, len(ds))):
            sample = ds[i]
            ids, mask = sample["input_ids"], sample["style_ref_mask"]
            assert len(ids) == len(mask)
            in_appendix = False
            for j, tok in enumerate(ids):
                tt = vocab.get_type(tok)
                if tt == _core.TokenType.Track:
                    in_appendix = False
                elif tt == _core.TokenType.TrackEnd:
                    in_appendix = True
                if not mask[j]:
                    continue
                saw_any_mask = True
                assert tt in expressive, (
                    f"style_ref_mask flagged a non-expressive token at position {j}"
                )
                assert not in_appendix, (
                    "style_ref_mask flagged a token at/after its track's TrackEnd "
                    "(appendix content, never in-place skeleton)"
                )
        assert saw_any_mask, "No sample among the first 30 had a non-empty style_ref_mask"

    def test_style_pretrain_mode_returns_two_expressive_only_views(
        self, humanize_tokenizer, training_parquet
    ):
        from midigpt.training.dataset import MidiGPTDataset

        ds = MidiGPTDataset(
            str(training_parquet),
            humanize_tokenizer,
            infill_probability=0.0,
            humanize_probability=0.0,
            mask_bar_config=None,
            max_seq_len=512,
            max_tracks=4,
            min_tracks=1,
            min_fill_ratio=0.5,
            style_pretrain_mode=True,
        )
        vocab = humanize_tokenizer._vocab
        expressive = (
            _core.TokenType.VelocityLevel,
            _core.TokenType.DeltaDirection,
            _core.TokenType.Delta,
        )
        for i in range(min(5, len(ds))):
            sample = ds[i]
            for key in ("anchor", "positive"):
                ids, mask = sample[f"{key}_ids"], sample[f"{key}_mask"]
                assert len(ids) == len(mask)
                # The mask must exactly equal "is this an expressive token" —
                # style_pretrain_mode has no target/reference split, the
                # whole window is the segment.
                for j, tok in enumerate(ids):
                    assert mask[j] == (vocab.get_type(tok) in expressive)

    def test_humanize_rejected_without_encoder_support(self, ghost_tokenizer, training_parquet):
        from midigpt.training.dataset import MidiGPTDataset

        with pytest.raises(ValueError, match="supports_humanize"):
            MidiGPTDataset(
                str(training_parquet),
                ghost_tokenizer,
                infill_probability=0.0,
                humanize_probability=0.5,
                mask_bar_config=None,
                max_seq_len=128,
            )


@pytest.mark.slow
@pytest.mark.inference
def test_training_smoke(ghost_tokenizer, training_parquet, tmp_path):
    """Full 1-step Lightning training loop — verifies end-to-end integration."""
    import dataclasses
    import math

    import lightning as L

    from midigpt.inference.model.gpt2 import GPT2Config, GPT2LMHeadModel
    from midigpt.training.data_module import MidiGPTDataModule
    from midigpt.training.lightning_module import MidiGPTLightningModule

    gpt2_cfg = GPT2Config(
        vocab_size=ghost_tokenizer.vocab_size(),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(gpt2_cfg)

    @dataclasses.dataclass
    class _Cfg:
        learning_rate: float = 1e-3
        weight_decay: float = 0.01
        warmup_steps: int = 0
        lr_scheduler_type: str = "constant"

    dm = MidiGPTDataModule(
        train_path=str(training_parquet),
        tokenizer=ghost_tokenizer,
        infill_probability=0.0,
        mask_bar_config=None,
        max_seq_len=64,
        max_tracks=4,
        min_tracks=1,
        min_fill_ratio=0.5,
        per_device_batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    lit = MidiGPTLightningModule(model, _Cfg())
    lit.total_steps = 2

    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_steps=2,
        precision="32",
        log_every_n_steps=1,
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, dm)
    # Verify the model saved to a bundle.
    out = tmp_path / "smoke.safetensors"
    import json as _json
    enc = _json.loads(ghost_tokenizer._vocab.config().to_json())
    model.save_pretrained(str(out), encoder_config=enc)
    assert out.exists() and out.stat().st_size > 0
