"""LightningModule for Audio-Conditioned MIDI-GPT Training."""

from __future__ import annotations

import math
from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from midigpt.adapters.audio_adapter import AudioToMidiAdapter
from midigpt.audio.base import BaseAudioExtractor
from midigpt.inference.model.gpt2 import GPT2LMHeadModel
from midigpt.inference.model.lora import (
    apply_lora,
    freeze_all_except_lora_and_adapter,
)

TrainingMode = Literal["adapter_only", "lora", "full"]


class AudioConditionedLightningModule(L.LightningModule):
    """PyTorch Lightning module for training audio-conditioned MIDI infilling.

    Supports 3 comparative training regimes:
      1. adapter_only: Freeze audio extractor + freeze MIDI-GPT; train only adapter.
      2. lora: Freeze audio extractor + freeze base MIDI-GPT; train adapter + LoRA adapters.
      3. full: Freeze audio extractor; train adapter + entire MIDI-GPT.
    """

    def __init__(
        self,
        gpt2_model: GPT2LMHeadModel,
        audio_adapter: AudioToMidiAdapter,
        audio_extractor: BaseAudioExtractor | None = None,
        training_mode: TrainingMode = "adapter_only",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        warmup_steps: int = 200,
        total_steps: int = 5000,
    ):
        super().__init__()
        self.gpt2_model = gpt2_model
        self.audio_adapter = audio_adapter
        self.audio_extractor = audio_extractor
        self.training_mode = training_mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self._configure_training_regime()

    def _configure_training_regime(self) -> None:
        """Freeze/unfreeze appropriate parameter sets based on training_mode."""
        # 1. Audio extractor is always frozen
        if self.audio_extractor is not None:
            for p in self.audio_extractor.parameters():
                p.requires_grad = False

        # 2. Audio adapter is always trainable
        for p in self.audio_adapter.parameters():
            p.requires_grad = True

        # 3. Configure MIDI-GPT backbone
        if self.training_mode == "adapter_only":
            for p in self.gpt2_model.parameters():
                p.requires_grad = False
        elif self.training_mode == "lora":
            # Apply LoRA layers to GPT-2
            apply_lora(
                self.gpt2_model,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_module_names=["c_attn", "c_proj", "c_fc"],
            )
            freeze_all_except_lora_and_adapter(
                self.gpt2_model, extra_trainable_modules=[self.audio_adapter]
            )
        elif self.training_mode == "full":
            for p in self.gpt2_model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

    def forward(
        self,
        input_ids: torch.Tensor,
        soft_token_replacements: dict[int, torch.Tensor] | list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Forward pass through MIDI-GPT with soft token replacement."""
        logits, _ = self.gpt2_model(
            input_ids=input_ids,
            soft_token_replacements=soft_token_replacements,
        )
        return logits

    def _compute_loss(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        # Generate one soft embedding per audio context bar.
        # Positions come from the tokenizer (which [Bar] tokens to replace).
        soft_replacements = None
        if "context_bar_frames" in batch and batch["context_bar_frames"]:
            soft_replacements = []
            samples = batch["context_bar_frames"]
            # A non-collated sample remains supported for unit tests.
            if samples and isinstance(samples[0], torch.Tensor):
                samples = [samples]
                track_samples = [batch.get("context_track_frames", batch.get("track_frames", []))]
                # Backward-compatible test path; production datasets always
                # provide tokenizer-derived audio_bar_positions.
                position_samples = [batch.get("audio_bar_positions", list(range(len(samples[0]))))]
            else:
                track_samples = batch.get("context_track_frames", [[] for _ in samples])
                position_samples = batch.get("audio_bar_positions", [[] for _ in samples])

            for sample_idx, (bar_frames_list, track_frames_list, positions) in enumerate(
                zip(samples, track_samples, position_samples)
            ):
                if len(bar_frames_list) != len(positions):
                    raise ValueError("Each audio bar must have exactly one tokenizer placeholder position.")
                for b_frames, t_frames, pos_idx in zip(bar_frames_list, track_frames_list, positions):
                    b_frames = b_frames.unsqueeze(0) if b_frames.ndim == 2 else b_frames
                    t_frames = t_frames.unsqueeze(0) if t_frames.ndim == 2 else t_frames
                    soft_emb = self.audio_adapter(b_frames, track_frames=t_frames)
                    if soft_emb.shape[1] != 1:
                        raise ValueError("Audio placeholder injection currently requires tokens_per_bar=1.")
                    soft_replacements.append((sample_idx, int(pos_idx), soft_emb))

        logits = self.forward(input_ids, soft_token_replacements=soft_replacements)

        # Shift labels for autoregressive next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # When no audio context bars exist in a batch, the frozen backbone
        # produces a loss with no grad_fn. Anchor to adapter params so
        # backward always has a gradient path.
        if not soft_replacements:
            loss = loss.detach() + 0.0 * sum(p.sum() for p in self.audio_adapter.parameters())

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._compute_loss(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._compute_loss(batch, "val")

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
