"""Online knowledge distillation trainer.

Teacher runs a forward pass each step (no pre-caching) so soft targets remain
correct under data augmentation (pitch/velocity changes alter token IDs).

Loss = α · KD(student ‖ teacher) + (1 - α) · CE(student, hard_labels)

where KD uses temperature-scaled KL divergence and CE is standard cross-entropy.
KD temperature T sharpens/softens the teacher distribution; higher T transfers
more information from near-uniform mass, lower T behaves more like hard labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    # ── loss weighting ─────────────────────────────────────────────────────────
    temperature: float = 4.0    # T: softens teacher/student distributions for KD
    alpha: float = 0.5          # weight on KD loss; (1-alpha) on hard CE loss

    # ── optimiser ──────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_train_steps: int = 10_000
    gradient_clip: float = 1.0

    # ── logging / checkpointing ────────────────────────────────────────────────
    log_every: int = 10
    checkpoint_every: int = 500
    checkpoint_dir: str = "checkpoints/distill"

    def validate(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


class DistillationTrainer:
    """Trains a student model to mimic a frozen teacher.

    Args:
        teacher : Larger (or differently-configured) model used as the oracle.
                  Must return (logits, ...) from forward(input_ids).
        student : Smaller model to be trained. Must share the same vocabulary.
        config  : DistillationConfig hyperparameters.
        device  : Torch device. Defaults to the student's current device.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig,
        device: torch.device | str | None = None,
    ):
        config.validate()
        self.teacher = teacher.eval()
        self.student = student
        self.config = config

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        if device is None:
            try:
                device = next(student.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.teacher.to(self.device)
        self.student.to(self.device)

        self._optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._scheduler = _warmup_linear_schedule(
            self._optimizer, config.warmup_steps, config.num_train_steps
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def train(
        self,
        dataloader,
        *,
        resume_step: int = 0,
    ) -> None:
        """Training loop over an iterable of batches.

        Each batch must be a dict with keys ``input_ids`` (B, T) and
        ``labels`` (B, T), matching the supervised training format.
        """
        cfg = self.config
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        step = resume_step

        for batch in dataloader:
            if step >= cfg.num_train_steps:
                break

            loss, metrics = self.train_step(batch)
            step += 1

            if step % cfg.log_every == 0:
                log.info(
                    "step %d | loss=%.4f ce=%.4f kd=%.4f",
                    step, metrics["loss"], metrics["ce_loss"], metrics["kd_loss"],
                )

            if step % cfg.checkpoint_every == 0:
                self._save_checkpoint(step)

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Single gradient step. Returns (loss, metrics_dict)."""
        cfg = self.config
        self.student.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        B, T = input_ids.shape
        V = _vocab_size(self.student)

        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out

        student_out = self.student(input_ids)
        student_logits = student_out[0] if isinstance(student_out, tuple) else student_out

        # Shift for next-token prediction.
        s_shift = student_logits[:, :-1].contiguous()  # (B, T-1, V)
        t_shift = teacher_logits[:, :-1].contiguous()
        l_shift = labels[:, 1:].contiguous()            # (B, T-1)

        # Hard cross-entropy loss.
        ce_loss = F.cross_entropy(
            s_shift.view(-1, V), l_shift.view(-1), ignore_index=-100
        )

        # Soft KD loss — only over non-padding positions.
        mask = (l_shift != -100).reshape(-1)
        T_kd = cfg.temperature
        kd_loss = F.kl_div(
            F.log_softmax(s_shift.reshape(-1, V)[mask] / T_kd, dim=-1),
            F.softmax(t_shift.reshape(-1, V)[mask] / T_kd, dim=-1),
            reduction="batchmean",
        ) * (T_kd ** 2)

        loss = cfg.alpha * kd_loss + (1.0 - cfg.alpha) * ce_loss

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.gradient_clip)
        self._optimizer.step()
        self._scheduler.step()

        return loss.detach(), {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "kd_loss": kd_loss.item(),
        }

    # ── internals ──────────────────────────────────────────────────────────────

    def _save_checkpoint(self, step: int) -> None:
        path = Path(self.config.checkpoint_dir) / f"step_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            path,
        )
        log.info("checkpoint saved: %s", path)


# ── helpers ────────────────────────────────────────────────────────────────────

def _vocab_size(model: nn.Module) -> int:
    for name, p in model.named_parameters():
        if "wte.weight" in name or "lm_head.weight" in name:
            return p.shape[0]
    raise RuntimeError("Cannot infer vocab size from model parameters.")


def _warmup_linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
