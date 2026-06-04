"""Lottery Ticket Hypothesis — Iterative Magnitude Pruning (IMP).

Algorithm (Frankle & Carlin, 2019):
  1. Record initial (or early "rewound") weights W₀.
  2. For each round:
     a. Train the model for `steps_per_round` steps.
     b. Prune the `prune_fraction` of remaining weights with smallest |W|.
     c. Reset surviving weights to W₀ (the "ticket" rewind).
  3. After all rounds the sparse subnetwork (the winning ticket) is returned.

The key idea: the sparse subnetwork with its *original* initialization trains
to the same accuracy as the dense network — it is the "winning ticket."

Weight rewinding is implemented by saving `weight_orig` (the pre-mask tensor
stored by PyTorch's pruning hooks) and overwriting it with W₀ after each prune
step. Pruning masks accumulate across rounds so the total sparsity compounds.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune

from midigpt.optimization.pruning import make_pruning_permanent, sparsity

log = logging.getLogger(__name__)


@dataclass
class LotteryConfig:
    # ── IMP schedule ──────────────────────────────────────────────────────────
    prune_fraction: float = 0.2   # fraction of *remaining* weights pruned each round
    num_rounds: int = 5           # number of prune-rewind cycles
    steps_per_round: int = 1_000  # training steps between prune rounds

    # ── weight rewinding ──────────────────────────────────────────────────────
    # Step at which to snapshot weights for rewinding. 0 = use random init.
    # Set to e.g. 100 for "late resetting" (Frankle et al. 2020) which is more
    # stable for large models.
    rewind_step: int = 0

    # ── optimiser ─────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # ── output ────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/lottery"
    save_each_round: bool = True

    def validate(self) -> None:
        if not (0.0 < self.prune_fraction < 1.0):
            raise ValueError(f"prune_fraction must be in (0, 1), got {self.prune_fraction}")
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if self.rewind_step < 0:
            raise ValueError("rewind_step must be >= 0")


class LotteryTicketTrainer:
    """Finds a winning lottery ticket via iterative magnitude pruning.

    Args:
        model    : The model to train and prune (modified in-place across rounds).
        train_fn : Callable(model, optimizer, dataloader, steps) → None.
                   Responsible for running exactly `steps` gradient updates.
                   Must handle its own loss, backward, and optimizer.step().
        config   : LotteryConfig hyperparameters.
        device   : Torch device. Defaults to the model's current device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_fn: Callable[[nn.Module, torch.optim.Optimizer, object, int], None],
        config: LotteryConfig,
        device: torch.device | str | None = None,
    ):
        config.validate()
        self.model = model
        self.train_fn = train_fn
        self.config = config

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        # Snapshot weights at round 0 for rewinding. These will be overwritten
        # at rewind_step > 0 after the first partial training pass.
        self._rewind_weights: dict[str, torch.Tensor] = _snapshot_weights(model)
        self._rewind_captured = config.rewind_step == 0

    # ── public API ─────────────────────────────────────────────────────────────

    def find_ticket(self, dataloader) -> nn.Module:
        """Run IMP and return the winning-ticket sparse model.

        Args:
            dataloader: Iterable yielding training batches, passed directly to
                        `train_fn`. Must be re-iterable across rounds.

        Returns the model with permanent pruning masks folded in.
        """
        cfg = self.config
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        global_step = 0

        for round_idx in range(cfg.num_rounds):
            log.info(
                "IMP round %d/%d — sparsity before training: %.1f%%",
                round_idx + 1, cfg.num_rounds, sparsity(self.model) * 100,
            )

            # ── optional rewind-step snapshot ─────────────────────────────────
            # Train a few steps first, snapshot weights, then continue.
            if not self._rewind_captured and round_idx == 0:
                warmup_steps = cfg.rewind_step
                self.train_fn(self.model, optimizer, dataloader, warmup_steps)
                global_step += warmup_steps
                self._rewind_weights = _snapshot_weights(self.model)
                self._rewind_captured = True
                remaining = cfg.steps_per_round - warmup_steps
                if remaining > 0:
                    self.train_fn(self.model, optimizer, dataloader, remaining)
                    global_step += remaining
            else:
                self.train_fn(self.model, optimizer, dataloader, cfg.steps_per_round)
                global_step += cfg.steps_per_round

            # ── prune ──────────────────────────────────────────────────────────
            _prune_round(self.model, cfg.prune_fraction)
            current_sparsity = sparsity(self.model)
            log.info(
                "IMP round %d/%d — sparsity after pruning: %.1f%%",
                round_idx + 1, cfg.num_rounds, current_sparsity * 100,
            )

            # ── rewind surviving weights ───────────────────────────────────────
            _rewind_weights(self.model, self._rewind_weights)

            # ── reset optimizer moments ────────────────────────────────────────
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

            if cfg.save_each_round:
                self._save_round(round_idx + 1, current_sparsity)

        make_pruning_permanent(self.model)
        final_sparsity = sparsity(self.model)
        log.info("IMP complete — final sparsity: %.1f%%", final_sparsity * 100)
        return self.model

    # ── internals ──────────────────────────────────────────────────────────────

    def _save_round(self, round_idx: int, sparsity_val: float) -> None:
        path = (
            Path(self.config.checkpoint_dir)
            / f"round_{round_idx:02d}_sparsity{sparsity_val:.2f}.pt"
        )
        torch.save({"round": round_idx, "sparsity": sparsity_val,
                    "model_state_dict": self.model.state_dict()}, path)
        log.info("round checkpoint saved: %s", path)


# ── internal helpers ───────────────────────────────────────────────────────────

def _prunable_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    try:
        from midigpt.inference.model.gpt2 import Conv1D
        target_types = (nn.Linear, Conv1D)
    except ImportError:
        target_types = (nn.Linear,)
    return [(m, "weight") for m in model.modules() if isinstance(m, target_types)]


def _prune_round(model: nn.Module, fraction: float) -> None:
    """Globally prune `fraction` of remaining non-zero weights by magnitude."""
    params = _prunable_params(model)
    if not params:
        return
    torch_prune.global_unstructured(
        params,
        pruning_method=torch_prune.L1Unstructured,
        amount=fraction,
    )


def _snapshot_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Deep-copy all weight(-related) parameters for later rewinding.

    Captures `weight_orig` when pruning hooks are active, else `weight`.
    """
    snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        snapshot[name] = param.data.clone()
    return snapshot


def _rewind_weights(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    """Reset surviving (unmasked) weights to their snapshotted values.

    When pruning hooks are active, PyTorch stores the original (pre-mask) weight
    as `weight_orig`. We overwrite `weight_orig` with the snapshot so the
    effective weight = mask * snapshot_weight after the hook runs.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name])
