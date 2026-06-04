"""Weight pruning utilities.

Two strategies:
  prune_magnitude  — global unstructured L1 magnitude pruning (torch.nn.utils.prune)
  prune_heads      — structured attention-head pruning (zero out full head weights)

Both strategies apply PyTorch's forward-hook masking so the sparsity is active
immediately without altering the state_dict layout. Call make_pruning_permanent()
to fold masks into weights before export or further fine-tuning.

Head importance heuristic: mean absolute value of a head's Q/K/V weight slice in
c_attn, combined with its output slice in c_proj. Lower score = less important.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

log = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    # ── unstructured pruning ───────────────────────────────────────────────────
    amount: float = 0.1         # fraction of weights to prune (0.0–1.0)

    # ── structured head pruning ────────────────────────────────────────────────
    head_amount: float = 0.25   # fraction of attention heads to prune per layer
    head_layers: list[int] = field(default_factory=list)  # [] = all layers

    def validate(self) -> None:
        if not (0.0 < self.amount < 1.0):
            raise ValueError(f"amount must be in (0, 1), got {self.amount}")
        if not (0.0 < self.head_amount < 1.0):
            raise ValueError(f"head_amount must be in (0, 1), got {self.head_amount}")


# ── unstructured magnitude pruning ────────────────────────────────────────────

def prune_magnitude(
    model: nn.Module,
    amount: float = 0.1,
) -> nn.Module:
    """Global unstructured L1 magnitude pruning.

    Prunes `amount` fraction of weights with the smallest absolute value across
    all Conv1D and nn.Linear weight tensors in the model. Uses PyTorch's forward-
    hook mechanism — call make_pruning_permanent() before export.

    Args:
        model  : The model to prune (modified in-place).
        amount : Fraction of total weights to zero out (0.0–1.0).

    Returns the model with pruning hooks applied.
    """
    try:
        from midigpt.inference.model.gpt2 import Conv1D
        target_types = (nn.Linear, Conv1D)
    except ImportError:
        target_types = (nn.Linear,)

    params = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, target_types)
    ]
    if not params:
        log.warning("prune_magnitude: no prunable layers found")
        return model

    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    total = sum(p.numel() for m, _ in params for p in [m.weight])
    zeroed = sum((m.weight == 0).sum().item() for m, _ in params)
    log.info(
        "magnitude pruning: %.1f%% of weights zeroed (%d / %d)",
        100.0 * zeroed / max(total, 1), zeroed, total,
    )
    return model


# ── structured head pruning ───────────────────────────────────────────────────

def prune_heads(
    model: nn.Module,
    head_amount: float = 0.25,
    layers: list[int] | None = None,
) -> dict[int, list[int]]:
    """Structured attention-head pruning by weight magnitude.

    Ranks heads within each layer by importance (mean |weight| of their
    Q/K/V slices in c_attn plus their output slice in c_proj). The bottom
    `head_amount` fraction are zeroed out structurally.

    Modifies weights in-place. No PyTorch pruning hooks are used — the
    zeroed weights persist in the state dict directly.

    Args:
        model       : GPT2LMHeadModel (or compatible).
        head_amount : Fraction of heads per layer to prune.
        layers      : Layer indices to prune. None = all layers.

    Returns {layer_idx: [pruned_head_indices, ...]} for inspection.
    """
    try:
        from midigpt.inference.model.gpt2 import GPT2Attention
    except ImportError:
        raise ImportError("prune_heads requires midigpt.inference.model.gpt2") from None

    pruned: dict[int, list[int]] = {}

    for layer_idx, block in enumerate(_iter_blocks(model)):
        if layers is not None and layer_idx not in layers:
            continue

        attn = block.attn
        n_head = attn.n_head
        head_dim = attn.head_dim
        n_embd = n_head * head_dim

        n_prune = max(1, round(n_head * head_amount))

        # ── importance score per head ──────────────────────────────────────────
        # c_attn weight: (n_embd, 3*n_embd)
        # Q slice: [:, 0:n_embd], K: [:, n_embd:2n], V: [:, 2n:3n]
        w_attn = attn.c_attn.weight.data  # (n_embd, 3*n_embd)
        w_proj = attn.c_proj.weight.data  # (n_embd, n_embd)

        scores = []
        for h in range(n_head):
            s = h * head_dim
            e = s + head_dim
            q_score = w_attn[:, s:e].abs().mean()
            k_score = w_attn[:, n_embd + s : n_embd + e].abs().mean()
            v_score = w_attn[:, 2 * n_embd + s : 2 * n_embd + e].abs().mean()
            # c_proj input: head outputs are in rows [s:e] (after _merge_heads)
            p_score = w_proj[s:e, :].abs().mean()
            scores.append(((q_score + k_score + v_score + p_score) / 4).item())

        least_important = sorted(range(n_head), key=lambda h: scores[h])[:n_prune]

        # ── zero out ───────────────────────────────────────────────────────────
        with torch.no_grad():
            for h in least_important:
                s = h * head_dim
                e = s + head_dim
                attn.c_attn.weight[:, s:e] = 0
                attn.c_attn.weight[:, n_embd + s : n_embd + e] = 0
                attn.c_attn.weight[:, 2 * n_embd + s : 2 * n_embd + e] = 0
                if attn.c_attn.bias is not None:
                    attn.c_attn.bias[s:e] = 0
                    attn.c_attn.bias[n_embd + s : n_embd + e] = 0
                    attn.c_attn.bias[2 * n_embd + s : 2 * n_embd + e] = 0
                attn.c_proj.weight[s:e, :] = 0

        pruned[layer_idx] = least_important
        log.info(
            "layer %d: pruned heads %s (importance scores: %s)",
            layer_idx, least_important,
            [f"{scores[h]:.4f}" for h in least_important],
        )

    return pruned


def head_importance_scores(model: nn.Module) -> dict[int, list[float]]:
    """Return {layer_idx: [score_head_0, score_head_1, ...]} for inspection."""
    try:
        from midigpt.inference.model.gpt2 import GPT2Attention  # noqa: F401
    except ImportError:
        raise ImportError("head_importance_scores requires midigpt.inference.model.gpt2") from None

    result: dict[int, list[float]] = {}
    for layer_idx, block in enumerate(_iter_blocks(model)):
        attn = block.attn
        n_head = attn.n_head
        head_dim = attn.head_dim
        n_embd = n_head * head_dim
        w_attn = attn.c_attn.weight.data
        w_proj = attn.c_proj.weight.data
        scores = []
        for h in range(n_head):
            s, e = h * head_dim, (h + 1) * head_dim
            q = w_attn[:, s:e].abs().mean()
            k = w_attn[:, n_embd + s : n_embd + e].abs().mean()
            v = w_attn[:, 2 * n_embd + s : 2 * n_embd + e].abs().mean()
            p = w_proj[s:e, :].abs().mean()
            scores.append(((q + k + v + p) / 4).item())
        result[layer_idx] = scores
    return result


# ── make permanent ────────────────────────────────────────────────────────────

def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """Fold all PyTorch pruning masks into weights and remove hooks.

    Call this before saving a checkpoint or exporting to ONNX so the state
    dict does not include _mask and _orig tensors.
    """
    try:
        from midigpt.inference.model.gpt2 import Conv1D
        target_types = (nn.Linear, Conv1D)
    except ImportError:
        target_types = (nn.Linear,)

    count = 0
    for module in model.modules():
        if isinstance(module, target_types):
            try:
                prune.remove(module, "weight")
                count += 1
            except ValueError:
                pass  # no pruning hook on this module

    log.info("make_pruning_permanent: removed hooks from %d modules", count)
    return model


def sparsity(model: nn.Module) -> float:
    """Return the global weight sparsity as a fraction in [0, 1]."""
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return zeros / max(total, 1)


# ── internal ──────────────────────────────────────────────────────────────────

def _iter_blocks(model: nn.Module):
    """Yield GPT2Block instances from transformer.h (or equivalent)."""
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError("Model has no 'transformer' attribute — expected GPT2LMHeadModel")
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        raise AttributeError("transformer has no 'h' attribute — expected ModuleList of GPT2Block")
    yield from blocks
