"""Low-Rank Adaptation (LoRA) modules and utilities for PEFT fine-tuning."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from midigpt.inference.model.gpt2 import Conv1D


class LoRAConv1D(nn.Module):
    """LoRA wrapper for HF-style Conv1D layers (weight is (nx, nf))."""

    def __init__(
        self,
        base_layer: Conv1D,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        nx, nf = base_layer.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(nx, r))
        self.lora_B = nn.Parameter(torch.zeros(r, nf))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize: A Gaussian, B zero so delta starts at 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward: x @ W + b
        base_out = self.base_layer(x)
        if self.r > 0:
            # LoRA delta: x @ (A @ B) * scaling
            # (..., nx) @ (nx, r) -> (..., r) @ (r, nf) -> (..., nf)
            delta = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
            return base_out + delta
        return base_out


class LoRALinear(nn.Module):
    """LoRA wrapper for standard nn.Linear layers."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.r > 0:
            delta = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scaling
            return base_out + delta
        return base_out


def apply_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    target_module_names: Sequence[str] = ("c_attn", "c_proj", "c_fc"),
) -> nn.Module:
    """Recursively wrap targeted Conv1D/Linear layers with LoRA adapters.

    Args:
        model: PyTorch model (e.g. GPT2LMHeadModel).
        r: Rank of the low-rank adaptation.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability.
        target_module_names: Submodule attribute names to target.

    Returns:
        The modified model with LoRA layers inserted.
    """
    for name, module in model.named_children():
        if any(target in name for target in target_module_names):
            if isinstance(module, Conv1D):
                setattr(
                    model,
                    name,
                    LoRAConv1D(
                        module,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    ),
                )
            elif isinstance(module, nn.Linear):
                setattr(
                    model,
                    name,
                    LoRALinear(
                        module,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    ),
                )
        else:
            apply_lora(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_module_names=target_module_names,
            )
    return model


def freeze_all_except_lora_and_adapter(
    model: nn.Module, extra_trainable_modules: Sequence[nn.Module] | None = None
) -> None:
    """Freeze all base model weights except LoRA parameters and specified adapter modules."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True

    # Unfreeze extra modules
    if extra_trainable_modules is not None:
        for mod in extra_trainable_modules:
            for param in mod.parameters():
                param.requires_grad = True
