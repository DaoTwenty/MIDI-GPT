"""Post-training quantization wrappers.

Two backends:
  quantize_dynamic  — torch-native CPU int8 (zero extra deps, export-friendly)
  quantize_bnb      — bitsandbytes GPU int8 or NF4 (requires: pip install bitsandbytes)

GPT-2 uses Conv1D (weight (nx, nf), applies x @ w + b) rather than nn.Linear.
torch.quantization.quantize_dynamic only handles nn.Linear, so we replace
Conv1D layers with equivalent nn.Linear modules before quantizing.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ── Conv1D → Linear conversion ─────────────────────────────────────────────────

def _replace_conv1d_with_linear(model: nn.Module) -> nn.Module:
    """Walk the model tree, replace every Conv1D with an equivalent nn.Linear.

    Conv1D(nx, nf): weight (nx, nf), forward x @ w + b  → maps nx → nf
    nn.Linear(nx, nf): weight (nf, nx), forward x @ w.T + b → same mapping
    """
    try:
        from midigpt.inference.model.gpt2 import Conv1D
    except ImportError:
        return model

    for name, child in list(model.named_children()):
        if isinstance(child, Conv1D):
            nx, nf = child.weight.shape
            lin = nn.Linear(nx, nf, bias=True)
            lin.weight = nn.Parameter(child.weight.data.T.contiguous())
            lin.bias = nn.Parameter(child.bias.data.clone())
            setattr(model, name, lin)
        else:
            _replace_conv1d_with_linear(child)
    return model


# ── dynamic quantization ───────────────────────────────────────────────────────

def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False,
) -> nn.Module:
    """Apply torch dynamic quantization (CPU int8).

    Converts all Conv1D layers to nn.Linear first (required for torch's
    quantization engine), then applies quantize_dynamic over {nn.Linear}.

    Args:
        model  : The model to quantize. Should be on CPU.
        dtype  : Quantization dtype — torch.qint8 (default) or torch.float16.
        inplace: If False (default), operates on a deep copy.

    Returns the quantized model. Suitable for CPU inference and ONNX export.
    Quantized models cannot be fine-tuned.
    """
    if not inplace:
        model = copy.deepcopy(model)

    model = model.cpu().eval()
    model = _replace_conv1d_with_linear(model)
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)
    log.info("dynamic quantization applied (dtype=%s)", dtype)
    return quantized


# ── bitsandbytes quantization ──────────────────────────────────────────────────

def quantize_bnb(
    model: nn.Module,
    bits: int = 8,
    inplace: bool = False,
) -> nn.Module:
    """Apply bitsandbytes in-place quantization (GPU int8 or NF4).

    Replaces nn.Linear and Conv1D layers with bitsandbytes equivalents.
    Requires: pip install bitsandbytes

    Args:
        model : The model to quantize. Should be on CUDA.
        bits  : 8 for LLM.int8() quantization, 4 for NF4 quantization.
        inplace: If False (default), operates on a deep copy.

    Returns the quantized model. Suitable for GPU inference with reduced
    memory footprint. Quantized models cannot be fine-tuned.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for GPU quantization: pip install bitsandbytes"
        ) from None

    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    if not inplace:
        model = copy.deepcopy(model)

    # Convert Conv1D to nn.Linear first so bnb can handle them uniformly.
    model = _replace_conv1d_with_linear(model)

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parent, attr = _resolve_parent(model, name)
        if parent is None:
            continue
        if bits == 8:
            replacement = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
        else:
            replacement = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                quant_type="nf4",
                compute_dtype=torch.float16,
            )
        replacement.weight = bnb.nn.Int8Params(
            module.weight.data,
            requires_grad=False,
            has_fp16_weights=(bits == 16),
        ) if bits == 8 else module.weight
        if module.bias is not None:
            replacement.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        setattr(parent, attr, replacement)

    log.info("bitsandbytes %d-bit quantization applied", bits)
    return model


# ── helper ─────────────────────────────────────────────────────────────────────

def _resolve_parent(
    root: nn.Module, dotted_name: str
) -> tuple[nn.Module | None, str]:
    """Return (parent_module, attribute_name) for a dotted module path."""
    parts = dotted_name.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None, ""
    return parent, parts[-1]
