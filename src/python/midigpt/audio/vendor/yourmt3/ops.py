# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Vendored subset of amt/src/model/ops.py (mimbres/YourMT3).

Trimmed to the two helpers spectrogram.py actually needs; the rest of the
upstream file (get_layer_norm, count_parameters, adjust_b_to_gcd,
optional_compiler_dynamic) is unused on the encoder-only extraction path and
was dropped. See ../NOTICE.
"""

from packaging.version import parse as VersionParse

import torch
from einops import rearrange


def optional_compiler_disable(func):
    if VersionParse(torch.__version__) >= VersionParse("2.1"):
        return torch.compiler.disable(func)
    else:
        return func


def minmax_normalize(x: torch.Tensor, eps: float = 0.008) -> torch.FloatTensor:
    """Min-max normalization: x_norm = (x - x_min) / (x_max - x_min + eps).

    Args:
        x: (B, T, F)
    Returns:
        (B, T, F) with output range of [0, 1]
    """
    x_max = rearrange(x, "b t f -> b (t f)").max(1, keepdim=True)[0]
    x_min = rearrange(x, "b t f -> b (f t)").min(1, keepdim=True)[0]
    x_max = x_max[:, None, :]
    x_min = x_min[:, None, :]
    return (x - x_min) / (x_max - x_min + eps)
