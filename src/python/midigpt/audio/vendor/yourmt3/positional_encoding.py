# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Vendored subset of amt/src/model/positional_encoding.py (mimbres/YourMT3).

Trimmed to FixedSinusoidalPositionalEmbedding only -- the file this class
came from also defines AlibiPositionalBias, RoPE wrappers, and Conformer
positional encodings for the perceiver-tf/conformer encoder variants, none
of which the plain-T5 checkpoint this extractor targets uses. See ../NOTICE.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn


class FixedSinusoidalPositionalEmbedding(nn.Embedding):
    """Sinusoidal absolute positional embeddings (APE) of any length.

    Adapted from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding.
    """

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not
        interleaved. The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)
