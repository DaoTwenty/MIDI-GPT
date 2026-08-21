"""Variant A: joint-conditioning via a soft-prefix.

Standalone wrapper around GPT2Transformer/GPT2Block (imported directly, not
a change to inference/model/gpt2.py -- stays in prototype scope per the plan
doc). Owns a StyleEncoder; forward projects the pooled style vector z into
the LM's embedding space and concatenates it as one extra position at index
0, ahead of the real token embeddings (shifting their position ids to
1..T). Causal self-attention then lets every real position see the style
prefix "for free" via ordinary attention -- no per-bar injection needed.

Training-only (full-sequence teacher-forced, no past_kv/KV-cache handling
here) -- this is not meant to be dropped into InferenceEngine as-is.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from midigpt.inference.model.gpt2 import GPT2Config, GPT2Transformer
from style_encoder import StyleEncoder, StyleEncoderConfig


class ConditionedGPT2(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, style_cfg: StyleEncoderConfig):
        super().__init__()
        self.cfg = gpt2_cfg
        self.style_cfg = style_cfg
        self.transformer = GPT2Transformer(gpt2_cfg)
        self.lm_head = nn.Linear(gpt2_cfg.n_embd, gpt2_cfg.vocab_size, bias=False)
        self.style_encoder = StyleEncoder(style_cfg)
        self.z_proj = nn.Linear(style_cfg.z_dim, gpt2_cfg.n_embd)
        # Fallback prefix for samples with no available reference segment
        # (MidiGPTDataset's own fallback -- see dataset.py's _encode_one).
        # Keeps every sample's sequence length uniform (always exactly one
        # prefix position) without conflating "no reference available" with
        # a real, learnable style signal the way a zero vector would.
        self.null_style_embed = nn.Parameter(torch.zeros(gpt2_cfg.n_embd))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.transformer.wte.weight, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        proj_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, std=proj_std)

    def load_pretrained_style_encoder(self, state_dict: dict, freeze: bool) -> None:
        self.style_encoder.load_state_dict(state_dict)
        if freeze:
            for p in self.style_encoder.parameters():
                p.requires_grad_(False)

    def encode_style_prefix(
        self, style_ids: torch.Tensor, style_key_mask: torch.Tensor, has_style: torch.Tensor
    ) -> torch.Tensor:
        """Returns the projected prefix embedding, (B, n_embd): the real
        projected z for samples with a reference segment, the learned
        fallback for those without."""
        B = style_ids.size(0)
        # StyleEncoder's pooling attention softmaxes to NaN over an
        # all-masked row -- samples with has_style=False get a dummy
        # non-empty mask fed through purely to keep the forward pass finite;
        # their actual output is discarded and replaced below.
        safe_mask = style_key_mask.clone()
        safe_mask[~has_style, 0] = True
        z = self.style_encoder(style_ids, safe_mask)
        prefix = self.z_proj(z)
        null = self.null_style_embed.expand(B, -1)
        return torch.where(has_style.unsqueeze(-1), prefix, null)

    def forward(
        self,
        input_ids: torch.Tensor,
        style_ids: torch.Tensor,
        style_key_mask: torch.Tensor,
        has_style: torch.Tensor,
    ) -> torch.Tensor:
        """Full-sequence teacher-forced forward.

        Returns logits of shape (B, T, V), position i predicting
        input_ids[:, i] directly (the style prefix supplies the "previous
        token" context for position 0, so -- unlike a prefix-less causal LM
        -- even the first real token gets a supervised prediction; no
        further shifting needed by the caller beyond the usual label
        ignore_index=-100 padding mask).
        """
        B, T = input_ids.shape
        prefix = self.encode_style_prefix(style_ids, style_key_mask, has_style)  # (B, D)

        real_pos = torch.arange(1, T + 1, device=input_ids.device).unsqueeze(0)
        tok_emb = self.transformer.wte(input_ids) + self.transformer.wpe(real_pos)

        prefix_pos = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)
        prefix_emb = prefix.unsqueeze(1) + self.transformer.wpe(prefix_pos)

        x = torch.cat([prefix_emb, tok_emb], dim=1)  # (B, T+1, D)
        for block in self.transformer.h:
            x, _, _ = block(x, past_kv=None, return_attn_weights=False, key_mask=None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits[:, :-1, :]  # drop the final position (predicts past end of input)
