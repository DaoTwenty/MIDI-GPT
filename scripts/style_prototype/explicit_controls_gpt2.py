"""Variant C: joint-conditioning via the same soft-prefix mechanism as
Variant A (conditioned_gpt2.py), but the prefix vector comes from
explicit_controls.py's hand-designed statistics instead of a learned
StyleEncoder -- a small per-control lookup-table embedding, not a
transformer over raw tokens.

Deliberately NOT sharing code with conditioned_gpt2.py: Variant A's GPU
training jobs read that file at process start, and this file is being added
while those jobs may still be queued -- duplication here trades a small
amount of repeated code for zero risk of an in-flight job importing a
half-edited shared module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from explicit_controls import N_BINS, N_CONTROLS
from midigpt.inference.model.gpt2 import GPT2Config, GPT2Transformer


class ExplicitControlsGPT2(nn.Module):
    def __init__(self, gpt2_cfg: GPT2Config, control_dim: int = 16, n_bins: int = N_BINS):
        super().__init__()
        self.cfg = gpt2_cfg
        self.n_bins = n_bins
        self.transformer = GPT2Transformer(gpt2_cfg)
        self.lm_head = nn.Linear(gpt2_cfg.n_embd, gpt2_cfg.vocab_size, bias=False)
        # One small embedding table per control -- each bin index is looked
        # up independently, then concatenated and projected. Interpretable
        # by construction: every dimension of the prefix traces back to one
        # named, hand-designed statistic, not a learned latent mixture.
        self.control_embeds = nn.ModuleList(
            [nn.Embedding(n_bins, control_dim) for _ in range(N_CONTROLS)]
        )
        self.combine = nn.Linear(N_CONTROLS * control_dim, gpt2_cfg.n_embd)
        # Fallback prefix for samples with no available reference segment
        # (mirrors ConditionedGPT2.null_style_embed).
        self.null_control_embed = nn.Parameter(torch.zeros(gpt2_cfg.n_embd))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.transformer.wte.weight, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        proj_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, std=proj_std)

    def encode_control_prefix(
        self, control_bins: torch.Tensor, has_controls: torch.Tensor
    ) -> torch.Tensor:
        """control_bins: (B, N_CONTROLS) long, each in [0, n_bins). Returns
        (B, n_embd): the projected control embedding for samples with a
        reference segment, the learned fallback for those without."""
        B = control_bins.size(0)
        parts = [self.control_embeds[i](control_bins[:, i]) for i in range(N_CONTROLS)]
        prefix = self.combine(torch.cat(parts, dim=-1))
        null = self.null_control_embed.expand(B, -1)
        return torch.where(has_controls.unsqueeze(-1), prefix, null)

    def forward(
        self,
        input_ids: torch.Tensor,
        control_bins: torch.Tensor,
        has_controls: torch.Tensor,
    ) -> torch.Tensor:
        """Same convention as ConditionedGPT2.forward: returns (B, T, V),
        position i predicting input_ids[:, i] directly (the control prefix
        supplies the "previous token" context for position 0)."""
        B, T = input_ids.shape
        prefix = self.encode_control_prefix(control_bins, has_controls)  # (B, D)

        real_pos = torch.arange(1, T + 1, device=input_ids.device).unsqueeze(0)
        tok_emb = self.transformer.wte(input_ids) + self.transformer.wpe(real_pos)

        prefix_pos = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)
        prefix_emb = prefix.unsqueeze(1) + self.transformer.wpe(prefix_pos)

        x = torch.cat([prefix_emb, tok_emb], dim=1)  # (B, T+1, D)
        for block in self.transformer.h:
            x, _, _ = block(x, past_kv=None, return_attn_weights=False, key_mask=None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits[:, :-1, :]
