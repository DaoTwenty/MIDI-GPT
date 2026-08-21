"""Style encoder: compresses an expressive-only token sequence (Velocity/
Delta, via StyleVocab's compact re-indexing) into a small, fixed-size vector
`z` -- "how it was played", never "what was played" (pitch/duration are
never part of its input vocabulary at all, by construction).

LM-independent: only depends on a StyleVocab-sized embedding table, so it can
be pretrained standalone (train_contrastive.py) before any GPT2 checkpoint
exists, or trained jointly with one (train_joint_conditioning.py). Both
variants share this exact module -- only the training procedure differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StyleEncoderConfig:
    vocab_size: int  # StyleVocab.size for the encoder config in use
    d_model: int = 128
    n_layer: int = 3
    n_head: int = 4
    max_len: int = 512
    # Deliberate bottleneck: small enough to resist becoming a verbatim copy
    # of the reference segment (on top of the input already excluding pitch/
    # duration), large enough to carry more than one or two scalar stats.
    z_dim: int = 32

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class _EncoderBlock(nn.Module):
    """Pre-LN transformer block with full (non-causal) self-attention --
    this is a fixed-input encoder, not autoregressive."""

    def __init__(self, cfg: StyleEncoderConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.ln_1 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln_2 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln_1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_mask is not None:
            # key_mask: (B, T) bool, True = real token. Broadcast to
            # (B, 1, 1, T) for scaled_dot_product_attention's bool-mask
            # convention (True = attend, no causal restriction here).
            attn_mask = key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(y)
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleEncoder(nn.Module):
    def __init__(self, cfg: StyleEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Position within the segment, not within the piece -- style should
        # be invariant to where in a performance a reference segment sits.
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([_EncoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5)
        # Single learned query for attention-pooling over all positions,
        # instead of mean-pooling -- lets the encoder learn to weight e.g.
        # downbeats or large deviations more heavily than filler notes.
        self.pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.proj = nn.Linear(cfg.d_model, cfg.z_dim)

    def forward(
        self, token_ids: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """token_ids: (B, T) local StyleVocab ids (see style_vocab.py).
        key_mask: (B, T) bool, True = real token (False = padding).
        Returns z: (B, z_dim), unit-norm.
        """
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device).clamp(max=self.cfg.max_len - 1)
        x = self.embed(token_ids) + self.pos_embed(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x, key_mask)
        x = self.ln_f(x)

        q = self.pool_query.expand(B, -1, -1)  # (B, 1, D)
        scale = self.cfg.d_model**-0.5
        scores = torch.matmul(q, x.transpose(-2, -1)) * scale  # (B, 1, T)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask.unsqueeze(1), float("-inf"))
        weights = scores.softmax(dim=-1)
        pooled = torch.matmul(weights, x).squeeze(1)  # (B, D)

        z = self.proj(pooled)
        return F.normalize(z, dim=-1)
