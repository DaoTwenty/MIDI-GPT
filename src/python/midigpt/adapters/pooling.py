"""Temporal pooling modules for condensing audio frame sequences into bar latent states."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanConvPooler(nn.Module):
    """1D Convolution + temporal average pooling.

    Passes frame hidden states through a temporal convolution layer to capture
    local transitions, followed by average pooling across the bar duration.
    """

    def __init__(self, feature_dim: int, num_outputs: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_outputs = num_outputs

        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )
        self.ln = nn.LayerNorm(feature_dim)
        if num_outputs > 1:
            self.expand = nn.Linear(feature_dim, num_outputs * feature_dim)
        else:
            self.expand = None

    def forward(self, frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool frame representations.

        Args:
            frames: Tensor of shape ``(B, T, D)``.
            mask: Optional boolean mask of shape ``(B, T)`` where True indicates valid frames.

        Returns:
            Pooled tensor of shape ``(B, num_outputs, D)``.
        """
        B, T, D = frames.shape
        if T == 0:
            return torch.zeros(B, self.num_outputs, D, device=frames.device, dtype=frames.dtype)

        # Conv over time dimension: (B, D, T)
        x = frames.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)  # (B, T, D)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).to(frames.dtype)
            pooled = (x * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        else:
            pooled = x.mean(dim=1)  # (B, D)

        pooled = self.ln(pooled)

        if self.expand is not None:
            pooled = self.expand(pooled).view(B, self.num_outputs, D)
        else:
            pooled = pooled.unsqueeze(1)  # (B, 1, D)

        return pooled


class PerceiverResampler(nn.Module):
    """Learned Cross-Attention Perceiver Resampler.

    Uses learnable latent query vectors to cross-attend to variable-length audio
    frames, learning dynamic saliency (e.g. note attacks, harmonic shifts) while
    ignoring quiet/irrelevant frames.
    """

    def __init__(
        self,
        feature_dim: int,
        num_latents: int = 1,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_latents = num_latents
        self.num_heads = num_heads

        # Learned latent query vectors: (1, num_latents, feature_dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, feature_dim) * 0.02)

        self.ln_latents = nn.LayerNorm(feature_dim)
        self.ln_frames = nn.LayerNorm(feature_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, ff_mult * feature_dim),
            nn.GELU(),
            nn.Linear(ff_mult * feature_dim, feature_dim),
        )

    def forward(self, frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Resample variable frames into fixed K latent tokens.

        Args:
            frames: Tensor of shape ``(B, T, D)``.
            mask: Optional boolean mask of shape ``(B, T)`` where True = valid, False = pad.

        Returns:
            Tensor of shape ``(B, num_latents, D)``.
        """
        B, T, D = frames.shape
        if T == 0:
            return self.latents.repeat(B, 1, 1)

        # Expand latents for batch: (B, K, D)
        q = self.latents.repeat(B, 1, 1)
        q_norm = self.ln_latents(q)
        k_norm = self.ln_frames(frames)

        key_padding_mask = ~mask if mask is not None else None

        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=k_norm,
            value=k_norm,
            key_padding_mask=key_padding_mask,
        )
        x = q + attn_out
        x = x + self.mlp(self.ln_ff(x))
        return x  # (B, num_latents, D)


class BeatGridPooler(nn.Module):
    """Beat-Grid segmented pooler emitting K tokens per bar (e.g. 4 for 4/4).

    Splits the temporal frame sequence into K equal beat windows and applies
    per-window convolutional pooling, preserving sub-bar temporal progression.
    """

    def __init__(self, feature_dim: int, beats_per_bar: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.beats_per_bar = beats_per_bar

        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool frame representations into beat-level tokens.

        Args:
            frames: Tensor of shape ``(B, T, D)``.
            mask: Optional boolean mask of shape ``(B, T)``.

        Returns:
            Tensor of shape ``(B, beats_per_bar, D)``.
        """
        B, T, D = frames.shape
        if T == 0:
            return torch.zeros(
                B, self.beats_per_bar, D, device=frames.device, dtype=frames.dtype
            )

        # Conv over time: (B, D, T)
        conv_frames = self.conv(frames.transpose(1, 2)).transpose(1, 2)  # (B, T, D)

        # Split T into beats_per_bar chunks
        beat_outputs = []
        chunk_size = max(1, T // self.beats_per_bar)

        for b in range(self.beats_per_bar):
            start_idx = b * chunk_size
            end_idx = (b + 1) * chunk_size if b < self.beats_per_bar - 1 else T
            start_idx = min(start_idx, T - 1)
            end_idx = max(start_idx + 1, min(end_idx, T))

            chunk = conv_frames[:, start_idx:end_idx, :]  # (B, T_chunk, D)
            if mask is not None:
                chunk_mask = mask[:, start_idx:end_idx].unsqueeze(-1).to(frames.dtype)
                chunk_sum = (chunk * chunk_mask).sum(dim=1)
                chunk_count = torch.clamp(chunk_mask.sum(dim=1), min=1.0)
                pooled_beat = chunk_sum / chunk_count
            else:
                pooled_beat = chunk.mean(dim=1)  # (B, D)

            beat_outputs.append(pooled_beat)

        out = torch.stack(beat_outputs, dim=1)  # (B, beats_per_bar, D)
        return self.ln(out)
