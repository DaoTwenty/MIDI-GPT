"""Hierarchical fusion modules for combining local bar latents with global track context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatLinearFusion(nn.Module):
    """Concatenation + linear projection fusion.

    Concatenates the local bar latents with the global track representation and
    projects back to feature_dim via an MLP with residual connection.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.ln_out = nn.LayerNorm(feature_dim)

    def forward(self, h_bar: torch.Tensor, h_track: torch.Tensor) -> torch.Tensor:
        """Fuse local bar features with global track context.

        Args:
            h_bar: Local bar features of shape ``(B, K, D)``.
            h_track: Global track feature of shape ``(B, D)`` or ``(B, 1, D)``.

        Returns:
            Fused features of shape ``(B, K, D)``.
        """
        if h_track.ndim == 2:
            h_track = h_track.unsqueeze(1)  # (B, 1, D)

        K = h_bar.shape[1]
        h_track_expanded = h_track.expand(-1, K, -1)  # (B, K, D)

        cat = torch.cat([h_bar, h_track_expanded], dim=-1)  # (B, K, 2D)
        fused = h_bar + self.proj(cat)
        return self.ln_out(fused)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) fusion.

    The global track context generates affine transformation parameters (gamma, beta)
    that dynamically scale and shift the local bar representation.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.film_gen = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )
        self.ln_out = nn.LayerNorm(feature_dim)

    def forward(self, h_bar: torch.Tensor, h_track: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h_bar: Local bar features of shape ``(B, K, D)``.
            h_track: Global track feature of shape ``(B, D)`` or ``(B, 1, D)``.

        Returns:
            Modulated features of shape ``(B, K, D)``.
        """
        if h_track.ndim == 3:
            h_track = h_track.squeeze(1)  # (B, D)

        film_params = self.film_gen(h_track)  # (B, 2D)
        gamma, beta = film_params.chunk(2, dim=-1)  # (B, D), (B, D)

        gamma = gamma.unsqueeze(1)  # (B, 1, D)
        beta = beta.unsqueeze(1)  # (B, 1, D)

        # Modulate: h_fused = (1 + gamma) * h_bar + beta
        fused = (1.0 + gamma) * h_bar + beta
        return self.ln_out(fused)


class ResidualAddFusion(nn.Module):
    """Additive residual fusion.

    Projects the global track vector and adds it as a residual bias to the local
    bar representations.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.track_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.ln_out = nn.LayerNorm(feature_dim)

    def forward(self, h_bar: torch.Tensor, h_track: torch.Tensor) -> torch.Tensor:
        """Add projected global track representation to bar latents.

        Args:
            h_bar: Local bar features of shape ``(B, K, D)``.
            h_track: Global track feature of shape ``(B, D)`` or ``(B, 1, D)``.

        Returns:
            Fused features of shape ``(B, K, D)``.
        """
        if h_track.ndim == 2:
            h_track = h_track.unsqueeze(1)  # (B, 1, D)

        proj_track = self.track_proj(h_track)  # (B, 1, D)
        fused = h_bar + proj_track
        return self.ln_out(fused)
