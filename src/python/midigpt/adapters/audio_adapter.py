"""Unified Audio-to-MIDI Adapter Module."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from midigpt.adapters.hierarchical_fusion import (
    ConcatLinearFusion,
    FiLMFusion,
    ResidualAddFusion,
)
from midigpt.adapters.pooling import (
    BeatGridPooler,
    MeanConvPooler,
    PerceiverResampler,
)

PoolingStrategy = Literal["mean_conv", "perceiver", "beat_grid"]
FusionStrategy = Literal["concat", "film", "residual", "none"]


class AudioToMidiAdapter(nn.Module):
    """Bridges continuous transcription model representations to MIDI-GPT token embeddings.

    Pipeline:
        1. Pool audio frame sequence H_bar (T frames) -> h_bar (K vectors)
        2. (Optional) Fuse global track context h_track -> h_fused
        3. Project h_fused (D_trans) -> e_bar (D_model)
    """

    def __init__(
        self,
        feature_dim: int = 768,
        model_dim: int = 512,
        num_tokens_per_bar: int = 1,
        pooling_strategy: PoolingStrategy = "perceiver",
        fusion_strategy: FusionStrategy = "concat",
        perceiver_heads: int = 8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.num_tokens_per_bar = num_tokens_per_bar
        self.pooling_strategy = pooling_strategy
        self.fusion_strategy = fusion_strategy

        # 1. Pooler
        if pooling_strategy == "mean_conv":
            self.pooler = MeanConvPooler(feature_dim, num_outputs=num_tokens_per_bar)
        elif pooling_strategy == "perceiver":
            self.pooler = PerceiverResampler(
                feature_dim, num_latents=num_tokens_per_bar, num_heads=perceiver_heads
            )
        elif pooling_strategy == "beat_grid":
            self.pooler = BeatGridPooler(feature_dim, beats_per_bar=num_tokens_per_bar)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        # Global track pooler (mean over time)
        self.track_pooler = nn.Sequential(
            nn.LayerNorm(feature_dim),
        )

        # 2. Hierarchical Fusion
        if fusion_strategy == "concat":
            self.fusion = ConcatLinearFusion(feature_dim)
        elif fusion_strategy == "film":
            self.fusion = FiLMFusion(feature_dim)
        elif fusion_strategy == "residual":
            self.fusion = ResidualAddFusion(feature_dim)
        elif fusion_strategy == "none" or fusion_strategy is None:
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # 3. Output Projection to MIDI-GPT embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 2 * model_dim),
            nn.GELU(),
            nn.Linear(2 * model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        bar_frames: torch.Tensor,
        track_frames: torch.Tensor | None = None,
        bar_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Translate audio frames into continuous soft bar token embeddings.

        Args:
            bar_frames: Tensor of shape ``(B, T, D_trans)`` representing audio frames of a bar.
            track_frames: Optional tensor of shape ``(B, T_track, D_trans)`` representing the whole stem.
            bar_mask: Optional boolean mask ``(B, T)`` indicating valid frame positions.

        Returns:
            Soft bar token embeddings of shape ``(B, num_tokens_per_bar, D_model)``.
        """
        # Step 1: Pool bar frames -> (B, K, D_trans)
        h_bar = self.pooler(bar_frames, mask=bar_mask)

        # Step 2: Hierarchical Fusion
        if self.fusion is not None and track_frames is not None:
            # Pool track frames globally -> (B, D_trans)
            h_track = track_frames.mean(dim=1)
            h_track = self.track_pooler(h_track)
            h_fused = self.fusion(h_bar, h_track)
        else:
            h_fused = h_bar

        # Step 3: Project to MIDI-GPT hidden dimension -> (B, K, D_model)
        e_bar = self.proj(h_fused)
        return e_bar
