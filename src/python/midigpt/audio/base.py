"""Base interface for audio transcription feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseAudioExtractor(nn.Module, ABC):
    """Abstract base class for audio transcription feature extractors.

    Subclasses wrap models such as YourMT3+ or MuScriptor to extract continuous
    frame-level representations before discrete token sampling.
    """

    def __init__(self, feature_dim: int, sample_rate: int = 16000, hop_length: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract frame-level continuous hidden states from raw audio waveform.

        Args:
            audio: Waveform tensor of shape ``(B, C, L)`` or ``(B, L)``,
                where L is samples at ``self.sample_rate``.

        Returns:
            Hidden states tensor of shape ``(B, T, feature_dim)``, where
            ``T`` is the number of temporal frames.
        """
        ...

    def extract_frames(self, audio: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Extract frame hidden states with automatic resampling if needed."""
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, L)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # (B, 1, L)

        sr = sample_rate or self.sample_rate
        if sr != self.sample_rate:
            # Resample if sample rates mismatch
            import torchaudio.functional as AF

            audio = AF.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

        # Average multichannel to mono if multi-channel
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        return self.forward(audio.squeeze(1))

    def extract_bar_frames(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        bar_intervals: list[tuple[float, float]],
    ) -> list[torch.Tensor]:
        """Extract sliced temporal frame tensors for each specified bar interval.

        Args:
            audio: Waveform tensor of shape ``(L,)`` or ``(1, L)``.
            sample_rate: Sample rate of the input waveform.
            bar_intervals: List of ``(start_time_sec, end_time_sec)`` tuples for each bar.

        Returns:
            List of tensors, each shaped ``(T_bar, feature_dim)`` corresponding
            to the frames within that bar's time window.
        """
        # Get frame sequence for the entire audio: (1, T_total, feature_dim)
        frames = self.extract_frames(audio, sample_rate=sample_rate).squeeze(0)  # (T_total, D)
        total_duration = audio.shape[-1] / (sample_rate or self.sample_rate)
        total_frames = frames.shape[0]

        if total_frames == 0 or total_duration <= 0:
            return [torch.zeros(1, self.feature_dim, device=frames.device) for _ in bar_intervals]

        bar_frames_list = []
        for start_sec, end_sec in bar_intervals:
            start_sec = max(0.0, start_sec)
            end_sec = min(total_duration, max(start_sec + 0.05, end_sec))

            # Map seconds to frame indices
            frame_start = int((start_sec / total_duration) * total_frames)
            frame_end = int((end_sec / total_duration) * total_frames)
            frame_end = max(frame_start + 1, min(total_frames, frame_end))

            bar_slice = frames[frame_start:frame_end]
            if bar_slice.shape[0] == 0:
                bar_slice = frames[frame_start : frame_start + 1]
            bar_frames_list.append(bar_slice)

        return bar_frames_list

    def extract_track_frames(
        self, audio: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Extract frame sequence for the entire audio stem / track."""
        return self.extract_frames(audio, sample_rate=sample_rate).squeeze(0)  # (T_track, D)
