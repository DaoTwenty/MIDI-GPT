# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""Vendored from amt/src/model/spectrogram.py (mimbres/YourMT3).

Unmodified except: the `config.config.audio_cfg` fallback default import was
dropped (callers always pass an explicit audio_cfg) and the unused
`nnAudio` backend's import guard is left as a lazy import, same as upstream.
See ../NOTICE.
"""

import importlib
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from midigpt.audio.vendor.yourmt3.ops import minmax_normalize, optional_compiler_disable


class Melspectrogram(nn.Module):
    def __init__(
        self,
        audio_backend: Literal["torchaudio", "nnaudio"] = "torchaudio",
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 128,
        f_min: int = 50,
        f_max: Optional[int] = 8000,
        n_mels: int = 512,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.audio_backend = audio_backend.lower()

        if self.audio_backend == "torchaudio":
            torchaudio = importlib.import_module("torchaudio")
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
            )
        elif self.audio_backend == "nnaudio":
            nnaudio = importlib.import_module("nnAudio.features")
            self.mel_stft_nnaudio = nnaudio.mel.MelSpectrogram(
                sr=sample_rate,
                win_length=n_fft,
                n_mels=n_mels,
                hop_length=hop_length,
                fmin=20,
                fmax=f_max,
            )
        else:
            raise NotImplementedError(audio_backend)
        self.eps = eps

    @optional_compiler_disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, T)
        """(B, 1, T) waveform -> (B, T, F) log-mel spectrogram."""
        if self.audio_backend == "torchaudio":
            x = self.mel_stft(x)  # (B, 1, F, T)
            x = rearrange(x, "b 1 f t -> b t f")
            x = minmax_normalize(torch.log(x + self.eps))
            return torch.nan_to_num(x)  # some torchaudio versions return nan on all-zero input
        elif self.audio_backend == "nnaudio":
            x = self.mel_stft_nnaudio(x)  # (B, F, T)
            x = rearrange(x, "b f t -> b t f")
            return minmax_normalize(torch.log(x + self.eps))


class Spectrogram(nn.Module):
    def __init__(
        self,
        audio_backend: Literal["torchaudio", "nnaudio"] = "torchaudio",
        n_fft: int = 2048,
        hop_length: int = 128,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.audio_backend = audio_backend.lower()

        if self.audio_backend == "torchaudio":
            torchaudio = importlib.import_module("torchaudio")
            self.stft = torchaudio.transforms.Spectrogram(
                n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, power=1.0
            )
        else:
            raise NotImplementedError(audio_backend)
        self.eps = eps

    @optional_compiler_disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, T)
        """(B, 1, T) waveform -> (B, T, F) log-magnitude spectrogram."""
        x = self.stft(x)[:, :, 1:, :]  # (B, 1, F, T), drop DC component
        x = rearrange(x, "b 1 f t -> b t f")
        x = minmax_normalize(torch.log(x + self.eps))
        return torch.nan_to_num(x)


def get_spectrogram_layer_from_audio_cfg(audio_cfg: Dict) -> Tuple[nn.Module, Tuple[int, ...]]:
    """Build the mel-/spectrogram frontend from an audio_cfg dict.

    Returns the layer and its output shape (T, F) (excluding batch dim),
    inferred by running one dummy forward pass -- mirrors upstream.
    """
    if audio_cfg["codec"] == "melspec":
        layer = Melspectrogram(**audio_cfg)
    elif audio_cfg["codec"] == "spec":
        layer = Spectrogram(**audio_cfg)
    else:
        raise NotImplementedError(audio_cfg["codec"])

    with torch.no_grad():
        output_shape = layer(torch.randn(1, 1, audio_cfg["input_frames"])).shape[1:]
    return layer, output_shape
