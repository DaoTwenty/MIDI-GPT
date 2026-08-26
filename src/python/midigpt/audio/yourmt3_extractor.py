"""YourMT3+ continuous hidden-state extractor."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor


class SpectrogramFrontend(nn.Module):
    """Log-mel spectrogram frontend matching MT3/YourMT3 input processing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 128,
        n_mels: int = 512,
        f_min: float = 20.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Register mel filterbank as buffer
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        # Mel basis
        import numpy as np

        # Construct mel filter bank
        weights = self._create_mel_filterbank(
            sample_rate, n_fft, n_mels, f_min, f_max
        )
        self.register_buffer("mel_basis", weights)

    def _create_mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float
    ) -> torch.Tensor:
        # Standard Slaney/HTK triangular mel filters
        def hz_to_mel(hz: float) -> float:
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(f_min)
        max_mel = hz_to_mel(f_max)
        mels = [min_mel + i * (max_mel - min_mel) / (n_mels + 1) for i in range(n_mels + 2)]
        freqs = [mel_to_hz(m) for m in mels]
        fft_freqs = [i * sr / n_fft for i in range(n_fft // 2 + 1)]

        weights = torch.zeros((n_mels, n_fft // 2 + 1), dtype=torch.float32)
        for i in range(n_mels):
            f_left = freqs[i]
            f_center = freqs[i + 1]
            f_right = freqs[i + 2]
            for j, f in enumerate(fft_freqs):
                if f_left <= f <= f_center and f_center > f_left:
                    weights[i, j] = (f - f_left) / (f_center - f_left)
                elif f_center <= f <= f_right and f_right > f_center:
                    weights[i, j] = (f_right - f) / (f_right - f_center)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram: (B, L) -> (B, T, n_mels)."""
        # x is (B, L)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # Power spectrogram
        magnitudes = stft.abs().pow(2.0)  # (B, n_fft // 2 + 1, T)
        # Mel spectrogram
        mel_spec = torch.matmul(self.mel_basis, magnitudes)  # (B, n_mels, T)
        # Log mel
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
        return log_mel.transpose(1, 2)  # (B, T, n_mels)


class YourMT3Extractor(BaseAudioExtractor):
    """YourMT3+ multi-track audio transcription feature extractor.

    Extracts continuous acoustic hidden states from the audio spectrogram
    using a multi-layer transformer encoder backbone.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        n_mels: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        sample_rate: int = 16000,
        hop_length: int = 128,
        checkpoint_path: str | None = None,
    ):
        super().__init__(feature_dim=feature_dim, sample_rate=sample_rate, hop_length=hop_length)
        self.frontend = SpectrogramFrontend(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.input_proj = nn.Linear(n_mels, feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=4 * feature_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(feature_dim)

        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract frame-level hidden states.

        Args:
            audio: (B, L) waveform at sample_rate.

        Returns:
            (B, T, feature_dim) hidden states.
        """
        # Frontend: (B, L) -> (B, T, n_mels)
        mel = self.frontend(audio)
        x = self.input_proj(mel)
        h = self.encoder(x)
        h = self.ln_out(h)
        return h

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from a local file (.pt, .safetensors, .ckpt)."""
        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch
            state = safetensors.torch.load_file(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in state:
            state = state["state_dict"]

        # Filter to encoder keys
        enc_state = {k.replace("encoder.", "").replace("model.", ""): v for k, v in state.items() if "encoder" in k or "frontend" in k}
        if enc_state:
            self.load_state_dict(enc_state, strict=False)
        else:
            self.load_state_dict(state, strict=False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        feature_dim: int = 768,
        device: str | None = None,
    ) -> "YourMT3Extractor":
        """Load a YourMT3+ extractor from a local checkpoint path or HuggingFace Hub repo ID.

        Examples::
            extractor = YourMT3Extractor.from_pretrained("mimbres/yourmt3-small")
            extractor = YourMT3Extractor.from_pretrained("checkpoints/yourmt3_full.safetensors")
        """
        import os
        from pathlib import Path

        if os.path.exists(model_name_or_path):
            ckpt_path = model_name_or_path
        else:
            # Try downloading from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                # Look for common model weights filenames
                for fname in ["model.safetensors", "pytorch_model.bin", "model.ckpt"]:
                    try:
                        ckpt_path = hf_hub_download(repo_id=model_name_or_path, filename=fname)
                        break
                    except Exception:
                        continue
                else:
                    raise FileNotFoundError(f"Could not resolve weights for {model_name_or_path}")
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint from {model_name_or_path}: {e}") from e

        instance = cls(feature_dim=feature_dim, checkpoint_path=ckpt_path)
        if device is not None:
            instance = instance.to(device)
        return instance
