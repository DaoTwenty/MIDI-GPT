"""MuScriptor continuous audio transcription feature extractor."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor
from midigpt.audio.yourmt3_extractor import SpectrogramFrontend


class MuScriptorExtractor(BaseAudioExtractor):
    """MuScriptor audio transcription feature extractor.

    Wraps MuScriptor's acoustic encoder module to extract continuous frame-level
    embeddings over audio stems.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        n_mels: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        sample_rate: int = 22050,
        hop_length: int = 256,
        checkpoint_path: str | None = None,
    ):
        super().__init__(feature_dim=feature_dim, sample_rate=sample_rate, hop_length=hop_length)
        self.frontend = SpectrogramFrontend(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_mels, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )

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
        """Extract frame-level hidden states from audio waveform."""
        # Frontend: (B, L) -> (B, T, n_mels)
        mel = self.frontend(audio)
        # Conv: (B, n_mels, T) -> (B, D, T) -> (B, T, D)
        x = self.input_conv(mel.transpose(1, 2)).transpose(1, 2)
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

        enc_state = {k.replace("encoder.", "").replace("model.", ""): v for k, v in state.items() if "encoder" in k or "input_conv" in k}
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
    ) -> "MuScriptorExtractor":
        """Load a MuScriptor extractor from a local checkpoint path or HuggingFace Hub repo ID."""
        import os

        if os.path.exists(model_name_or_path):
            ckpt_path = model_name_or_path
        else:
            try:
                from huggingface_hub import hf_hub_download
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
