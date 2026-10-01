"""Real MuScriptor decoder-only feature extractor.

Wraps the actual pretrained MuScriptor checkpoint (arXiv:2607.08168,
Kyutai x Mirelo, https://github.com/muscriptor/muscriptor, PyPI package
``muscriptor``) via that package's own ``LMModel`` -- not a hand-vendored
reimplementation.

MuScriptor has no separate encoder: it is a single causal decoder stack.
At inference the mel-spectrogram is linearly projected to the model's
latent dim and used, together with instrument/dataset class embeddings, as
a PREFIX prepended to the token stream (see ``muscriptor.models.lm.LMModel
.forward``): the real prepend order is
``[mel frames][dataset_name][instrument_group][tokens]`` (mel frames occupy
the *earliest* positions -- they are prepended *last* in that loop, which
puts them leftmost). Because attention is strictly causal, the hidden
states at the mel-frame positions are mathematically identical whether or
not any class conditioning or tokens are appended afterward: later
positions can never influence earlier ones. So a correct "encoder-only"
read of MuScriptor's audio features needs nothing but the mel-spectrogram
conditioner and a prefix-only forward pass through the transformer
stack -- no instrument/dataset conditioning and no token embedding are
required to get bit-identical results at those positions.
"""

from __future__ import annotations

import torch

from midigpt.audio.base import BaseAudioExtractor

_SAMPLE_RATE = 16000
_FRAME_RATE = 100
_HOP_LENGTH = _SAMPLE_RATE // _FRAME_RATE  # 160
_MODEL_SIZES = ("small", "medium", "large")
_DEFAULT_MODEL_SIZE = "medium"


class MuScriptorExtractor(BaseAudioExtractor):
    """Extracts MuScriptor's decoder hidden states at the mel-spectrogram
    prefix positions.

    Loads the real ``muscriptor`` package's ``LMModel`` (via
    ``muscriptor.TranscriptionModel.load_model``, which downloads/locates
    the checkpoint, builds the exact architecture from its ``config.json``,
    and does a *strict* ``load_state_dict`` -- any key mismatch raises
    immediately instead of silently dropping weights) and runs only its
    mel-spectrogram conditioner + transformer blocks, stopping after
    ``layer`` blocks (default: all of them, i.e. the same residual stream
    MuScriptor's own LM head reads from).

    Requires the ``muscriptor`` PyPI package (``pip install muscriptor``);
    imported lazily so it stays an optional dependency, matching this
    module's existing ``huggingface_hub``/``safetensors`` convention.
    """

    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL_SIZE,
        weights_path: str | None = None,
        layer: int = -1,
        device: str | None = None,
        dtype: str | None = None,
    ):
        if model_size not in _MODEL_SIZES:
            raise ValueError(f"model_size must be one of {_MODEL_SIZES}, got {model_size!r}")

        try:
            from muscriptor import TranscriptionModel
        except ImportError as e:
            raise ImportError(
                "MuScriptorExtractor requires the 'muscriptor' package: "
                "pip install muscriptor"
            ) from e

        transcription_model = TranscriptionModel.load_model(
            weights_path=weights_path or model_size, device=device, dtype=dtype
        )
        # TranscriptionModel doesn't expose its LMModel publicly; ._model is the
        # only handle onto it. Reaching past the public API here (rather than
        # reimplementing _build_model/_resolve_config ourselves) keeps the
        # download/config-resolution/strict-load logic single-sourced from the
        # muscriptor package instead of duplicated and liable to drift from it.
        lm = transcription_model._model
        num_layers = len(lm.transformer.layers)
        resolved_layer = num_layers if layer in (-1, None) else layer
        if not (0 < resolved_layer <= num_layers):
            raise ValueError(f"layer must be in [1, {num_layers}] (or -1 for last), got {layer}")

        super().__init__(feature_dim=lm.dim, sample_rate=_SAMPLE_RATE, hop_length=_HOP_LENGTH)
        self.lm = lm
        self.model_size = model_size
        self.layer = resolved_layer
        self.num_layers = num_layers

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract decoder hidden states at the mel-spectrogram prefix positions.

        Args:
            audio: (B, L) waveform at self.sample_rate (16kHz).

        Returns:
            (B, T, feature_dim) residual-stream activations after ``self.layer``
            transformer blocks, T = number of 100Hz mel frames covering L samples.
            When ``self.layer == self.num_layers`` (the default), this is exactly
            the input MuScriptor's own LM head would see at those positions.
        """
        from muscriptor.modules.conditioners import WavCondition
        from muscriptor.modules.transformer import create_sin_embedding

        B, L = audio.shape
        device = self.lm.emb.weight.device
        wav = audio.to(device).unsqueeze(1)  # (B, 1, L)
        length = torch.full((B,), L, device=device, dtype=torch.long)

        mel_cond = self.lm.condition_provider.conditioners["self_wav"]

        with self.lm.autocast:
            embeds, _mask = mel_cond(
                WavCondition(wav=wav, length=length, sample_rate=[self.sample_rate] * B)
            )
            x = embeds.to(self.lm.emb.weight.dtype)

            T = x.shape[1]
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            pos_emb = create_sin_embedding(
                positions, x.shape[-1], max_period=self.lm.transformer.max_period, dtype=torch.float32
            ).to(x.dtype)
            x = x + pos_emb

            for block in self.lm.transformer.layers[: self.layer]:
                x = block(x, model_state=None)

            if self.layer == self.num_layers:
                x = self.lm.out_norm(x)

        return x
