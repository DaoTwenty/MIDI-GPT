"""Real SAME audio autoencoder feature extractor (arXiv:2605.18613, Stability AI).

Wraps the actual pretrained ``stabilityai/SAME-L`` / ``stabilityai/SAME-S``
checkpoints via the ``stable_audio_3`` package (the dedicated, first-party
companion repo to the Stable Audio 3 model family, MIT licensed,
https://github.com/Stability-AI/stable-audio-3) -- not a hand-vendored
reimplementation. That repo's own ``stable_audio_3.models.autoencoders`` module
defines ``SAMEEncoder``/``SAMEDecoder`` by name and is the canonical reference
implementation for these exact checkpoints, so there is no architecture to
keep in sync here.

Chose ``stable_audio_3`` over the older, much larger ``stable_audio_tools``
repo: it is purpose-built for this model family (small, tested, actively
maintained dependency surface -- ``tests/test_autoencoder.py`` exercises
``encode``/``decode`` directly against real checkpoints), whereas
``stable_audio_tools`` is a general Stable-Audio-1/2-era training framework
whose SAME support would be secondary/adapted rather than the reference
source.

SAME is a genuine encoder-bottleneck-decoder autoencoder (not encoder-only):
raw 44.1kHz audio -> ``PatchedPretransform`` (256-sample non-overlapping
patching) -> ``SAMEEncoder`` (a stack of ``TransformerResamplingBlock``s,
chunked-attention transformers with further striding) -> ``SoftNormBottleneck``,
reaching an overall ~4096x downsampling (~10.77Hz frame rate). We only need
the encoder half's continuous latent frames, but ``AudioAutoencoder`` doesn't
expose an encoder-only entry point separate from the full
pretransform+encoder+bottleneck pipeline, so ``encode_audio`` is called as-is
-- these stages are deterministic and lossless-in-architecture (unlike the
decoder, they add no sampling/noise at inference), so this is exactly the
"encoder" reading of the model, just under the autoencoder's own combined API.

SAME's encoder is stereo-native (``in_channels=2`` in both SAME-L/SAME-S
checkpoint configs). This project's own audio synthesis pipeline
(``midigpt/audio/synth.py`` -- ``FastAudioSynthesizer``/``SoundFontSynthesizer``)
only ever renders mono waveforms, so there is no genuine stereo signal for this
extractor to preserve; ``BaseAudioExtractor.extract_frames`` also always
downmixes to mono before calling ``forward``, matching every other extractor in
this codebase. Rather than hand-roll a mono->stereo upmix here, ``forward``
relies on the vendored ``AudioAutoencoder.preprocess_audio_list_for_encoder`` ->
``prepare_audio`` -> ``set_audio_channels`` codepath, which already duplicates
a single input channel to both stereo channels when ``target_channels=2``
(``stable_audio_3/inference/audio_utils.py::set_audio_channels``) -- this is
the library's own documented behavior for mono input, not a workaround we
invented, so it's reused as-is.
"""

from __future__ import annotations

from pathlib import Path

import torch

from midigpt.audio.base import BaseAudioExtractor

_VARIANTS = ("same-l", "same-s")
_DEFAULT_VARIANT = "same-l"
_SIZE_ALIASES = {"l": "same-l", "large": "same-l", "s": "same-s", "small": "same-s"}
_CONFIG_FILENAME = "model_config.json"
_CKPT_FILENAME = "model.safetensors"


def _resolve_checkpoint_files(weights_path: str) -> tuple[str, str]:
    """Resolve (config_path, ckpt_path) from a local checkpoint dir or bare HF Hub repo ID."""
    local_dir = Path(weights_path)
    if local_dir.is_dir():
        config_path = local_dir / _CONFIG_FILENAME
        ckpt_path = local_dir / _CKPT_FILENAME
        if not config_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"{weights_path} is missing {_CONFIG_FILENAME}/{_CKPT_FILENAME} "
                "(expected a flat HuggingFace snapshot directory, e.g. via "
                "`huggingface-cli download stabilityai/SAME-L --local-dir ...`)"
            )
        return str(config_path), str(ckpt_path)

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=weights_path, filename=_CONFIG_FILENAME)
    ckpt_path = hf_hub_download(repo_id=weights_path, filename=_CKPT_FILENAME)
    return config_path, ckpt_path


class SAMEExtractor(BaseAudioExtractor):
    """Extracts SAME's continuous latent frame sequence (post-bottleneck).

    Loads the real ``AudioAutoencoder`` (via ``stable_audio_3``'s own
    ``load_autoencoder``, which builds the exact architecture from the
    checkpoint's ``model_config.json`` and loads ``model.safetensors`` --
    without a ``weights_path``, falls back to ``AutoencoderModel.from_pretrained``,
    which resolves ``variant`` ('same-l'/'same-s') against the gated
    ``stabilityai/SAME-L``/``stabilityai/SAME-S`` Hub repos) and runs its full
    encode pipeline (patch pretransform -> TRB encoder -> soft-norm
    bottleneck), returning the resulting latent sequence.
    """

    def __init__(
        self,
        variant: str = _DEFAULT_VARIANT,
        weights_path: str | None = None,
        device: str | None = None,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 32,
    ):
        variant = _SIZE_ALIASES.get(variant, variant)
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")

        try:
            from stable_audio_3 import AutoencoderModel
            from stable_audio_3.loading_utils import load_autoencoder
        except ImportError as e:
            raise ImportError(
                "SAMEExtractor requires the 'stable_audio_3' package: pip install midigpt[same]"
            ) from e

        if weights_path is not None:
            config_path, ckpt_path = _resolve_checkpoint_files(weights_path)
            autoencoder = load_autoencoder(config_path, ckpt_path, device=device or "cpu")
            autoencoder.eval().requires_grad_(False)
        else:
            # Network path: resolves the gated stabilityai/SAME-L / SAME-S Hub repo
            # via huggingface_hub (requires HF_TOKEN with the license accepted).
            autoencoder = AutoencoderModel.from_pretrained(variant, device=device).autoencoder

        super().__init__(
            feature_dim=autoencoder.latent_dim,
            sample_rate=autoencoder.sample_rate,
            hop_length=int(autoencoder.downsampling_ratio),
        )
        self.autoencoder = autoencoder
        self.variant = variant
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.overlap = overlap

        if device is not None:
            self.autoencoder = self.autoencoder.to(device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode mono waveform through SAME's full encode pipeline.

        Args:
            audio: (B, L) waveform at self.sample_rate (44.1kHz). Duplicated to
                stereo internally (see module docstring) since SAME's encoder is
                stereo-native (in_channels=2) but this project only renders mono.

        Returns:
            (B, T, feature_dim) continuous latent frames, T = number of ~10.77Hz
            frames covering L samples (feature_dim = self.autoencoder.latent_dim,
            256 for both same-l and same-s).
        """
        if audio.ndim != 2:
            raise ValueError(f"expected (B, L) waveform, got shape {tuple(audio.shape)}")

        device = next(self.autoencoder.parameters()).device
        audio = audio.to(device).unsqueeze(1)  # (B, 1, L)

        preprocessed = self.autoencoder.preprocess_audio_list_for_encoder(
            list(audio), in_sr_list=self.sample_rate
        )
        latents = self.autoencoder.encode_audio(
            preprocessed, chunked=self.chunked, chunk_size=self.chunk_size, overlap=self.overlap
        )
        return latents.transpose(1, 2)  # (B, latent_dim, T) -> (B, T, latent_dim)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        variant: str = _DEFAULT_VARIANT,
        device: str | None = None,
    ) -> "SAMEExtractor":
        """Load a SAMEExtractor from a local checkpoint directory or HuggingFace Hub repo ID.

        Examples::
            extractor = SAMEExtractor.from_pretrained(
                "/scratch/triana24/MIDI-GPT/hf_cache/same-l"
            )
        """
        return cls(variant=variant, weights_path=model_name_or_path, device=device)
