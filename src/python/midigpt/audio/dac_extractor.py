"""Real DAC (Descript Audio Codec) audio feature extractor (arXiv:2306.06546,
"High-Fidelity Audio Compression with Improved RVQGAN," Kumar et al. 2023,
Descript Inc.).

Wraps the actual pretrained ``descript/dac_44khz`` checkpoint via
``transformers``' own first-party ``DacModel`` (``from transformers import
DacModel`` -- no ``trust_remote_code``, this is a library-integrated model,
not vendored or remote code) -- not a hand-vendored reimplementation.

DAC is a neural audio codec, not a transformer: conv encoder/decoder +
residual vector quantization (RVQ). Unlike MERT-v2/YourMT3+'s transformer
``last_hidden_state``, DAC's useful continuous frame-level representation is
the *pre-quantization* continuous latent -- ``DacModel.encode()``'s
``projected_latents`` output (shape ``(batch, n_codebooks * codebook_dim,
time_steps)``), not ``audio_codes`` (discrete, post-argmax codebook indices)
and not ``quantized_representation`` (post-quantization -- still a continuous
tensor, but already lossy in effect since it's built by summing looked-up
codebook vectors). We call ``encode()`` only; the full ``forward()``
encode+quantize+decode round trip is never run.

Default checkpoint ``descript/dac_44khz``: ``hop_length=512``,
``downsampling_ratios=[2,4,8,8]`` (product 512), ``sampling_rate=44100`` ->
~86.1Hz frame rate; ``n_codebooks=9``, ``codebook_dim=8`` -> ``projected_latents``
is 72-dim. Note this is *not* ``config.hidden_size`` -- that attribute is the
post-quantizer-projection conv width the decoder consumes (1024 for
``dac_44khz``), an unrelated number. ``dac_24khz``/``dac_16khz`` sibling
checkpoints exist as lower-sample-rate ``variant=`` alternatives, same
architecture and encode-path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor

_VARIANTS = ("44khz", "24khz", "16khz")
_DEFAULT_VARIANT = "44khz"
_REPO_IDS = {
    "44khz": "descript/dac_44khz",
    "24khz": "descript/dac_24khz",
    "16khz": "descript/dac_16khz",
}


class DACExtractor(BaseAudioExtractor):
    """Extracts DAC's pre-quantization continuous latent (``projected_latents``).

    Loads the real ``DacModel`` (via ``transformers.DacModel.from_pretrained``,
    pointed at either a HuggingFace Hub repo ID or a local checkpoint directory
    -- e.g. the pre-downloaded ``/scratch/triana24/MIDI-GPT/hf_cache/dac_44khz``)
    and returns ``DacModel.encode(...).projected_latents``, transposed from
    DAC's own channels-first ``(B, D, T)`` layout to this codebase's ``(B, T, D)``
    convention.
    """

    def __init__(
        self,
        variant: str = _DEFAULT_VARIANT,
        weights_path: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")

        try:
            from transformers import DacModel
        except ImportError as e:
            raise ImportError("DACExtractor requires the 'transformers' package: pip install midigpt[dac]") from e

        # transformers renamed from_pretrained's dtype kwarg (torch_dtype -> dtype)
        # around the 4.x/5.x boundary; only pass one when the caller actually
        # asked for a specific dtype, trying the current name first so a
        # fresh install (transformers>=4.45, unpinned upper bound -- see
        # pyproject.toml) doesn't print a deprecation warning on every load.
        dtype_kwargs = {"dtype": getattr(torch, dtype)} if dtype else {}

        try:
            model = DacModel.from_pretrained(weights_path or _REPO_IDS[variant], **dtype_kwargs)
        except TypeError:
            if "dtype" not in dtype_kwargs:
                raise
            model = DacModel.from_pretrained(
                weights_path or _REPO_IDS[variant], torch_dtype=dtype_kwargs["dtype"]
            )

        # projected_latents' channel width is n_codebooks * codebook_dim, not
        # config.hidden_size (that's the post-quantizer-projection conv width
        # feeding the decoder -- an unrelated number, confirmed empirically
        # against a tiny same-downsampling_ratios DacModel; see dac_extractor
        # module docstring).
        feature_dim = model.config.n_codebooks * model.config.codebook_dim
        super().__init__(feature_dim=feature_dim, sample_rate=model.config.sampling_rate, hop_length=model.config.hop_length)
        self.model = model
        self.variant = variant
        # Empirically confirmed (against a tiny DacModel sharing the real
        # checkpoint's downsampling_ratios=[2,4,8,8], which is what fixes the
        # encoder's conv kernel/stride/padding shapes regardless of channel
        # widths): the strided encoder convs require at least one hop_length
        # worth of samples, independent of encoder_hidden_size. Below this,
        # DacEncoder's conv1d layers raise ("kernel size can't be greater than
        # actual input size").
        self.minimum_input_samples = model.config.hop_length

        if device is not None:
            self.model = self.model.to(device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract DAC's pre-quantization continuous latent.

        Args:
            audio: (B, L) raw waveform at self.sample_rate (44.1kHz).

        Returns:
            (B, T, feature_dim) continuous hidden states, T = floor(L / hop_length)
            -- exactly ``DacModel.encode(...).projected_latents``, transposed
            from DAC's own ``(B, feature_dim, T)`` layout to ``(B, T, feature_dim)``.
        """
        if audio.ndim != 2:
            raise ValueError(f"expected (B, L) waveform, got shape {tuple(audio.shape)}")

        device = next(self.model.parameters()).device
        audio = audio.to(device)
        if audio.shape[-1] < self.minimum_input_samples:
            audio = F.pad(audio, (0, self.minimum_input_samples - audio.shape[-1]))

        input_values = audio.unsqueeze(1)  # (B, 1, L) -- DacModel wants an explicit channel dim
        output = self.model.encode(input_values, return_dict=True)
        return output.projected_latents.transpose(1, 2)  # (B, D, T) -> (B, T, D)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        variant: str = _DEFAULT_VARIANT,
        device: str | None = None,
    ) -> "DACExtractor":
        """Load a DACExtractor from a local checkpoint directory or HuggingFace Hub repo ID.

        Examples::
            extractor = DACExtractor.from_pretrained(
                "/scratch/triana24/MIDI-GPT/hf_cache/dac_44khz"
            )
        """
        return cls(variant=variant, weights_path=model_name_or_path, device=device)
