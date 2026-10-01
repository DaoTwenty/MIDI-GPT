"""Real Music2Latent audio feature extractor (arXiv:2408.06500, Sony CSL / QMUL, ISMIR 2024).

Wraps the actual pretrained ``SonyCSLParis/music2latent`` consistency-autoencoder
checkpoint via the ``music2latent`` PyPI package's own ``EncoderDecoder.encode`` --
not a hand-vendored reimplementation. The package ships its own architecture
(``models.py``: a 2D-conv U-Net encoder over complex-valued real/imag STFT frames,
plus a consistency-model decoder we never call here) and downloads its single
233MB checkpoint (``music2latent.pt``) from the ``SonyCSLParis/music2latent``
HuggingFace Hub repo the first time ``EncoderDecoder()`` is constructed -- no
manual weights download or gating step needed, unlike every other extractor in
this codebase so far.

Music2Latent is a single end-to-end consistency autoencoder, not a transformer:
there is no notion of hidden layers/blocks to select between (contrast MERT's
`layer`), only one fixed encoder output. Its real hparams (hop=512,
freq_downsample_list=[1,0,0,0] -> downscaling_factor=2**3=8) fix the frame hop
at hop*downscaling_factor=4096 samples, i.e. ~10.77Hz at 44.1kHz ("~11Hz" in the
paper), and the bottleneck at bottleneck_channels=64 (`d_lat=64`, confirmed
verbatim in the paper).

``EncoderDecoder.encode()``'s leading dimension is documented as "audio_channels"
but is handled batch-agnostically throughout ``encode_audio_inference`` (each
entry along that axis is independently chunked-if-needed and reassembled), so
passing a genuine ``(B, L)`` batch of unrelated clips works exactly like a real
batch dimension -- just capped by ``max_batch_size_encode`` (package default: 1,
i.e. one entry through the model at a time) unless overridden, which is why
`forward()` below passes `max_batch_size=B` by default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor

_SAMPLE_RATE = 44100
_HOP_LENGTH = 4096  # hop(512) * downscaling_factor(8) -- fixed by the checkpoint's architecture
_FEATURE_DIM = 64  # d_lat / bottleneck_channels -- fixed by the checkpoint's architecture
# Smallest input that reliably survives encode_audio_inference's internal
# cropping (`cropped_length = (((L - 3*hop) // hop) // downscaling_factor) *
# hop * downscaling_factor + 3*hop`, hop=512, downscaling_factor=8): the
# formula needs L >= 3*hop + hop*downscaling_factor = 5632 to yield a single
# non-empty latent frame. 16384 (4 * hop_length) gives comfortable margin
# while still being under 0.4s at 44.1kHz.
_MINIMUM_INPUT_SAMPLES = 16384


class Music2LatentExtractor(BaseAudioExtractor):
    """Extracts Music2Latent's 64-channel consistency-autoencoder latents.

    Loads the real ``EncoderDecoder`` (``music2latent`` package), which downloads
    its own checkpoint from the ``SonyCSLParis/music2latent`` HuggingFace Hub repo
    on first construction if not already cached -- no manual download/gating step,
    unlike this codebase's other extractors. ``weights_path``, if given, is passed
    through as ``EncoderDecoder(load_path_inference=...)`` to load a pre-downloaded
    checkpoint from a specific local path (e.g. on a network-restricted compute
    node). Note this does NOT skip the package's own auto-download attempt --
    ``EncoderDecoder.__init__`` unconditionally calls ``download_model()`` first,
    which only no-ops if ``<site-packages>/music2latent/models/music2latent.pt``
    already exists, regardless of ``load_path_inference``. On a fully offline
    node, pre-place the checkpoint at that exact path rather than relying on
    ``weights_path`` alone.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        max_batch_size: int | None = None,
    ):
        try:
            from music2latent import EncoderDecoder
        except ImportError as e:
            raise ImportError(
                "Music2LatentExtractor requires the 'music2latent' package: pip install midigpt[music2latent]"
            ) from e

        super().__init__(feature_dim=_FEATURE_DIM, sample_rate=_SAMPLE_RATE, hop_length=_HOP_LENGTH)
        self.encdec = EncoderDecoder(load_path_inference=weights_path, device=device or "cpu")
        self.gen = self.encdec.gen  # register as a real submodule so .to()/.cuda()/.cpu() reach the real weights
        self.max_batch_size = max_batch_size
        self.minimum_input_samples = _MINIMUM_INPUT_SAMPLES

    def _apply(self, fn, recurse=True):
        # EncoderDecoder caches its own `.device` and re-`.to(self.device)`s
        # `self.gen` on every encode() call (see music2latent's
        # encode_audio_inference) -- without this override, an external
        # .to()/.cuda()/.cpu() on this module would move `self.gen`'s
        # parameters (registered above) but leave `self.encdec.device` stale,
        # so the very next forward() would silently move the weights right
        # back to the old device.
        result = super()._apply(fn, recurse=recurse)
        self.encdec.device = next(self.gen.parameters()).device
        return result

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract Music2Latent's consistency-autoencoder latents.

        Args:
            audio: (B, L) waveform at self.sample_rate (44.1kHz), mono.

        Returns:
            (B, T, feature_dim) latents, T ~= L / hop_length (4096 samples, ~11Hz).
        """
        if audio.ndim != 2:
            raise ValueError(f"expected (B, L) waveform, got shape {tuple(audio.shape)}")

        device = next(self.gen.parameters()).device
        audio = audio.to(device)
        if audio.shape[-1] < self.minimum_input_samples:
            audio = F.pad(audio, (0, self.minimum_input_samples - audio.shape[-1]))

        latent = self.encdec.encode(audio, max_batch_size=self.max_batch_size or audio.shape[0])  # (B, feature_dim, T)
        return latent.transpose(1, 2)  # (B, T, feature_dim)

    @classmethod
    def from_pretrained(cls, weights_path: str, device: str | None = None) -> "Music2LatentExtractor":
        """Load a Music2LatentExtractor from a pre-downloaded local checkpoint path.

        Examples::
            extractor = Music2LatentExtractor.from_pretrained(
                "/scratch/triana24/MIDI-GPT/hf_cache/music2latent/music2latent.pt"
            )
        """
        return cls(weights_path=weights_path, device=device)
