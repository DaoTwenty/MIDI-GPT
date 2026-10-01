"""Real MERT-v2 audio feature extractor (arXiv:2609.33757, m-a-p / YuE2 project).

Wraps the actual pretrained ``m-a-p/MERT-v2-FullSong`` / ``m-a-p/MERT-v2-30s``
checkpoints via ``transformers.AutoModel.from_pretrained(..., trust_remote_code=True)``
-- not a hand-vendored reimplementation. The repo ships its own modeling code
(``modeling_mert2.py``, ``MERT2Model``, ``model_type="mert2"``) that
``trust_remote_code`` loads and runs as-is, so there is no architecture to
keep in sync here.

MERT-v2 is a bidirectional Conformer encoder (log-mel frontend -> 3-stage
ConvNeXt subsampler -> 24 rotary-attention Conformer blocks) operating on
raw, un-normalized 24kHz mono waveforms and producing 25Hz frame
representations (hidden_size=1024). It is also the frozen backbone behind
``m-a-p/SheetSage2``'s transcription head, but we want its own frame-level
hidden states directly, not SheetSage2's downstream output.

Unlike YourMT3+'s fixed-window sinusoidal positional encoding, MERT-v2 uses
rotary embeddings and imposes no fixed chunk size in its modeling code, so
no chunking is needed here -- the full waveform is run through in one pass.
``context_seconds`` (360s FullSong / 30s 30s-variant) is the length the
checkpoint was trained/evaluated at, not a hard input-length constraint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor

_VARIANTS = ("fullsong", "30s")
_DEFAULT_VARIANT = "fullsong"
_REPO_IDS = {
    "fullsong": "m-a-p/MERT-v2-FullSong",
    "30s": "m-a-p/MERT-v2-30s",
}


class MERTExtractor(BaseAudioExtractor):
    """Extracts MERT-v2 Conformer hidden states.

    Loads the real ``MERT2Model`` (via ``transformers.AutoModel.from_pretrained``
    with ``trust_remote_code=True``, pointed at either a HuggingFace Hub repo ID
    or a local checkpoint directory -- e.g. the pre-downloaded
    ``/scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong``) and returns hidden
    states at ``layer`` (default: the last of the 24 Conformer blocks, i.e.
    ``output.last_hidden_state``, identical to what the model's own consumers
    read).
    """

    def __init__(
        self,
        variant: str = _DEFAULT_VARIANT,
        weights_path: str | None = None,
        layer: int = -1,
        device: str | None = None,
        dtype: str | None = None,
    ):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("MERTExtractor requires the 'transformers' package: pip install midigpt[mert]") from e

        # transformers renamed from_pretrained's dtype kwarg (torch_dtype -> dtype)
        # around the 4.x/5.x boundary; only pass one when the caller actually
        # asked for a specific dtype, trying the current name first so a
        # fresh install (transformers>=4.53, unpinned upper bound -- see
        # pyproject.toml) doesn't print a deprecation warning on every load.
        dtype_kwargs = {"dtype": getattr(torch, dtype)} if dtype else {}

        try:
            model = AutoModel.from_pretrained(
                weights_path or _REPO_IDS[variant],
                trust_remote_code=True,
                attn_implementation="sdpa",
                **dtype_kwargs,
            )
        except TypeError:
            if "dtype" not in dtype_kwargs:
                raise
            model = AutoModel.from_pretrained(
                weights_path or _REPO_IDS[variant],
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=dtype_kwargs["dtype"],
            )

        num_layers = model.config.num_hidden_layers
        resolved_layer = num_layers if layer in (-1, None) else layer
        if not (0 < resolved_layer <= num_layers):
            raise ValueError(f"layer must be in [1, {num_layers}] (or -1 for last), got {layer}")

        hop_length = int(round(model.config.sampling_rate / model.config.frame_rate))
        super().__init__(feature_dim=model.config.hidden_size, sample_rate=model.config.sampling_rate, hop_length=hop_length)
        self.model = model
        self.variant = variant
        self.layer = resolved_layer
        self.num_layers = num_layers
        self.minimum_input_samples = model.config.minimum_input_samples

        if device is not None:
            self.model = self.model.to(device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract Conformer hidden states at ``self.layer``.

        Args:
            audio: (B, L) raw (un-normalized) waveform at self.sample_rate (24kHz).

        Returns:
            (B, T, feature_dim) hidden states, T = number of 25Hz frames covering
            L samples. When ``self.layer == self.num_layers`` (the default), this
            is exactly ``output.last_hidden_state``.
        """
        if audio.ndim != 2:
            raise ValueError(f"expected (B, L) waveform, got shape {tuple(audio.shape)}")

        device = next(self.model.parameters()).device
        audio = audio.to(device)
        if audio.shape[-1] < self.minimum_input_samples:
            audio = F.pad(audio, (0, self.minimum_input_samples - audio.shape[-1]))

        output = self.model(input_values=audio, output_hidden_states=True, return_dict=True)

        if self.layer == self.num_layers:
            return output.last_hidden_state
        # hidden_states[0] is block 1's output (no separate input-embedding entry).
        return output.hidden_states[self.layer - 1]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        variant: str = _DEFAULT_VARIANT,
        layer: int = -1,
        device: str | None = None,
    ) -> "MERTExtractor":
        """Load a MERTExtractor from a local checkpoint directory or HuggingFace Hub repo ID.

        Examples::
            extractor = MERTExtractor.from_pretrained(
                "/scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong"
            )
        """
        return cls(variant=variant, weights_path=model_name_or_path, layer=layer, device=device)
