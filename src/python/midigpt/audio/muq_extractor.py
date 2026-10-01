"""Real MuQ audio feature extractor (arXiv:2501.01108, Tencent AI Lab).

Wraps the actual pretrained ``OpenMuQ/MuQ-large-msd-iter`` checkpoint via the
official ``muq`` PyPI package's own ``MuQ`` class (``pip install muq``) --
not a hand-vendored reimplementation, and NOT ``transformers.AutoModel``:
``muq.MuQ`` is a plain ``nn.Module`` that mixes in
``huggingface_hub.PyTorchModelHubMixin`` directly (see
``github.com/tencent-ailab/MuQ`` at ``src/muq/muq/muq.py``), so loading goes
through ``MuQ.from_pretrained(...)``, which natively accepts either a
HuggingFace Hub repo ID or a local checkpoint directory (it does
``os.path.isdir(model_id)`` internally and reads ``model.safetensors``
straight off disk when it is one) -- there is no separate local-path
branch to write here.

MuQ is a 12-layer Conformer (``transformers.models.wav2vec2_conformer``'s
real ``Wav2Vec2ConformerEncoder``, not a reimplementation of it) over a
128-bin log-mel frontend, operating on raw 24kHz mono waveforms it strictly
requires (hardcoded in its own ``MelSTFT`` frontend, not configurable) and
producing 25Hz frame representations (``encoder_dim=1024``). Its own
``MuQModel.encoder()`` calls the transformers Conformer encoder directly
with ``output_hidden_states=True`` and reads back ``out["hidden_states"]``/
``out["last_hidden_state"]`` -- this only works on transformers versions
where ``Wav2Vec2ConformerEncoder.forward`` still collects hidden states
itself (verified present at 4.53.2, verified gone by 4.54.0, where that
collection moved into a wrapping decorator on the outer ``Wav2Vec2ConformerModel``
that MuQ's code never calls); see the ``muq`` extra's pin in pyproject.toml.

The real encoder's ``hidden_states`` tuple has ``encoder_depth + 1 = 13``
entries: index 0 is the pre-Conformer (post-conv-subsampling) embedding,
and index ``k`` for ``k in [1, 12]`` is Conformer block ``k``'s output --
index 12 has the encoder's final ``layer_norm`` applied and is exactly
``last_hidden_state`` (confirmed against transformers' own
``modeling_wav2vec2_conformer.py`` source, not assumed). So ``layer=k``
below indexes ``hidden_states[k]`` directly, unlike MERT-v2's own
(differently-indexed) ``modeling_mert2.py`` convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor

_REPO_ID = "OpenMuQ/MuQ-large-msd-iter"
_SAMPLE_RATE = 24000  # hardcoded in muq's own MelSTFT frontend, not a config field


class MuQExtractor(BaseAudioExtractor):
    """Extracts MuQ Conformer hidden states.

    Loads the real ``muq.MuQ`` model (via ``MuQ.from_pretrained``, pointed at
    either a HuggingFace Hub repo ID or a local checkpoint directory -- e.g.
    the pre-downloaded ``/scratch/triana24/MIDI-GPT/hf_cache/muq-large-msd-iter``)
    and returns hidden states at ``layer`` (default: the last of the 12
    Conformer blocks, i.e. ``output.last_hidden_state``, identical to what
    the model's own consumers read).
    """

    def __init__(
        self,
        weights_path: str | None = None,
        layer: int = -1,
        device: str | None = None,
        dtype: str | None = None,
    ):
        try:
            from muq import MuQ
        except ImportError as e:
            raise ImportError("MuQExtractor requires the 'muq' package: pip install midigpt[muq]") from e

        model = MuQ.from_pretrained(weights_path or _REPO_ID)

        num_layers = model.config.encoder_depth
        resolved_layer = num_layers if layer in (-1, None) else layer
        if not (0 < resolved_layer <= num_layers):
            raise ValueError(f"layer must be in [1, {num_layers}] (or -1 for last), got {layer}")

        hop_length = int(round(_SAMPLE_RATE / model.config.label_rate))
        super().__init__(feature_dim=model.config.encoder_dim, sample_rate=_SAMPLE_RATE, hop_length=hop_length)
        self.model = model
        self.layer = resolved_layer
        self.num_layers = num_layers
        # preprocessing() drops the last raw mel frame ([..., :-1]) before the
        # conv subsampler ever sees it, so a raw waveform under two mel hops
        # long can trim down to zero frames -- pad up to that floor first.
        self.minimum_input_samples = 2 * model.config.hop_length

        if device is not None:
            self.model = self.model.to(device)
        if dtype is not None:
            # MuQ's own README recommends fp32 to avoid NaNs; only cast if
            # the caller explicitly asks for something else.
            self.model = self.model.to(getattr(torch, dtype))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract Conformer hidden states at ``self.layer``.

        Args:
            audio: (B, L) raw waveform at self.sample_rate (24kHz).

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

        output = self.model(audio, output_hidden_states=True)

        if self.layer == self.num_layers:
            return output.last_hidden_state
        return output.hidden_states[self.layer]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        layer: int = -1,
        device: str | None = None,
    ) -> "MuQExtractor":
        """Load a MuQExtractor from a local checkpoint directory or HuggingFace Hub repo ID.

        Examples::
            extractor = MuQExtractor.from_pretrained(
                "/scratch/triana24/MIDI-GPT/hf_cache/muq-large-msd-iter"
            )
        """
        return cls(weights_path=model_name_or_path, layer=layer, device=device)
