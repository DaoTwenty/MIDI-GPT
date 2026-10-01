"""Real SheetSage2 audio feature extractor (arXiv:2609.33757, m-a-p project).

Wraps the actual pretrained ``m-a-p/SheetSage2`` checkpoint via
``transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`` --
not a hand-vendored reimplementation. The repo ships its own modeling code
(``modeling_sheetsage2.py``, ``SheetSage2Model``, ``model_type="sheetsage2"``)
that ``trust_remote_code`` loads and runs as-is.

SheetSage2 is *not* a separate architecture from MERT-v2: it is the exact
``m-a-p/MERT-v2-FullSong`` backbone (frozen ``MERT2Model``, 24 Conformer
blocks, hidden_size=1024, 25Hz) with a rank-64/alpha-128 LoRA adapter fused
into every block's ``query_proj``/``key_proj``/``value_proj``/``out_proj``
(see ``modeling_sheetsage2.SheetSage2Model.merge_lora``), fine-tuned for
melody/chord/beat transcription, plus a small BART decoder that
autoregressively emits ABC/MIDI tokens. We only want the encoder side.

``SheetSage2Model.get_audio_features`` returns a per-layer softmax-weighted
mix of the LoRA-merged Conformer block outputs (``mixed_hidden_state``,
still 1024-dim, the backbone's own width) and then linearly projects that
down to ``config.hidden_size`` (512) as ``encoder_last_hidden_state``:

    memory = self.encoder_projection(mixed)
    ...
    output = SheetSage2EncoderOutput(encoder_last_hidden_state=memory, ...)

That 512-dim ``encoder_last_hidden_state`` -- not the 1024-dim
``mixed_hidden_state`` -- is what ``SheetSage2Model.forward``/``generate``
actually feeds the decoder's cross-attention as ``encoder_hidden_states``,
and it is exactly what the model's own ``export_embeddings=True`` tensor
dump (``tensors_sheetsage2.WindowTensorWriter.audio_features``) saves. So
D=512 here, projected down from the shared 1024-dim MERT2 width -- not
1024 as the raw backbone's own hidden_size might suggest.

``get_audio_features`` always zero-pads its input up to one full
``config.input_audio_length`` window (300s @ 24kHz = 7,200,000 samples)
before running the encoder, regardless of the caller's actual audio
length -- whole-song transcription chunks externally via
``pipeline_sheetsage2.Transcriber``'s sliding-window plan, a level this
extractor does not replicate. We only take one window per call (audio
longer than 300s raises, matching the model's own single-window
contract) and slice the returned frames back down to the count implied
by the real (unpadded) input length, so callers relying on
``BaseAudioExtractor``'s proportional bar/track frame slicing see a
frame sequence whose length actually corresponds to their audio's
duration instead of the padded window's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor

_REPO_ID = "m-a-p/SheetSage2"


class SheetSage2Extractor(BaseAudioExtractor):
    """Extracts SheetSage2's LoRA-merged, cross-attended encoder hidden states.

    Loads the real ``SheetSage2Model`` (via
    ``transformers.AutoModel.from_pretrained`` with ``trust_remote_code=True``,
    pointed at either a HuggingFace Hub repo ID or a local checkpoint directory
    -- e.g. the pre-downloaded ``/scratch/triana24/MIDI-GPT/hf_cache/sheetsage2``)
    and returns ``encoder_last_hidden_state`` from ``get_audio_features`` --
    the same tensor the model's own BART decoder cross-attends to.

    ``weights_path`` checkpoints saved with ``weights_format="adapter"`` (the
    default distributed format) hold only the LoRA adapter + decoder weights;
    loading them also downloads/locates the frozen MERT-v2-FullSong parent
    backbone separately (sha256-verified against the adapter checkpoint's
    pinned revision) and merges the adapter into it. ``base_model_path`` lets
    that parent be resolved from a local directory (e.g. the same
    ``mert-v2-fullsong`` directory ``MERTExtractor`` uses) instead of the
    HuggingFace Hub, avoiding a second network fetch of the 632M-param
    backbone. Checkpoints already saved with ``weights_format="merged"``
    (post ``SheetSage2Model.merge_lora``/``save_pretrained``) embed the merged
    encoder directly and ignore ``base_model_path``.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        base_model_path: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ):
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "SheetSage2Extractor requires the 'transformers' package: pip install midigpt[sheetsage2]"
            ) from e

        base_model_kwargs = {"base_model_path": base_model_path} if base_model_path else {}
        # Unlike MERT2Model (plain transformers.AutoModel/PreTrainedModel
        # machinery, which renamed torch_dtype -> dtype), SheetSage2Model hand-
        # writes its own from_pretrained override that only ever pops
        # "torch_dtype" (see modeling_sheetsage2.py) -- no dtype/torch_dtype
        # fallback dance needed here.
        dtype_kwargs = {"torch_dtype": getattr(torch, dtype)} if dtype else {}

        model = AutoModel.from_pretrained(
            weights_path or _REPO_ID,
            trust_remote_code=True,
            attn_implementation="sdpa",
            **base_model_kwargs,
            **dtype_kwargs,
        )

        backbone_config = model.config.backbone_config
        hop_length = int(round(backbone_config["sampling_rate"] / backbone_config["frame_rate"]))
        super().__init__(
            feature_dim=model.config.hidden_size,
            sample_rate=model.config.sampling_rate,
            hop_length=hop_length,
        )
        self.model = model
        self.minimum_input_samples = backbone_config["minimum_input_samples"]
        self.window_samples = round(model.config.input_audio_length * model.config.sampling_rate)

        if device is not None:
            self.model = self.model.to(device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract the LoRA-merged, cross-attended encoder hidden states.

        Args:
            audio: (B, L) raw (un-normalized) waveform at self.sample_rate (24kHz),
                L up to one ``input_audio_length`` window (300s = 7,200,000 samples).

        Returns:
            (B, T, feature_dim) hidden states, T = number of 25Hz frames covering
            L samples -- i.e. ``get_audio_features().encoder_last_hidden_state``
            sliced back down from the model's internally-padded full window to
            just the frames covering the real input length.
        """
        if audio.ndim != 2:
            raise ValueError(f"expected (B, L) waveform, got shape {tuple(audio.shape)}")
        if audio.shape[-1] > self.window_samples:
            raise ValueError(
                f"SheetSage2 processes one {self.window_samples}-sample window at a time "
                f"({self.model.config.input_audio_length:.0f}s @ {self.sample_rate}Hz); got "
                f"{audio.shape[-1]} samples. Chunk audio externally (see the vendored "
                "pipeline_sheetsage2.Transcriber's sliding-window plan) before calling this extractor."
            )

        device = next(self.model.parameters()).device
        audio = audio.to(device)
        if audio.shape[-1] < self.minimum_input_samples:
            audio = F.pad(audio, (0, self.minimum_input_samples - audio.shape[-1]))

        output = self.model.get_audio_features(audio, output_hidden_states=False, return_dict=True)
        memory = output.encoder_last_hidden_state

        valid_frames = -(-audio.shape[-1] // self.hop_length)  # ceil(L / hop_length)
        return memory[:, :valid_frames, :]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        base_model_path: str | None = None,
        device: str | None = None,
    ) -> "SheetSage2Extractor":
        """Load a SheetSage2Extractor from a local checkpoint directory or HuggingFace Hub repo ID.

        Examples::
            extractor = SheetSage2Extractor.from_pretrained(
                "/scratch/triana24/MIDI-GPT/hf_cache/sheetsage2",
                base_model_path="/scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong",
            )
        """
        return cls(weights_path=model_name_or_path, base_model_path=base_model_path, device=device)
