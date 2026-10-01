"""YourMT3+ continuous hidden-state extractor -- real pretrained architecture.

Wraps the vendored plain-T5 encoder variant of YourMT3+ (see
``midigpt/audio/vendor/yourmt3/``) that the
``notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72`` checkpoint ("YMT3+"
in the upstream HuggingFace Space demo) was trained with. Extraction point is
the T5 encoder's ``last_hidden_state`` -- no decoder, no token decoding, this
is a frame-level feature extractor, not a transcriber.

Config values below (``AUDIO_CFG``, ``ENCODER_CONFIG``, ``T5_BASE_CFG``) are
the exact defaults the checkpoint was trained with; see
``vendor/yourmt3/NOTICE.md`` for how each was derived.
"""

from __future__ import annotations

import math
import pickle
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from midigpt.audio.base import BaseAudioExtractor

# transformers/einops are only needed for this extractor (optional [yourmt3]
# extra, see pyproject.toml) -- kept out of module-level imports so plain
# `import midigpt.audio` doesn't force the dependency on callers who only
# want e.g. the MuScriptor or proxy extractors. The vendor modules pull them
# in transitively, so the deferral happens here, at first use.

AUDIO_CFG = {
    "codec": "melspec",
    "audio_backend": "torchaudio",
    "sample_rate": 16000,
    "hop_length": 128,
    "n_fft": 2048,
    "n_mels": 512,
    "f_min": 50.0,
    "f_max": 8000.0,
    "input_frames": 32767,  # ~2.048s @ 16kHz; fixed chunk size, baked into the encoder's positional encoding
}

ENCODER_CONFIG = {
    "d_model": 512,
    "num_heads": 6,
    "num_layers": 8,
    "dropout_rate": 0.05,
    "position_encoding_type": "sinusoidal",
    "ff_widening_factor": 2,
    "ff_layer_type": "t5_gmlp",
    # max(feat_length=256, event_length=1024) + max_task_token_length(mt3_full_plus)=0 + 10
    "num_max_positions": 1034,
    # Irrelevant to the encoder (T5StackYMT3 takes inputs_embeds, has no token embedding
    # layer) -- left at T5Config's default, doesn't affect any weight shape we load.
}

T5_BASE_CFG = {
    "architectures": ["T5ForConditionalGeneration"],
    "d_ff": 1024,
    "d_kv": 64,
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "use_cache": True,
}

_STATE_DICT_PREFIXES = ("spectrogram.", "encoder.")


class _ForgivingUnpickler(pickle.Unpickler):
    """Stand in a harmless dummy object for any class this repo doesn't have.

    The real YourMT3+ Lightning checkpoint's non-tensor metadata (optimizer
    state, hyperparameters, callback state -- everything except
    ``state_dict``) can reference classes from the original training
    codebase (e.g. a custom ``utils`` module) that we don't vendor and never
    read. Rather than requiring the whole unrelated training codebase just to
    unpickle metadata we discard, resolve unknown classes to an inert stub so
    loading succeeds; the tensors in ``state_dict`` are always plain
    ``torch.Tensor``s and never go through this path.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {"__init__": lambda self, *a, **kw: None, "__setstate__": lambda self, state: None})


class _ForgivingPickleModule:
    Unpickler = _ForgivingUnpickler
    Pickler = pickle.Pickler
    UnpicklingError = pickle.UnpicklingError
    load = pickle.load
    dump = pickle.dump


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    """Load a raw state_dict out of a .ckpt (PyTorch Lightning) or .safetensors file."""
    if checkpoint_path.endswith(".safetensors"):
        import safetensors.torch

        return safetensors.torch.load_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, pickle_module=_ForgivingPickleModule
    )
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def key_match_report(module: nn.Module, state_dict: dict) -> dict:
    """Match a raw YourMT3 Lightning checkpoint state_dict against `module`.

    Restricted to the ``spectrogram.``/``encoder.`` key prefixes -- the only
    submodules this vendored extractor implements (no pre_encoder/decoder/
    embed_tokens/lm_head/pitchshift, all present in the full upstream
    LightningModule but irrelevant to encoder-only feature extraction).

    Returns a dict with matched/missing/unexpected/shape_mismatched keys and
    counts, so a caller can both load the matched weights and print exact
    diagnostics (see scripts/verify_yourmt3_extractor.py).
    """
    own = module.state_dict()
    relevant_ckpt = {
        k: v for k, v in state_dict.items() if k.startswith(_STATE_DICT_PREFIXES) and "pitchshift" not in k
    }

    matched: dict[str, torch.Tensor] = {}
    shape_mismatched: list[tuple[str, tuple, tuple]] = []
    for k, v in relevant_ckpt.items():
        if k not in own:
            continue
        if tuple(own[k].shape) == tuple(v.shape):
            matched[k] = v
        else:
            shape_mismatched.append((k, tuple(own[k].shape), tuple(v.shape)))

    missing = sorted(set(own) - set(matched))
    unexpected = sorted(set(relevant_ckpt) - set(matched) - {k for k, *_ in shape_mismatched})

    return {
        "matched": matched,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatched": shape_mismatched,
        "n_own_keys": len(own),
        "n_relevant_checkpoint_keys": len(relevant_ckpt),
    }


class YourMT3PlusExtractor(BaseAudioExtractor):
    """YourMT3+ (plain-T5 encoder variant) audio feature extractor.

    Vendored real architecture matching the
    ``notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72`` checkpoint:
    log-mel spectrogram frontend -> 8-layer T5 encoder (d_model=512,
    sinusoidal absolute positional encoding) -> ``last_hidden_state``.

    Input audio is chunked into non-overlapping ``~2.048s`` (32767-sample)
    windows -- the fixed size baked into this checkpoint's positional
    encoding -- run through the encoder independently (mirroring how
    upstream always used this model), and the resulting per-chunk hidden
    states are concatenated along time. The trailing padding added to fill
    out the last chunk is cropped back out of the returned sequence.
    """

    CHUNK_SAMPLES = AUDIO_CFG["input_frames"]

    def __init__(self, checkpoint_path: str | None = None):
        from transformers import T5Config

        from midigpt.audio.vendor.yourmt3.spectrogram import get_spectrogram_layer_from_audio_cfg
        from midigpt.audio.vendor.yourmt3.t5mod import T5EncoderYMT3

        spectrogram, spec_output_shape = get_spectrogram_layer_from_audio_cfg(AUDIO_CFG)
        super().__init__(
            feature_dim=ENCODER_CONFIG["d_model"],
            sample_rate=AUDIO_CFG["sample_rate"],
            hop_length=AUDIO_CFG["hop_length"],
        )
        self.spectrogram = spectrogram
        self._frames_per_chunk = spec_output_shape[0]

        t5_config = T5Config(**T5_BASE_CFG)
        self.encoder = T5EncoderYMT3(dict(ENCODER_CONFIG), t5_config)

        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract frame-level hidden states.

        Args:
            audio: (B, L) waveform at sample_rate.

        Returns:
            (B, T, feature_dim) hidden states.
        """
        if audio.ndim != 2:
            raise ValueError(f"expected (B, L) waveform, got shape {tuple(audio.shape)}")

        batch_size, num_samples = audio.shape
        n_chunks = max(1, math.ceil(num_samples / self.CHUNK_SAMPLES))
        padded_len = n_chunks * self.CHUNK_SAMPLES
        if padded_len > num_samples:
            audio = F.pad(audio, (0, padded_len - num_samples))

        x = audio.reshape(batch_size * n_chunks, 1, self.CHUNK_SAMPLES)
        mel = self.spectrogram(x)  # (B*n_chunks, frames_per_chunk, n_mels)
        enc_hs = self.encoder(inputs_embeds=mel)["last_hidden_state"]  # (B*n_chunks, frames_per_chunk, d_model)
        enc_hs = enc_hs.reshape(batch_size, n_chunks * enc_hs.shape[1], enc_hs.shape[2])

        # Crop the trailing padding-derived frames back out, so the frame count stays
        # proportional to the real (unpadded) audio duration -- BaseAudioExtractor's
        # slice_bar_frames maps bar times to frame indices assuming that proportionality.
        if num_samples > 0:
            real_frames = min(enc_hs.shape[1], num_samples // self.hop_length + 1)
        else:
            real_frames = 1
        real_frames = max(real_frames, 1)
        return enc_hs[:, :real_frames, :]

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from a local file (.ckpt or .safetensors).

        Raises if literally nothing matches (silently loading near-random
        weights is the exact failure mode this class replaces); warns (but
        still loads) on a partial match.
        """
        state_dict = _load_checkpoint_state_dict(checkpoint_path)
        report = key_match_report(self, state_dict)

        if not report["matched"]:
            raise RuntimeError(
                f"No spectrogram.*/encoder.* keys matched between {checkpoint_path!r} and "
                f"YourMT3PlusExtractor's architecture ({report['n_relevant_checkpoint_keys']} "
                f"candidate checkpoint keys, {report['n_own_keys']} own keys). Refusing to "
                "silently load near-random weights -- run scripts/verify_yourmt3_extractor.py "
                "against this checkpoint for full diagnostics."
            )
        if report["missing"] or report["unexpected"] or report["shape_mismatched"]:
            warnings.warn(
                f"Partial YourMT3+ checkpoint load from {checkpoint_path!r}: "
                f"{len(report['matched'])} matched, {len(report['missing'])} missing, "
                f"{len(report['unexpected'])} unexpected, {len(report['shape_mismatched'])} "
                "shape-mismatched keys. Run scripts/verify_yourmt3_extractor.py for details.",
                stacklevel=2,
            )
        self.load_state_dict(report["matched"], strict=False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str | None = None,
    ) -> "YourMT3PlusExtractor":
        """Load a YourMT3PlusExtractor from a local checkpoint path or HuggingFace Hub repo ID.

        Examples::
            extractor = YourMT3PlusExtractor.from_pretrained(
                "/scratch/.../notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt"
            )
        """
        import os

        if os.path.exists(model_name_or_path):
            ckpt_path = model_name_or_path
        else:
            try:
                from huggingface_hub import hf_hub_download

                filename = "amt/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt"
                try:
                    ckpt_path = hf_hub_download(
                        repo_id=model_name_or_path, filename=filename, repo_type="space"
                    )
                except Exception:
                    ckpt_path = hf_hub_download(repo_id=model_name_or_path, filename="model.ckpt")
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint from {model_name_or_path}: {e}") from e

        instance = cls(checkpoint_path=ckpt_path)
        if device is not None:
            instance = instance.to(device)
        return instance
