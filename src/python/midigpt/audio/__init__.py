from midigpt.audio.base import BaseAudioExtractor
from midigpt.audio.dac_extractor import DACExtractor
from midigpt.audio.mert_extractor import MERTExtractor
from midigpt.audio.muq_extractor import MuQExtractor
from midigpt.audio.music2latent_extractor import Music2LatentExtractor
from midigpt.audio.muscriptor_extractor import MuScriptorExtractor
from midigpt.audio.same_extractor import SAMEExtractor
from midigpt.audio.sheetsage2_extractor import SheetSage2Extractor
from midigpt.audio.generic_proxy_extractor import GenericProxyAudioExtractor
from midigpt.audio.render_cache import AudioRenderCache
from midigpt.audio.synth import (
    FastAudioSynthesizer,
    SoundFontSynthesizer,
    get_bar_intervals,
    safe_pitch_shift,
)
from midigpt.audio.yourmt3_extractor import YourMT3Extractor
from midigpt.audio.yourmt3_real_extractor import YourMT3PlusExtractor

# MuScriptorExtractor's feature_dim is checkpoint-determined, not a free
# architecture choice -- this only lets a caller's existing --feature-dim
# flag (e.g. test_audio_midi_pipeline.py's hardcoded 768) keep selecting a
# self-consistent size without every caller having to learn --model-size too.
_MUSCRIPTOR_DIM_TO_SIZE = {768: "small", 1024: "medium", 1536: "large"}


def load_audio_extractor(
    extractor_type: str = "yourmt3",
    weights_path: str | None = None,
    feature_dim: int = 768,
    device: str | None = None,
    model_size: str | None = None,
) -> BaseAudioExtractor:
    """Factory helper to instantiate an audio transcription extractor via CLI arguments.

    Args:
        extractor_type: 'yourmt3' (or 'yourmt3+'), 'muscriptor', 'mert'
            (aliases: 'mert2', 'yue2' -- MERT-v2 is YuE2's frame-level audio
            backbone), 'sheetsage2' (alias: 'sheetsage' -- the same MERT-v2
            backbone with a fine-tuned LoRA adapter merged in), 'dac', 'muq',
            'music2latent', or 'same' --
            unlike the others (all transcription-style encoders whose
            frame-level output is a transformer's hidden state), DAC (Descript
            Audio Codec) is a neural audio codec: this returns its
            pre-quantization continuous latent (``projected_latents``), an
            audio embedding/latent-space representation rather than a
            transcription one. Same audio-embedding/latent-space family for
            'muq' (arXiv:2501.01108, Tencent AI Lab): a 12-layer Conformer
            trained via self-supervised Mel-RVQ pretraining (not a
            transcription objective) -- architecturally close to 'mert'
            (also a Conformer over 24kHz audio at 25Hz), but loaded via the
            ``muq`` PyPI package's own ``MuQ.from_pretrained``, not
            ``transformers.AutoModel``. Also in this family: 'music2latent'
            (arXiv:2408.06500, Sony CSL / QMUL) -- not a transformer at all,
            but a single consistency-model autoencoder; returns its 64-channel
            encoder bottleneck latent (``d_lat=64``) at ~11Hz, loaded via the
            ``music2latent`` PyPI package's own ``EncoderDecoder``, whose
            weights auto-download on first use (no manual HF cache step,
            unlike every other extractor here). Also in this family: 'same'
            (arXiv:2605.18613, Stability AI) -- SAME is a genuine
            encoder-bottleneck-decoder autoencoder (the Stable Audio 3 family's
            audio codec, not encoder-only like the others), loaded via the
            ``stable_audio_3`` package's ``AudioAutoencoder``; returns its
            256-dim post-bottleneck latent at ~10.77Hz. Stereo-native
            (in_channels=2) unlike every other extractor here (all mono) --
            this project's mono-only synth pipeline is duplicated to both
            channels internally, matching ``stable_audio_3``'s own documented
            behavior for mono input (see same_extractor.py's module docstring).
        weights_path: Optional path to a local checkpoint (.pt, .safetensors, .ckpt),
            a HuggingFace Hub repository ID, or (for 'muscriptor') a local
            model.safetensors path / hf:// URL / size keyword. For 'yourmt3'/
            'yourmt3+' this routes to the real vendored YourMT3PlusExtractor
            (see yourmt3_real_extractor.py) -- its architecture (and 512-dim
            feature output) is fixed by the checkpoint, `feature_dim` is
            ignored in that case. Without a weights_path, 'yourmt3'/'yourmt3+'
            fall back to the shape-compatible YourMT3Extractor proxy (fast,
            randomly initialized, configurable feature_dim) for smoke tests.
            For 'mert', this is a local checkpoint directory (e.g.
            ``/scratch/triana24/MIDI-GPT/hf_cache/mert-v2-fullsong``) or a
            HuggingFace Hub repo ID; without it, resolves via `model_size` to
            the default 'm-a-p/MERT-v2-FullSong' / 'm-a-p/MERT-v2-30s' repo ID
            (requires network access, and 'transformers' trust_remote_code).
            For 'sheetsage2', this is a local checkpoint directory (e.g.
            ``/scratch/triana24/MIDI-GPT/hf_cache/sheetsage2``) or a
            HuggingFace Hub repo ID; without it, defaults to 'm-a-p/SheetSage2'
            (requires network access).
            For 'dac', this is a local checkpoint directory (e.g.
            ``/scratch/triana24/MIDI-GPT/hf_cache/dac_44khz``) or a
            HuggingFace Hub repo ID; without it, resolves via `model_size` to
            the default 'descript/dac_44khz' repo ID (requires network access;
            no trust_remote_code, 'dac' is a first-party transformers model).
            For 'muq', this is a local checkpoint directory (e.g.
            ``/scratch/triana24/MIDI-GPT/hf_cache/muq-large-msd-iter``) or a
            HuggingFace Hub repo ID -- ``muq.MuQ.from_pretrained`` accepts
            either natively; without it, defaults to
            'OpenMuQ/MuQ-large-msd-iter' (requires network access, and the
            'muq' package).
            For 'music2latent', this is a local ``music2latent.pt`` checkpoint
            path, passed through as ``EncoderDecoder(load_path_inference=...)``;
            without it, the ``music2latent`` package auto-downloads its own
            checkpoint from 'SonyCSLParis/music2latent' on first use (requires
            network access, and the 'music2latent' package) -- note this
            auto-download attempt happens unconditionally even when a
            weights_path IS given, only skipped if the package's own default
            cache path already has the checkpoint (see Music2LatentExtractor's
            docstring).
            For 'same', this is a local checkpoint directory containing
            ``model_config.json``/``model.safetensors`` (e.g.
            ``/scratch/triana24/MIDI-GPT/hf_cache/same-l``) or a HuggingFace
            Hub repo ID (e.g. ``stabilityai/SAME-L``); without it, resolves
            via `model_size` through ``stable_audio_3``'s own
            ``AutoencoderModel.from_pretrained`` keyword lookup (requires
            network access and an HF_TOKEN with the gated Stability AI
            Community License accepted on the repo).
        feature_dim: Feature embedding dimension (default 768). For 'muscriptor'
            this is only used to infer model_size when model_size is unset
            (768/1024/1536 -> small/medium/large); the extractor's actual
            feature_dim always comes from the loaded checkpoint's config.
            Ignored by real-checkpoint YourMT3+ loads (see above); used as-is
            by the no-weights YourMT3Extractor proxy. Ignored by 'mert' (fixed
            at 1024-dim by the checkpoint), 'sheetsage2' (fixed at 512-dim,
            the checkpoint's cross-attended encoder projection width -- not
            the 1024-dim backbone it's projected down from), 'dac' (fixed
            at 72-dim for 'dac_44khz' -- ``n_codebooks * codebook_dim`` from
            the checkpoint's config, not its unrelated ``hidden_size``), and
            'muq' (fixed at 1024-dim, ``encoder_dim`` from the checkpoint's
            config), 'music2latent' (fixed at 64-dim, the checkpoint's
            ``bottleneck_channels`` / ``d_lat``), and 'same' (fixed at
            256-dim, the checkpoint's ``latent_dim`` -- same for both
            same-l/same-s).
        device: Device to load the extractor onto.
        model_size: 'muscriptor' -- 'small', 'medium' (default), or 'large'.
            'mert' -- 'fullsong' (default) or '30s', selecting between the
            MERT-v2-FullSong (360s context) and MERT-v2-30s (30s context)
            checkpoints. 'dac' -- '44khz' (default), '24khz', or '16khz',
            selecting the sample-rate variant (also sets `sample_rate`/
            `hop_length` accordingly, all read from that checkpoint's config).
            Unused by 'muq' (only one released checkpoint size, ~310M params)
            and by 'music2latent' (only one released checkpoint). 'same' --
            'same-l' (default, 852M params) or 'same-s' (108M params,
            CPU-deployable); 'l'/'large' and 's'/'small' are accepted aliases.
            Unused by 'sheetsage2'; construct SheetSage2Extractor
            directly (passing `base_model_path`) if a `weights_format="adapter"`
            checkpoint's frozen MERT-v2-FullSong parent must be resolved from a
            local directory rather than the HuggingFace Hub.

    Returns:
        Configured and loaded BaseAudioExtractor instance.
    """
    ext_lower = extractor_type.lower().replace("+", "").strip()

    if ext_lower in ("yourmt3", "mt3"):
        if weights_path:
            if feature_dim != 768:
                import warnings

                warnings.warn(
                    f"feature_dim={feature_dim} was requested but is ignored: real YourMT3+ "
                    "checkpoints fix the encoder output at 512-dim.",
                    stacklevel=2,
                )
            extractor = YourMT3PlusExtractor.from_pretrained(weights_path, device=device)
        else:
            extractor = YourMT3Extractor(feature_dim=feature_dim)
    elif ext_lower in ("muscriptor", "muscript"):
        size = model_size or _MUSCRIPTOR_DIM_TO_SIZE.get(feature_dim, "medium")
        extractor = MuScriptorExtractor(
            model_size=size, weights_path=weights_path, device=device
        )
    elif ext_lower in ("mert", "mert2", "yue2"):
        variant = model_size or "fullsong"
        extractor = MERTExtractor(variant=variant, weights_path=weights_path, device=device)
    elif ext_lower in ("sheetsage2", "sheetsage"):
        extractor = SheetSage2Extractor(weights_path=weights_path, device=device)
    elif ext_lower == "dac":
        variant = model_size or "44khz"
        extractor = DACExtractor(variant=variant, weights_path=weights_path, device=device)
    elif ext_lower == "muq":
        extractor = MuQExtractor(weights_path=weights_path, device=device)
    elif ext_lower == "music2latent":
        extractor = Music2LatentExtractor(weights_path=weights_path, device=device)
    elif ext_lower == "same":
        variant = model_size or "same-l"
        extractor = SAMEExtractor(variant=variant, weights_path=weights_path, device=device)
    else:
        raise ValueError(
            f"Unknown audio extractor type: {extractor_type!r}. Choose 'yourmt3', 'muscriptor', "
            "'mert', 'sheetsage2', 'dac', 'muq', 'music2latent', or 'same'."
        )

    if device is not None:
        extractor = extractor.to(device)

    return extractor


__all__ = [
    "BaseAudioExtractor",
    "YourMT3Extractor",
    "YourMT3PlusExtractor",
    "MuScriptorExtractor",
    "MERTExtractor",
    "SheetSage2Extractor",
    "DACExtractor",
    "MuQExtractor",
    "Music2LatentExtractor",
    "SAMEExtractor",
    "GenericProxyAudioExtractor",
    "FastAudioSynthesizer",
    "SoundFontSynthesizer",
    "AudioRenderCache",
    "get_bar_intervals",
    "safe_pitch_shift",
    "load_audio_extractor",
]
