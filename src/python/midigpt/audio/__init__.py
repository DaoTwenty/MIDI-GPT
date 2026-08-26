from midigpt.audio.base import BaseAudioExtractor
from midigpt.audio.muscriptor_extractor import MuScriptorExtractor
from midigpt.audio.synth import FastAudioSynthesizer, get_bar_intervals
from midigpt.audio.yourmt3_extractor import YourMT3Extractor


def load_audio_extractor(
    extractor_type: str = "yourmt3",
    weights_path: str | None = None,
    feature_dim: int = 768,
    device: str | None = None,
) -> BaseAudioExtractor:
    """Factory helper to instantiate an audio transcription extractor via CLI arguments.

    Args:
        extractor_type: 'yourmt3' (or 'yourmt3+') or 'muscriptor'.
        weights_path: Optional path to a local checkpoint (.pt, .safetensors, .ckpt)
            or a HuggingFace Hub repository ID (e.g. 'mimbres/yourmt3-small').
        feature_dim: Feature embedding dimension (default 768).
        device: Device to load the extractor onto.

    Returns:
        Configured and loaded BaseAudioExtractor instance.
    """
    ext_lower = extractor_type.lower().replace("+", "").strip()

    if ext_lower in ("yourmt3", "mt3"):
        if weights_path:
            extractor = YourMT3Extractor.from_pretrained(
                weights_path, feature_dim=feature_dim, device=device
            )
        else:
            extractor = YourMT3Extractor(feature_dim=feature_dim)
    elif ext_lower in ("muscriptor", "muscript"):
        if weights_path:
            extractor = MuScriptorExtractor.from_pretrained(
                weights_path, feature_dim=feature_dim, device=device
            )
        else:
            extractor = MuScriptorExtractor(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown audio extractor type: {extractor_type!r}. Choose 'yourmt3' or 'muscriptor'.")

    if device is not None:
        extractor = extractor.to(device)

    return extractor


__all__ = [
    "BaseAudioExtractor",
    "YourMT3Extractor",
    "MuScriptorExtractor",
    "FastAudioSynthesizer",
    "get_bar_intervals",
    "load_audio_extractor",
]
