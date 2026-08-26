"""Audio-to-MIDI Adapters and Pooling Modules."""

from midigpt.adapters.audio_adapter import (
    AudioToMidiAdapter,
    FusionStrategy,
    PoolingStrategy,
)
from midigpt.adapters.hierarchical_fusion import (
    ConcatLinearFusion,
    FiLMFusion,
    ResidualAddFusion,
)
from midigpt.adapters.pooling import (
    BeatGridPooler,
    MeanConvPooler,
    PerceiverResampler,
)

__all__ = [
    "AudioToMidiAdapter",
    "PoolingStrategy",
    "FusionStrategy",
    "MeanConvPooler",
    "PerceiverResampler",
    "BeatGridPooler",
    "ConcatLinearFusion",
    "FiLMFusion",
    "ResidualAddFusion",
]
