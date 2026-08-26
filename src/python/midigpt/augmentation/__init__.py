from midigpt.augmentation.base import AugmentationPipeline, BaseTransform
from midigpt.augmentation.instrument_swap import InstrumentSwap
from midigpt.augmentation.mask_bar import MaskBar, MaskBarConfig, MaskMode
from midigpt.augmentation.score_window import (
    select_window,
    select_window_dispatch,
    select_window_uniform,
    select_window_weighted,
)
from midigpt.augmentation.track_permutation import TrackPermutation
from midigpt.augmentation.transpose import Transpose
from midigpt.augmentation.velocity import VelocityScale

__all__ = [
    "AugmentationPipeline",
    "BaseTransform",
    "InstrumentSwap",
    "MaskBar",
    "MaskBarConfig",
    "MaskMode",
    "TrackPermutation",
    "Transpose",
    "VelocityScale",
    "select_window",
    "select_window_dispatch",
    "select_window_uniform",
    "select_window_weighted",
]
