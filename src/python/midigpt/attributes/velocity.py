from midigpt._types import Score
from midigpt.attributes.base import BaseAttribute
from midigpt.attributes.quantile import _bar_notes


class VelocityRange(BaseAttribute):
    """Dynamic range (max - min velocity) of the notes in a bar.

    Bar-level by design, not track-level: Humanize's generation unit is the
    bar (multi_humanize targets are (track, bar) pairs), so this control is
    only meaningful at the same granularity as what actually gets
    regenerated -- a single track-wide value would be a flat constant across
    bars whose dynamics naturally vary (accents, phrasing, crescendo).
    """

    name = "velocity_range"
    token_type = "BarLevelVelocityRange"
    level = "bar"
    track_type = "both"
    size = 10

    def compute(self, score: Score, track_idx: int, bar_idx: int | None = None) -> float | int:
        if bar_idx is None:
            return 0
        track = score.tracks[track_idx]
        notes = _bar_notes(track.bars[bar_idx], score)
        if len(notes) < 2:
            return 0
        velocities = [n.velocity for n in notes]
        return max(velocities) - min(velocities)

    def quantize(self, value: float | int) -> int:
        return max(0, min(self.size - 1, int(value) * self.size // 128))

    def dequantize(self, quantized: int) -> float | int:
        return quantized * 128 // self.size
