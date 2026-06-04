from __future__ import annotations

from midigpt._types import Score
from midigpt.attributes.base import AttributeAnalyzer


class AttributeReward:
    """Reward signal measuring realized vs. requested track-level attributes.

    Reward is 1.0 when the model achieves all requested attributes, and
    degrades as realized values diverge from requested ones.

    soft   : r_attr = 1 - |realized - requested| / (domain_size - 1)
             Final reward = mean over requested attributes.
    binary : 1.0 if ALL requested attributes match exactly, else 0.0.
    """

    def __init__(self, analyzer: AttributeAnalyzer, scale: str = "soft"):
        if scale not in ("soft", "binary"):
            raise ValueError(f"scale must be 'soft' or 'binary', got {scale!r}")
        self._analyzer = analyzer
        self._scale = scale

    def compute(
        self,
        score: Score,
        track_idx: int,
        requested_attrs: dict[str, int],
    ) -> float:
        """Return scalar reward in [0, 1].

        Args:
            score: The result score after generation.
            track_idx: Index of the generated track within `score`.
            requested_attrs: {attr_name: quantized_value} that was requested.
        """
        if not requested_attrs:
            return 1.0

        realized = self._analyzer.compute_track_tokens(score, track_idx)

        rewards: list[float] = []
        for name, req_val in requested_attrs.items():
            real_val = realized.get(name)
            if real_val is None:
                continue
            if self._scale == "binary":
                rewards.append(1.0 if real_val == req_val else 0.0)
            else:
                attr = self._analyzer.get(name)
                domain_size = int(getattr(attr, "size", 1)) if attr is not None else 1
                denom = max(domain_size - 1, 1)
                r = 1.0 - abs(real_val - req_val) / denom
                rewards.append(max(0.0, r))

        if not rewards:
            return 1.0

        if self._scale == "binary":
            return 1.0 if all(r == 1.0 for r in rewards) else 0.0
        return sum(rewards) / len(rewards)
