"""Compact expressive-token vocabulary for the style encoder.

The LM's VelocityLevel/DeltaDirection/Delta token ids sit at large, sparse
offsets inside the full LM vocabulary (a few hundred entries wide). The style
encoder is a separate, much smaller model with no reason to share embedding-
table size with the LM's full vocab, so this re-indexes just those three
token types into a small, dense 0..size-1 id space.

Reuses midigpt.training.dataset's EXPRESSIVE_TYPES/predicate rather than
redefining "what counts as expressive" in a second place.
"""

from __future__ import annotations

import midigpt._core as _core
from midigpt.training.dataset import _EXPRESSIVE_TYPES


class StyleVocab:
    """Bidirectional mapping between raw LM token ids (VelocityLevel/
    DeltaDirection/Delta only) and a compact 0..size-1 id space, derived from
    a given Vocabulary's token ranges for those three types."""

    def __init__(self, vocab: "_core.Vocabulary"):
        self._raw_to_local: dict[int, int] = {}
        self._local_to_raw: list[int] = []
        for tt in _EXPRESSIVE_TYPES:
            if not vocab.has(tt):
                continue
            start, end = vocab.range(tt)
            for raw in range(start, end):
                self._raw_to_local[raw] = len(self._local_to_raw)
                self._local_to_raw.append(raw)

    @property
    def size(self) -> int:
        return len(self._local_to_raw)

    def is_expressive(self, raw_token_id: int) -> bool:
        return raw_token_id in self._raw_to_local

    def to_local(self, raw_token_id: int) -> int:
        return self._raw_to_local[raw_token_id]

    def to_local_seq(self, raw_token_ids: list[int]) -> list[int]:
        """Filter a raw token sequence down to just its expressive tokens,
        re-indexed into the compact local id space, in order."""
        return [self._raw_to_local[t] for t in raw_token_ids if t in self._raw_to_local]

    def to_raw(self, local_id: int) -> int:
        return self._local_to_raw[local_id]
