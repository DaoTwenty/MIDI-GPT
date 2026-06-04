from __future__ import annotations

import glob as _glob
import random
from typing import Any

from midigpt._types import Score
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.augmentation.score_window import select_window
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.rl.config import GRPOConfig


class RLVPDataset:
    """Dataset for RLVP training.

    Each sample is a dict:
        "score"          : windowed Score (model_dim bars, up to max_tracks tracks)
        "agent_id"       : track index in `score` that will be generated
        "requested_attrs": {attr_name: quantized_value} randomly drawn from valid
                           track-level attributes for the agent track's type

    The agent track is always melodic (drums don't carry the attribute set that
    we currently reward). If a score has no melodic track after windowing, the
    sample is skipped.
    """

    def __init__(
        self,
        parquet_path: str | list[str],
        tokenizer: Tokenizer,
        analyzer: AttributeAnalyzer,
        config: GRPOConfig,
    ):
        try:
            import datasets as hf
        except ImportError:
            raise ImportError("pip install midigpt[train]") from None

        paths = _resolve_paths(parquet_path)
        self._data = hf.load_dataset("parquet", data_files=paths, split="train")
        self._tokenizer = tokenizer
        self._analyzer = analyzer
        self._config = config

        # Gather melodic track-level attributes and their domain sizes.
        attr_sizes = analyzer.attribute_sizes()
        attr_levels = analyzer.attribute_levels()
        attr_track_types = analyzer.attribute_track_types()
        self._melodic_attrs: list[tuple[str, int]] = [
            (name, attr_sizes[name])
            for name, level in attr_levels.items()
            if level == "track"
            and attr_track_types.get(name, "both") in ("melodic", "both")
            and attr_sizes[name] > 1
        ]
        if not self._melodic_attrs:
            raise ValueError(
                "No melodic track-level attributes found in analyzer. "
                "RLVP requires at least one attribute to reward."
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Return a sample dict or None if this index cannot produce a valid window."""
        row = self._data[idx]
        try:
            score = Score.from_bytes(row["music"])
        except Exception:
            return None

        cfg = self._config
        windowed = select_window(
            score,
            n_bars=cfg.model_dim,
            n_tracks=min(cfg.max_tracks, len(score.tracks)),
            min_fill_ratio=cfg.min_fill_ratio,
        )
        if windowed is None:
            return None

        # Pick a random melodic track as the agent.
        from midigpt._core import TrackType
        melodic_ids = [
            i for i, t in enumerate(windowed.tracks)
            if getattr(t, "track_type", None) != "drum"
            and getattr(t, "type", None) != TrackType.Drum
        ]
        if not melodic_ids:
            return None

        agent_id = random.choice(melodic_ids)

        # Randomly sample num_attrs_per_prompt attribute names and target values.
        n = min(cfg.num_attrs_per_prompt, len(self._melodic_attrs))
        chosen = random.sample(self._melodic_attrs, n)
        requested_attrs = {
            name: random.randint(0, size - 1) for name, size in chosen
        }

        return {
            "score": windowed,
            "agent_id": agent_id,
            "requested_attrs": requested_attrs,
        }

    def sample_batch(self, size: int) -> list[dict[str, Any]]:
        """Draw `size` valid samples, retrying on None returns."""
        n = len(self)
        out: list[dict] = []
        tried: set[int] = set()
        while len(out) < size and len(tried) < n:
            idx = random.randint(0, n - 1)
            if idx in tried:
                continue
            tried.add(idx)
            sample = self[idx]
            if sample is not None:
                out.append(sample)
        return out


def _resolve_paths(parquet_path: str | list[str]) -> list[str]:
    if isinstance(parquet_path, list):
        return sorted(parquet_path)
    expanded = sorted(_glob.glob(parquet_path))
    return expanded if expanded else [parquet_path]
