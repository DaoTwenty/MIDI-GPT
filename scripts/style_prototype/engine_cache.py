"""Per-checkpoint InferenceEngine cache for humanize_style_server.py, covering
three request shapes: plain (unconditioned), steered (activation steering on
top of a plain base checkpoint), and soft-prefix (a real, KV-cache-capable
ConditionedGPT2/ExplicitControlsGPT2 running directly through the engine, see
prefix_conditioned_gpt2.py).

Bounded LRU keyed at the base-checkpoint level for plain/steered entries:
loading a checkpoint (`tokenizer.checkpoint.load_checkpoint`) is the
expensive part, so the underlying `bundle.model` is loaded once and shared
between the plain engine and every steered variant built on top of it.
Steered sub-entries live NESTED inside their base's cache node (not a flat
separate cache) specifically so evicting a base evicts every steered engine
built on it in one shot -- no steered engine can ever outlive (and hold a
stale reference to) an evicted base model.

Soft-prefix entries own their full model weights independently (the
conditioned checkpoint IS the whole model, not a base+injection composition)
so they get their own separate top-level LRU, keyed by (path, variant).

Callers must hold the server's single request semaphore while calling any
`get_*`/mutating method here -- this cache is not itself thread-safe.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from midigpt.attributes.base import AttributeAnalyzer
from midigpt.inference.engine import InferenceEngine
from midigpt.tokenizer.checkpoint import load_checkpoint
from midigpt.tokenizer.tokenizer import Tokenizer
from steered_forward import SteeredGPT2LMHeadModel


class _BaseEntry:
    __slots__ = ("bundle", "plain_engine", "steered")

    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.plain_engine: InferenceEngine | None = None
        self.steered: dict[tuple, tuple[SteeredGPT2LMHeadModel, InferenceEngine]] = {}


class EngineCache:
    def __init__(self, device: str | None = None, max_entries: int = 4) -> None:
        self._device = device
        self._max_entries = max_entries
        self._bases: OrderedDict[str, _BaseEntry] = OrderedDict()
        self._soft_prefix: OrderedDict[tuple, tuple] = OrderedDict()

    @staticmethod
    def _resolve(path: str) -> str:
        return str(Path(path).resolve())

    def _get_base_entry(self, checkpoint_path: str) -> _BaseEntry:
        key = self._resolve(checkpoint_path)
        if key in self._bases:
            self._bases.move_to_end(key)
            return self._bases[key]
        bundle = load_checkpoint(key, device=self._device)
        entry = _BaseEntry(bundle)
        self._bases[key] = entry
        if len(self._bases) > self._max_entries:
            self._bases.popitem(last=False)
        return entry

    def get_plain_engine(self, checkpoint_path: str) -> InferenceEngine:
        entry = self._get_base_entry(checkpoint_path)
        if entry.plain_engine is None:
            analyzer = AttributeAnalyzer.from_config(entry.bundle.encoder_config)
            tokenizer = Tokenizer(entry.bundle.encoder_config, analyzer)
            engine = InferenceEngine(entry.bundle.model, tokenizer, analyzer)
            engine.warmup()
            entry.plain_engine = engine
        return entry.plain_engine

    def get_steered_engine(
        self,
        base_checkpoint_path: str,
        conditioning_checkpoint_path: str,
        variant: str,
        layers: set[int],
    ) -> tuple[SteeredGPT2LMHeadModel, InferenceEngine]:
        """Returns the (wrapper, engine) pair for this (base, conditioning,
        variant, layers) key, building it once and reusing thereafter.
        Caller must set `wrapper.steer_vec`/`wrapper.alpha` per request --
        see steered_forward.py's build_steered_engine docstring for why this
        is safe under the engine's cached warmup state."""
        entry = self._get_base_entry(base_checkpoint_path)
        key = (self._resolve(conditioning_checkpoint_path), variant, tuple(sorted(layers)))
        if key not in entry.steered:
            base_model = entry.bundle.model
            n_embd = base_model.cfg.n_embd
            wrapper = SteeredGPT2LMHeadModel(
                base_model, torch.zeros(n_embd), set(layers), 1.0
            ).to(self._device or "cpu")
            wrapper.eval()
            analyzer = AttributeAnalyzer.from_config(entry.bundle.encoder_config)
            tokenizer = Tokenizer(entry.bundle.encoder_config, analyzer)
            engine = InferenceEngine(wrapper, tokenizer, analyzer)
            engine.warmup()
            entry.steered[key] = (wrapper, engine)
        return entry.steered[key]

    def get_soft_prefix_engine(self, conditioning_checkpoint_path: str, variant: str):
        """Returns the (wrapper, engine) pair for a real, KV-cache-capable
        soft-prefix model (prefix_conditioned_gpt2.py). Caller must set the
        wrapper's conditioning instance attrs per request before calling
        engine.session(...).run() -- same mutate-under-semaphore pattern as
        get_steered_engine."""
        from prefix_conditioned_gpt2 import build_prefix_conditioned_engine

        key = (self._resolve(conditioning_checkpoint_path), variant)
        if key in self._soft_prefix:
            self._soft_prefix.move_to_end(key)
            return self._soft_prefix[key]
        wrapper, engine = build_prefix_conditioned_engine(
            conditioning_checkpoint_path, variant, device=self._device
        )
        self._soft_prefix[key] = (wrapper, engine)
        if len(self._soft_prefix) > self._max_entries:
            self._soft_prefix.popitem(last=False)
        return (wrapper, engine)
