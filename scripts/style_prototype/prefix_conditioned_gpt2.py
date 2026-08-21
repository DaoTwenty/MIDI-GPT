"""Real, incremental, KV-cache-capable soft-prefix conditioning: makes
ConditionedGPT2 (Variant A) / ExplicitControlsGPT2 (Variant C) usable as a
genuine InferenceEngine model, so soft-prefix generation runs through actual
SamplingSession (grammar-constrained, KV-cached, one note's conditioning-
aware choice compounding into the next), not just the eval scripts'
one-shot single-forward-pass resampler.

Both source classes (conditioned_gpt2.py / explicit_controls_gpt2.py) are
explicitly documented as "not meant to be dropped into InferenceEngine as-is"
-- their own forward() is a single-shot, full-sequence, teacher-forced call
with no past_kv handling at all. This module supplies the missing
incremental path by reusing their transformer/lm_head/prefix-encoding logic
directly (encode_style_prefix / encode_control_prefix), injecting the
computed prefix as an EXTRA KV position 0 on the very first (prefill) call
only.

The exact ModelBase protocol mirrored here (confirmed this session by
reading inference/base.py + GPT2LMHeadModel's concrete implementation,
inference/model/gpt2.py):
    forward(input_ids, past_kv=None, key_mask=None, position_ids=None) -> (logits, presents)
    make_empty_kv() -> tuple            # per-layer (zeros(1,n_head,0,head_dim), zeros(1,n_head,0,head_dim)) -- NEVER None
    kv_length(past_kv) -> int
    kv_null_positions(past_kv, spans) -> None   # in-place
    max_context() -> int
    forward_with_hooks(input_ids, past_kv, hooks) -> (logits, presents, dict)

Design (once past_kv[0][0].shape[2] == 0, i.e. the very first call):
  1. Compute prefix_vec (B, n_embd) from the wrapper's own mutable
     conditioning-input instance attributes (style_ids/style_mask/has_style
     for variant A, control_bins/has_controls for variant C -- these are
     set once per request, before engine.session(...).run(), exactly like
     SteeredGPT2LMHeadModel's steer_vec/alpha pattern in steered_forward.py
     -- forward()'s signature must match what _KVRunner calls verbatim, no
     room for extra required args).
  2. Embed prefix_vec at position 0, run it through every block with
     past_kv=None to seed a genuine 1-position KV cache.
  3. Embed the real input_ids at position_ids = arange(1, 1+T) (manual --
     GPT2's own position_ids=None default of arange(past_len, past_len+T)
     is wrong here since past_len is still 0 at this point), run them
     through the blocks using the seeded 1-position KV as past_kv.
  4. Return logits for the T real-token positions only (the prefix's own
     position never needs a prediction), and past_kv = the full (1+T)-length
     cache (prefix + real tokens).

On every SUBSEQUENT call, past_kv[0][0].shape[2] > 0 already reflects
"prefix + tokens so far", so GPT2's ordinary incremental behavior
(position_ids = arange(past_len, past_len+T)) is already correct with zero
special-casing -- the prefix is invisible to every call after the first.

kv_null_positions/max_context both need a permanent +1/-1 offset since
physical KV position 0 is always the prefix, never a real, maskable
position -- called out explicitly below, easy to miss.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

import midigpt._core as _core
from conditioned_gpt2 import ConditionedGPT2
from explicit_controls import N_CONTROLS
from explicit_controls_gpt2 import ExplicitControlsGPT2
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.tokenizer.tokenizer import Tokenizer


def _run_blocks(transformer, x, past_kv, key_mask):
    """One pass of every transformer block, mirroring GPT2LMHeadModel.
    forward()'s block loop exactly. `past_kv`: per-layer (k,v) tuple or
    None (first-ever call for that sub-pass)."""
    presents = []
    for i, block in enumerate(transformer.h):
        pkv = past_kv[i] if past_kv is not None else None
        x, present, _ = block(x, past_kv=pkv, return_attn_weights=False, key_mask=key_mask)
        presents.append(present)
    return x, tuple(presents)


class PrefixConditionedGPT2(nn.Module):
    """Wraps a trained ConditionedGPT2 (variant="A") or ExplicitControlsGPT2
    (variant="C") instance for real incremental generation. See module
    docstring for the full design."""

    def __init__(self, base: ConditionedGPT2 | ExplicitControlsGPT2, variant: str):
        super().__init__()
        if variant not in ("A", "C"):
            raise ValueError(f"variant must be 'A' or 'C', got {variant!r}")
        self.base = base
        self.variant = variant
        self.cfg = base.cfg
        # Conditioning input, set per request under the server's semaphore
        # (mirrors SteeredGPT2LMHeadModel.steer_vec/alpha). Defaults below
        # are the "no reference available" fallback for each variant --
        # NOT arbitrary: InferenceEngine.warmup() fires one real dummy
        # forward() call (input_ids=[[0]], empty kv) before any request
        # ever sets these, so they must be valid tensors from construction,
        # not None (see engine.py's _compute_initial_kv -- discards the
        # warmup call's output, but the call itself must not crash).
        if variant == "A":
            self.style_ids = torch.zeros(1, 1, dtype=torch.long)
            self.style_mask = torch.zeros(1, 1, dtype=torch.bool)
            self.has_style = torch.zeros(1, dtype=torch.bool)
        else:
            self.control_bins = torch.zeros(1, N_CONTROLS, dtype=torch.long)
            self.has_controls = torch.zeros(1, dtype=torch.bool)

    def _compute_prefix(self, device: torch.device) -> torch.Tensor:
        if self.variant == "A":
            return self.base.encode_style_prefix(
                self.style_ids.to(device), self.style_mask.to(device), self.has_style.to(device)
            )
        return self.base.encode_control_prefix(self.control_bins.to(device), self.has_controls.to(device))

    def forward(self, input_ids, past_kv=None, key_mask=None, position_ids=None):
        transformer = self.base.transformer
        lm_head = self.base.lm_head
        B, T = input_ids.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None and past_kv[0][0].shape[2] > 0 else 0

        if past_len == 0:
            prefix_vec = self._compute_prefix(input_ids.device)  # (B, n_embd)
            prefix_pos = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)
            x_prefix = prefix_vec.unsqueeze(1) + transformer.wpe(prefix_pos)
            # Prefix's own self-attention pass: no key_mask (a length-1
            # sequence attending only to itself has nothing to mask), no
            # past_kv (it IS the first cached position).
            _, presents_prefix = _run_blocks(transformer, x_prefix, past_kv=None, key_mask=None)

            pos = (
                position_ids.to(input_ids.device)
                if position_ids is not None
                else torch.arange(1, 1 + T, device=input_ids.device).unsqueeze(0)
            )
            x = transformer.wte(input_ids) + transformer.wpe(pos)
            x, presents = _run_blocks(transformer, x, past_kv=presents_prefix, key_mask=key_mask)
        else:
            pos = (
                position_ids.to(input_ids.device)
                if position_ids is not None
                else torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
            )
            x = transformer.wte(input_ids) + transformer.wpe(pos)
            x, presents = _run_blocks(transformer, x, past_kv=past_kv, key_mask=key_mask)

        x = transformer.ln_f(x)
        logits = lm_head(x)
        return logits, presents

    def make_empty_kv(self) -> tuple:
        cfg = self.cfg
        dev = next(self.base.parameters()).device
        return tuple(
            (
                torch.zeros(1, cfg.n_head, 0, cfg.head_dim, device=dev),
                torch.zeros(1, cfg.n_head, 0, cfg.head_dim, device=dev),
            )
            for _ in range(cfg.n_layer)
        )

    def kv_length(self, past_kv) -> int:
        if past_kv is None or len(past_kv) == 0:
            return 0
        return int(past_kv[0][0].shape[2])

    def kv_null_positions(self, past_kv, spans: list[tuple[int, int]]) -> None:
        """`spans` arrive in real-token-relative coordinates (the caller
        has no concept of the prefix); +1 shifts into physical KV indices,
        since physical position 0 is permanently the prefix."""
        if past_kv is None or not spans:
            return
        for k_c, v_c in past_kv:
            for s, e in spans:
                k_c[:, :, s + 1 : e + 1, :] = -1e4
                v_c[:, :, s + 1 : e + 1, :] = 0.0

    def max_context(self) -> int:
        # Position 0 is permanently reserved for the prefix -- one less
        # real token fits than the base architecture's raw n_positions.
        return self.cfg.n_positions - 1

    def forward_with_hooks(self, input_ids, past_kv, hooks: dict):
        """Real-token pass only fires hooks -- the prefix-seeding sub-pass
        is internal bookkeeping, not part of the caller-visible sequence.
        Whether this server's actual request modes (plain AR/humanize
        generation) ever invoke this at all is unverified; revisit if a
        real caller needs it."""
        transformer = self.base.transformer
        lm_head = self.base.lm_head
        B, T = input_ids.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None and past_kv[0][0].shape[2] > 0 else 0
        want_attn = "attn" in hooks
        want_hidden = "hidden" in hooks
        hook_outputs: dict[str, list] = {k: [] for k in hooks}

        if past_len == 0:
            prefix_vec = self._compute_prefix(input_ids.device)
            prefix_pos = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)
            x = prefix_vec.unsqueeze(1) + transformer.wpe(prefix_pos)
            presents_prefix = []
            for block in transformer.h:
                x, present, _ = block(x, past_kv=None, return_attn_weights=False, key_mask=None)
                presents_prefix.append(present)
            pos = torch.arange(1, 1 + T, device=input_ids.device).unsqueeze(0)
            x = transformer.wte(input_ids) + transformer.wpe(pos)
            base_past_kv = presents_prefix
        else:
            pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
            x = transformer.wte(input_ids) + transformer.wpe(pos)
            base_past_kv = past_kv

        presents = []
        for i, block in enumerate(transformer.h):
            x, present, attn_w = block(x, past_kv=base_past_kv[i], return_attn_weights=want_attn)
            presents.append(present)
            if want_attn and attn_w is not None:
                hooks["attn"](i, attn_w)
                hook_outputs["attn"].append(attn_w)
            if want_hidden:
                hooks["hidden"](i, x)
                hook_outputs["hidden"].append(x)

        x = transformer.ln_f(x)
        logits = lm_head(x)
        if "logits" in hooks:
            hooks["logits"](logits)
            hook_outputs["logits"].append(logits)
        return logits, tuple(presents), hook_outputs


def build_prefix_conditioned_engine(conditioning_checkpoint_path: str, variant: str, device: str | None = None):
    """Loads a Variant A (ConditionedGPT2) or Variant C (ExplicitControlsGPT2)
    checkpoint fresh (these live outside the arch registry/load_checkpoint
    system entirely -- confirmed no existing loader handles them), wraps it
    in PrefixConditionedGPT2, and builds a warmed InferenceEngine around it.
    Returns (wrapper, engine) -- the caller sets conditioning attrs on
    `wrapper` per request, mirroring steered_forward.py's build_steered_engine."""
    from midigpt.inference.engine import InferenceEngine

    ckpt = torch.load(conditioning_checkpoint_path, map_location="cpu", weights_only=False)
    gpt2_cfg = ckpt["gpt2_config"]
    enc_cfg = _core.EncoderConfig.from_json(ckpt["encoder_config_json"])

    if variant == "A":
        style_cfg = ckpt["style_encoder_config"]
        base = ConditionedGPT2(gpt2_cfg, style_cfg)
    else:
        control_dim = ckpt["control_dim"]
        base = ExplicitControlsGPT2(gpt2_cfg, control_dim=control_dim)
    base.load_state_dict(ckpt["model_state_dict"])
    base.to(device or "cpu")
    base.eval()

    wrapper = PrefixConditionedGPT2(base, variant).to(device or "cpu")
    wrapper.eval()

    analyzer = AttributeAnalyzer.from_config(enc_cfg)
    tokenizer = Tokenizer(enc_cfg, analyzer)
    engine = InferenceEngine(wrapper, tokenizer, analyzer)
    engine.warmup()
    return wrapper, engine
