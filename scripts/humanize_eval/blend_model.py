"""BlendedModel -- a true probability-space mixture of two GPT2LMHeadModel
checkpoints (see chat discussion: ship A for trustworthy-context precision,
E for mechanical-context robustness, blend between them per generation
instead of hard-switching checkpoints).

Duck-types the same "ModelBase" surface InferenceEngine/SamplingSession
actually call on `engine._model` (confirmed by reading session.py, not
assumed): `forward(input_ids, past_kv, key_mask=None, position_ids=None) ->
(logits, presents)`, `make_empty_kv()`, `kv_null_positions(past_kv, spans)`,
`max_context()`, plus `.parameters()` (free via nn.Module). Because of this,
`InferenceEngine(blended, tokenizer, analyzer)` works completely unmodified
-- no changes to production inference/session code.

KV packing: `make_empty_kv()`/`forward()` concatenate model_A's and model_E's
per-layer (k, v) tuples into ONE flat tuple (A's layers first, then E's),
rather than nesting `(kv_A, kv_E)`. This matters: InferenceEngine's device
autodetection does `isinstance(kv[0][0], torch.Tensor)` -- a flat
concatenation keeps `kv[0][0]` a real tensor (first layer, k), so that check
(and everything else in the codebase that treats past_kv as "a tuple of
per-layer (k,v) tensor pairs") keeps working with zero special-casing.

Blending math: mixing raw logits (`alpha*logits_A + (1-alpha)*logits_E`) is
NOT the same operation as mixing predictive distributions -- it's closer to
a product-of-experts than a mixture. A true mixture-of-experts is
probability-space: `p = alpha*softmax(logits_A) + (1-alpha)*softmax(logits_E)`.
Computed here via logsumexp in log-space for numerical stability; the result
is returned as "logits" but is actually already-normalized log-probabilities
-- correct either way, since any downstream softmax recovers the same `p`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlendedModel(nn.Module):
    def __init__(self, model_a: nn.Module, model_e: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.model_a = model_a
        self.model_e = model_e
        self.alpha = alpha  # 1.0 = pure A (trust context), 0.0 = pure E (robust to bad context)
        self._n_layer_a = model_a.cfg.n_layer
        self._n_layer_e = model_e.cfg.n_layer

    def make_empty_kv(self) -> tuple:
        return self.model_a.make_empty_kv() + self.model_e.make_empty_kv()

    def _split_kv(self, past_kv):
        if past_kv is None:
            return None, None
        return past_kv[: self._n_layer_a], past_kv[self._n_layer_a :]

    def forward(self, input_ids, past_kv=None, key_mask=None, position_ids=None):
        kv_a, kv_e = self._split_kv(past_kv)
        logits_a, presents_a = self.model_a(
            input_ids, kv_a, key_mask=key_mask, position_ids=position_ids
        )
        logits_e, presents_e = self.model_e(
            input_ids, kv_e, key_mask=key_mask, position_ids=position_ids
        )

        a = float(self.alpha)
        # Exact boundaries: a 1e-8-style floor on log(alpha) leaks a small
        # but real amount of the "excluded" model into tail probabilities
        # (confirmed by smoke test -- max abs diff ~0.27 nats at alpha=0.0
        # without this), since model logits are confident enough that some
        # tail log-probs are more extreme than the floor. alpha will always
        # be a clean 0/1 for the A-alone/E-alone comparison arms of the eval,
        # so this needs to be exact there, not just close.
        if a >= 1.0:
            blended = torch.log_softmax(logits_a, dim=-1)
        elif a <= 0.0:
            blended = torch.log_softmax(logits_e, dim=-1)
        else:
            logp_a = torch.log_softmax(logits_a, dim=-1)
            logp_e = torch.log_softmax(logits_e, dim=-1)
            log_a = torch.log(torch.tensor(a, device=logits_a.device))
            log_1ma = torch.log(torch.tensor(1.0 - a, device=logits_a.device))
            blended = torch.logsumexp(
                torch.stack([log_a + logp_a, log_1ma + logp_e], dim=0), dim=0
            )
        return blended, presents_a + presents_e

    def kv_null_positions(self, past_kv, spans):
        kv_a, kv_e = self._split_kv(past_kv)
        self.model_a.kv_null_positions(kv_a, spans)
        self.model_e.kv_null_positions(kv_e, spans)

    def max_context(self) -> int:
        return min(self.model_a.max_context(), self.model_e.max_context())
