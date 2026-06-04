"""GRPO trainer for RLVP (Reinforcement Learning with Verifiable Policy).

Algorithm overview
------------------
For each training step:
1. Sample `prompts_per_step` prompts from RLVPDataset.
2. For each prompt, run G=group_size rollouts of the policy (no_grad), collecting
   (context_ids, generated_ids, result_score) for each completion.
3. Score each completion with AttributeReward → scalar in [0, 1].
4. Normalise rewards within the group as GRPO advantages:
     adv = (r - mean(r)) / (std(r) + eps)
5. Teacher-force each completion: forward pass over (context_ids + generated_ids)
   to compute per-token log probs under the current policy and the frozen reference.
6. GRPO loss per completion:
     pg_loss = -adv * mean(policy_log_probs)
     kl      = mean(policy_log_probs - ref_log_probs)   # ≥0 when policy diverges
     loss    = pg_loss + kl_coef * kl
   Total loss = mean over all completions.
7. Backward + gradient clip + optimizer step.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import torch

import midigpt._core as _core
from midigpt._converters import from_cpp, to_cpp
from midigpt._types import Score
from midigpt.attributes.base import AttributeAnalyzer
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.rl.config import GRPOConfig
from midigpt.training.rl.dataset import RLVPDataset
from midigpt.training.rl.reward import AttributeReward

log = logging.getLogger(__name__)


# ── constraint helpers ─────────────────────────────────────────────────────────

def _grammar_constraints(step, score: Score) -> _core.ConstraintGraph:
    """Grammar-only ConstraintGraph — no attribute constraints.

    This is intentional for RLVP: we want the policy to freely choose attribute
    tokens so the reward measures genuine adherence, not forced compliance.
    """
    graph = _core.ConstraintGraph()
    grammar = _core.GrammarConstraint()
    if step.is_autoregressive:
        grammar.set_exact_bars(step.end_bar - step.start_bar)
        grammar.set_autoregressive_mode(True)
    grammar.set_max_tracks(len(score.tracks))
    grammar.set_require_notes(True)
    graph.add_constraint(grammar)
    return graph


# ── rollout ────────────────────────────────────────────────────────────────────

def _rollout(
    model,
    tokenizer: Tokenizer,
    score: Score,
    agent_id: int,
    requested_attrs: dict[str, int],
    temperature: float,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], list[int], Score] | None:
    """Run one autoregressive rollout for `agent_id` in `score`.

    Attribute tokens for `requested_attrs` are injected into the score's prompt
    so the model sees what is being requested. The grammar does NOT force attribute
    values — the policy chooses freely, and the reward measures adherence.

    Returns (context_ids, generated_ids, result_score) or None on failure.
    """
    # Inject requested attributes into the agent track's prompt so attribute
    # tokens appear in the encoded sequence context.
    working_score = copy.deepcopy(score)
    working_score.tracks[agent_id].attributes.update(requested_attrs)

    n_bars = len(working_score.tracks[0].bars)

    # SelectionMask: agent track is autoregressive, all its bars are targets.
    n_tracks = len(working_score.tracks)
    mask = _core.SelectionMask()
    mask.selected = [[False] * n_bars for _ in range(n_tracks)]
    mask.autoregressive = [False] * n_tracks
    mask.ignore = [False] * n_tracks
    for b in range(n_bars):
        mask.selected[agent_id][b] = True
    mask.autoregressive[agent_id] = True

    enc_cfg = tokenizer._vocab.config()
    enc_cfg.model_dim = n_bars

    planner = _core.StepPlanner(mask, enc_cfg, n_bars, 1)
    steps = list(planner.plan())
    if not steps:
        return None

    # RLVP always processes exactly one step (all bars, one track).
    step = steps[0]
    cpp_score = to_cpp(working_score)

    try:
        state = _core.SessionState(
            cpp_score, step,
            tokenizer._vocab,
            _grammar_constraints(step, working_score),
            tokenizer._encoder,
            tokenizer._decoder,
        )
    except Exception as exc:
        log.debug("SessionState creation failed: %s", exc)
        return None

    context_ids = list(state.context_tokens())
    generated_ids: list[int] = []

    with torch.no_grad():
        while not state.complete() and len(generated_ids) < max_new_tokens:
            all_ids = context_ids + generated_ids
            ctx = torch.tensor([all_ids], dtype=torch.long, device=device)
            try:
                out = model(ctx)
            except Exception:
                return None
            logits = out[0][0, -1] if isinstance(out, tuple) else out[0, -1]

            mask_bool = torch.as_tensor(state.logit_mask(), dtype=torch.bool, device=device)
            if not mask_bool.any():
                return None

            masked = logits.masked_fill(~mask_bool, float("-inf"))
            probs = (masked / temperature).softmax(-1)
            if probs.sum() < 1e-6 or torch.isnan(probs).any():
                probs = mask_bool.float() / mask_bool.float().sum()

            token = int(torch.multinomial(probs, 1).item())
            state.advance(token)
            generated_ids.append(token)

    if not generated_ids:
        return None

    try:
        result = from_cpp(state.result())
    except Exception:
        return None

    return context_ids, generated_ids, result


# ── log prob computation ───────────────────────────────────────────────────────

def _log_probs_teacher_forcing(
    model,
    context_ids: list[int],
    generated_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Teacher-forcing log probs for `generated_ids` given `context_ids`.

    Forward pass over the full sequence (context + generated), then slice out
    log probs at the generated positions.

    Returns tensor of shape (gen_len,) with requires_grad when model params do.
    """
    full = context_ids + generated_ids
    input_ids = torch.tensor([full[:-1]], dtype=torch.long, device=device)

    out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out  # (1, S-1, vocab)

    ctx_len = len(context_ids)
    gen_len = len(generated_ids)
    # Position ctx_len-1 in logits predicts generated_ids[0].
    gen_logits = logits[0, ctx_len - 1 : ctx_len - 1 + gen_len]  # (gen_len, vocab)

    log_p = gen_logits.log_softmax(-1)
    token_ids = torch.tensor(generated_ids, dtype=torch.long, device=device)
    return log_p.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # (gen_len,)


# ── trainer ────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """GRPO trainer for RLVP attribute-adherence fine-tuning.

    Args:
        model     : The policy model (GPT2LMHeadModel or compatible).
        tokenizer : Tokenizer wrapping the C++ vocab/encoder/decoder.
        analyzer  : AttributeAnalyzer used for reward computation.
        config    : GRPOConfig hyperparameters.
        device    : Torch device for model and rollouts. Defaults to model's device.
    """

    def __init__(
        self,
        model,
        tokenizer: Tokenizer,
        analyzer: AttributeAnalyzer,
        config: GRPOConfig,
        device: torch.device | str | None = None,
    ):
        config.validate()
        self.model = model
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.config = config

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        self._ref_model = copy.deepcopy(model).eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)

        self._reward_fn = AttributeReward(analyzer, scale=config.reward_scale)
        self._optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def train(
        self,
        dataset: RLVPDataset,
        *,
        resume_step: int = 0,
    ) -> None:
        """Main training loop.

        Args:
            dataset    : RLVPDataset yielding (score, agent_id, requested_attrs).
            resume_step: Step to resume from (for checkpointing).
        """
        cfg = self.config
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for step in range(resume_step, cfg.num_train_steps):
            prompts = dataset.sample_batch(cfg.prompts_per_step)
            if not prompts:
                log.warning("step %d: no valid prompts sampled, skipping", step)
                continue

            loss, metrics = self.train_step(prompts)

            if step % cfg.log_every == 0:
                log.info(
                    "step %d | loss=%.4f reward=%.3f kl=%.4f",
                    step,
                    metrics["loss"],
                    metrics["mean_reward"],
                    metrics["mean_kl"],
                )

            if (step + 1) % cfg.checkpoint_every == 0:
                self._save_checkpoint(step + 1)

    def train_step(
        self, prompts: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One gradient step over a list of prompt dicts.

        Each prompt dict: {"score": Score, "agent_id": int, "requested_attrs": dict}

        Returns (loss_tensor, metrics_dict).
        """
        cfg = self.config
        self.model.train()

        all_losses: list[torch.Tensor] = []
        all_rewards: list[float] = []
        all_kls: list[float] = []

        self._optimizer.zero_grad()

        for prompt in prompts:
            score: Score = prompt["score"]
            agent_id: int = prompt["agent_id"]
            requested_attrs: dict[str, int] = prompt["requested_attrs"]

            # ── rollout group ──────────────────────────────────────────────────
            completions: list[tuple[list[int], list[int], Score]] = []
            rewards: list[float] = []

            for _ in range(cfg.group_size):
                result = _rollout(
                    self.model,
                    self.tokenizer,
                    score,
                    agent_id,
                    requested_attrs,
                    temperature=cfg.temperature,
                    max_new_tokens=cfg.max_new_tokens,
                    device=self.device,
                )
                if result is None:
                    continue
                ctx_ids, gen_ids, result_score = result
                r = self._reward_fn.compute(result_score, agent_id, requested_attrs)
                completions.append((ctx_ids, gen_ids, result_score))
                rewards.append(r)

            if len(completions) < 2:
                # Need at least 2 completions to normalize GRPO advantages.
                continue

            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            mean_r = rewards_t.mean().item()
            std_r = rewards_t.std().item()
            advantages = (rewards_t - mean_r) / (std_r + cfg.advantage_eps)

            all_rewards.extend(rewards)

            # ── policy gradient + KL per completion ────────────────────────────
            for (ctx_ids, gen_ids, _), adv in zip(completions, advantages.tolist()):
                policy_lp = _log_probs_teacher_forcing(
                    self.model, ctx_ids, gen_ids, self.device
                )
                with torch.no_grad():
                    ref_lp = _log_probs_teacher_forcing(
                        self._ref_model, ctx_ids, gen_ids, self.device
                    )

                pg_loss = -adv * policy_lp.mean()
                kl = (policy_lp - ref_lp).mean()
                comp_loss = pg_loss + cfg.kl_coef * kl

                all_losses.append(comp_loss)
                all_kls.append(kl.item())

        if not all_losses:
            dummy = torch.tensor(0.0, requires_grad=True)
            return dummy, {"loss": 0.0, "mean_reward": 0.0, "mean_kl": 0.0}

        total_loss = torch.stack(all_losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
        self._optimizer.step()

        metrics = {
            "loss": total_loss.item(),
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "mean_kl": sum(all_kls) / len(all_kls),
        }
        return total_loss.detach(), metrics

    def _save_checkpoint(self, step: int) -> None:
        path = Path(self.config.checkpoint_dir) / f"step_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            path,
        )
        log.info("checkpoint saved: %s", path)
