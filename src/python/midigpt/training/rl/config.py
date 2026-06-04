from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    # ── rollout ────────────────────────────────────────────────────────────────
    group_size: int = 4          # G: number of completions sampled per prompt
    temperature: float = 1.0    # sampling temperature during rollout
    max_new_tokens: int = 512   # hard cap on tokens generated per rollout

    # ── training ───────────────────────────────────────────────────────────────
    learning_rate: float = 1e-6
    num_train_steps: int = 1_000
    prompts_per_step: int = 4   # distinct prompts accumulated per gradient step
    gradient_clip: float = 1.0
    kl_coef: float = 0.1        # weight for KL(policy ‖ reference) penalty
    advantage_eps: float = 1e-8 # std floor to avoid division-by-zero

    # ── reward ─────────────────────────────────────────────────────────────────
    # "soft"  : continuous reward in [0,1] proportional to absolute error
    # "binary": 1.0 if all requested attributes match exactly, else 0.0
    reward_scale: str = "soft"
    num_attrs_per_prompt: int = 2  # attributes randomly requested per prompt

    # ── dataset / windowing ────────────────────────────────────────────────────
    model_dim: int = 4          # bars per generation window
    max_tracks: int = 8         # max tracks to include from each score
    min_fill_ratio: float = 0.75

    # ── checkpointing / logging ────────────────────────────────────────────────
    checkpoint_every: int = 100
    log_every: int = 10
    checkpoint_dir: str = "checkpoints/rl"

    def validate(self) -> None:
        if self.reward_scale not in ("soft", "binary"):
            raise ValueError(f"reward_scale must be 'soft' or 'binary', got {self.reward_scale!r}")
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 (GRPO needs at least two samples to normalise)")
        if not (0.0 <= self.kl_coef):
            raise ValueError("kl_coef must be >= 0")
        if self.num_attrs_per_prompt < 1:
            raise ValueError("num_attrs_per_prompt must be >= 1")
