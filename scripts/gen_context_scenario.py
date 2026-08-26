#!/usr/bin/env python3
"""Generate a random ContextView scenario for review.

Mirrors the geometry in
`src/python/midigpt/osc/studio/static/js/views/ContextView.js::_draw`
and the state rules in `_getBarState`, so each scenario tells you what
the studio canvas *should* render for the rolled params + playhead.

Usage:
    python scripts/gen_context_scenario.py              # fresh random scenario
    python scripts/gen_context_scenario.py --seed 42    # reproducible
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field


POLICIES = ("a_empty", "a_masked", "b")
TRACK_KINDS = ("loop", "live", "agent")


@dataclass
class Track:
    idx: int
    kind: str          # "loop" | "live" | "agent"
    label: str

    @property
    def is_agent(self) -> bool:
        return self.kind == "agent"


def scenario_to_dict(s: "Scenario") -> dict:
    """Serialize a scenario + its state grid for the review UI."""
    grid = []
    for t in s.tracks:
        grid.append({
            "track": {"idx": t.idx, "kind": t.kind, "label": t.label,
                      "is_agent": t.is_agent},
            "cells": [
                {"col": c, "bar": s.col_to_bar(c),
                 "state": bar_state(s, t, c)}
                for c in range(s.model_dim)
            ],
        })
    return {
        "seed": s.seed,
        "params": {
            "model_dim": s.model_dim,
            "lookahead": s.lookahead,
            "num_anticip": s.num_anticip,
            "buffer_bars": s.buffer_bars,
            "adapt_buffer": s.adapt_buffer,
            "warmup_policy": s.warmup_policy,
            "playhead": s.playhead,
        },
        "derived": {
            "regime": "WARMUP" if s.in_warmup else "ESTABLISHED",
            "steady_col": s.steady_col,
            "playhead_col": s.playhead_col,
            "target_col": s.target_col,
            "to_gen_cols": [s.to_gen_first_col, s.to_gen_last_col],
            "lookahead_cols": [s.lookahead_first_col, s.lookahead_last_col],
            "first_gen_playhead": s.first_gen_playhead,
            "first_target_bar": s.first_target_bar,
            "failed_gen_bars": sorted(s.failed_gen_bars),
            "in_warmup": s.in_warmup,
        },
        "grid": grid,
    }


@dataclass
class Scenario:
    seed: int
    tracks: list[Track]
    model_dim: int
    lookahead: int
    num_anticip: int
    buffer_bars: int
    adapt_buffer: bool
    warmup_policy: str
    playhead: int
    # Bars where the agent gen failed (no notes). Absolute bar indices.
    failed_gen_bars: set[int] = field(default_factory=set)

    # --- derived geometry (matches ContextView.js) -----------------------

    @property
    def steady_col(self) -> int:
        return self.model_dim - self.num_anticip - self.lookahead

    @property
    def in_warmup(self) -> bool:
        return self.playhead < self.steady_col

    @property
    def playhead_col(self) -> int:
        return min(self.playhead, self.steady_col)

    @property
    def target_col(self) -> int:
        return self.playhead_col + self.lookahead

    @property
    def gen_active(self) -> bool:
        return self.playhead >= self.first_gen_playhead

    @property
    def num_completed_ticks(self) -> int:
        """Ticks whose output is already committed at this playhead.

        Ticks fire every `num_anticip` bars starting at `first_gen_playhead`.
        At the boundary playhead `P0 + k*N`, tick k is in-flight (not yet
        completed) so `num_completed_ticks == k`.
        """
        if not self.gen_active:
            return 0
        delta = self.playhead - self.first_gen_playhead
        return (delta + self.num_anticip - 1) // self.num_anticip

    @property
    def to_gen_first_bar(self) -> int:
        return self.first_target_bar + self.num_completed_ticks * self.num_anticip

    @property
    def to_gen_last_bar(self) -> int:
        return self.to_gen_first_bar + self.num_anticip - 1

    @property
    def to_gen_first_col(self) -> int:
        return self.bar_to_col(self.to_gen_first_bar)

    @property
    def to_gen_last_col(self) -> int:
        return self.bar_to_col(self.to_gen_last_bar)

    @property
    def lookahead_first_col(self) -> int:
        return self.playhead_col + 1

    @property
    def lookahead_last_col(self) -> int:
        return self.to_gen_first_col - 1

    @property
    def first_gen_playhead(self) -> int:
        return max(0, self.buffer_bars - self.lookahead) \
            if self.adapt_buffer else self.buffer_bars

    @property
    def first_target_bar(self) -> int:
        # Target on the FIRST gen ever fired = first_gen_playhead + lookahead.
        return self.first_gen_playhead + self.lookahead

    def col_to_bar(self, col: int) -> int:
        return self.playhead + (col - self.playhead_col)

    def bar_to_col(self, bar: int) -> int:
        return self.playhead_col + (bar - self.playhead)


# ──────────────────────────────────────────────────────────────────────────
# State decision (mirrors _getBarState)
# ──────────────────────────────────────────────────────────────────────────

def bar_state(s: Scenario, track: Track, col: int) -> str:
    bar = s.col_to_bar(col)
    if bar < 0:
        return "OUTSIDE"

    gen_active = s.gen_active

    # Conditioning tracks.
    if not track.is_agent:
        if bar < s.playhead:
            return "CONTEXT"
        return "MASKED_FUTURE" if gen_active else "PENDING"

    # Agent track — failures override everything else.
    if bar in s.failed_gen_bars:
        return "MASKED_FAILED"

    # Pre-history: every bar that no tick ever targets independently.
    # Boundary is `first_target_bar` (not `first_gen_playhead`) because the
    # first-gen lookahead bars `[first_gen_playhead, first_target_bar)` are
    # part of that same bootstrap window, not bars produced by their own gens.
    if bar < s.first_target_bar:
        if not gen_active:
            return "PENDING"
        if s.warmup_policy == "a_empty":
            return "CONTEXT"
        if s.warmup_policy == "a_masked":
            return "MASKED_PAST"
        # policy b → bootstrap fills these bars
        if s.playhead == s.first_gen_playhead:
            return "TO_GENERATE"
        return "CONTEXT"

    # Output committed by ticks that have already fired.
    if bar < s.to_gen_first_bar:
        return "GENERATED"

    # Current (or next-upcoming) tick: a contiguous block of `num_anticip` bars.
    if bar <= s.to_gen_last_bar:
        return "TO_GENERATE" if gen_active else "PENDING"

    # Beyond the in-flight tick.
    if s.in_warmup:
        return "WASTED" if gen_active else "PENDING"
    return "MASKED_FUTURE"


# ──────────────────────────────────────────────────────────────────────────
# Scenario generation
# ──────────────────────────────────────────────────────────────────────────

MAX_TICKS = 128  # stable failure flag per tick (one tick == num_anticip bars)
FAILURE_HORIZON = MAX_TICKS  # kept for back-compat exports


def _roll_base(seed: int | None):
    """Roll everything except the playhead + (filtered) failure set.

    Returns the structural params plus a horizon-long deterministic
    list of failure flags so the same seed produces a stable failure
    pattern no matter where the playhead slider sits.
    """
    rng  = random.Random(seed)
    seed = seed if seed is not None else rng.randrange(1 << 30)
    rng  = random.Random(seed)

    n_tracks = rng.randint(2, 4)
    agent_idx = rng.randrange(n_tracks)
    tracks: list[Track] = []
    for i in range(n_tracks):
        if i == agent_idx:
            tracks.append(Track(i, "agent", f"T{i} agent"))
        else:
            kind = rng.choice(("loop", "live"))
            tracks.append(Track(i, kind, f"T{i} human {kind}"))

    model_dim   = rng.choice((4, 8, 12, 16))
    lookahead   = rng.randint(1, min(4, model_dim - 2))
    num_anticip = rng.randint(1, min(4, model_dim - lookahead - 1))
    buffer_bars = rng.randint(2, model_dim - 1)
    adapt_buf   = rng.random() < 0.7
    policy      = rng.choice(POLICIES)

    steady_col   = model_dim - num_anticip - lookahead
    initial_ph   = rng.randint(steady_col, steady_col + rng.randint(0, 25)) \
        if rng.random() < 0.7 and steady_col > 0 else \
        rng.randint(0, max(0, steady_col - 1))

    # Stable per-TICK failure flags. A tick produces `num_anticip` bars in
    # one shot, so failure is decided per tick, not per bar.
    fail_p = rng.random() * 0.3
    fail_flags = [rng.random() < fail_p for _ in range(MAX_TICKS)]

    return {
        "seed":         seed,
        "tracks":       tracks,
        "model_dim":    model_dim,
        "lookahead":    lookahead,
        "num_anticip":  num_anticip,
        "buffer_bars":  buffer_bars,
        "adapt_buffer": adapt_buf,
        "warmup_policy": policy,
        "initial_playhead": initial_ph,
        "fail_flags":   fail_flags,
    }


def apply_playhead(base: dict, playhead: int) -> "Scenario":
    """Build a Scenario from base params + a chosen playhead.

    Failures are decided per tick. A tick is "completed" iff its output
    has been committed at this playhead (the in-flight tick at a boundary
    is NOT yet completed — its targets are still TO_GENERATE).
    """
    lookahead    = base["lookahead"]
    num_anticip  = base["num_anticip"]
    buffer_bars  = base["buffer_bars"]
    adapt_buffer = base["adapt_buffer"]
    first_gen_ph = max(0, buffer_bars - lookahead) if adapt_buffer else buffer_bars
    first_target = first_gen_ph + lookahead

    if playhead >= first_gen_ph:
        completed = (playhead - first_gen_ph + num_anticip - 1) // num_anticip
    else:
        completed = 0

    fail_flags = base["fail_flags"]
    failed: set[int] = set()
    for k in range(min(completed, len(fail_flags))):
        if fail_flags[k]:
            tick_start = first_target + k * num_anticip
            for bar in range(tick_start, tick_start + num_anticip):
                failed.add(bar)

    return Scenario(
        seed=base["seed"],
        tracks=base["tracks"],
        model_dim=base["model_dim"],
        lookahead=lookahead,
        num_anticip=num_anticip,
        buffer_bars=buffer_bars,
        adapt_buffer=adapt_buffer,
        warmup_policy=base["warmup_policy"],
        playhead=playhead,
        failed_gen_bars=failed,
    )


def roll(seed: int | None = None, playhead: int | None = None) -> "Scenario":
    """Compatibility wrapper: roll a scenario at an explicit or default playhead."""
    base = _roll_base(seed)
    ph   = base["initial_playhead"] if playhead is None else playhead
    return apply_playhead(base, ph)


# ──────────────────────────────────────────────────────────────────────────
# Pretty-print
# ──────────────────────────────────────────────────────────────────────────

STATE_GLYPH = {
    "OUTSIDE":       "·",
    "CONTEXT":       "CTX",
    "GENERATED":     "GEN",
    "TO_GENERATE":   "→T←",
    "GENERATING":    "···",
    "WASTED":        "WST",
    "PENDING":       "···",
    "MASKED_FUTURE": "M·F",
    "MASKED_PAST":   "M·P",
    "MASKED_FAILED": "M·X",
}


def render(s: Scenario) -> str:
    out: list[str] = []
    out.append(f"=== Scenario seed={s.seed} ===")
    out.append(
        f"tracks={len(s.tracks)}  model_dim={s.model_dim}  "
        f"lookahead={s.lookahead}  num_anticip={s.num_anticip}  "
        f"buffer={s.buffer_bars}  adapt_buffer={s.adapt_buffer}  "
        f"policy={s.warmup_policy}"
    )
    out.append(f"playhead={s.playhead}  regime={'WARMUP' if s.in_warmup else 'ESTABLISHED'}")
    out.append(
        f"steady_col={s.steady_col}  playhead_col={s.playhead_col}  "
        f"target_col={s.target_col}  "
        f"toGen_cols=[{s.to_gen_first_col},{s.to_gen_last_col}]  "
        f"lookahead_cols=[{s.lookahead_first_col},{s.lookahead_last_col}]"
    )
    out.append(
        f"first_gen_playhead={s.first_gen_playhead}  "
        f"first_target_bar={s.first_target_bar}  "
        f"failed_gen_bars={sorted(s.failed_gen_bars) or '∅'}"
    )
    out.append("")

    # Header row of bar indices.
    header = "       " + " ".join(f"{s.col_to_bar(c):>4}" for c in range(s.model_dim))
    out.append(header)
    rule   = "       " + " ".join("----" for _ in range(s.model_dim))
    out.append(rule)
    for t in s.tracks:
        cells = " ".join(
            f"{STATE_GLYPH[bar_state(s, t, c)]:>4}" for c in range(s.model_dim)
        )
        tag = "★" if t.is_agent else " "
        out.append(f"{tag} {t.label:<5} {cells}")

    out.append("")
    out.append("Legend:")
    out.append("  CTX=context  GEN=generated  →T←=this tick's target")
    out.append("  M·F=masked future  M·P=masked pre-history  M·X=masked failed gen")
    out.append("  WST=wasted (warmup tail)  ···=pending/generating")
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Replay a specific scenario (prints seed on every run)")
    args = ap.parse_args()
    s = roll(args.seed)
    print(render(s))


if __name__ == "__main__":
    main()
