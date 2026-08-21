"""Dedicated HTTP server for Humanize requests.

Same request/response shape as `midigpt.http.server` on purpose: a single
JSON body, `score` is `Score.to_dict()` (the exact structure that server's
`/generate` already uses), response is `{"score": ..., "timing": ...}`. A
caller already talking to the generic server needs no second representation
for the piece itself.

What's actually different here, and the whole reason this is a separate
server rather than just another mode on the generic one:

  1. `targets` -- humanize's own version of a prompt. Which track(s) and
     bar(s) to actually humanize, addressed directly as
     `[{"track": 0, "bars": [4,5,6,7]}]` instead of requiring the caller to
     build a full TrackPrompt list (bars/ignore/humanize flags per track).
     Everything not listed is passed through as real, untouched context.
     Omitting `targets` entirely humanizes every track that has any notes.

  2. Dual-checkpoint mode/alpha selection. Loads BOTH the A and E Humanize
     checkpoints (see RESULTS_LOG.md's final verdict -- A is most precise
     under trustworthy context, E is most robust to mechanical/unreliable
     context) and picks between them per request:

       mode=robust      (default) -- E alone. Never collapses under
                         unreliable context, small precision cost.
       mode=expressive  -- A alone. Best precision, but only trustworthy
                           when EVERY context track is a real performance.
       mode=blend       -- BlendedModel (scripts/humanize_eval/
                           blend_model.py) at a caller-supplied `alpha`
                           (1.0=pure A, 0.0=pure E).
       mode=auto        -- per request, alpha is set from a lightweight
                           heuristic estimate of how "mechanical" the
                           target's context tracks look (reuses
                           mechanize.py's own onset-grid definition, so the
                           detector can't drift from the transform it's
                           detecting). The BLENDING mechanism itself is
                           validated (E9, 2026-08-09: given the correct
                           alpha, interpolation behaves sensibly, doesn't
                           degrade toward either extreme). The DETECTOR's
                           accuracy at estimating that alpha from real,
                           unlabeled data is a separate, still-open question
                           (E9b, in progress) -- treat auto as experimental
                           until that lands.

  3. Chunking. A real piece is usually longer than the model's bar window,
     so pieces are split into sequential windows (see `_iter_windows`), each
     humanized using its own real neighbouring bars as context and spliced
     back at the right offset. There's no cross-window memory beyond one
     window's width -- a piece's bar 50 can't be influenced by bar 2
     directly. Later windows DO see earlier windows' real generated output
     as context, though, so context quality improves as generation proceeds.

Usage::

    midigpt-humanize-http --ckpt-a models/humanize_tiny.json ... --port 8001

    # humanize only track 0, bars 4-7, and all of track 2 -- everything else
    # is left exactly as sent and used as real context
    curl -X POST http://localhost:8001/humanize -H 'Content-Type: application/json' -d '{
        "score": {...},
        "mode": "blend", "alpha": 0.7, "temperature": 0.9,
        "targets": [{"track": 0, "bars": [4,5,6,7]}, {"track": 2, "bars": "all"}]
    }'
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi and uvicorn are required for the HTTP server. "
        "Install with: pip install midigpt[http]"
    ) from None

from midigpt._types import Score
from midigpt.augmentation.mechanize import mechanize_track, quantized_onset
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.engine import InferenceEngine
from midigpt.inference.validation import RequestValidationError

log = logging.getLogger(__name__)

_VALID_MODES = {"robust", "expressive", "blend", "auto"}


class _TargetSpec(BaseModel):
    track: int
    # None (the default, e.g. an entry with no "bars" key) = this track is
    # NOT a humanize target, context only -- the only reason to still list
    # it is `mechanize_before`. "all" or an explicit bar list = humanize
    # those bars.
    bars: list[int] | str | None = None
    # Actually transforms this track's notes (flatten velocity, snap onsets
    # to the grid) before generation runs -- a real transformation, not a
    # hint. Valid whether or not this track is also a target.
    mechanize_before: bool = False


class _HumanizeBody(BaseModel):
    # Same `Score.to_dict()` shape the generic server's `_GenerateBody.score`
    # uses -- deliberately, so a caller already talking to that server needs
    # no second representation for the piece itself.
    score: dict
    # Named checkpoint_mode (not `mode`) to avoid colliding with a possible
    # future per-target mode field.
    checkpoint_mode: str = "robust"
    alpha: float | None = None
    # This server's actual value-add over the generic one: which track(s)/
    # bar(s) to humanize (and which to mechanize first), addressed directly
    # instead of requiring the caller to build a full TrackPrompt list.
    # None = every track with any notes is a target, nothing pre-mechanized
    # (the original default, kept for backward compatibility).
    targets: list[_TargetSpec] | None = None

    temperature: float = 0.85
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    max_attempts: int = 3
    novelty_check: bool = True
    bars_per_step: int | None = None
    tracks_per_step: int | None = None
    model_dim: int | None = None


@dataclass
class GenParams:
    """The generation-control knobs that actually matter for humanizing --
    not a mirror of every InferenceConfig field. Deliberately excluded and
    why: `mask_mode`/`silence_check` (nothing is ever masked or can go
    silent -- Humanize never changes note presence, only existing notes'
    velocity/timing), `polyphony_hard_limit`/`density_hard_limit` (polyphony
    and onset density are fully determined by the fixed skeleton, these
    constraints have nothing to act on), `shuffle` (no ordering ambiguity
    across a single humanize call). `novelty_check` IS kept -- whether the
    regenerated velocity/timing actually differs from the input is a real,
    observable failure mode (confirmed firing in testing: "candidate
    identical to original"). `bars_per_step`/`tracks_per_step`/`model_dim`
    default to None = auto: this server picks a sensible value from the
    actual target spec (all target bars/tracks in one step) and the
    encoder's window size -- set them explicitly only to override that."""

    temperature: float = 0.85
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    max_attempts: int = 3
    novelty_check: bool = True
    bars_per_step: int | None = None
    tracks_per_step: int | None = None
    model_dim: int | None = None  # bar-window size; must be one of the encoder's num_bars_map


def _num_bars_map(engine: InferenceEngine) -> list[int]:
    cfg = json.loads(engine._tokenizer._vocab.config().to_json())
    return sorted(cfg.get("num_bars_map") or [4])


def _window_bar_count(engine: InferenceEngine) -> int:
    return max(_num_bars_map(engine))


def _track_mechanicalness(track, resolution: int) -> float | None:
    """Fraction of notes in `track` that look mechanical: exactly on the
    same onset-quantization grid mechanize.py snaps to, reusing that exact
    function so detection can't silently drift from the transform it's
    trying to recognize. Returns None if the track has no notes (caller
    should then fall back to a neutral default, not a spuriously confident
    score)."""
    total, on_grid = 0, 0
    for bar in track.bars:
        for note in bar.notes:
            total += 1
            if note.onset_ticks == quantized_onset(note, bar, resolution):
                on_grid += 1
    if total == 0:
        return None
    return on_grid / total


def _iter_windows(n_bars_total: int, window_bars: int):
    """Yields (window_start, new_start, new_end) triples covering the whole
    piece with the model's context window (`window_start` to
    `window_start + window_bars`), where only [new_start, new_end) within
    that window should actually be targeted for humanization.

    `model_dim` (the window size the tokenizer sees) must be one of a fixed
    set of valid values (the encoder's `num_bars_map`, e.g. [4, 8, 12, 16]),
    not an arbitrary leftover bar count -- confirmed the hard way: a naive
    "clip the last window to what's left" approach produced a 9-bar final
    window on a 40-bar piece with window_bars=16 and failed validation
    ("model_dim=9 not in vocab domain [4, 8, 12, 16]"). Fix: every window is
    always exactly `window_bars` wide; the final window instead SHIFTS
    backward to overlap the previous one so its width stays valid, and only
    the not-yet-covered tail bars are marked as new targets -- the
    overlapped bars are included purely as (already-humanized, not
    re-generated) context.
    """
    if n_bars_total <= window_bars:
        yield 0, 0, n_bars_total
        return
    start = 0
    covered = 0
    while covered < n_bars_total:
        if start + window_bars > n_bars_total:
            start = n_bars_total - window_bars  # shift final window to stay full-width
        new_start = covered
        new_end = min(start + window_bars, n_bars_total)
        yield start, new_start, new_end
        covered = new_end
        start = covered


class HumanizeServer:
    _WATCHDOG_INTERVAL = 10

    def __init__(
        self,
        engine_a: InferenceEngine,
        engine_e: InferenceEngine,
        blended_model,
        engine_blend: InferenceEngine,
        idle_timeout: float = 0,
    ) -> None:
        self._engine_a = engine_a
        self._engine_e = engine_e
        self._blended_model = blended_model  # exposes .alpha, mutated per-request
        self._engine_blend = engine_blend
        self._semaphore = asyncio.Semaphore(1)
        self._idle_timeout = float(idle_timeout) if idle_timeout else 0.0
        self._last_activity = time.monotonic()
        self._window_bars = _window_bar_count(engine_a)
        self._app = self._build_app()

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    async def _idle_watchdog(self) -> None:
        log.info("Idle watchdog started (timeout=%.0fs).", self._idle_timeout)
        while True:
            await asyncio.sleep(self._WATCHDOG_INTERVAL)
            if time.monotonic() - self._last_activity >= self._idle_timeout:
                log.info("Idle timeout reached — shutting down.")
                os._exit(0)

    @property
    def app(self) -> FastAPI:
        return self._app

    def _engine_for(self, mode: str, alpha: float | None):
        if mode == "expressive":
            return self._engine_a
        if mode == "robust":
            return self._engine_e
        if mode == "blend":
            self._blended_model.alpha = 1.0 if alpha is None else alpha
            return self._engine_blend
        raise ValueError(f"unreachable mode {mode!r}")

    def _humanize_window(self, window_score: Score, targets_local: dict[int, set[int] | None], new_range: tuple[int, int], mode: str, alpha: float | None, params: GenParams):
        """`targets_local` maps track id -> wanted bar indices, ALREADY in
        this window's local (0-based within the window) coordinates, or
        None meaning "every bar in this track that has notes." `new_range` =
        (local_start, local_end) bars within this window that are actually
        new (not already humanized by a previous, overlapping window) --
        only bars satisfying both the caller's target spec AND this range
        get generated; anything else is real context (either the piece's
        original notes, on the very first window, or already-generated
        output, on a later window that overlaps backward -- see
        `_iter_windows`)."""
        lo, hi = new_range
        target_cells: dict[int, list[int]] = {}
        for t, wanted in targets_local.items():
            bars = [
                b for b, bar in enumerate(window_score.tracks[t].bars)
                if lo <= b < hi and bar.notes and (wanted is None or b in wanted)
            ]
            if bars:
                target_cells[t] = bars
        if not target_cells:
            return window_score, {}  # nothing new to humanize in this window

        if mode == "auto":
            resolution = window_score.resolution
            other_tracks = [t for t in range(len(window_score.tracks)) if t not in target_cells]
            fracs = [
                f for t in other_tracks
                if (f := _track_mechanicalness(window_score.tracks[t], resolution)) is not None
            ]
            # alpha = 1 - mechanicalness: no real context tracks at all (a
            # single-track piece) has nothing to judge trustworthiness from,
            # so default to the safe (robust) end rather than assume A's
            # best case.
            alpha_eff = 1.0 - (sum(fracs) / len(fracs)) if fracs else 0.0
            engine = self._engine_for("blend", alpha_eff)
        else:
            engine = self._engine_for(mode, alpha)

        tracks_req = [
            TrackPrompt(id=t, bars=sorted(bars), humanize=True) if t in target_cells
            else TrackPrompt(id=t, bars=[], ignore=True)
            for t in range(len(window_score.tracks))
        ]
        bars_per_step = params.bars_per_step or max(1, max(len(b) for b in target_cells.values()))
        tracks_per_step = params.tracks_per_step or max(1, len(target_cells))
        req = GenerationRequest(
            tracks=tracks_req,
            config=InferenceConfig(
                temperature=params.temperature, top_p=params.top_p, top_k=params.top_k,
                bars_per_step=bars_per_step, tracks_per_step=tracks_per_step,
                model_dim=len(window_score.tracks[0].bars),
                seed=params.seed if params.seed is not None else -1,
                max_attempts=params.max_attempts, novelty_check=params.novelty_check,
                mask_mode="remove", silence_check=False,
            ),
        )
        return engine.session(copy.deepcopy(window_score), req).run(), target_cells

    def _pad_bars(self, score: Score, target_len: int) -> None:
        """In-place: pad every track's bars up to `target_len` with empty
        (silent, real -- not future/masked) bars matching the last real
        bar's time signature. Needed because `model_dim` must be an exact
        entry in the encoder's `num_bars_map`; a short piece (e.g. a 2-bar
        loop against a [4,8,12,16] domain) doesn't naturally land on one."""
        from midigpt._types import Bar

        for tr in score.tracks:
            if not tr.bars:
                tr.bars.append(Bar())  # 4/4 default -- nothing to copy a time signature from
            last = tr.bars[-1]
            while len(tr.bars) < target_len:
                tr.bars.append(Bar(notes=[], ts_numerator=last.ts_numerator, ts_denominator=last.ts_denominator, beat_length=last.beat_length))

    @staticmethod
    def _localize_targets(targets: dict[int, set[int] | None], window_start: int, window_width: int) -> dict[int, set[int] | None]:
        """Translate a caller's global bar indices into this window's local
        (0-based within the window) coordinates, dropping any that fall
        outside the window entirely. `None` (meaning "all notes-bearing
        bars") passes through unchanged -- it's already window-relative by
        construction, resolved later against each bar's actual content."""
        local: dict[int, set[int] | None] = {}
        for t, wanted in targets.items():
            if wanted is None:
                local[t] = None
            else:
                shifted = {b - window_start for b in wanted if window_start <= b < window_start + window_width}
                if shifted:
                    local[t] = shifted
        return local

    def _humanize_full(self, score: Score, targets: dict[int, set[int] | None] | None, mechanize_ids: set[int], mode: str, alpha: float | None, params: GenParams) -> Score:
        """`targets`: {track_id: wanted_bar_indices_or_None}, global bar
        coordinates against the whole piece. `None` for a track means
        "every bar with notes in that track" (the default when `targets`
        itself is omitted entirely: every track that has any notes at all).
        This is the "prompt" -- same idea as GenerationRequest's TrackPrompt
        list in the generic server, just addressed by (track, bars) instead
        of requiring the caller to construct the whole Score/request JSON.

        `mechanize_ids`: tracks to actually flatten (constant velocity,
        grid-quantized onsets) before any generation happens, independent of
        whether they're also targets. Applied ONCE, up front, to the whole
        track -- not re-applied per window. If a mechanized track is also a
        target, its target bars still get correctly regenerated when their
        window comes up (Humanize always overwrites a target bar's
        velocity/timing regardless of its starting state, so pre-mechanizing
        it first is harmless there); the only real effect is that this
        track reads as mechanical context wherever it appears *before* its
        own target bars are reached -- which is the intended behaviour, not
        a side effect to guard against.
        """
        if not score.tracks:
            return score
        if targets is None:
            targets = {t: None for t, tr in enumerate(score.tracks) if any(bar.notes for bar in tr.bars)}
        for t in targets:
            if t < 0 or t >= len(score.tracks):
                raise ValueError(f"target track {t} out of range (piece has {len(score.tracks)} tracks)")
        for t in mechanize_ids:
            if t < 0 or t >= len(score.tracks):
                raise ValueError(f"mechanize track {t} out of range (piece has {len(score.tracks)} tracks)")
        if not targets and not mechanize_ids:
            return copy.deepcopy(score)  # nothing requested -- pass the piece through unchanged

        valid_sizes = _num_bars_map(self._engine_a)
        if params.model_dim is not None and params.model_dim not in valid_sizes:
            raise ValueError(f"model_dim={params.model_dim} not in the encoder's num_bars_map {valid_sizes}")
        window_bars = params.model_dim or self._window_bars

        n_bars_total = max(len(tr.bars) for tr in score.tracks)
        output = copy.deepcopy(score)
        for t in mechanize_ids:
            mechanize_track(output.tracks[t], output.resolution)
        if not targets:
            return output  # mechanize-only request, nothing to generate

        if n_bars_total <= window_bars:
            # Explicit model_dim: pad to exactly that (respects the caller's
            # chosen context budget). Auto: pad only to the smallest valid
            # size that fits, so a short piece doesn't waste context on
            # padding nobody asked for.
            padded_len = params.model_dim if params.model_dim is not None else next(
                (s for s in valid_sizes if s >= n_bars_total), max(valid_sizes)
            )
            window = copy.deepcopy(score)
            self._pad_bars(window, padded_len)
            local_targets = self._localize_targets(targets, 0, n_bars_total)
            result, hit_cells = self._humanize_window(window, local_targets, (0, n_bars_total), mode, alpha, params)
            # Splice back ONLY the bars actually targeted (`hit_cells`), never a
            # whole range: a bar the model wasn't asked to touch is not
            # guaranteed to decode back byte-identical (confirmed the hard way --
            # untouched trailing bars in a partially-targeted track came back
            # with onset_ticks reset to 0). Blindly copying a whole range
            # silently corrupts real, untouched content.
            for t, bars in hit_cells.items():
                for local_b in bars:
                    output.tracks[t].bars[local_b] = result.tracks[t].bars[local_b]
            return output

        for window_start, new_start, new_end in _iter_windows(n_bars_total, window_bars):
            window = copy.deepcopy(output)  # deepcopy OUTPUT: later windows see earlier windows' real generated bars as context
            for tr in window.tracks:
                tr.bars = tr.bars[window_start : window_start + window_bars]
            local_range = (new_start - window_start, new_end - window_start)
            local_targets = self._localize_targets(targets, window_start, window_bars)
            result, hit_cells = self._humanize_window(window, local_targets, local_range, mode, alpha, params)
            for t, bars in hit_cells.items():
                for local_b in bars:
                    output.tracks[t].bars[window_start + local_b] = result.tracks[t].bars[local_b]
        return output

    def _capabilities(self) -> dict:
        return {
            "modes": sorted(_VALID_MODES),
            "window_bars": self._window_bars,
            "valid_model_dims": _num_bars_map(self._engine_a),
            # E9 (2026-08-09): given the correct alpha, blending interpolates
            # sensibly -- validated. E9b (2026-08-09): the heuristic detector
            # estimates that alpha accurately from real, unlabeled data too
            # (pearson r=0.959 vs. oracle alpha, mean abs error 0.026 on 200
            # pieces; Auto_Blend nearly matches Oracle_Blend on both
            # degeneracy and Wasserstein) -- also validated.
            "blend_mechanism_validated": True,
            "auto_detector_validated": True,
            "targets_format": (
                '[{"track": <int>, "bars": [<int>, ...] | "all" | omitted, '
                '"mechanize_before": bool}, ...] -- global bar indices; omitting "bars" '
                "means this track is context-only (only useful with mechanize_before=true); "
                "omitting `targets` entirely humanizes every track with notes, nothing mechanized"
            ),
        }

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            watchdog_task = asyncio.create_task(self._idle_watchdog()) if self._idle_timeout > 0 else None
            yield
            if watchdog_task is not None:
                watchdog_task.cancel()

        app = FastAPI(
            title="MIDI-GPT Humanize Server",
            description="JSON REST API dedicated to the Humanize feature -- same Score.to_dict() shape as midigpt.http.server, plus track/bar-level targets and dual-checkpoint mode selection.",
            version="0.1.0",
            lifespan=lifespan,
        )

        @app.get("/health", tags=["meta"])
        def health():
            self._touch()
            return {"status": "ok"}

        @app.get("/info", tags=["meta"])
        def info():
            return self._capabilities()

        @app.post("/humanize", tags=["generation"])
        async def humanize(body: _HumanizeBody):
            self._touch()
            if body.checkpoint_mode not in _VALID_MODES:
                raise HTTPException(status_code=400, detail=f"checkpoint_mode must be one of {sorted(_VALID_MODES)}")
            if body.checkpoint_mode == "blend" and body.alpha is None:
                raise HTTPException(status_code=400, detail="checkpoint_mode=blend requires alpha (0.0-1.0)")
            if body.alpha is not None and not (0.0 <= body.alpha <= 1.0):
                raise HTTPException(status_code=400, detail="alpha must be in [0.0, 1.0]")
            if not (0.0 < body.top_p <= 1.0):
                raise HTTPException(status_code=400, detail="top_p must be in (0.0, 1.0]")
            if body.top_k < 0:
                raise HTTPException(status_code=400, detail="top_k must be >= 0")
            if body.max_attempts < 1:
                raise HTTPException(status_code=400, detail="max_attempts must be >= 1")
            params = GenParams(
                temperature=body.temperature, top_p=body.top_p, top_k=body.top_k,
                seed=body.seed, max_attempts=body.max_attempts, novelty_check=body.novelty_check,
                bars_per_step=body.bars_per_step, tracks_per_step=body.tracks_per_step, model_dim=body.model_dim,
            )

            try:
                score = Score.from_dict(body.score)
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid score: {exc}") from exc

            target_map: dict[int, set[int] | None] | None = None
            mechanize_ids: set[int] = set()
            if body.targets is not None:
                target_map = {}
                for t in body.targets:
                    if t.mechanize_before:
                        mechanize_ids.add(t.track)
                    if t.bars is not None:
                        target_map[t.track] = None if t.bars == "all" else {int(b) for b in t.bars}

            t0 = time.perf_counter()
            async with self._semaphore:
                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(
                        None, self._humanize_full, score, target_map, mechanize_ids,
                        body.checkpoint_mode, body.alpha, params,
                    )
                except (RequestValidationError, ValueError) as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                except Exception as exc:
                    log.exception("Humanize failed")
                    raise HTTPException(status_code=500, detail=str(exc)) from exc

            return {
                "score": result.to_dict(),
                "timing": {"total_s": time.perf_counter() - t0},
            }

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        uvicorn.run(self._app, host=host, port=port)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIDI-GPT Humanize HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt-a", required=True, metavar="PATH", help="Path to the 'expressive' checkpoint (A)")
    p.add_argument("--ckpt-e", required=True, metavar="PATH", help="Path to the 'robust' checkpoint (E)")
    p.add_argument("--device", default=None, metavar="DEVICE", help='"cpu", "cuda", "mps", or "auto"')
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--idle-timeout", type=float, default=0, metavar="SECONDS")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "humanize_eval"))
    from blend_model import BlendedModel  # noqa: E402

    log.info("Loading checkpoint A (expressive): %s", args.ckpt_a)
    engine_a = InferenceEngine.from_checkpoint(args.ckpt_a, device=args.device)
    log.info("Loading checkpoint E (robust): %s", args.ckpt_e)
    engine_e = InferenceEngine.from_checkpoint(args.ckpt_e, device=args.device)

    if engine_a._tokenizer.vocab_size() != engine_e._tokenizer.vocab_size():
        raise SystemExit("ckpt-a and ckpt-e must share a vocab (same encoder config) to blend.")

    blended_model = BlendedModel(engine_a._model, engine_e._model, alpha=1.0)
    engine_blend = InferenceEngine(blended_model, engine_a._tokenizer, engine_a._analyzer)
    engine_blend.warmup()

    server = HumanizeServer(engine_a, engine_e, blended_model, engine_blend, idle_timeout=args.idle_timeout)
    log.info("Starting Humanize HTTP server on %s:%d (window=%d bars)", args.host, args.port, server._window_bars)
    if args.idle_timeout:
        log.info("Auto-shutdown after %.0fs of inactivity.", args.idle_timeout)
    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
