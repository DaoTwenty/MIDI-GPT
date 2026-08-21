"""Flexible Humanize HTTP server: unlike src/python/midigpt/http/
humanize_server.py (fixed dual-checkpoint at startup, humanize-only, no
conditioning), this server supports BOTH autoregressive and humanize modes
per track, per-bar (not just per-track) expressive/quantized context
control, an arbitrary checkpoint chosen PER REQUEST (not fixed at startup),
and the style-conditioning prototype variants built this session (both
mechanisms: activation steering, and real soft-prefix generation via
prefix_conditioned_gpt2.py's incremental-KV-cache-capable ConditionedGPT2/
ExplicitControlsGPT2). Plus /parquet/* endpoints to fetch a real MIDI piece
from a dataset by row index, for testing without uploading your own MIDI.

Standalone script (not a pyproject.toml entry point) -- prototype scope,
matching the rest of scripts/style_prototype/. Run directly:

    python3 humanize_style_server.py --data-root $SCRATCH/MIDI-GPT/data --port 8002

Request/response shape and error-mapping conventions (400=bad request,
422=RequestValidationError/ValueError from generation, 500=unexpected)
pattern-matched from src/python/midigpt/http/humanize_server.py -- read that
file for the concrete precedent this mirrors.

See scripts/style_prototype/prefix_conditioned_gpt2.py and steered_forward.py
for the two conditioning mechanisms' internals, and engine_cache.py for how
checkpoints are loaded once and reused across requests.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Response
    from pydantic import BaseModel, model_validator
except ImportError:
    raise ImportError(
        "fastapi and uvicorn are required for the HTTP server. Install with: pip install midigpt[http]"
    ) from None

import torch

from engine_cache import EngineCache
from explicit_controls import CONTROL_NAMES, ControlTypeRanges, compute_controls, quantize_controls
from midigpt._converters import to_cpp
from midigpt._types import Score
from midigpt.augmentation.mechanize import mechanize_bar
from midigpt.inference.config import GenerationRequest, InferenceConfig, TrackPrompt
from midigpt.inference.validation import RequestValidationError
from parquet_retrieval import ParquetIndexCache, resolve_under_root
from style_vocab import StyleVocab

import midigpt._core as _core

log = logging.getLogger(__name__)

_MODES = {"context", "autoregressive", "humanize"}


class _TargetSpec(BaseModel):
    track: int
    mode: Literal["context", "autoregressive", "humanize"] = "context"
    bars: list[int] | Literal["all"] | None = None
    mechanize_before: bool = False
    mechanize_bars: list[int] = []
    expressive_bars: list[int] = []
    attributes: dict[str, int] = {}
    bar_attributes: dict[int, dict[str, int]] = {}
    controls: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_bars(self):
        if self.mode == "context" and self.bars is not None:
            raise ValueError(f"track {self.track}: mode='context' must not set bars")
        if self.mode != "context" and self.bars is None:
            raise ValueError(f"track {self.track}: mode={self.mode!r} requires bars")
        if set(self.mechanize_bars) & set(self.expressive_bars):
            raise ValueError(f"track {self.track}: mechanize_bars and expressive_bars overlap")
        return self


class _ReferenceSpec(BaseModel):
    score: dict
    track: int
    bars: list[int] | Literal["all"] = "all"


class _ConditioningSpec(BaseModel):
    mechanism: Literal["steering", "soft_prefix"]
    checkpoint: str
    variant: Literal["A", "C"]
    reference: _ReferenceSpec | None = None
    controls: list[float] | None = None
    control_bins: list[int] | None = None
    alpha: float = 1.0
    layers: list[int] | Literal["all"] = "all"

    @model_validator(mode="after")
    def _check_variant_fields(self):
        if self.variant == "A" and self.reference is None:
            raise ValueError("variant='A' requires reference")
        if self.variant == "C" and self.controls is None and self.control_bins is None:
            raise ValueError("variant='C' requires controls or control_bins")
        if self.mechanism == "soft_prefix" and (self.alpha != 1.0 or self.layers != "all"):
            raise ValueError("alpha/layers only apply to mechanism='steering'")
        return self


class _GenerateBody(BaseModel):
    score: dict
    checkpoint: str | None = None
    targets: list[_TargetSpec]
    conditioning: _ConditioningSpec | None = None

    temperature: float = 0.85
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    max_attempts: int = 3
    novelty_check: bool = True
    silence_check: bool = False
    mask_mode: str = "remove"
    bars_per_step: int | None = None
    tracks_per_step: int | None = None
    model_dim: int | None = None

    @model_validator(mode="after")
    def _check_checkpoint(self):
        soft_prefix = self.conditioning is not None and self.conditioning.mechanism == "soft_prefix"
        if soft_prefix and self.checkpoint is not None:
            raise ValueError("checkpoint must be omitted when conditioning.mechanism='soft_prefix' (the conditioning checkpoint IS the full model)")
        if not soft_prefix and self.checkpoint is None:
            raise ValueError("checkpoint is required unless conditioning.mechanism='soft_prefix'")
        return self


def _resolve_layers(layers_spec, n_layer: int) -> set[int]:
    layers = set(range(n_layer)) if layers_spec == "all" else {int(x) for x in layers_spec}
    if not layers.issubset(set(range(n_layer))):
        raise ValueError(f"layers {sorted(layers)} outside model's {n_layer} blocks")
    return layers


def _iter_windows(n_bars_total: int, window_bars: int):
    """Identical to humanize_server.py's _iter_windows -- see that docstring
    for why the final window shifts backward instead of clipping short."""
    if n_bars_total <= window_bars:
        yield 0, 0, n_bars_total
        return
    start = 0
    covered = 0
    while covered < n_bars_total:
        if start + window_bars > n_bars_total:
            start = n_bars_total - window_bars
        new_start = covered
        new_end = min(start + window_bars, n_bars_total)
        yield start, new_start, new_end
        covered = new_end
        start = covered


class FlexServer:
    _WATCHDOG_INTERVAL = 10

    def __init__(self, device: str | None, cache_size: int, data_root: Path, idle_timeout: float = 0) -> None:
        self._device = device
        self._engines = EngineCache(device=device, max_entries=cache_size)
        self._parquets = ParquetIndexCache(max_entries=8)
        self._data_root = data_root
        self._semaphore = asyncio.Semaphore(1)
        self._idle_timeout = float(idle_timeout) if idle_timeout else 0.0
        self._last_activity = time.monotonic()
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

    # ---- mechanize resolution (once, up front, mirrors humanize_server.py's _humanize_full) ----

    @staticmethod
    def _resolve_mechanize_set(score: Score, targets: list[_TargetSpec]) -> dict[tuple[int, int], bool]:
        out: dict[tuple[int, int], bool] = {}
        for spec in targets:
            if not (spec.mechanize_before or spec.mechanize_bars):
                continue
            n_bars = len(score.tracks[spec.track].bars)
            expressive = set(spec.expressive_bars)
            for b in range(n_bars):
                if b in spec.mechanize_bars or (spec.mechanize_before and b not in expressive):
                    out[(spec.track, b)] = True
        return out

    # ---- window-target resolution ----

    def _resolve_window_targets(
        self,
        window_score: Score,
        targets_local: dict[int, tuple[str, set[int] | None, dict, dict, dict]],
        new_range: tuple[int, int],
    ) -> tuple[dict[int, list[int]], dict[int, tuple[str, dict, dict, dict]]]:
        """mirrors humanize_server.py's _humanize_window target-cell logic,
        with the one real fix this server needed: the bar.notes gate is
        correct for mode='humanize' (needs an existing skeleton) but wrong
        for mode='autoregressive' (an empty bar is the normal AR target)."""
        lo, hi = new_range
        target_cells: dict[int, list[int]] = {}
        extra: dict[int, tuple[str, dict, dict, dict]] = {}
        for t, (mode, wanted, attributes, bar_attributes, controls) in targets_local.items():
            if mode == "humanize":
                bars = [
                    b for b, bar in enumerate(window_score.tracks[t].bars)
                    if lo <= b < hi and bar.notes and (wanted is None or b in wanted)
                ]
            else:  # autoregressive
                bars = [b for b in range(lo, hi) if wanted is None or b in wanted]
            if bars:
                target_cells[t] = bars
                extra[t] = (mode, attributes, bar_attributes, controls)
        return target_cells, extra

    @staticmethod
    def _localize_targets(
        targets: dict[int, tuple[str, set[int] | None, dict, dict, dict]], window_start: int, window_width: int
    ) -> dict[int, tuple[str, set[int] | None, dict, dict, dict]]:
        local = {}
        for t, (mode, wanted, attributes, bar_attributes, controls) in targets.items():
            if wanted is None:
                local[t] = (mode, None, attributes, bar_attributes, controls)
            else:
                shifted = {b - window_start for b in wanted if window_start <= b < window_start + window_width}
                if shifted:
                    local[t] = (mode, shifted, attributes, bar_attributes, controls)
        return local

    def _generate_window(self, window_score: Score, target_cells, extra, engine, body: _GenerateBody):
        tracks_req = []
        for t in range(len(window_score.tracks)):
            if t in target_cells:
                mode, attributes, bar_attributes, controls = extra[t]
                tracks_req.append(TrackPrompt(
                    id=t, bars=sorted(target_cells[t]),
                    autoregressive=(mode == "autoregressive"), humanize=(mode == "humanize"),
                    attributes=attributes, bar_attributes=bar_attributes, controls=controls,
                ))
            else:
                tracks_req.append(TrackPrompt(id=t, bars=[], ignore=True))
        bars_per_step = body.bars_per_step or max(1, max(len(b) for b in target_cells.values()))
        tracks_per_step = body.tracks_per_step or max(1, len(target_cells))
        req = GenerationRequest(
            tracks=tracks_req,
            config=InferenceConfig(
                temperature=body.temperature, top_p=body.top_p, top_k=body.top_k,
                bars_per_step=bars_per_step, tracks_per_step=tracks_per_step,
                model_dim=len(window_score.tracks[0].bars),
                seed=body.seed if body.seed is not None else -1,
                max_attempts=body.max_attempts, novelty_check=body.novelty_check,
                silence_check=body.silence_check, mask_mode=body.mask_mode,
            ),
        )
        return engine.session(copy.deepcopy(window_score), req).run()

    def _pad_bars(self, score: Score, target_len: int) -> None:
        from midigpt._types import Bar

        for tr in score.tracks:
            if not tr.bars:
                tr.bars.append(Bar())
            last = tr.bars[-1]
            while len(tr.bars) < target_len:
                tr.bars.append(Bar(notes=[], ts_numerator=last.ts_numerator, ts_denominator=last.ts_denominator, beat_length=last.beat_length))

    def _generate_full(self, score: Score, body: _GenerateBody, engine, window_bars: int, valid_sizes: list[int]) -> Score:
        if not score.tracks:
            return score
        for spec in body.targets:
            if spec.track < 0 or spec.track >= len(score.tracks):
                raise ValueError(f"target track {spec.track} out of range (piece has {len(score.tracks)} tracks)")

        mechanize_set = self._resolve_mechanize_set(score, body.targets)
        output = copy.deepcopy(score)
        for (t, b), should in mechanize_set.items():
            if should and b < len(output.tracks[t].bars):
                mechanize_bar(output.tracks[t].bars[b], output.resolution)

        targets: dict[int, tuple[str, set[int] | None, dict, dict, dict]] = {}
        for spec in body.targets:
            if spec.mode == "context":
                continue
            wanted = None if spec.bars == "all" else {int(b) for b in spec.bars}
            targets[spec.track] = (spec.mode, wanted, spec.attributes, spec.bar_attributes, spec.controls)
        if not targets:
            return output  # mechanize-only request

        if body.model_dim is not None and body.model_dim not in valid_sizes:
            raise ValueError(f"model_dim={body.model_dim} not in the encoder's num_bars_map {valid_sizes}")
        window_bars = body.model_dim or window_bars

        n_bars_total = max(len(tr.bars) for tr in score.tracks)
        if n_bars_total <= window_bars:
            padded_len = body.model_dim if body.model_dim is not None else next(
                (s for s in valid_sizes if s >= n_bars_total), max(valid_sizes)
            )
            window = copy.deepcopy(output)
            self._pad_bars(window, padded_len)
            local_targets = self._localize_targets(targets, 0, n_bars_total)
            target_cells, extra = self._resolve_window_targets(window, local_targets, (0, n_bars_total))
            if not target_cells:
                return output
            result = self._generate_window(window, target_cells, extra, engine, body)
            for t, bars in target_cells.items():
                for local_b in bars:
                    output.tracks[t].bars[local_b] = result.tracks[t].bars[local_b]
            return output

        for window_start, new_start, new_end in _iter_windows(n_bars_total, window_bars):
            window = copy.deepcopy(output)
            for tr in window.tracks:
                tr.bars = tr.bars[window_start : window_start + window_bars]
            local_range = (new_start - window_start, new_end - window_start)
            local_targets = self._localize_targets(targets, window_start, window_bars)
            target_cells, extra = self._resolve_window_targets(window, local_targets, local_range)
            if not target_cells:
                continue
            result = self._generate_window(window, target_cells, extra, engine, body)
            for t, bars in target_cells.items():
                for local_b in bars:
                    output.tracks[t].bars[window_start + local_b] = result.tracks[t].bars[local_b]
        return output

    # ---- conditioning resolution ----

    def _derive_reference_tokens(self, engine, ref: _ReferenceSpec) -> list[int]:
        """Reduced sub-Score -> raw expressive token ids, via the same
        _expressive_mask mechanism dataset.py uses for reference_bars (not
        MidiGPTDataset's training-time appendix machinery -- simpler, and
        cross-checked against it in the test plan, not assumed identical)."""
        from midigpt.training.dataset import _expressive_mask

        ref_score = Score.from_dict(ref.score)
        tokens = engine._tokenizer.encode(copy.deepcopy(ref_score), compute_attributes=False)
        vocab = engine._tokenizer._vocab
        bars = (
            {(ref.track, b) for b in range(len(ref_score.tracks[ref.track].bars))}
            if ref.bars == "all"
            else {(ref.track, int(b)) for b in ref.bars}
        )
        mask = _expressive_mask(tokens, vocab, bars)
        return [t for t, m in zip(tokens, mask, strict=True) if m]

    def _resolve_engine(self, body: _GenerateBody):
        """Returns (engine, window_bars, valid_sizes, conditioning_summary)."""
        from steered_forward import compute_steer_vector, compute_steer_vector_c

        if body.conditioning is None:
            engine = self._engines.get_plain_engine(body.checkpoint)
            return engine, self._window_bars(engine), self._valid_sizes(engine), None

        c = body.conditioning
        if c.mechanism == "steering":
            base_engine = self._engines.get_plain_engine(body.checkpoint)
            layers = _resolve_layers(c.layers, base_engine._model.cfg.n_layer)
            wrapper, engine = self._engines.get_steered_engine(body.checkpoint, c.checkpoint, c.variant, layers)
            if c.variant == "A":
                from steered_forward import load_steering_source

                encoder, z_proj, style_cfg = load_steering_source(c.checkpoint, self._device or "cpu")
                style_vocab = StyleVocab(engine._tokenizer._vocab)
                if style_cfg.vocab_size != style_vocab.size:
                    raise ValueError(
                        f"conditioning checkpoint's StyleVocab size={style_cfg.vocab_size} != base's {style_vocab.size}"
                    )
                raw = self._derive_reference_tokens(base_engine, c.reference)
                style_ids = style_vocab.to_local_seq(raw)
                if not style_ids:
                    raise ValueError("reference produced no usable expressive tokens")
                steer_vec = compute_steer_vector(encoder, z_proj, style_ids, self._device or "cpu")
            else:
                from steered_forward import load_steering_source_c

                control_embeds, combine = load_steering_source_c(c.checkpoint, self._device or "cpu")
                if c.control_bins is not None:
                    bins = c.control_bins
                else:
                    ranges = ControlTypeRanges(base_engine._tokenizer._vocab)
                    raw = self._derive_reference_tokens(base_engine, c.reference) if c.reference else []
                    values = c.controls if c.controls is not None else compute_controls(raw, ranges)
                    if values is None:
                        raise ValueError("could not derive control values (no expressive tokens found)")
                    bins = quantize_controls(values)
                steer_vec = compute_steer_vector_c(control_embeds, combine, bins, self._device or "cpu")
            wrapper.steer_vec = steer_vec
            wrapper.alpha = c.alpha
            return engine, self._window_bars(engine), self._valid_sizes(engine), {"mechanism": "steering", "variant": c.variant, "alpha": c.alpha}

        # soft_prefix
        wrapper, engine = self._engines.get_soft_prefix_engine(c.checkpoint, c.variant)
        if c.variant == "A":
            style_vocab = StyleVocab(engine._tokenizer._vocab)
            raw = self._derive_reference_tokens(engine, c.reference)
            style_ids = style_vocab.to_local_seq(raw)
            if not style_ids:
                raise ValueError("reference produced no usable expressive tokens")
            wrapper.style_ids = torch.tensor([style_ids], dtype=torch.long)
            wrapper.style_mask = torch.ones(1, len(style_ids), dtype=torch.bool)
            wrapper.has_style = torch.ones(1, dtype=torch.bool)
        else:
            if c.control_bins is not None:
                bins = c.control_bins
            else:
                ranges = ControlTypeRanges(engine._tokenizer._vocab)
                raw = self._derive_reference_tokens(engine, c.reference) if c.reference else []
                values = c.controls if c.controls is not None else compute_controls(raw, ranges)
                if values is None:
                    raise ValueError("could not derive control values (no expressive tokens found)")
                bins = quantize_controls(values)
            wrapper.control_bins = torch.tensor([bins], dtype=torch.long)
            wrapper.has_controls = torch.ones(1, dtype=torch.bool)
        return engine, self._window_bars(engine), self._valid_sizes(engine), {"mechanism": "soft_prefix", "variant": c.variant}

    @staticmethod
    def _valid_sizes(engine) -> list[int]:
        import json

        cfg = json.loads(engine._tokenizer._vocab.config().to_json())
        return sorted(cfg.get("num_bars_map") or [4])

    def _window_bars(self, engine) -> int:
        return max(self._valid_sizes(engine))

    # ---- app ----

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            watchdog_task = asyncio.create_task(self._idle_watchdog()) if self._idle_timeout > 0 else None
            yield
            if watchdog_task is not None:
                watchdog_task.cancel()

        app = FastAPI(
            title="MIDI-GPT Flexible Humanize Server",
            description="Per-request checkpoint choice (including style-conditioning variants), autoregressive + humanize per track, per-bar mechanize control, and parquet MIDI retrieval.",
            version="0.1.0",
            lifespan=lifespan,
        )

        @app.get("/health", tags=["meta"])
        def health():
            self._touch()
            return {"status": "ok"}

        @app.post("/generate", tags=["generation"])
        async def generate(body: _GenerateBody):
            self._touch()
            if not (0.0 < body.top_p <= 1.0):
                raise HTTPException(400, "top_p must be in (0.0, 1.0]")
            if body.top_k < 0:
                raise HTTPException(400, "top_k must be >= 0")
            if body.max_attempts < 1:
                raise HTTPException(400, "max_attempts must be >= 1")
            try:
                score = Score.from_dict(body.score)
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, f"Invalid score: {exc}") from exc

            checkpoint_to_check = body.checkpoint or (body.conditioning.checkpoint if body.conditioning else None)
            if checkpoint_to_check and not Path(checkpoint_to_check).exists():
                raise HTTPException(400, f"checkpoint not found: {checkpoint_to_check}")
            if body.conditioning is not None and body.conditioning.mechanism == "steering":
                if not Path(body.conditioning.checkpoint).exists():
                    raise HTTPException(400, f"conditioning checkpoint not found: {body.conditioning.checkpoint}")

            t0 = time.perf_counter()
            async with self._semaphore:
                loop = asyncio.get_running_loop()
                try:
                    engine, window_bars, valid_sizes, cond_summary = await loop.run_in_executor(
                        None, self._resolve_engine, body
                    )
                    result = await loop.run_in_executor(
                        None, self._generate_full, score, body, engine, window_bars, valid_sizes
                    )
                except (RequestValidationError, ValueError) as exc:
                    raise HTTPException(422, str(exc)) from exc
                except Exception as exc:
                    log.exception("Generation failed")
                    raise HTTPException(500, str(exc)) from exc

            return {
                "score": result.to_dict(),
                "timing": {"total_s": time.perf_counter() - t0},
                "checkpoint": checkpoint_to_check,
                "conditioning": cond_summary,
            }

        @app.get("/parquet/info", tags=["dataset"])
        def parquet_info(path: str):
            self._touch()
            try:
                resolved = resolve_under_root(self._data_root, path)
                idx = self._parquets.get(resolved)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return {"path": str(resolved.relative_to(self._data_root.resolve())), "num_rows": idx.num_rows, "columns": idx.columns}

        @app.get("/parquet/row", tags=["dataset"])
        def parquet_row(path: str, index: int, format: Literal["score", "midi"] = "score"):
            self._touch()
            try:
                resolved = resolve_under_root(self._data_root, path)
                idx = self._parquets.get(resolved)
                row = idx.read_row(index)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except IndexError as exc:
                raise HTTPException(400, str(exc)) from exc
            try:
                score = Score.from_bytes(row["music"])
            except Exception as exc:
                raise HTTPException(422, f"row {index}: could not decode MIDI bytes: {exc}") from exc

            meta = {k: v for k, v in row.items() if k != "music"}
            if format == "score":
                return {
                    "score": score.to_dict(), "meta": meta, "row_index": index,
                    "source_path": str(resolved.relative_to(self._data_root.resolve())),
                }
            raw_midi = bytes(_core.MidiWriter().write_bytes(to_cpp(score)))
            filename = f"{resolved.stem}_row{index}.mid"
            return Response(
                content=raw_midi, media_type="audio/midi",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8002) -> None:
        uvicorn.run(self._app, host=host, port=port)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIDI-GPT flexible Humanize HTTP server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default=None, metavar="DEVICE", help='"cpu", "cuda", "mps", or None=auto')
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--idle-timeout", type=float, default=0, metavar="SECONDS")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--cache-size", type=int, default=4, help="Max distinct base checkpoints kept warm in memory")
    p.add_argument("--data-root", required=True, type=Path, help="Allowed root directory for /parquet/* retrieval")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not args.data_root.is_dir():
        raise SystemExit(f"--data-root {args.data_root} is not a directory")

    server = FlexServer(device=args.device, cache_size=args.cache_size, data_root=args.data_root, idle_timeout=args.idle_timeout)
    log.info("Starting flexible Humanize HTTP server on %s:%d (data_root=%s)", args.host, args.port, args.data_root)
    if args.idle_timeout:
        log.info("Auto-shutdown after %.0fs of inactivity.", args.idle_timeout)
    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
