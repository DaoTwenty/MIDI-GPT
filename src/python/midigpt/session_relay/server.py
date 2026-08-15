"""The session-relay: a stateful WebSocket relay sitting in front of the
stateless midigpt-http inference server. See the architecture sketch for
the full design (protocol, locking policy, worked example):
https://claude.ai/code/artifact/982da690-b1d3-432d-9130-8c6efc85feaa

Endpoints
---------
GET  /health                       liveness probe
GET  /models                       passthrough of midigpt-http's GET /models, short-TTL cached
POST /sessions                     create a session, returns {session_id, ws_url}
GET  /sessions/{session_id}/score  fetch the session's current full score (404 if unknown/dropped)
WS   /ws/{session_id}              join a session; first frame must be {"type": "join", ...}

Save/resume: sessions are NOT durably persisted while live. To resume one,
fetch its score (via GET /sessions/{id}/score before disconnecting, or from
the best-effort disk snapshot written when it becomes empty — see
SessionRegistry) and POST it back to /sessions as a fresh session's initial
score. A resumed session gets a new session_id with fresh participants/locks.

WebSocket protocol (JSON frames)
---------------------------------
Client -> relay:
  {"type": "join", "user": {"id"?, "name"?}}           first frame only
  {"type": "generate", "track", "bars": [a, b], "request": <GenerationRequest>, "model"?}
  {"type": "cancel", "request_id"}
  {"type": "presence", "track"?, "bar"?}
  {"type": "edit", "track", "bar", "ops": [{"op": "add"|"delete"|"move", ...}, ...]}

Relay -> clients:
  {"type": "state_sync", "score", "participants", "locks"}      to the joiner only
  {"type": "participant_joined"/"participant_left", "user_id", "name"?}
  {"type": "lock_acquired", "track", "bars", "holder", "request_id"}
  {"type": "notes_streaming", "request_id", "notes"}
  {"type": "generation_done"/"generation_cancelled", "request_id", "track", "bars", "score_patch", "model"}
  {"type": "generation_deferred", "request_id", "track", "bars", "attempt", "max_retries", "retry_after"}
  {"type": "generation_error", "request_id", "track", "bars", "error"}
  {"type": "generation_rejected", "track", "bars", "reason", "held_by"}   to the requester only
                                                                           (also used to reject a
                                                                           conflicting `edit`)
  {"type": "edit_applied", "track", "bars", "editor", "score_patch"}
  {"type": "presence", "user_id", "track", "bar"}
  {"type": "error", "error"}

"model" (on `generate`, optional): id of a model loaded on the midigpt-http
backend (see GET /models) to run this specific request against — omit to
use midigpt-http's own default_model. Per-request, not per-session/sticky.

"generation_deferred": midigpt-http returned 503 (its own server-wide
request cap was hit) for this attempt; the relay is retrying automatically
(holding the lock throughout) — broadcast to the whole session, not just
the requester, since it reflects backend-wide load. A `cancel` sent while
waiting between retries takes effect immediately rather than waiting out
the current retry_after.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn

from midigpt.session_relay.edit_ops import apply_edit_ops
from midigpt.session_relay.inference_client import InferenceClient
from midigpt.session_relay.models import Participant, Session, SessionRegistry
from midigpt.session_relay.score_merge import apply_patch, extract_patch

log = logging.getLogger(__name__)


class _CreateSessionBody(BaseModel):
    score: dict


class RelayServer:
    _MODELS_CACHE_TTL = 300.0  # seconds

    def __init__(
        self,
        inference_url: str = "http://localhost:8000",
        idle_timeout: float = 1800,
        busy_max_retries: int = 2,
        snapshot_dir: str | None = None,
    ) -> None:
        self._registry = SessionRegistry(idle_timeout=idle_timeout, snapshot_dir=snapshot_dir)
        self._inference = InferenceClient(inference_url)
        self._busy_max_retries = busy_max_retries
        # GET /models is static for the life of a midigpt-http process; this
        # cache is a backstop against re-fetching on every call, not real
        # invalidation logic (a midigpt-http restart with a different
        # --config just means up to _MODELS_CACHE_TTL seconds of staleness).
        self._models_cache: dict | None = None
        self._models_cache_at: float = 0.0
        self._app = self._build_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    # ---------------- broadcast ----------------

    async def _broadcast(self, session: Session, message: dict, exclude: str | None = None) -> None:
        dead = []
        for user_id, participant in list(session.participants.items()):
            if user_id == exclude:
                continue
            try:
                await participant.ws.send_json(message)
            except Exception:
                dead.append(user_id)
        for user_id in dead:
            session.participants.pop(user_id, None)

    # ---------------- generation ----------------

    async def _handle_generate(self, session: Session, participant: Participant, msg: dict) -> None:
        track = msg.get("track")
        bars = msg.get("bars")
        request = msg.get("request")
        model = msg.get("model")
        if not isinstance(track, int) or not bars or not isinstance(request, dict):
            await participant.ws.send_json({
                "type": "error",
                "error": "generate requires an int 'track', a non-empty 'bars' list, and a dict 'request'",
            })
            return
        if model is not None and not isinstance(model, str):
            await participant.ws.send_json({"type": "error", "error": "'model' must be a string if provided"})
            return
        bar_lo, bar_hi = min(bars), max(bars)

        existing = session.find_lock(track, bar_lo, bar_hi)
        if existing is not None:
            await participant.ws.send_json({
                "type": "generation_rejected",
                "track": track,
                "bars": [bar_lo, bar_hi],
                "reason": f"locked by {existing.holder}",
                "held_by": existing.holder,
            })
            return

        request_id = uuid.uuid4().hex
        session.acquire(track, bar_lo, bar_hi, participant.user_id, request_id)
        await self._broadcast(session, {
            "type": "lock_acquired",
            "track": track,
            "bars": [bar_lo, bar_hi],
            "holder": participant.user_id,
            "request_id": request_id,
        })

        # Runs independently of the WS message loop — a slow generation
        # must not block this participant (or anyone else) from sending
        # more messages, including a `cancel` for THIS request_id.
        asyncio.create_task(
            self._run_generation(session, track, bar_lo, bar_hi, request, request_id, model)
        )

    async def _stream_one_attempt(
        self, session: Session, track: int, bar_lo: int, bar_hi: int,
        request: dict, request_id: str, model: str | None,
    ) -> float | None:
        """Runs ONE attempt against midigpt-http. Returns None if a
        terminal event was reached (done/cancelled/error — already
        broadcast, lock already released by this call) or a retry_after
        (float, seconds) if midigpt-http returned 503 for this attempt —
        the lock is NOT released and nothing is broadcast in that case;
        the caller's retry loop (_run_generation) owns what happens next.
        """
        async for event in self._inference.generate_stream(
            session.score, request, request_id, model=model
        ):
            event_type = event.get("type")
            if event_type == "server_busy":
                return event.get("retry_after", 5.0)
            if event_type == "notes":
                await self._broadcast(session, {
                    "type": "notes_streaming",
                    "request_id": request_id,
                    "notes": event["notes"],
                })
            elif event_type in ("done", "cancelled"):
                response = event["response"]
                patch = None
                if response.get("score") is not None:
                    patch = extract_patch(response["score"], track, bar_lo, bar_hi)
                    apply_patch(session.score, patch)
                session.release(request_id)
                await self._broadcast(session, {
                    "type": "generation_done" if event_type == "done" else "generation_cancelled",
                    "request_id": request_id,
                    "track": track,
                    "bars": [bar_lo, bar_hi],
                    "score_patch": patch,
                    "model": response.get("model"),
                })
                return None
            elif event_type == "error":
                session.release(request_id)
                await self._broadcast(session, {
                    "type": "generation_error",
                    "request_id": request_id,
                    "track": track,
                    "bars": [bar_lo, bar_hi],
                    "error": event.get("error"),
                })
                return None
        # Stream ended without a terminal event — shouldn't happen given
        # midigpt-http's contract, but don't hang the lock forever if it does.
        session.release(request_id)
        await self._broadcast(session, {
            "type": "generation_error",
            "request_id": request_id,
            "track": track,
            "bars": [bar_lo, bar_hi],
            "error": "inference stream ended without a terminal event",
        })
        return None

    async def _run_generation(
        self, session: Session, track: int, bar_lo: int, bar_hi: int, request: dict,
        request_id: str, model: str | None = None,
    ) -> None:
        attempt = 0
        try:
            while True:
                retry_after = await self._stream_one_attempt(
                    session, track, bar_lo, bar_hi, request, request_id, model
                )
                if retry_after is None:
                    return  # terminal event already broadcast + lock released above

                attempt += 1
                if attempt > self._busy_max_retries:
                    session.release(request_id)
                    await self._broadcast(session, {
                        "type": "generation_error",
                        "request_id": request_id,
                        "track": track,
                        "bars": [bar_lo, bar_hi],
                        "error": f"inference server busy after {attempt - 1} retries",
                    })
                    return

                retry_after = min(retry_after, 30.0)  # don't trust an arbitrary upstream Retry-After
                await self._broadcast(session, {
                    "type": "generation_deferred",
                    "request_id": request_id,
                    "track": track,
                    "bars": [bar_lo, bar_hi],
                    "attempt": attempt,
                    "max_retries": self._busy_max_retries,
                    "retry_after": retry_after,
                })

                flag = session.cancel_flags.get(request_id)
                if flag is None:
                    await asyncio.sleep(retry_after)
                    continue
                sleep_task = asyncio.create_task(asyncio.sleep(retry_after))
                flag_task = asyncio.create_task(flag.wait())
                done, pending = await asyncio.wait(
                    {sleep_task, flag_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                if flag_task in done:
                    session.release(request_id)
                    await self._broadcast(session, {
                        "type": "generation_cancelled",
                        "request_id": request_id,
                        "track": track,
                        "bars": [bar_lo, bar_hi],
                        "score_patch": None,
                        "model": model,
                    })
                    return
        except Exception as exc:
            log.exception("generation relay failed (request_id=%s)", request_id)
            session.release(request_id)
            await self._broadcast(session, {
                "type": "generation_error",
                "request_id": request_id,
                "track": track,
                "bars": [bar_lo, bar_hi],
                "error": str(exc),
            })

    async def _handle_cancel(self, session: Session, participant: Participant, msg: dict) -> None:
        request_id = msg.get("request_id")
        if not request_id or request_id not in session.inflight:
            await participant.ws.send_json({
                "type": "error",
                "error": f"no in-flight request {request_id!r} in this session",
            })
            return
        # Any participant may cancel any in-flight request — it's a shared
        # session, not a private job queue (see the design sketch, §04).
        # Two cancellation paths, both signalled here: the local cancel_flag
        # wakes an in-progress busy-retry wait immediately (midigpt-http
        # gives no upstream cancel target for a request that got 503'd
        # before ever being admitted, so this is the only way to cancel one
        # mid-backoff — see the retry loop in _run_generation); the upstream
        # call cancels an already-admitted, actively-streaming generation.
        # Whichever path actually applies is the one that matters — the
        # lock release + generation_cancelled broadcast happens when
        # _run_generation observes it, not here, so an upstream 404 (the
        # expected outcome while still mid-backoff) isn't treated as a
        # failure worth reporting back.
        flag = session.cancel_flags.get(request_id)
        if flag is not None:
            flag.set()
        await self._inference.cancel(request_id)

    async def _handle_presence(self, session: Session, participant: Participant, msg: dict) -> None:
        participant.focus_track = msg.get("track")
        participant.focus_bar = msg.get("bar")
        await self._broadcast(session, {
            "type": "presence",
            "user_id": participant.user_id,
            "track": participant.focus_track,
            "bar": participant.focus_bar,
        }, exclude=participant.user_id)

    # ---------------- manual edits ----------------

    async def _handle_edit(self, session: Session, participant: Participant, msg: dict) -> None:
        """Direct manual note edits (add/delete/move), as opposed to a
        model `generate` call — e.g. a participant dragging a note in
        their piano-roll UI. Scoped to one bar per message (notes are
        addressed by their index in that bar's note list — Note has no
        persistent id). Unlike `generate`, the lock here is acquired,
        applied, broadcast, and released synchronously within this one
        call — it never spans an asyncio.create_task, since an edit is a
        single instantaneous mutation rather than a long-running stream.
        """
        track = msg.get("track")
        bar = msg.get("bar")
        ops = msg.get("ops")
        if not isinstance(track, int) or not isinstance(bar, int) or not isinstance(ops, list) or not ops:
            await participant.ws.send_json({
                "type": "error",
                "error": "edit requires an int 'track', an int 'bar', and a non-empty 'ops' list",
            })
            return

        tracks = session.score.get("tracks", [])
        if not (0 <= track < len(tracks)) or not (0 <= bar < len(tracks[track].get("bars", []))):
            await participant.ws.send_json({
                "type": "error",
                "error": f"no such (track={track}, bar={bar}) in this session's score",
            })
            return

        # Reuses the same Lock/find_lock machinery as `generate` — an edit
        # conflicting with an in-flight generation (or another edit, though
        # that race is impossible in practice since this whole handler runs
        # without ever awaiting between acquire and release) gets rejected
        # exactly like a conflicting generate would, via the same message
        # type (the client reaction is identical either way).
        existing = session.find_lock(track, bar, bar)
        if existing is not None:
            await participant.ws.send_json({
                "type": "generation_rejected",
                "track": track,
                "bars": [bar, bar],
                "reason": f"locked by {existing.holder}",
                "held_by": existing.holder,
            })
            return

        request_id = uuid.uuid4().hex
        session.acquire(track, bar, bar, participant.user_id, request_id)
        try:
            bar_dict = tracks[track]["bars"][bar]
            patched_bar, err = apply_edit_ops(bar_dict, ops)
            if err is not None:
                await participant.ws.send_json({"type": "error", "error": err})
                return
            patch = {"track": track, "bars": {bar: patched_bar}}
            apply_patch(session.score, patch)
            await self._broadcast(session, {
                "type": "edit_applied",
                "track": track,
                "bars": [bar, bar],
                "editor": participant.user_id,
                "score_patch": patch,
            })
        finally:
            session.release(request_id)

    async def _handle_message(self, session: Session, participant: Participant, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "generate":
            await self._handle_generate(session, participant, msg)
        elif msg_type == "cancel":
            await self._handle_cancel(session, participant, msg)
        elif msg_type == "presence":
            await self._handle_presence(session, participant, msg)
        elif msg_type == "edit":
            await self._handle_edit(session, participant, msg)
        else:
            await participant.ws.send_json({"type": "error", "error": f"unknown message type {msg_type!r}"})

    # ---------------- connection lifecycle ----------------

    async def _handle_ws(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        session = self._registry.get(session_id)
        if session is None:
            await websocket.send_json({"type": "error", "error": f"no session {session_id!r}"})
            await websocket.close(code=4404)
            return

        try:
            first = await websocket.receive_json()
        except Exception:
            await websocket.close(code=4400)
            return
        if not isinstance(first, dict) or first.get("type") != "join":
            await websocket.send_json({"type": "error", "error": "first message must be {'type': 'join', ...}"})
            await websocket.close(code=4400)
            return

        user = first.get("user") or {}
        user_id = str(user.get("id") or uuid.uuid4().hex)
        name = str(user.get("name") or user_id)
        # Reconnecting under the same user_id replaces the stale connection
        # rather than producing two participants for one person.
        participant = Participant(user_id=user_id, name=name, ws=websocket)
        session.participants[user_id] = participant
        session.last_activity = time.monotonic()

        await websocket.send_json({
            "type": "state_sync",
            "score": session.score,
            "participants": [
                {"user_id": p.user_id, "name": p.name} for p in session.participants.values()
            ],
            "locks": [
                {
                    "track": lock.track,
                    "bars": [lock.bar_lo, lock.bar_hi],
                    "holder": lock.holder,
                    "request_id": lock.request_id,
                }
                for lock in session.locks
            ],
        })
        await self._broadcast(
            session, {"type": "participant_joined", "user_id": user_id, "name": name}, exclude=user_id
        )

        try:
            while True:
                msg = await websocket.receive_json()
                session.last_activity = time.monotonic()
                if not isinstance(msg, dict):
                    await websocket.send_json({"type": "error", "error": "message must be a JSON object"})
                    continue
                await self._handle_message(session, participant, msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("ws loop error (session=%s user=%s)", session_id, user_id)
        finally:
            # Only remove if this connection is still the registered one —
            # a reconnect under the same user_id would already have
            # replaced `participant` in session.participants with a fresh
            # object, and we must not tear that down here.
            if session.participants.get(user_id) is participant:
                session.participants.pop(user_id, None)
                await self._broadcast(session, {"type": "participant_left", "user_id": user_id})
            self._registry.drop_if_empty(session_id)

    # ---------------- models ----------------

    async def _get_models(self) -> dict:
        now = time.monotonic()
        if self._models_cache is not None and (now - self._models_cache_at) < self._MODELS_CACHE_TTL:
            return self._models_cache
        payload = await self._inference.list_models()
        self._models_cache = payload
        self._models_cache_at = now
        return payload

    # ---------------- app wiring ----------------

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            watchdog = asyncio.create_task(self._registry.idle_watchdog())
            yield
            watchdog.cancel()
            await self._inference.aclose()

        app = FastAPI(
            title="MIDI-GPT Session Relay",
            description="Stateful WebSocket relay for multi-user collaborative sessions, "
            "fronting a stateless midigpt-http inference server.",
            lifespan=lifespan,
        )

        @app.get("/health", tags=["meta"])
        def health():
            return {"status": "ok"}

        @app.get("/models", tags=["meta"])
        async def list_models():
            return await self._get_models()

        @app.post("/sessions", tags=["sessions"])
        async def create_session(body: _CreateSessionBody):
            session = self._registry.create(body.score)
            return {"session_id": session.session_id, "ws_url": f"/ws/{session.session_id}"}

        @app.get("/sessions/{session_id}/score", tags=["sessions"])
        async def get_session_score(session_id: str):
            session = self._registry.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
            return {"score": session.score}

        @app.websocket("/ws/{session_id}")
        async def ws_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_ws(websocket, session_id)

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8100) -> None:
        uvicorn.run(self._app, host=host, port=port)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIDI-GPT session-relay server")
    p.add_argument(
        "--inference-url",
        default="http://localhost:8000",
        metavar="URL",
        help="Base URL of the midigpt-http inference server this relay fronts",
    )
    p.add_argument("--host", default="0.0.0.0", help="Host/IP to bind")
    p.add_argument("--port", type=int, default=8100, help="TCP port to listen on")
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=1800,
        metavar="SECONDS",
        help="Drop a session after it has had zero participants for this long",
    )
    p.add_argument(
        "--busy-max-retries",
        type=int,
        default=2,
        metavar="N",
        help="How many times to retry a generation that got a 503 (busy) from midigpt-http "
        "before giving up and reporting generation_error",
    )
    p.add_argument(
        "--snapshot-dir",
        default="./session_snapshots",
        metavar="PATH",
        help="Directory to write a best-effort score snapshot to when a session becomes empty "
        "(save/resume backstop — see module docstring). Pass '' to disable.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = RelayServer(
        inference_url=args.inference_url,
        idle_timeout=args.idle_timeout,
        busy_max_retries=args.busy_max_retries,
        snapshot_dir=args.snapshot_dir or None,
    )
    log.info(
        "Starting session-relay on %s:%d (inference=%s, idle_timeout=%.0fs, "
        "busy_max_retries=%d, snapshot_dir=%s)",
        args.host, args.port, args.inference_url, args.idle_timeout,
        args.busy_max_retries, args.snapshot_dir or "(disabled)",
    )
    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
