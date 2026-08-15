"""In-memory state owned by the relay: sessions, participants, and the
per-(track, bar range) lock table. None of this is known to midigpt-http —
the inference server stays exactly as stateless as it already is.

Sessions are NOT durably persisted while live — see SessionRegistry's
snapshot-on-drop behavior below for the save/resume story (a best-effort
disk snapshot written when a session becomes empty, not a live store).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Lock:
    """One in-flight generation or edit's claim on a (track, bar range)."""

    track: int
    bar_lo: int
    bar_hi: int  # inclusive
    holder: str  # user_id
    request_id: str

    def overlaps(self, track: int, bar_lo: int, bar_hi: int) -> bool:
        return self.track == track and not (bar_hi < self.bar_lo or bar_lo > self.bar_hi)


@dataclass
class Participant:
    user_id: str
    name: str
    ws: object  # starlette.websockets.WebSocket — untyped here to keep this module fastapi-free
    focus_track: int | None = None
    focus_bar: int | None = None


@dataclass
class Session:
    session_id: str
    score: dict
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    participants: dict[str, Participant] = field(default_factory=dict)
    locks: list[Lock] = field(default_factory=list)
    # request_id -> Lock, so cancel/done/error handlers can release the
    # right lock without re-deriving (track, bar range) from the message.
    inflight: dict[str, Lock] = field(default_factory=dict)
    # request_id -> Event, set by a `cancel` message so a generation
    # currently sleeping through a busy-retry backoff (see RelayServer's
    # retry loop) can wake up and cancel immediately instead of waiting out
    # the full retry_after — midigpt-http gives no upstream cancel target
    # for a request that hasn't been admitted yet (503 fires before it
    # registers a cancel_events entry), so this has to be relay-local.
    cancel_flags: dict[str, asyncio.Event] = field(default_factory=dict)

    def find_lock(self, track: int, bar_lo: int, bar_hi: int) -> Lock | None:
        for lock in self.locks:
            if lock.overlaps(track, bar_lo, bar_hi):
                return lock
        return None

    def acquire(self, track: int, bar_lo: int, bar_hi: int, holder: str, request_id: str) -> Lock:
        lock = Lock(track, bar_lo, bar_hi, holder, request_id)
        self.locks.append(lock)
        self.inflight[request_id] = lock
        self.cancel_flags[request_id] = asyncio.Event()
        return lock

    def release(self, request_id: str) -> Lock | None:
        lock = self.inflight.pop(request_id, None)
        if lock is not None and lock in self.locks:
            self.locks.remove(lock)
        self.cancel_flags.pop(request_id, None)
        return lock


class SessionRegistry:
    """Holds every live Session in memory. A single relay process, single
    dict — a multi-instance deployment (multiple relay processes sharing
    session/lock state) isn't needed yet: this process already serves an
    unlimited number of concurrent *sessions* fine on its own, and nothing
    here requires the session/lock state itself to be durable while live —
    see the snapshot-on-drop behavior below for what does survive.
    """

    _IDLE_CHECK_INTERVAL = 30  # seconds between sweeps

    def __init__(self, idle_timeout: float = 1800, snapshot_dir: Path | str | None = None):
        self._sessions: dict[str, Session] = {}
        self._idle_timeout = idle_timeout
        # Best-effort disk snapshot written whenever a session becomes
        # empty (see _snapshot_and_drop) — the save/resume story. Not a
        # live store: nothing is written while a session is still active,
        # and nothing reads it back automatically. To resume, a client
        # reads the snapshot (or its own last-known score via
        # GET /sessions/{id}/score before disconnecting) and POSTs it back
        # as a fresh session's initial score. None disables snapshotting.
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self._snapshot_dir is not None:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create(self, initial_score: dict) -> Session:
        session_id = uuid.uuid4().hex
        session = Session(session_id=session_id, score=initial_score)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def _snapshot(self, session: Session) -> None:
        """Best-effort write of session.score to disk. Never raises — a
        failed snapshot (disk full, permissions) is operator-recoverable
        insurance, not the primary save path (see GET /sessions/{id}/score
        for that), and must never take down the relay.
        """
        if self._snapshot_dir is None:
            return
        path = self._snapshot_dir / f"{session.session_id}.json"
        try:
            path.write_text(json.dumps({
                "session_id": session.session_id,
                "score": session.score,
                "saved_at": time.time(),
            }))
        except OSError:
            log.exception("Could not write session snapshot to %s", path)

    def _snapshot_and_drop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._snapshot(session)

    def drop_if_empty(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None and not session.participants:
            self._snapshot_and_drop(session_id)

    async def idle_watchdog(self) -> None:
        """Drop sessions that have had zero participants for longer than
        idle_timeout. A session with at least one participant is never
        touched here regardless of how long it's been open — "idle" means
        "empty," not "quiet."
        """
        while True:
            await asyncio.sleep(self._IDLE_CHECK_INTERVAL)
            now = time.monotonic()
            dead = [
                sid
                for sid, session in self._sessions.items()
                if not session.participants and (now - session.last_activity) > self._idle_timeout
            ]
            for sid in dead:
                self._snapshot_and_drop(sid)
