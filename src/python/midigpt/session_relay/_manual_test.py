"""Manual multi-client smoke test against REAL running servers — not a
pytest suite, a throwaway script to exercise the actual WebSocket protocol
end to end.

Usage:
    .venv/bin/python -m midigpt.session_relay._manual_test

Assumes a relay is already running on localhost:8100, fronting a
midigpt-http on localhost:8000 that has been launched from this repo root
(the save/resume scenario checks for a disk snapshot under
./session_snapshots, relative to wherever the relay process's cwd was).
For the model-selection scenario to actually exercise model routing (not
just skip), midigpt-http needs >1 model loaded (e.g. "yellow_medium" +
"yellow_small").

The busy/retry scenario is self-contained: it launches its own throwaway
midigpt-http (max_queue=1) and a relay pointed at it, on scratch ports, so
a 503 can be forced deterministically without touching the servers above.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile

import httpx
import websockets
import yaml

RELAY = "http://localhost:8100"
RELAY_WS = "ws://localhost:8100"
SNAPSHOT_DIR = os.environ.get("SESSION_RELAY_SNAPSHOT_DIR", "session_snapshots")


async def recv_until(ws, want_type: str, timeout: float = 30.0) -> dict:
    """Read frames until one with type == want_type shows up; returns it.
    Prints every frame seen along the way so failures are legible."""
    async with asyncio.timeout(timeout):
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            print(f"  <- {msg.get('type')}: "
                  f"{ {k: v for k, v in msg.items() if k not in ('score', 'score_patch')} }")
            if msg.get("type") == want_type:
                return msg


async def drain_notes(ws, request_id: str, final_types=("generation_done", "generation_cancelled", "generation_error")):
    """Consume notes_streaming/generation_deferred events for request_id
    until one of the final event types arrives; returns that final event."""
    note_count = 0
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("type") == "notes_streaming" and msg.get("request_id") == request_id:
            note_count += len(msg["notes"])
            continue
        if msg.get("type") in final_types:
            print(f"  streamed {note_count} notes before {msg['type']}")
            return msg
        print(f"  (ignoring unrelated frame: {msg.get('type')})")


def _request_for_bars(base_request: dict, track: int, bars: list[int]) -> dict:
    """Clone base_request but target exactly one track's bars=`bars`,
    everything else emptied out — keeps single-scenario generations short
    (the shared fixture's own request targets all 8 bars of one track)."""
    req = json.loads(json.dumps(base_request))
    for tp in req["tracks"]:
        if tp["id"] == track:
            tp["bars"] = bars
            tp["autoregressive"] = True
        else:
            tp["bars"] = []
            tp["autoregressive"] = False
    return req


# ---------------- scenarios against the live relay + midigpt-http ----------------

async def scenario_core(alice, bob, base_req) -> str:
    """Original lock/generate/cancel scenario. Returns alice's session_id
    isn't known here (caller already created it) — kept for reference."""
    gen_track4 = {
        "type": "generate",
        "track": 4,
        "bars": list(range(8)),
        "request": base_req["request"],
    }
    await alice.send(json.dumps(gen_track4))
    lock_a = await recv_until(alice, "lock_acquired")
    req_id_a = lock_a["request_id"]
    assert lock_a["holder"] == "alice"
    print(f"PASS: alice's generate acquired lock, request_id={req_id_a}")

    lock_seen_by_bob = await recv_until(bob, "lock_acquired")
    assert lock_seen_by_bob["request_id"] == req_id_a
    print("PASS: bob saw alice's lock via broadcast")

    await bob.send(json.dumps(gen_track4))
    rejected = await recv_until(bob, "generation_rejected")
    assert rejected["held_by"] == "alice"
    print("PASS: bob's overlapping generate was rejected (held_by=alice)")

    done_a = await drain_notes(alice, req_id_a)
    assert done_a["type"] == "generation_done"
    assert done_a["score_patch"]["track"] == 4
    print(f"PASS: alice's generation completed, patch has {len(done_a['score_patch']['bars'])} bar(s)")

    done_seen_by_bob = await drain_notes(bob, req_id_a)
    assert done_seen_by_bob["type"] == "generation_done"
    print("PASS: bob saw the same completion via broadcast")

    await bob.send(json.dumps(gen_track4))
    lock_b = await recv_until(bob, "lock_acquired")
    req_id_b = lock_b["request_id"]
    await recv_until(alice, "lock_acquired")

    await asyncio.sleep(1.0)
    await bob.send(json.dumps({"type": "cancel", "request_id": req_id_b}))
    cancelled = await drain_notes(bob, req_id_b)
    assert cancelled["type"] == "generation_cancelled"
    print("PASS: bob's own cancel produced generation_cancelled")

    await alice.send(json.dumps(gen_track4))
    relock = await recv_until(alice, "lock_acquired")
    assert relock["holder"] == "alice"
    print("PASS: lock was released after cancel — alice re-acquired it")
    await alice.send(json.dumps({"type": "cancel", "request_id": relock["request_id"]}))
    await drain_notes(alice, relock["request_id"])

    await alice.send(json.dumps({"type": "bogus"}))
    err = await recv_until(alice, "error")
    assert "unknown message type" in err["error"]
    print("PASS: unknown message type produced a clean error, not a crash")


async def scenario_model_selection(alice, base_req) -> None:
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{RELAY}/models")
        resp.raise_for_status()
        models_resp = resp.json()
    non_default = [m["id"] for m in models_resp["models"] if m["id"] != models_resp["default_model"]]
    if not non_default:
        print("SKIP: model-selection scenario needs >1 model loaded on midigpt-http")
        return
    model = non_default[0]

    gen = {
        "type": "generate",
        "track": 4,
        "bars": [7],
        "request": _request_for_bars(base_req["request"], 4, [7]),
        "model": model,
    }
    await alice.send(json.dumps(gen))
    lock = await recv_until(alice, "lock_acquired")
    done = await drain_notes(alice, lock["request_id"])
    assert done["type"] == "generation_done"
    assert done["model"] == model, f"expected model={model!r}, got {done.get('model')!r}"
    print(f"PASS: generate with model={model!r} echoed back correctly")


async def scenario_edit(alice, bob, base_req) -> None:
    note = {"pitch": 60, "velocity": 90, "onset_ticks": 0, "duration_ticks": 24, "delta": 0}
    await alice.send(json.dumps({
        "type": "edit", "track": 1, "bar": 0,
        "ops": [{"op": "add", "note": note}],
    }))
    applied = await recv_until(alice, "edit_applied")
    assert applied["track"] == 1 and applied["bars"] == [0, 0]
    assert applied["score_patch"]["bars"]["0"]["notes"], "expected the added note in the patch"
    print("PASS: edit applied, patch contains the added note")

    seen = await recv_until(bob, "edit_applied")
    assert seen["editor"] == "alice"
    print("PASS: bob saw alice's edit via broadcast")

    # edit-vs-generate contention: bob locks track 4 bar 7 via generate
    # (bar 7 = a valid single-bar right-suffix for an autoregressive track —
    # see _request_for_bars), alice's edit on the same (track, bar) must be
    # rejected.
    gen = {
        "type": "generate", "track": 4, "bars": [7],
        "request": _request_for_bars(base_req["request"], 4, [7]),
    }
    await bob.send(json.dumps(gen))
    lock = await recv_until(bob, "lock_acquired")
    await recv_until(alice, "lock_acquired")

    await alice.send(json.dumps({
        "type": "edit", "track": 4, "bar": 7,
        "ops": [{"op": "add", "note": note}],
    }))
    rejected = await recv_until(alice, "generation_rejected")
    assert rejected["held_by"] == "bob"
    print("PASS: edit correctly rejected while a generation holds the same range")

    await bob.send(json.dumps({"type": "cancel", "request_id": lock["request_id"]}))
    await drain_notes(bob, lock["request_id"])

    # generate-vs-edit contention, the reverse direction: alice edits track
    # 2 bar 7 (holds the lock synchronously — too fast for bob to race it
    # meaningfully, but confirm bob's generate on the SAME range afterward
    # sees a clean, unlocked state — i.e. the edit's lock was released).
    await alice.send(json.dumps({
        "type": "edit", "track": 2, "bar": 7,
        "ops": [{"op": "add", "note": note}],
    }))
    await recv_until(alice, "edit_applied")
    gen2 = {
        "type": "generate", "track": 2, "bars": [7],
        "request": _request_for_bars(base_req["request"], 2, [7]),
    }
    await bob.send(json.dumps(gen2))
    lock2 = await recv_until(bob, "lock_acquired")
    assert lock2["holder"] == "bob", "lock from a completed edit should not linger"
    print("PASS: edit's lock was released — a subsequent generate on the same range succeeded")
    await bob.send(json.dumps({"type": "cancel", "request_id": lock2["request_id"]}))
    await drain_notes(bob, lock2["request_id"])


async def scenario_get_score(session_id: str) -> dict:
    """Fetches the session's current score via the client-driven save path.
    Must be called while the session still has >=1 participant — a session
    with zero participants is dropped (and snapshotted, see
    scenario_resume_from_snapshot) immediately, so GET would 404."""
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{RELAY}/sessions/{session_id}/score")
        resp.raise_for_status()
        saved_score = resp.json()["score"]
    assert saved_score["tracks"], "fetched score should have tracks"
    print(f"PASS: GET /sessions/{session_id}/score returned a full score")
    return saved_score


async def scenario_resume_from_snapshot(session_id: str, expected_score: dict) -> None:
    """Must be called AFTER every participant of `session_id` has
    disconnected — checks the relay's auto-snapshot-on-empty backstop, then
    resumes from it via the ordinary POST /sessions score payload."""
    await asyncio.sleep(0.5)  # give the disconnect-triggered cleanup a moment to run
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{session_id}.json")
    assert os.path.exists(snapshot_path), f"expected snapshot at {snapshot_path}"
    with open(snapshot_path) as f:
        snap = json.load(f)
    assert snap["session_id"] == session_id
    assert snap["score"]["tracks"] == expected_score["tracks"]
    print(f"PASS: disk snapshot written to {snapshot_path} on session-empty")

    async with httpx.AsyncClient() as http:
        resp = await http.post(f"{RELAY}/sessions", json={"score": snap["score"]})
        resp.raise_for_status()
        new_session_id = resp.json()["session_id"]
    print(f"PASS: resumed as new session {new_session_id}")

    ws_url = f"{RELAY_WS}/ws/{new_session_id}"
    async with websockets.connect(ws_url) as carol:
        await carol.send(json.dumps({"type": "join", "user": {"id": "carol", "name": "Carol"}}))
        sync = await recv_until(carol, "state_sync")
        assert sync["score"]["tracks"] == expected_score["tracks"], \
            "resumed session's score should match the saved snapshot"
    print("PASS: resumed session's state_sync matches the saved score")


async def main() -> None:
    base_req = json.load(open("/tmp/rhythm_test_isolate.json"))
    score = base_req["score"]

    async with httpx.AsyncClient() as http:
        resp = await http.post(f"{RELAY}/sessions", json={"score": score})
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        print(f"created session {session_id}")

    ws_url = f"{RELAY_WS}/ws/{session_id}"

    async with websockets.connect(ws_url) as alice, websockets.connect(ws_url) as bob:
        await alice.send(json.dumps({"type": "join", "user": {"id": "alice", "name": "Alice"}}))
        sync = await recv_until(alice, "state_sync")
        assert sync["score"]["tracks"], "state_sync should carry the initial score"
        print("PASS: alice joined, got state_sync")

        await bob.send(json.dumps({"type": "join", "user": {"id": "bob", "name": "Bob"}}))
        await recv_until(bob, "state_sync")
        joined = await recv_until(alice, "participant_joined")
        assert joined["user_id"] == "bob"
        print("PASS: bob joined, alice was notified")

        print("\n--- core lock/generate/cancel scenario ---")
        await scenario_core(alice, bob, base_req)

        print("\n--- model selection scenario ---")
        await scenario_model_selection(alice, base_req)

        print("\n--- edit scenario ---")
        await scenario_edit(alice, bob, base_req)

        print("\n--- save/resume scenario (fetch while still connected) ---")
        current_score = await scenario_get_score(session_id)

    # alice and bob have now both disconnected (the `async with` block
    # above exited) -> session_id should be empty and dropped+snapshotted.
    print("\n--- save/resume scenario (snapshot + resume after disconnect) ---")
    await scenario_resume_from_snapshot(session_id, current_score)

    print("\nALL LIVE-SERVER CHECKS PASSED")


# ---------------- self-contained busy/retry scenario ----------------

async def _wait_healthy(url: str, timeout: float = 60.0) -> None:
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient() as http:
            while True:
                try:
                    resp = await http.get(url)
                    if resp.status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.5)


async def scenario_busy_retry() -> None:
    """Self-contained: launches a throwaway midigpt-http (max_queue=1) and
    a relay pointed at it, on scratch ports, so a 503 can be forced
    deterministically. Exercises generation_deferred + cancel-during-
    busy-wait, neither of which the live-server scenarios above can hit
    reliably (they'd need to actually saturate a shared production server).
    """
    port_inf, port_relay = 8091, 8191
    cfg = {
        "port": port_inf,
        "device": "cpu",
        "max_queue": 1,
        "default_model": "yellow_medium",
        "models": [{"id": "yellow_medium", "pretrained": "yellow_medium"}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        cfg_path = f.name

    inf_proc = subprocess.Popen(
        [".venv/bin/midigpt-http", "--config", cfg_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    relay_proc = None
    try:
        await _wait_healthy(f"http://localhost:{port_inf}/health")
        relay_proc = subprocess.Popen(
            [
                ".venv/bin/python", "-m", "midigpt.session_relay",
                "--inference-url", f"http://localhost:{port_inf}",
                "--port", str(port_relay),
                "--busy-max-retries", "1",
                "--snapshot-dir", "",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        await _wait_healthy(f"http://localhost:{port_relay}/health")

        base_req = json.load(open("/tmp/rhythm_test_isolate.json"))
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"http://localhost:{port_relay}/sessions", json={"score": base_req["score"]}
            )
            resp.raise_for_status()
            session_id = resp.json()["session_id"]

        ws_url = f"ws://localhost:{port_relay}/ws/{session_id}"
        async with websockets.connect(ws_url) as x, websockets.connect(ws_url) as y:
            await x.send(json.dumps({"type": "join", "user": {"id": "x"}}))
            await recv_until(x, "state_sync")
            await y.send(json.dumps({"type": "join", "user": {"id": "y"}}))
            await recv_until(y, "state_sync")
            await recv_until(x, "participant_joined")

            # x takes the only admission slot on the throwaway midigpt-http
            # (max_queue=1) with a long-ish generation on track 4.
            await x.send(json.dumps({
                "type": "generate", "track": 4, "bars": list(range(8)),
                "request": _request_for_bars(base_req["request"], 4, list(range(8))),
            }))
            lock_x = await recv_until(x, "lock_acquired")
            await recv_until(y, "lock_acquired")

            # y's generate (different track -> no relay-side lock conflict)
            # should get 503'd by midigpt-http and deferred by the relay.
            await y.send(json.dumps({
                "type": "generate", "track": 3, "bars": [7],
                "request": _request_for_bars(base_req["request"], 3, [7]),
            }))
            lock_y = await recv_until(y, "lock_acquired")
            await recv_until(x, "lock_acquired")
            deferred = await recv_until(y, "generation_deferred", timeout=30.0)
            assert deferred["request_id"] == lock_y["request_id"]
            assert deferred["attempt"] == 1
            print(f"PASS: y's generate was deferred (busy) — retry_after={deferred['retry_after']}")

            # cancel while y's request is sleeping through the retry wait —
            # should resolve immediately, not after the full retry_after.
            start = asyncio.get_event_loop().time()
            await y.send(json.dumps({"type": "cancel", "request_id": lock_y["request_id"]}))
            cancelled = await recv_until(y, "generation_cancelled", timeout=10.0)
            elapsed = asyncio.get_event_loop().time() - start
            assert cancelled["request_id"] == lock_y["request_id"]
            assert elapsed < deferred["retry_after"], (
                f"cancel took {elapsed:.1f}s, should short-circuit well under "
                f"retry_after={deferred['retry_after']}"
            )
            print(f"PASS: cancel during busy-retry wait resolved in {elapsed:.1f}s "
                  f"(< retry_after={deferred['retry_after']}) instead of waiting it out")

            # y's lock should be released — a fresh generate on the same
            # range should succeed in acquiring it.
            await y.send(json.dumps({
                "type": "generate", "track": 3, "bars": [7],
                "request": _request_for_bars(base_req["request"], 3, [7]),
            }))
            relock = await recv_until(y, "lock_acquired")
            assert relock["holder"] == "y"
            print("PASS: lock was released after busy-wait cancel — y re-acquired it")
            await y.send(json.dumps({"type": "cancel", "request_id": relock["request_id"]}))
            await drain_notes(y, relock["request_id"])

            await x.send(json.dumps({"type": "cancel", "request_id": lock_x["request_id"]}))
            await drain_notes(x, lock_x["request_id"])

        print("PASS: busy/retry scenario complete")
    finally:
        if relay_proc is not None:
            relay_proc.terminate()
            try:
                relay_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                relay_proc.kill()
        inf_proc.terminate()
        try:
            inf_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            inf_proc.kill()
        os.unlink(cfg_path)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n--- busy/retry scenario (self-contained, own throwaway servers) ---")
        asyncio.run(scenario_busy_retry())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
