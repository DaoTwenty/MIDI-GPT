#!/usr/bin/env python3
"""Tiny local web app for reviewing ContextView scenarios.

Run:
    python scripts/context_review/server.py [--port 5051]

Open http://localhost:5051/ in a browser. Each click of "New scenario"
rolls a fresh random config via `gen_context_scenario.roll()` and
renders the bar-state grid with the same diagonal-stripe styling the
studio uses. Type feedback in the textarea, click "Save", and it gets
appended to `docs/context_review_feedback.jsonl`. Later, just tell
Claude "analyze docs/context_review_feedback.jsonl" instead of pasting
scenarios back and forth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Allow importing the sibling generator without installing the package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gen_context_scenario import (  # noqa: E402
    MAX_TICKS,
    _roll_base,
    apply_playhead,
    scenario_to_dict,
)


def _scenario_payload(base: dict, playhead: int) -> dict:
    s = apply_playhead(base, playhead)
    payload = scenario_to_dict(s)
    payload["initial_playhead"] = base["initial_playhead"]
    L = base["lookahead"]; N = base["num_anticip"]
    first_gen_ph = max(0, base["buffer_bars"] - L) \
        if base["adapt_buffer"] else base["buffer_bars"]
    payload["max_playhead"] = first_gen_ph + MAX_TICKS * N - 1
    return payload

REPO_ROOT     = HERE.parent.parent
STATIC_DIR    = HERE / "static"
FEEDBACK_PATH = REPO_ROOT / "docs" / "context_review_feedback.jsonl"


class Handler(BaseHTTPRequestHandler):
    # Keep the console quiet — only show our own log lines.
    def log_message(self, fmt, *args):  # noqa: A003
        return

    # -- helpers ----------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel: str) -> None:
        path = (STATIC_DIR / rel).resolve()
        if not str(path).startswith(str(STATIC_DIR)) or not path.is_file():
            self.send_error(404, "Not found")
            return
        mime = {
            ".html": "text/html; charset=utf-8",
            ".js":   "application/javascript; charset=utf-8",
            ".css":  "text/css; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routes -----------------------------------------------------------
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_static("index.html")
            return
        if parsed.path == "/app.js":
            self._send_static("app.js")
            return
        if parsed.path == "/style.css":
            self._send_static("style.css")
            return
        if parsed.path == "/roll":
            qs   = parse_qs(parsed.query)
            seed = int(qs["seed"][0]) if "seed" in qs else None
            base = _roll_base(seed)
            self._send_json(_scenario_payload(base, base["initial_playhead"]))
            return
        if parsed.path == "/state":
            qs   = parse_qs(parsed.query)
            if "seed" not in qs or "playhead" not in qs:
                self._send_json({"error": "seed and playhead required"}, status=400)
                return
            seed = int(qs["seed"][0])
            ph   = max(0, int(qs["playhead"][0]))
            base = _roll_base(seed)
            self._send_json(_scenario_payload(base, ph))
            return
        if parsed.path == "/feedback":
            entries = []
            if FEEDBACK_PATH.exists():
                with FEEDBACK_PATH.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            self._send_json({"entries": entries})
            return
        self.send_error(404, "Not found")

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/feedback":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, status=400)
            return

        entry = {
            "ts":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed":     body.get("seed"),
            "params":   body.get("params"),
            "derived":  body.get("derived"),
            "grid":     body.get("grid"),
            "feedback": (body.get("feedback") or "").strip(),
        }
        if not entry["feedback"]:
            self._send_json({"error": "feedback is empty"}, status=400)
            return

        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[review] saved feedback for seed={entry['seed']} "
              f"→ {FEEDBACK_PATH}")
        self._send_json({"ok": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5051)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"[review] serving http://{args.host}:{args.port}/  "
          f"(feedback → {FEEDBACK_PATH.relative_to(REPO_ROOT)})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] stopped")


if __name__ == "__main__":
    main()
