"""Async client for the stateless midigpt-http inference server.

The relay is a pure consumer of midigpt-http's existing API — streaming
generation (`stream: true`), mid-flight cancellation
(`POST /generate/{request_id}/cancel`), model selection (an optional
`model` field, `GET /models`), and backpressure (`503` once the server's
admitted-request cap is hit) — all shipped independently of this service.
Nothing here requires or expects any change on that side.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


class InferenceClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=None)

    async def generate_stream(
        self, score: dict, request: dict, request_id: str, model: str | None = None
    ) -> AsyncIterator[dict]:
        """Yields the same event dicts midigpt-http's SSE stream carries —
        {"type": "notes", ...}, {"type": "done"/"cancelled", "response": {...}},
        or {"type": "error", ...} — parsed straight off the wire, plus one
        synthetic event type this client adds itself: {"type": "server_busy",
        "request_id", "retry_after"} when midigpt-http responds 503 (its
        server-wide admitted-request cap was hit before the stream even
        opened) — kept distinct from "error" so the caller can retry instead
        of failing outright. Frames are `data: <json>\\n\\n`, exactly what
        http/server.py's _stream_generate emits, so no generic SSE library
        is needed for this one shape.

        `model` selects which loaded model midigpt-http runs this request
        against (an id from GET /models) — omit to use its default_model.
        """
        payload = {"score": score, "request": request, "request_id": request_id, "stream": True}
        if model is not None:
            payload["model"] = model
        async with self._client.stream("POST", f"{self._base_url}/generate", json=payload) as resp:
            if resp.status_code == 503:
                retry_after = 5.0
                try:
                    retry_after = float(resp.headers.get("Retry-After", retry_after))
                except ValueError:
                    pass
                yield {"type": "server_busy", "request_id": request_id, "retry_after": retry_after}
                return
            if resp.status_code != 200:
                body = await resp.aread()
                yield {
                    "type": "error",
                    "request_id": request_id,
                    "error": f"inference server returned HTTP {resp.status_code}: "
                    f"{body.decode(errors='replace')}",
                }
                return
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    frame = frame.strip()
                    if frame.startswith("data:"):
                        yield json.loads(frame[len("data:") :].strip())

    async def cancel(self, request_id: str) -> bool:
        resp = await self._client.post(f"{self._base_url}/generate/{request_id}/cancel")
        return resp.status_code == 200

    async def list_models(self) -> dict:
        """Passthrough of midigpt-http's GET /models —
        {"default_model": str, "models": [{"id", "checkpoint"}, ...]}."""
        resp = await self._client.get(f"{self._base_url}/models")
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
