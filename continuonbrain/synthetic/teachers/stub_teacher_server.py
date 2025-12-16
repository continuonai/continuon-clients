"""
Stub teacher server (no heavy deps).

Purpose:
- Provide a local HTTP endpoint compatible with `HttpTeacher`.
- Emit a small "symbolic search / imagination" trace (planner + tool_calls) so
  RLDS episodes can contain supervision targets for advanced reasoning.

This is NOT a real VLA/LLM. It's a deterministic scaffold to prove the wiring:
- embedding: simple image-derived vector (mean pooled bytes -> obs_dim)
- caption: short heuristic caption
- planner: plan_steps that represent a symbolic search loop
- action_command: zeros

Run:
  python -m continuonbrain.synthetic.teachers.stub_teacher_server --host 0.0.0.0 --port 8099

Then record with:
  python -m continuonbrain.scripts.record_owner_realdepth_episode --teacher-url http://localhost:8099/infer ...
"""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Tuple

import numpy as np


def _parse_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


def _make_embedding(rgb_bytes: bytes, obs_dim: int) -> list[float]:
    # Cheap deterministic embedding: byte histogram-ish mean pooling
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8).astype(np.float32)
    if arr.size == 0:
        return [0.0] * obs_dim
    # Fold bytes into obs_dim buckets and normalize
    buckets = np.zeros(obs_dim, dtype=np.float32)
    idx = np.arange(arr.size) % obs_dim
    np.add.at(buckets, idx, arr)
    buckets /= (255.0 * (arr.size / obs_dim))
    return buckets.clip(0.0, 1.0).tolist()


def _infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    obs_dim = int(payload.get("obs_dim") or 128)
    action_dim = int(payload.get("action_dim") or 32)
    prompt = str(payload.get("prompt") or "").lower()

    rgb_b64 = payload.get("rgb_b64") or ""
    rgb_bytes = base64.b64decode(rgb_b64.encode("ascii")) if rgb_b64 else b""
    embedding = _make_embedding(rgb_bytes, obs_dim)

    # Minimal "symbolic search" plan scaffold.
    plan_steps = [
        "Observe the scene and detect the most salient human (owner candidate).",
        "Ask a clarifying question to confirm identity and intent.",
        "Extract the owner’s instruction and constraints.",
        "Search memory for relevant skills/tools; propose next action.",
        "Execute or request confirmation if safety/uncertainty is high.",
    ]

    caption = "A person is likely present; focus on identifying the owner and asking for guidance."
    if "owner" in prompt or "craig" in prompt:
        caption = "Owner-identification episode; be curious, polite, and ask for a concrete next task."

    return {
        "embedding": embedding,
        "caption": caption,
        "planner": {
            "intent": "owner_identity_and_guidance",
            "plan_steps": plan_steps,
            "selected_skill": "ask_clarifying_question",
            "confidence": 0.6,
        },
        "action_command": [0.0] * action_dim,
        "extra": {
            "symbolic_search.enabled": "true",
            "teacher.name": "stub_teacher_server",
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/infer":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        try:
            payload = _parse_body(self)
            out = _infer(payload)
            body = json.dumps(out).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Tuple[Any, ...]) -> None:  # noqa: D401
        # Keep console quiet by default.
        return


def main() -> None:
    p = argparse.ArgumentParser(description="Stub teacher server for RLDS enrichment.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    args = p.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Stub teacher listening on http://{args.host}:{args.port}/infer")
    server.serve_forever()


if __name__ == "__main__":
    main()


