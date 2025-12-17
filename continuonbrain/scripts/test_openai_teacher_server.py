"""
Quick connectivity test for an OpenAI-compatible teacher backend.

This helps validate local servers like vLLM/OpenWebUI/LM Studio before recording
teacher-enriched RLDS episodes.
"""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def _get(url: str, headers: dict[str, str], timeout_s: float) -> tuple[int, str]:
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return int(getattr(resp, "status", 200)), body


def _post(url: str, headers: dict[str, str], payload: dict, timeout_s: float) -> tuple[int, str]:
    req = Request(
        url,
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return int(getattr(resp, "status", 200)), body


def main() -> None:
    p = argparse.ArgumentParser(description="Test OpenAI-compatible server endpoints for teacher use.")
    p.add_argument("--base-url", required=True, help="Example: http://localhost:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--embed-model", default=None)
    p.add_argument("--chat-model", default=None)
    p.add_argument("--timeout-s", type=float, default=5.0)
    args = p.parse_args()

    base = str(args.base_url).rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"base_url={base}")

    try:
        code, body = _get(f"{base}/v1/models", headers=headers, timeout_s=args.timeout_s)
        print(f"GET /v1/models -> {code}")
        print(body[:800])
    except Exception as exc:  # noqa: BLE001
        print(f"GET /v1/models failed: {exc}")

    if args.embed_model:
        try:
            payload = {"model": args.embed_model, "input": "hello teacher"}
            code, body = _post(f"{base}/v1/embeddings", headers=headers, payload=payload, timeout_s=args.timeout_s)
            print(f"POST /v1/embeddings -> {code}")
            print(body[:800])
        except Exception as exc:  # noqa: BLE001
            print(f"POST /v1/embeddings failed: {exc}")
    else:
        print("skip embeddings (no --embed-model)")

    if args.chat_model:
        try:
            payload = {
                "model": args.chat_model,
                "messages": [{"role": "user", "content": "Return JSON only: {\"ok\": true}"}],
                "temperature": 0.0,
            }
            code, body = _post(
                f"{base}/v1/chat/completions", headers=headers, payload=payload, timeout_s=args.timeout_s
            )
            print(f"POST /v1/chat/completions -> {code}")
            print(body[:800])
        except Exception as exc:  # noqa: BLE001
            print(f"POST /v1/chat/completions failed: {exc}")
    else:
        print("skip chat (no --chat-model)")


if __name__ == "__main__":
    main()


