from __future__ import annotations

import base64
import json
from dataclasses import asdict
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

import numpy as np

from .base import Teacher, TeacherResult


def _b64_jpeg(rgb_bgr: np.ndarray) -> str:
    """
    Encode as JPEG if OpenCV is available; otherwise fall back to raw bytes (larger).
    """
    try:
        import cv2  # type: ignore

        ok, buf = cv2.imencode(".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass

    return base64.b64encode(rgb_bgr.tobytes()).decode("ascii")


class HttpTeacher(Teacher):
    """
    Calls an external teacher server over HTTP so we don't take heavyweight deps.

    Expected request JSON:
      {
        "prompt": "...",
        "obs_dim": 128,
        "action_dim": 32,
        "rgb_b64": "...",
        "rgb_shape": [H, W, 3],
        "depth_b64": "...",        # optional
        "depth_shape": [H, W],     # optional
        "depth_dtype": "uint16",   # optional
        "context": {...}           # optional
      }

    Expected response JSON (any fields optional):
      {
        "embedding": [..],
        "caption": "...",
        "planner": {...},
        "action_command": [..],
        "extra": {...}
      }
    """

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)

    def infer(
        self,
        *,
        rgb_bgr: np.ndarray,
        depth: Optional[np.ndarray],
        prompt: str,
        obs_dim: int,
        action_dim: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TeacherResult:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "rgb_b64": _b64_jpeg(rgb_bgr),
            "rgb_shape": list(rgb_bgr.shape),
            "context": context or {},
        }
        if depth is not None:
            payload["depth_b64"] = base64.b64encode(depth.tobytes()).decode("ascii")
            payload["depth_shape"] = list(depth.shape)
            payload["depth_dtype"] = str(depth.dtype)

        req = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))

        # Coerce into TeacherResult
        result = TeacherResult(
            embedding=data.get("embedding"),
            caption=data.get("caption"),
            planner=data.get("planner"),
            action_command=data.get("action_command"),
            extra=data.get("extra"),
        )

        # Basic shape sanity
        if result.embedding is not None and len(result.embedding) != int(obs_dim):
            # If teacher returns wrong size, ignore it.
            result.embedding = None
        if result.action_command is not None and len(result.action_command) != int(action_dim):
            result.action_command = None

        return result


