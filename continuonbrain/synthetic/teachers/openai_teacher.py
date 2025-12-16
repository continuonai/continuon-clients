from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

import numpy as np

from .base import Teacher, TeacherResult


def _b64_jpeg(rgb_bgr: np.ndarray) -> str:
    try:
        import cv2  # type: ignore

        ok, buf = cv2.imencode(".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass
    return base64.b64encode(rgb_bgr.tobytes()).decode("ascii")


class OpenAITeacher(Teacher):
    """
    OpenAI-compatible teacher. Works with vLLM/OpenWebUI/any server exposing:
      - POST {base_url}/v1/embeddings
      - POST {base_url}/v1/chat/completions  (optional, for planner/caption)

    We keep this dependency-light (stdlib urllib) and tolerant of missing endpoints.
    """

    def __init__(
        self,
        *,
        base_url: str,
        embed_base_url: Optional[str] = None,
        chat_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embed_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        timeout_s: float = 10.0,
        allow_insecure_http: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_base_url = (embed_base_url or base_url).rstrip("/")
        self.chat_base_url = (chat_base_url or base_url).rstrip("/")
        self.api_key = api_key
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.timeout_s = float(timeout_s)
        self.allow_insecure_http = bool(allow_insecure_http)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))

    def _get_embedding(self, *, rgb_bgr: np.ndarray, obs_dim: int) -> Optional[List[float]]:
        # In the upgraded flow, embeddings come from caption text (not raw image bytes).
        # Keep a legacy fallback only if caption embedding isn't available.
        if not self.embed_model:
            return None
        rgb_b64 = _b64_jpeg(rgb_bgr)
        payload = {"model": self.embed_model, "input": f"image_b64_hint:{rgb_b64[:2000]}"}
        data = self._post_json(f"{self.embed_base_url}/v1/embeddings", payload)
        vec = None
        try:
            vec = data["data"][0]["embedding"]
        except Exception:
            return None
        if not isinstance(vec, list) or not vec:
            return None

        arr = np.array(vec, dtype=np.float32).flatten()
        # Pad/truncate to obs_dim for direct use as observation.command
        if arr.size > obs_dim:
            arr = arr[:obs_dim]
        elif arr.size < obs_dim:
            arr = np.pad(arr, (0, obs_dim - arr.size))
        return arr.astype(np.float32).tolist()

    def _get_chat_json(self, *, prompt: str, rgb_bgr: np.ndarray) -> Dict[str, Any]:
        if not self.chat_model:
            return {}

        rgb_b64 = _b64_jpeg(rgb_bgr)
        # OpenAI multimodal schema (works with vLLM for vision models and many OpenAI-compatible servers)
        messages = [
            {"role": "system", "content": "Return STRICT JSON only. No markdown. No prose outside JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_b64}"}},
                ],
            },
        ]
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.2,
        }
        try:
            data = self._post_json(f"{self.chat_base_url}/v1/chat/completions", payload)
        except Exception:
            # Fallback for servers that don't support multimodal content arrays.
            messages2 = [
                {"role": "system", "content": "Return STRICT JSON only. No markdown."},
                {"role": "user", "content": f"{prompt}\n\nimage_b64:{rgb_b64[:4000]}"},
            ]
            payload2 = {"model": self.chat_model, "messages": messages2, "temperature": 0.2}
            data = self._post_json(f"{self.chat_base_url}/v1/chat/completions", payload2)
        text = ""
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {"caption": text[:500]}

    def _embed_text(self, *, text: str, obs_dim: int) -> Optional[List[float]]:
        if not self.embed_model:
            return None
        payload = {"model": self.embed_model, "input": text}
        data = self._post_json(f"{self.embed_base_url}/v1/embeddings", payload)
        try:
            vec = data["data"][0]["embedding"]
        except Exception:
            return None
        if not isinstance(vec, list) or not vec:
            return None
        arr = np.array(vec, dtype=np.float32).flatten()
        if arr.size > obs_dim:
            arr = arr[:obs_dim]
        elif arr.size < obs_dim:
            arr = np.pad(arr, (0, obs_dim - arr.size))
        return arr.astype(np.float32).tolist()

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
        embedding: Optional[List[float]] = None
        caption = None
        planner = None
        tool_calls = None

        # Best-effort planner/caption/tool traces
        extra: Dict[str, Any] = {"teacher.backend": "openai_compatible"}
        if context:
            extra["teacher.context"] = context

        try:
            out = self._get_chat_json(
                prompt=(
                    "Given an owner-identity training episode, produce JSON with optional keys:\n"
                    "caption (string), planner (object: intent, plan_steps[], selected_skill, confidence),\n"
                    "tool_calls (list of objects: tool,name,args_json,result_json,ok).\n"
                    "Be concise and safe."
                ),
                rgb_bgr=rgb_bgr,
            )
            if isinstance(out, dict):
                caption = out.get("caption")
                planner = out.get("planner")
                tool_calls = out.get("tool_calls")
        except Exception:
            pass

        # Preferred: embed the caption (text embedding model) so observation.command is meaningful even when
        # the VLA model is only used for perception/captioning.
        if isinstance(caption, str) and caption.strip():
            try:
                embedding = self._embed_text(text=caption.strip(), obs_dim=obs_dim)
            except Exception:
                embedding = None

        # Legacy fallback: some servers offer embeddings but no caption; try a weak image hint embedding.
        if embedding is None:
            try:
                embedding = self._get_embedding(rgb_bgr=rgb_bgr, obs_dim=obs_dim)
            except Exception:
                embedding = None

        if tool_calls is not None:
            extra["tool_calls"] = tool_calls

        return TeacherResult(
            embedding=embedding,
            caption=caption if isinstance(caption, str) else None,
            planner=planner if isinstance(planner, dict) else None,
            action_command=None,  # optional: could be added later
            extra=extra,
        )


