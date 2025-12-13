from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ChatRldsLogConfig:
    """
    Configuration for logging chat turns into RLDS JSON episodes.

    Notes:
    - Offline-first: this only writes local files.
    - Intended to be opt-in via env flags from the caller.
    """

    episodes_dir: Path
    # If set, turns with the same session_id append to a stable episode file.
    group_by_session: bool = True


def _safe_session_id(session_id: str) -> str:
    # Keep filenames safe and short.
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (session_id or ""))
    return out[:96] or "session"


def _load_or_init_episode(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"metadata": {"created_unix_s": int(time.time())}, "steps": []}
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            return payload
    except Exception:
        pass
    return {"metadata": {"created_unix_s": int(time.time()), "recovered": True}, "steps": []}


def log_chat_turn(
    cfg: ChatRldsLogConfig,
    *,
    user_message: str,
    assistant_response: str,
    structured: Optional[Dict[str, Any]] = None,
    status_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    model_hint: Optional[str] = None,
    agent_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Append a chat turn as an RLDS step.

    The step uses the RLDS-style {observation, action, step_metadata} convention so it can be
    consumed by downstream trainers/evals even if it is not a control-policy episode.
    """
    cfg.episodes_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()

    if cfg.group_by_session and session_id:
        fname = f"chat_session_{_safe_session_id(session_id)}.json"
        out_path = cfg.episodes_dir / fname
    else:
        out_path = cfg.episodes_dir / f"chat_{int(now)}.json"

    episode = _load_or_init_episode(out_path)
    steps = episode.get("steps")
    if not isinstance(steps, list):
        steps = []
        episode["steps"] = steps

    step = {
        "step_index": len(steps),
        "timestamp_ns": int(now * 1e9),
        "observation": {
            "type": "chat_turn",
            "user_message": user_message,
            "session_id": session_id,
            "status_context": status_context or {},
        },
        "action": {
            "assistant_response": assistant_response,
            "structured": structured or {},
            "model_hint": model_hint,
            "agent": agent_label,
        },
        "reward": 0.0,
        "is_terminal": False,
        "step_metadata": {
            "source": "human_chat",
            "timestamp": now,
        },
    }
    steps.append(step)
    out_path.write_text(json.dumps(episode, indent=2, default=str))
    return {"path": str(out_path), "steps": len(steps)}


