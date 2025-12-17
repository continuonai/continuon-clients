"""
RLDS Normalization Utilities

Goal: convert the various episode shapes produced in this repo into a single
canonical on-disk layout:

<episode_dir>/
  metadata.json
  steps/000000.jsonl
  blobs/ (optional)

This intentionally keeps steps permissive (observation/action are dicts) and uses
`step_metadata` (map<string,string>) for non-schema signals.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CANONICAL_STEPS_NAME = "000000.jsonl"


@dataclass(frozen=True)
class DetectResult:
    kind: str
    path: Path


def detect_variant(path: Path) -> DetectResult:
    """
    Detect the episode variant from a path.

    Supported kinds:
    - episode_dir: directory containing metadata.json and steps/000000.jsonl
    - single_json: json file containing {"steps":[...]} or similar
    - single_jsonl: jsonl file containing one step dict per line
    """
    path = path.expanduser().resolve()
    if path.is_dir():
        meta = path / "metadata.json"
        steps = path / "steps" / CANONICAL_STEPS_NAME
        if meta.exists() and steps.exists():
            return DetectResult(kind="episode_dir", path=path)
        return DetectResult(kind="unknown_dir", path=path)

    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            return DetectResult(kind="single_jsonl", path=path)
        if path.suffix.lower() == ".json":
            return DetectResult(kind="single_json", path=path)
        return DetectResult(kind="unknown_file", path=path)

    return DetectResult(kind="missing", path=path)


def _ensure_episode_dir(output_root: Path, episode_id: Optional[str] = None) -> Path:
    episode_id = episode_id or f"ep_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    episode_dir = output_root / episode_id
    (episode_dir / "steps").mkdir(parents=True, exist_ok=True)
    (episode_dir / "blobs").mkdir(parents=True, exist_ok=True)
    return episode_dir


def _write_metadata(episode_dir: Path, metadata: Dict[str, Any]) -> None:
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _write_steps_jsonl(episode_dir: Path, steps: Iterable[Dict[str, Any]]) -> Path:
    steps_path = episode_dir / "steps" / CANONICAL_STEPS_NAME
    with steps_path.open("w", encoding="utf-8") as handle:
        for step in steps:
            handle.write(json.dumps(step, sort_keys=True))
            handle.write("\n")
    return steps_path


def _canonicalize_step(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure step has canonical keys and types.
    We keep this permissive: unknown keys are ignored and can be moved to step_metadata later.
    """
    observation = raw.get("observation") or raw.get("obs") or {}
    action = raw.get("action") or {}
    reward = raw.get("reward", 0.0)
    is_terminal = raw.get("is_terminal", raw.get("done", False))
    step_metadata = raw.get("step_metadata") or {}

    # HOPE eval uses {"obs": {...}, "action": {...}, "step_metadata": {...}}
    # Chat episodes may include "action.text"; keep as-is.
    step = {
        "observation": observation if isinstance(observation, dict) else {"value": observation},
        "action": action if isinstance(action, dict) else {"value": action},
        "reward": float(reward) if isinstance(reward, (int, float)) else 0.0,
        "is_terminal": bool(is_terminal),
        "step_metadata": {str(k): str(v) for k, v in (step_metadata.items() if isinstance(step_metadata, dict) else [])},
    }
    return step


def normalize_to_episode_dir(
    input_path: Path,
    *,
    output_root: Path,
    episode_id: Optional[str] = None,
    default_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Normalize any supported input into canonical episode_dir.

    Returns the created/normalized episode directory.
    """
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detected = detect_variant(input_path)

    if detected.kind == "episode_dir":
        # Already canonical.
        return detected.path

    episode_dir = _ensure_episode_dir(output_root, episode_id=episode_id)

    base_metadata = default_metadata or {
        "xr_mode": "unknown",
        "control_role": "unknown",
        "environment_id": "unknown",
        "tags": ["normalized"],
        "software": {"xr_app": "n/a", "continuonbrain_os": "dev", "glove_firmware": "n/a"},
    }

    if detected.kind == "single_json":
        payload = json.loads(detected.path.read_text(encoding="utf-8"))
        # Support either {"metadata":..., "steps":[...]} or {"steps":[...]} or chat-style episode envelope.
        steps_raw = payload.get("steps") if isinstance(payload, dict) else None
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict):
            metadata = dict(base_metadata)
        # Preserve provenance.
        metadata.setdefault("tags", [])
        if isinstance(metadata["tags"], list):
            metadata["tags"].append(f"source_file:{detected.path.name}")
        metadata["tags"] = metadata["tags"] if isinstance(metadata["tags"], list) else [str(metadata["tags"])]

        _write_metadata(episode_dir, metadata)

        if not isinstance(steps_raw, list):
            steps_raw = []
        steps = [_canonicalize_step(step) for step in steps_raw if isinstance(step, dict)]
        _write_steps_jsonl(episode_dir, steps)
        return episode_dir

    if detected.kind == "single_jsonl":
        # Each line is a step dict. We synthesize minimal metadata.
        metadata = dict(base_metadata)
        tags = metadata.get("tags")
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        tags.append(f"source_file:{detected.path.name}")
        metadata["tags"] = tags
        _write_metadata(episode_dir, metadata)

        steps: List[Dict[str, Any]] = []
        with detected.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                if isinstance(raw, dict):
                    steps.append(_canonicalize_step(raw))
        _write_steps_jsonl(episode_dir, steps)
        return episode_dir

    raise ValueError(f"Unsupported input for normalization: kind={detected.kind} path={detected.path}")


