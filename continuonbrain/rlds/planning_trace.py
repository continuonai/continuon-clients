"""RLDS trace writer for planning/search episodes (canonical episode_dir).

This logs planning/search diagnostics in a replayable way without requiring the
full robotics strict schema.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_planning_episode(
    *,
    rlds_root: Path,
    environment_id: str,
    control_role: str,
    goal: Dict[str, Any],
    start_state: Dict[str, Any],
    plan: Dict[str, Any],
    execute: Dict[str, Any],
    model_debug: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Path:
    """
    Write a canonical episode_dir containing one step per planned step, plus a terminal summary step.
    """
    ts = time.time()
    episode_id = f"planning_arm_search_{int(ts)}_{uuid.uuid4().hex[:8]}"
    episode_dir = Path(rlds_root) / episode_id
    steps_dir = episode_dir / "steps"
    _ensure_dir(steps_dir)
    _ensure_dir(episode_dir / "blobs")

    metadata = {
        "xr_mode": "planning",
        "control_role": control_role,
        "environment_id": environment_id,
        "tags": (tags or []) + ["planning", "arm_search"],
        "software": {
            "xr_app": "n/a",
            "continuonbrain_os": "dev",
            "glove_firmware": "n/a",
        },
        "start_time_unix_ms": int(ts * 1000),
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    steps: List[Dict[str, Any]] = []
    plan_steps = (plan or {}).get("steps") or []
    for idx, step in enumerate(plan_steps):
        steps.append(
            {
                "observation": {
                    "start_state": start_state if idx == 0 else None,
                    "goal": goal if idx == 0 else None,
                    "predicted_state": {"joint_pos": step.get("predicted_joint_pos")},
                },
                "action": {
                    "kind": "planned_delta",
                    "joint_delta": step.get("joint_delta"),
                    "source": "arm_search",
                },
                "reward": 0.0,
                "is_terminal": False,
                "step_metadata": {
                    "step_index": str(idx),
                    "score": str(step.get("score")),
                    "uncertainty": str(step.get("uncertainty")),
                },
            }
        )

    # Terminal summary step.
    steps.append(
        {
            "observation": {
                "start_state": start_state,
                "goal": goal,
                "plan": plan,
                "execute": execute,
                "model_debug": model_debug or {},
            },
            "action": {"kind": "planning_summary", "source": "arm_search"},
            "reward": 0.0,
            "is_terminal": True,
            "step_metadata": {
                "timestamp": str(ts),
                "planned_steps": str(len(plan_steps)),
                "executed": str(bool(execute.get("executed"))).lower() if isinstance(execute, dict) else "false",
            },
        }
    )

    steps_path = steps_dir / "000000.jsonl"
    with steps_path.open("w", encoding="utf-8") as handle:
        for step in steps:
            handle.write(json.dumps(step, sort_keys=True))
            handle.write("\n")

    return episode_dir


