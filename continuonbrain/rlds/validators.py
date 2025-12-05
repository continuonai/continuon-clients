"""RLDS validation helpers for mock-mode Studio episodes."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ValidationResult:
    """Collects validation issues and warnings for an episode."""

    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _require_fields(obj: Dict[str, Any], required: List[str], prefix: str, issues: List[str]):
    for field in required:
        if field not in obj:
            issues.append(f"Missing required field: {prefix}{field}")


def _validate_pose(name: str, pose: Dict[str, Any], issues: List[str]):
    _require_fields(pose, ["position", "orientation_quat", "valid"], f"{name}.", issues)
    if len(pose.get("orientation_quat", [])) not in {0, 4}:
        issues.append(f"{name}.orientation_quat must contain 4 values")
    if len(pose.get("position", [])) not in {0, 3}:
        issues.append(f"{name}.position must contain 3 values")


def _validate_observation(step_idx: int, observation: Dict[str, Any], issues: List[str]):
    prefix = f"steps[{step_idx}].observation."
    _require_fields(
        observation,
        [
            "headset_pose",
            "right_hand_pose",
            "left_hand_pose",
            "gaze",
            "robot_state",
            "glove",
            "video_frame_id",
            "depth_frame_id",
            "diagnostics",
        ],
        prefix,
        issues,
    )

    for pose_name in ["headset_pose", "right_hand_pose", "left_hand_pose"]:
        if pose := observation.get(pose_name):
            _validate_pose(f"{prefix}{pose_name}", pose, issues)

    robot_state = observation.get("robot_state", {})
    _require_fields(
        robot_state,
        ["timestamp_nanos", "joint_positions", "frame_id", "wall_time_millis"],
        f"{prefix}robot_state.",
        issues,
    )

    diagnostics = observation.get("diagnostics") or {}
    _require_fields(
        diagnostics,
        ["latency_ms", "glove_drops", "glove_sample_rate_hz", "mock_mode"],
        f"{prefix}diagnostics.",
        issues,
    )

    if observation.get("video_frame_id") != observation.get("depth_frame_id"):
        issues.append(f"{prefix}video_frame_id and depth_frame_id must match for RLDS alignment")


def _validate_action(step_idx: int, action: Dict[str, Any], issues: List[str]):
    prefix = f"steps[{step_idx}].action."
    _require_fields(action, ["command", "source"], prefix, issues)
    if isinstance(action.get("command"), list) and not action["command"]:
        issues.append(f"{prefix}command must include at least one control value")


def validate_episode(episode: Dict[str, Any]) -> ValidationResult:
    """Validate a mock-mode episode against RLDS expectations."""

    errors: List[str] = []
    warnings: List[str] = []

    metadata = episode.get("episode_metadata", {})
    _require_fields(metadata, ["environment_id", "tags", "continuon"], "episode_metadata.", errors)

    continuon_block = metadata.get("continuon") or {}
    _require_fields(continuon_block, ["xr_mode", "control_role"], "episode_metadata.continuon.", errors)

    steps = episode.get("steps")
    if not steps:
        errors.append("Episode must contain at least one step")
        return ValidationResult(errors=errors, warnings=warnings)

    for idx, step in enumerate(steps):
        observation = step.get("observation")
        action = step.get("action")
        if observation is None:
            errors.append(f"steps[{idx}] missing observation block")
        else:
            _validate_observation(idx, observation, errors)

        if action is None:
            errors.append(f"steps[{idx}] missing action block")
        else:
            _validate_action(idx, action, errors)

    return ValidationResult(errors=errors, warnings=warnings)


def load_episode(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_cli():
    parser = argparse.ArgumentParser(description="Validate RLDS episode JSON for Studio mock mode")
    parser.add_argument("--episode", type=Path, required=True, help="Path to an episode.json file")
    args = parser.parse_args()

    episode = load_episode(args.episode)
    result = validate_episode(episode)
    if result.ok:
        print("✅ Episode passed RLDS validation")
    else:
        print("❌ Episode failed RLDS validation:")
        for issue in result.errors:
            print(f" - {issue}")

    if result.warnings:
        print("⚠️ Warnings:")
        for warning in result.warnings:
            print(f" - {warning}")

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_run_cli())
