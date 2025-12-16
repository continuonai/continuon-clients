"""RLDS validation helpers for mock-mode Studio episodes.

The checks in this module mirror the proto contract defined in
`proto/continuonxr/rlds/v1/rlds_episode.proto` so mock-mode episodes stay
aligned with the serialized Episode message. The validator favors explicit
errors over permissive parsing to catch schema drift in fixtures and mock
generators.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class ValidationResult:
    """Collects validation issues and warnings for an episode."""

    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _require_fields(obj: Dict[str, Any], required: Iterable[str], prefix: str, issues: List[str]):
    for field in required:
        if field not in obj:
            issues.append(f"Missing required field: {prefix}{field}")


def _flag_unexpected_fields(obj: Dict[str, Any], allowed: Iterable[str], prefix: str, issues: List[str]):
    allowed_set = set(allowed)
    for field in obj:
        if field not in allowed_set:
            issues.append(f"Unexpected field: {prefix}{field}")


def _validate_pose(name: str, pose: Dict[str, Any], issues: List[str]):
    _require_fields(pose, ["position", "orientation_quat", "valid"], f"{name}.", issues)
    _flag_unexpected_fields(pose, ["position", "orientation_quat", "valid"], f"{name}.", issues)
    if len(pose.get("orientation_quat", [])) not in {0, 4}:
        issues.append(f"{name}.orientation_quat must contain 4 values")
    if len(pose.get("position", [])) not in {0, 3}:
        issues.append(f"{name}.position must contain 3 values")


def _validate_gaze(prefix: str, gaze: Dict[str, Any], issues: List[str]):
    _require_fields(gaze, ["origin", "direction", "confidence"], prefix, issues)
    _flag_unexpected_fields(gaze, ["origin", "direction", "confidence", "target_id"], prefix, issues)
    if len(gaze.get("origin", [])) not in {0, 3}:
        issues.append(f"{prefix}origin must contain 3 values")
    if len(gaze.get("direction", [])) not in {0, 3}:
        issues.append(f"{prefix}direction must contain 3 values")


def _validate_robot_state(prefix: str, robot_state: Dict[str, Any], issues: List[str]):
    required_fields = [
        "timestamp_nanos",
        "joint_positions",
        "end_effector_pose",
        "gripper_open",
        "frame_id",
        "joint_velocities",
        "end_effector_twist",
        "wall_time_millis",
    ]
    allowed_fields = required_fields + ["joint_efforts"]
    _require_fields(robot_state, required_fields, prefix, issues)
    _flag_unexpected_fields(robot_state, allowed_fields, prefix, issues)

    if robot_state.get("end_effector_pose"):
        _validate_pose(f"{prefix}end_effector_pose", robot_state["end_effector_pose"], issues)

    if len(robot_state.get("end_effector_twist", [])) not in {0, 6}:
        issues.append(f"{prefix}end_effector_twist must contain 6 values")


def _validate_glove(prefix: str, glove: Dict[str, Any], issues: List[str]):
    _require_fields(glove, ["timestamp_nanos", "flex", "fsr", "orientation_quat", "accel", "valid"], prefix, issues)
    _flag_unexpected_fields(glove, ["timestamp_nanos", "flex", "fsr", "orientation_quat", "accel", "valid"], prefix, issues)
    if len(glove.get("flex", [])) not in {0, 5}:
        issues.append(f"{prefix}flex must contain 5 values")
    if len(glove.get("fsr", [])) not in {0, 8}:
        issues.append(f"{prefix}fsr must contain 8 values")
    if len(glove.get("orientation_quat", [])) not in {0, 4}:
        issues.append(f"{prefix}orientation_quat must contain 4 values")
    if len(glove.get("accel", [])) not in {0, 3}:
        issues.append(f"{prefix}accel must contain 3 values")


def _validate_diagnostics(prefix: str, diagnostics: Dict[str, Any], issues: List[str]):
    required_fields = ["latency_ms", "glove_drops", "ble_rssi", "glove_sample_rate_hz"]
    _require_fields(diagnostics, required_fields, prefix, issues)
    _flag_unexpected_fields(diagnostics, required_fields, prefix, issues)


def _validate_audio(prefix: str, audio: Dict[str, Any], issues: List[str]):
    _flag_unexpected_fields(audio, ["uri", "sample_rate_hz", "num_channels", "format", "frame_id"], prefix, issues)
    if uri := audio.get("uri"):
        if not isinstance(uri, str):
            issues.append(f"{prefix}uri must be a string when provided")


def _validate_ui_context(prefix: str, ui_context: Dict[str, Any], issues: List[str]):
    _flag_unexpected_fields(ui_context, ["active_panel", "layout", "focus_context"], prefix, issues)
    if layout := ui_context.get("layout"):
        if not isinstance(layout, dict):
            issues.append(f"{prefix}layout must be a map of strings to strings")
    if focus_context := ui_context.get("focus_context"):
        if not isinstance(focus_context, dict):
            issues.append(f"{prefix}focus_context must be a map of strings to strings")


def _validate_observation(step_idx: int, observation: Dict[str, Any], issues: List[str]):
    prefix = f"steps[{step_idx}].observation."
    required_fields = [
        "headset_pose",
        "right_hand_pose",
        "left_hand_pose",
        "gaze",
        "robot_state",
        "glove",
        "video_frame_id",
        "depth_frame_id",
        "diagnostics",
    ]
    _require_fields(observation, required_fields, prefix, issues)
    # v1.1 additions are optional; allow them while keeping the validator strict.
    _flag_unexpected_fields(
        observation,
        required_fields + ["audio", "ui_context", "media", "dialog", "world_model", "command", "segmentation"],
        prefix,
        issues,
    )

    for pose_name in ["headset_pose", "right_hand_pose", "left_hand_pose"]:
        if pose := observation.get(pose_name):
            _validate_pose(f"{prefix}{pose_name}", pose, issues)

    if gaze := observation.get("gaze"):
        _validate_gaze(f"{prefix}gaze.", gaze, issues)

    if robot_state := observation.get("robot_state"):
        _validate_robot_state(f"{prefix}robot_state.", robot_state, issues)

    if glove := observation.get("glove"):
        _validate_glove(f"{prefix}glove.", glove, issues)

    if diagnostics := observation.get("diagnostics"):
        _validate_diagnostics(f"{prefix}diagnostics.", diagnostics, issues)

    if audio := observation.get("audio"):
        _validate_audio(f"{prefix}audio.", audio, issues)

    if ui_context := observation.get("ui_context"):
        _validate_ui_context(f"{prefix}ui_context.", ui_context, issues)

    if observation.get("video_frame_id") != observation.get("depth_frame_id"):
        issues.append(f"{prefix}video_frame_id and depth_frame_id must match for RLDS alignment")


def _validate_action(step_idx: int, action: Dict[str, Any], issues: List[str]):
    prefix = f"steps[{step_idx}].action."
    _require_fields(action, ["command", "source"], prefix, issues)
    # v1.1 additions are optional; allow them while keeping the validator strict.
    _flag_unexpected_fields(
        action,
        ["command", "source", "annotation", "ui_action", "dialog", "planner", "tool_calls"],
        prefix,
        issues,
    )

    if isinstance(action.get("command"), list) and not action["command"]:
        issues.append(f"{prefix}command must include at least one control value")

    if annotation := action.get("annotation"):
        _flag_unexpected_fields(annotation, ["kind", "fields"], f"{prefix}annotation.", issues)
        if fields := annotation.get("fields"):
            if not isinstance(fields, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in fields.items()
            ):
                issues.append(f"{prefix}annotation.fields must map strings to strings")

    if ui_action := action.get("ui_action"):
        _flag_unexpected_fields(ui_action, ["action_type", "context"], f"{prefix}ui_action.", issues)
        if context := ui_action.get("context"):
            if not isinstance(context, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in context.items()
            ):
                issues.append(f"{prefix}ui_action.context must map strings to strings")


def validate_episode(episode: Dict[str, Any]) -> ValidationResult:
    """Validate a mock-mode episode against RLDS expectations."""

    errors: List[str] = []
    warnings: List[str] = []

    metadata = episode.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}
    _require_fields(metadata, ["xr_mode", "control_role", "environment_id", "tags", "software"], "metadata.", errors)
    # v1.1 additions are optional; allow them while keeping the validator strict.
    _flag_unexpected_fields(
        metadata,
        [
            "xr_mode",
            "control_role",
            "environment_id",
            "tags",
            "software",
            "schema_version",
            "episode_id",
            "robot_id",
            "robot_model",
            "frame_convention",
            "start_time_unix_ms",
            "duration_ms",
            "owner",
            "safety",
            "share",
            "provenance",
            "capabilities",
        ],
        "metadata.",
        errors,
    )

    software_block = metadata.get("software") or {}
    _require_fields(software_block, ["xr_app", "continuonbrain_os", "glove_firmware"], "metadata.software.", errors)
    _flag_unexpected_fields(software_block, ["xr_app", "continuonbrain_os", "glove_firmware"], "metadata.software.", errors)

    tags = metadata.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append("metadata.tags must be a list of strings")

    steps = episode.get("steps")
    if not steps:
        errors.append("Episode must contain at least one step")
        return ValidationResult(errors=errors, warnings=warnings)

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"steps[{idx}] must be an object")
            continue

        observation = step.get("observation")
        action = step.get("action")
        _flag_unexpected_fields(step, ["observation", "action", "is_terminal", "step_metadata"], f"steps[{idx}].", errors)
        if observation is None:
            errors.append(f"steps[{idx}] missing observation block")
        else:
            if not isinstance(observation, dict):
                errors.append(f"steps[{idx}].observation must be an object")
            else:
                _validate_observation(idx, observation, errors)

        if action is None:
            errors.append(f"steps[{idx}] missing action block")
        else:
            if not isinstance(action, dict):
                errors.append(f"steps[{idx}].action must be an object")
            else:
                _validate_action(idx, action, errors)

        if step_metadata := step.get("step_metadata"):
            if not isinstance(step_metadata, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in step_metadata.items()
            ):
                errors.append(f"steps[{idx}].step_metadata must be a map of strings to strings")

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
