"""Mock-mode RLDS episode generator for Continuon Brain Studio.

The generator mirrors the RLDS schema defined in ``docs/rlds-schema.md`` and the
``proto/continuonxr/rlds/v1/rlds_episode.proto`` contract. It is intended for
local Studio mock-mode sessions where hardware is absent but schema compliance
is still required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import time


@dataclass
class MockGeneratorConfig:
    """Configuration for mock-mode episode generation."""

    episode_id: str = "studio_mock_mode_episode"
    environment_id: str = "studio-mock"
    xr_mode: str = "workstation"
    control_role: str = "human_dev_xr"
    action_source: str = "human_teleop_xr"
    step_count: int = 3
    step_period_ms: int = 50


def _pose(x: float, y: float, z: float) -> Dict[str, Any]:
    return {
        "position": [x, y, z],
        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
        "valid": True,
    }


def _glove(valid: bool, timestamp_ns: int) -> Dict[str, Any]:
    return {
        "timestamp_nanos": timestamp_ns,
        "flex": [0.2, 0.3, 0.4, 0.5, 0.6],
        "fsr": [0.1] * 8,
        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
        "accel": [0.0, 9.81, 0.0],
        "valid": valid,
    }


def _robot_state(timestamp_ns: int, frame_id: str) -> Dict[str, Any]:
    return {
        "timestamp_nanos": timestamp_ns,
        "joint_positions": [0.0, 0.1, 0.2, 0.1, 0.0, -0.1],
        "joint_velocities": [0.0] * 6,
        "end_effector_pose": _pose(0.1, 0.2, 0.3),
        "gripper_open": True,
        "frame_id": frame_id,
        "end_effector_twist": [0.0] * 6,
        "wall_time_millis": int(timestamp_ns / 1e6),
    }


def _diagnostics(mock_mode: bool) -> Dict[str, Any]:
    return {
        "latency_ms": 8.5,
        "glove_drops": 0,
        "ble_rssi": -58,
        "glove_sample_rate_hz": 90.0,
        "mock_mode": mock_mode,
        "transport": "loopback",
    }


def _action(command_scale: float, source: str) -> Dict[str, Any]:
    return {
        "command": [0.0, command_scale, 0.0, 0.0, -command_scale, 0.0],
        "source": source,
        "annotation": {"kind": "mock_note", "fields": {"reason": "studio-mock"}},
        "ui_action": {"action_type": "panel_interaction", "context": {"panel": "timeline"}},
    }


def _ui_context(step_index: int) -> Dict[str, Any]:
    return {
        "active_panel": "mock_editor",
        "layout": {"pane": "left", "tab": "rlds"},
        "focus_context": {"cursor": f"frame_{step_index:04d}"},
    }


def generate_mock_mode_episode(config: MockGeneratorConfig | None = None) -> Dict[str, Any]:
    """Generate a mock-mode episode aligned with RLDS contracts.

    Returns a dictionary matching the JSON structure emitted by the Studio
    editor so validators and downstream trainers can exercise the same schema.
    """

    cfg = config or MockGeneratorConfig()
    start_ns = time.time_ns()
    steps: List[Dict[str, Any]] = []

    for idx in range(cfg.step_count):
        timestamp_ns = start_ns + idx * cfg.step_period_ms * 1_000_000
        frame_id = f"frame_{idx:04d}"

        observation = {
            "headset_pose": _pose(0.0, 0.0, 0.0),
            "right_hand_pose": _pose(0.02, -0.02, 0.0),
            "left_hand_pose": _pose(-0.02, 0.02, 0.0),
            "gaze": {
                "origin": [0.0, 0.0, 0.0],
                "direction": [0.0, 0.0, 1.0],
                "confidence": 0.95,
                "target_id": "mock_widget",
            },
            "robot_state": _robot_state(timestamp_ns, frame_id),
            "glove": _glove(valid=True, timestamp_ns=timestamp_ns),
            "video_frame_id": frame_id,
            "depth_frame_id": frame_id,
            "diagnostics": _diagnostics(mock_mode=True),
            "audio": {
                "uri": f"mock://audio/{frame_id}.wav",
                "sample_rate_hz": 16000,
                "num_channels": 1,
                "format": "pcm16le",
                "frame_id": frame_id,
            },
            "ui_context": _ui_context(idx),
        }

        step = {
            "observation": observation,
            "action": _action(command_scale=0.1 * (idx + 1), source=cfg.action_source),
            "is_terminal": idx == cfg.step_count - 1,
            "step_metadata": {
                "language_instruction": "Mock Studio episode",
                "safety_violations": "none",
            },
        }
        steps.append(step)

    episode_metadata = {
        "episode_id": cfg.episode_id,
        "environment_id": cfg.environment_id,
        "continuon": {
            "xr_mode": cfg.xr_mode,
            "control_role": cfg.control_role,
        },
        "software": {
            "xr_app": "studio-mock",
            "continuonbrain_os": "mock",
            "glove_firmware": "mock",
        },
        "tags": ["studio", "mock", f"continuon.xr_mode:{cfg.xr_mode}"],
    }

    return {"episode_metadata": episode_metadata, "steps": steps}
