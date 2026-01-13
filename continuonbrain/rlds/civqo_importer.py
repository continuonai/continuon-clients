"""Import RLDS episodes from Civqo agent sessions.

This module fetches or consumes pre-fetched RLDS episodes exported from
Civqo (https://civqo.com) agent sessions and emits episodes that align with
our RLDS JSON contract (``metadata.json`` + ``steps/*.jsonl``). Episodes
from Civqo contain tool calls, user inputs, and agent outputs from Claude Code
sessions in the Code City visualization environment.

Usage:
    # Import from Civqo API (requires API key)
    python -m continuonbrain.rlds.civqo_importer --api-key $CIVQO_API_KEY

    # Import from pre-exported JSON files
    python -m continuonbrain.rlds.civqo_importer --input-dir ./civqo_exports/

    # Import specific sessions
    python -m continuonbrain.rlds.civqo_importer --session-ids session_123,session_456
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from . import validators

logger = logging.getLogger(__name__)


@dataclass
class CivqoImportConfig:
    """Configuration for importing Civqo RLDS episodes."""

    # API Configuration
    api_base_url: str = "https://api.civqo.com/api"
    api_key: Optional[str] = None

    # Input/Output
    input_dir: Optional[Path] = None
    output_dir: Path = Path("continuonbrain/rlds/episodes")
    session_ids: Optional[List[str]] = None

    # Processing Options
    origin_tag: str = "origin:civqo"
    environment_id_prefix: str = "civqo"
    xr_mode: str = "trainer"
    control_role: str = "human_supervisor"
    software_xr_app: str = "civqo.com"
    software_runtime: str = "continuonbrain-civqo-importer"
    software_glove: str = "absent"
    default_frame_prefix: str = "civqo"
    default_action_source: str = "agent"

    # Limits
    max_episodes: Optional[int] = None
    max_steps_per_episode: Optional[int] = None

    # Anonymization
    re_anonymize: bool = False
    anonymization_salt: str = "civqo-continuon-import"

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def _hash_value(value: str, salt: str) -> str:
    """Hash a value with salt for anonymization."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def _default_pose(valid: bool = False) -> Dict[str, Any]:
    """Create default XR pose placeholder."""
    return {
        "position": [0.0, 0.0, 0.0],
        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
        "valid": valid,
    }


def _default_robot_state(timestamp_nanos: int, frame_id: str) -> Dict[str, Any]:
    """Create default robot state placeholder."""
    return {
        "timestamp_nanos": int(timestamp_nanos),
        "joint_positions": [0.0, 0.0],
        "joint_velocities": [0.0, 0.0],
        "joint_efforts": [],
        "end_effector_pose": _default_pose(valid=False),
        "end_effector_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_open": True,
        "frame_id": frame_id,
        "wall_time_millis": int(timestamp_nanos / 1_000_000),
    }


def _default_glove(timestamp_nanos: int) -> Dict[str, Any]:
    """Create default glove sensor placeholder."""
    return {
        "timestamp_nanos": int(timestamp_nanos),
        "flex": [0, 0, 0, 0, 0],
        "fsr": [0, 0, 0, 0, 0, 0, 0, 0],
        "orientation_quat": [0, 0, 0, 0],
        "accel": [0, 0, 0],
        "valid": False,
    }


def _default_diagnostics(latency_ms: int = 0) -> Dict[str, Any]:
    """Create default diagnostics placeholder."""
    return {
        "latency_ms": latency_ms,
        "glove_drops": 0,
        "ble_rssi": 0,
        "glove_sample_rate_hz": 0,
    }


def _normalize_observation(
    obs: Mapping[str, Any],
    frame_id: str,
    timestamp_nanos: int,
) -> Dict[str, Any]:
    """Normalize Civqo observation to RLDS schema with placeholders for missing XR data."""

    # Start with defaults
    normalized: Dict[str, Any] = {
        "headset_pose": _default_pose(),
        "right_hand_pose": _default_pose(),
        "left_hand_pose": _default_pose(),
        "gaze": {"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0], "confidence": 0.0},
        "robot_state": _default_robot_state(timestamp_nanos, frame_id),
        "glove": _default_glove(timestamp_nanos),
        "video_frame_id": frame_id,
        "depth_frame_id": frame_id,
        "diagnostics": _default_diagnostics(),
    }

    # Merge incoming observation data
    if "headset_pose" in obs:
        normalized["headset_pose"] = obs["headset_pose"]
    if "right_hand_pose" in obs:
        normalized["right_hand_pose"] = obs["right_hand_pose"]
    if "left_hand_pose" in obs:
        normalized["left_hand_pose"] = obs["left_hand_pose"]
    if "gaze" in obs:
        normalized["gaze"] = obs["gaze"]
    if "robot_state" in obs:
        # Merge robot state
        rs = normalized["robot_state"]
        incoming_rs = obs["robot_state"]
        for key in ("joint_positions", "joint_velocities", "joint_efforts",
                    "end_effector_pose", "end_effector_twist", "gripper_open", "frame_id"):
            if key in incoming_rs:
                rs[key] = incoming_rs[key]
        normalized["robot_state"] = rs
    if "glove" in obs:
        normalized["glove"] = {**normalized["glove"], **obs["glove"]}
    if "diagnostics" in obs:
        normalized["diagnostics"] = {**normalized["diagnostics"], **obs["diagnostics"]}
    if "video_frame_id" in obs:
        normalized["video_frame_id"] = obs["video_frame_id"]
        normalized["depth_frame_id"] = obs.get("depth_frame_id", obs["video_frame_id"])
    if "depth_frame_id" in obs:
        normalized["depth_frame_id"] = obs["depth_frame_id"]

    # Civqo-specific fields
    if "command" in obs:
        normalized["command"] = obs["command"]
    if "ui_context" in obs:
        normalized["ui_context"] = obs["ui_context"]
    if "world_model" in obs:
        normalized["world_model"] = obs["world_model"]

    return normalized


def _normalize_action(action: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize Civqo action to RLDS schema."""
    normalized: Dict[str, Any] = {
        "command": action.get("command", [0.0]),
        "source": action.get("source", "agent"),
    }

    if "annotation" in action:
        normalized["annotation"] = action["annotation"]
    if "ui_action" in action:
        normalized["ui_action"] = action["ui_action"]
    if "tool_calls" in action:
        normalized["tool_calls"] = action["tool_calls"]

    return normalized


def _normalize_step(
    step: Mapping[str, Any],
    episode_id: str,
    step_idx: int,
    cfg: CivqoImportConfig,
) -> Dict[str, Any]:
    """Normalize a Civqo step to RLDS schema."""

    frame_id = f"{cfg.default_frame_prefix}_{episode_id}_{step_idx:06d}"
    timestamp_nanos = step.get("observation", {}).get(
        "robot_state", {}
    ).get("timestamp_nanos", step_idx * 1_000_000)

    observation = _normalize_observation(
        step.get("observation", {}),
        frame_id,
        timestamp_nanos,
    )

    action = _normalize_action(step.get("action", {}))

    # Build step metadata
    step_metadata: Dict[str, str] = {}
    if "step_metadata" in step:
        step_metadata = {k: str(v) for k, v in step["step_metadata"].items()}
    step_metadata["frame_id"] = frame_id
    step_metadata["source"] = "civqo"
    step_metadata["source_step"] = str(step_idx)

    return {
        "observation": observation,
        "action": action,
        "is_terminal": step.get("is_terminal", False),
        "step_metadata": step_metadata,
    }


def _normalize_metadata(
    metadata: Mapping[str, Any],
    cfg: CivqoImportConfig,
) -> Dict[str, Any]:
    """Normalize Civqo episode metadata to RLDS schema."""

    # Build tags
    tags = list(metadata.get("tags", []))
    if cfg.origin_tag not in tags:
        tags.insert(0, cfg.origin_tag)
    if "imported:continuonbrain" not in tags:
        tags.append("imported:continuonbrain")

    normalized = {
        "episode_id": metadata.get("episode_id", "unknown"),
        "environment_id": metadata.get("environment_id", f"{cfg.environment_id_prefix}:unknown"),
        "xr_mode": metadata.get("xr_mode", cfg.xr_mode),
        "control_role": metadata.get("control_role", cfg.control_role),
        "tags": tags,
        "software": {
            "xr_app": metadata.get("software", {}).get("xr_app", cfg.software_xr_app),
            "continuonbrain_os": cfg.software_runtime,
            "glove_firmware": metadata.get("software", {}).get("glove_firmware", cfg.software_glove),
        },
        "schema_version": metadata.get("schema_version", "1.1"),
    }

    # Optional fields
    if "start_time_unix_ms" in metadata:
        normalized["start_time_unix_ms"] = metadata["start_time_unix_ms"]
    if "duration_ms" in metadata:
        normalized["duration_ms"] = metadata["duration_ms"]
    if "owner" in metadata:
        # Re-hash owner for privacy
        if cfg.re_anonymize:
            normalized["owner"] = _hash_value(str(metadata["owner"]), cfg.anonymization_salt)[:16]
        else:
            normalized["owner"] = metadata["owner"]
    if "safety" in metadata:
        normalized["safety"] = metadata["safety"]
    if "share" in metadata:
        normalized["share"] = metadata["share"]
    if "provenance" in metadata:
        prov = dict(metadata["provenance"])
        prov["importer"] = "continuonbrain.rlds.civqo_importer"
        prov["import_timestamp"] = __import__("datetime").datetime.now().isoformat()
        normalized["provenance"] = prov

    return normalized


def normalize_episode(
    episode: Mapping[str, Any],
    cfg: CivqoImportConfig,
) -> Dict[str, Any]:
    """Normalize a Civqo RLDS episode to ContinuonBrain schema."""

    metadata = _normalize_metadata(episode.get("metadata", {}), cfg)
    episode_id = metadata["episode_id"]

    steps = []
    raw_steps = episode.get("steps", [])
    for idx, step in enumerate(raw_steps):
        if cfg.max_steps_per_episode and idx >= cfg.max_steps_per_episode:
            break
        normalized_step = _normalize_step(step, episode_id, idx, cfg)
        steps.append(normalized_step)

    # Mark last step as terminal
    if steps and not steps[-1].get("is_terminal"):
        steps[-1]["is_terminal"] = True

    return {"metadata": metadata, "steps": steps}


def write_episode(payload: Mapping[str, Any], output_dir: Path, episode_id: str) -> Path:
    """Write an RLDS episode to disk in the standard directory format."""
    episode_dir = output_dir / str(episode_id)
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    (episode_dir / "metadata.json").write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True),
        "utf-8",
    )

    # Write steps as JSONL
    steps_path = steps_dir / "000000.jsonl"
    with steps_path.open("w", encoding="utf-8") as handle:
        for step in payload.get("steps", []):
            handle.write(json.dumps(step, sort_keys=True))
            handle.write("\n")

    return episode_dir


def fetch_episodes_from_api(cfg: CivqoImportConfig) -> Iterable[Dict[str, Any]]:
    """Fetch RLDS episodes from Civqo API."""

    if not cfg.api_key:
        raise ValueError("API key required for Civqo API access")

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(base_url=cfg.api_base_url, headers=headers, timeout=60) as client:
        # Get list of eligible sessions
        if cfg.session_ids:
            session_ids = cfg.session_ids
        else:
            resp = client.get("/rlds-export/sessions", params={"limit": cfg.max_episodes or 100})
            resp.raise_for_status()
            data = resp.json()
            session_ids = [s["id"] for s in data.get("sessions", [])]

        logger.info("Found %d sessions to export", len(session_ids))

        # Export each session
        for session_id in session_ids:
            try:
                resp = client.post(f"/rlds-export/session/{session_id}")
                resp.raise_for_status()
                result = resp.json()

                if result.get("success") and result.get("episode"):
                    yield result["episode"]
                else:
                    logger.warning("Failed to export session %s: %s", session_id, result.get("error"))

            except httpx.HTTPError as exc:
                logger.error("HTTP error exporting session %s: %s", session_id, exc)
            except json.JSONDecodeError:
                logger.error("Invalid JSON response for session %s", session_id)


def load_episodes_from_files(cfg: CivqoImportConfig) -> Iterable[Dict[str, Any]]:
    """Load RLDS episodes from pre-exported JSON files."""

    if not cfg.input_dir:
        raise ValueError("Input directory required for file-based import")

    input_path = cfg.input_dir
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Look for JSON and JSONL files
    for json_file in sorted(input_path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text("utf-8"))
            if isinstance(data, list):
                # Multiple episodes in one file
                for episode in data:
                    yield episode
            elif "metadata" in data and "steps" in data:
                # Single episode
                yield data
            else:
                logger.warning("Unrecognized format in %s", json_file)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", json_file, exc)

    # Also check for JSONL files
    for jsonl_file in sorted(input_path.glob("*.jsonl")):
        try:
            metadata = None
            steps = []
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if "metadata" in record:
                        metadata = record["metadata"]
                    elif "observation" in record and "action" in record:
                        steps.append(record)

            if metadata and steps:
                yield {"metadata": metadata, "steps": steps}
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", jsonl_file, exc)


def import_episodes(cfg: CivqoImportConfig) -> List[Path]:
    """Import Civqo episodes and write to output directory."""

    cfg.ensure_output_dir()

    # Determine source
    if cfg.input_dir:
        episodes = load_episodes_from_files(cfg)
    elif cfg.api_key:
        episodes = fetch_episodes_from_api(cfg)
    else:
        raise ValueError("Either input_dir or api_key must be provided")

    written: List[Path] = []

    for idx, episode in enumerate(episodes):
        if cfg.max_episodes and idx >= cfg.max_episodes:
            break

        try:
            # Normalize to ContinuonBrain schema
            normalized = normalize_episode(episode, cfg)
            episode_id = normalized["metadata"]["episode_id"]

            # Validate
            report = validators.validate_episode(normalized)
            if not report.ok:
                logger.error("Episode %s failed validation: %s", episode_id, "; ".join(report.errors))
                continue

            # Write to disk
            episode_path = write_episode(normalized, cfg.output_dir, episode_id)
            written.append(episode_path)
            logger.info("Wrote episode %s to %s", episode_id, episode_path)

        except Exception as exc:
            logger.error("Failed to process episode %d: %s", idx, exc)

    return written


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Import RLDS episodes from Civqo agent sessions",
    )

    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--api-key",
        help="Civqo API key for direct API access",
    )
    source_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing pre-exported Civqo episodes",
    )

    # API options
    parser.add_argument(
        "--api-url",
        default="https://api.civqo.com/api",
        help="Civqo API base URL",
    )
    parser.add_argument(
        "--session-ids",
        help="Comma-separated list of specific session IDs to import",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("continuonbrain/rlds/episodes"),
        help="Output directory for RLDS episodes",
    )

    # Processing options
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Maximum number of episodes to import",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--re-anonymize",
        action="store_true",
        help="Re-hash PII fields with local salt",
    )

    return parser.parse_args(argv)


def _build_config_from_args(args: argparse.Namespace) -> CivqoImportConfig:
    """Build config from parsed arguments."""
    session_ids = None
    if args.session_ids:
        session_ids = [s.strip() for s in args.session_ids.split(",")]

    return CivqoImportConfig(
        api_base_url=args.api_url,
        api_key=args.api_key,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        session_ids=session_ids,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        re_anonymize=args.re_anonymize,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entry
    """Main entry point for CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args(argv)
    cfg = _build_config_from_args(args)

    try:
        written = import_episodes(cfg)
        if not written:
            logger.warning("No episodes were written; check source availability or validation errors")
            return 1
        logger.info("Successfully imported %d episodes to %s", len(written), cfg.output_dir)
        return 0
    except Exception as exc:
        logger.exception("Import failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
