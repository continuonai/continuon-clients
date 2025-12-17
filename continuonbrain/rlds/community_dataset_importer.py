"""Ingest the HuggingFace VLA community dataset into Continuon RLDS layout.

This module downloads or consumes pre-fetched records from
``HuggingFaceVLA/community_dataset_v3`` and emits episodes that align with
our RLDS JSON contract (``metadata.json`` + ``steps/*.jsonl``). Missing XR or
robotics signals are padded with schema-stable placeholders so the output
passes our validators and downstream trainers.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from . import validators

logger = logging.getLogger(__name__)


@dataclass
class CommunityDatasetIngestConfig:
    """Configuration for converting the community dataset into RLDS episodes."""

    dataset_id: str = "HuggingFaceVLA/community_dataset_v3"
    split: str = "train"
    output_dir: Path = Path("continuonbrain/rlds/episodes")
    origin_tag: str = "origin:huggingface_vla:community_dataset_v3"
    environment_id: str = "hf_community_dataset_v3"
    xr_mode: str = "trainer"
    control_role: str = "human_supervisor"
    software_xr_app: str = "continuonai.com-importer"
    software_runtime: str = "continuonbrain-runtime"
    software_glove: str = "absent"
    default_frame_prefix: str = "hf_vla"
    default_action_source: str = "human_teleop_xr"
    add_instruction_tag: bool = True
    max_episodes: Optional[int] = None
    max_steps_per_episode: Optional[int] = None

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def _default_pose(valid: bool = False) -> Dict[str, Any]:
    return {"position": [0.0, 0.0, 0.0], "orientation_quat": [0.0, 0.0, 0.0, 1.0], "valid": valid}


def _default_robot_state(timestamp_nanos: int, frame_id: str) -> Dict[str, Any]:
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
    return {
        "timestamp_nanos": int(timestamp_nanos),
        "flex": [0, 0, 0, 0, 0],
        "fsr": [0, 0, 0, 0, 0, 0, 0, 0],
        "orientation_quat": [0, 0, 0, 0],
        "accel": [0, 0, 0],
        "valid": False,
    }


def _default_diagnostics() -> Dict[str, Any]:
    return {"latency_ms": 0, "glove_drops": 0, "ble_rssi": 0, "glove_sample_rate_hz": 0}


def _extract_episode_id(sample: Mapping[str, Any], fallback_idx: int) -> str:
    for key in ("episode_id", "trajectory_id", "episode", "id"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return f"episode_{fallback_idx:04d}"


def _extract_instruction(sample: Mapping[str, Any]) -> Optional[str]:
    for key in ("instruction", "language_instruction", "task", "goal"):
        if isinstance(sample.get(key), str) and sample[key].strip():
            return sample[key].strip()
    return None


def _extract_action_command(sample: Mapping[str, Any]) -> List[float]:
    candidate = None
    for key in ("action", "actions", "action_command", "command"):
        if key not in sample:
            continue
        candidate = sample[key]
        break

    if isinstance(candidate, Mapping) and "command" in candidate:
        candidate = candidate["command"]

    if isinstance(candidate, (list, tuple)):
        try:
            vec = [float(x) for x in candidate]
        except Exception:  # pragma: no cover - defensive conversion
            vec = []
        if not vec:
            return [0.0]
        return vec

    return [0.0]


def _merge_robot_state(base: Dict[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
    merged = {**base}
    for key in (
        "joint_positions",
        "joint_velocities",
        "joint_efforts",
        "end_effector_pose",
        "end_effector_twist",
        "gripper_open",
        "frame_id",
    ):
        if key in incoming:
            merged[key] = incoming[key]
    if "timestamp_nanos" in incoming:
        merged["timestamp_nanos"] = int(incoming["timestamp_nanos"])
        merged["wall_time_millis"] = int(int(incoming["timestamp_nanos"]) / 1_000_000)
    return merged


def _merge_observation_placeholders(
    sample: Mapping[str, Any],
    frame_id: str,
    timestamp_nanos: int,
) -> Dict[str, Any]:
    observation: Dict[str, Any] = {
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

    if isinstance(sample.get("observation"), Mapping):
        incoming_obs = sample["observation"]
        for pose_key in ("headset_pose", "right_hand_pose", "left_hand_pose", "gaze"):
            if pose_key in incoming_obs:
                observation[pose_key] = incoming_obs[pose_key]
        if "robot_state" in incoming_obs and isinstance(incoming_obs["robot_state"], Mapping):
            observation["robot_state"] = _merge_robot_state(
                observation["robot_state"], incoming_obs["robot_state"]
            )
        if "glove" in incoming_obs and isinstance(incoming_obs["glove"], Mapping):
            observation["glove"] = {**observation["glove"], **incoming_obs["glove"]}
        if "diagnostics" in incoming_obs and isinstance(incoming_obs["diagnostics"], Mapping):
            observation["diagnostics"] = {**observation["diagnostics"], **incoming_obs["diagnostics"]}
        if incoming_obs.get("video_frame_id"):
            observation["video_frame_id"] = incoming_obs["video_frame_id"]
            observation["depth_frame_id"] = incoming_obs.get("depth_frame_id", incoming_obs["video_frame_id"])
        if incoming_obs.get("depth_frame_id"):
            observation["depth_frame_id"] = incoming_obs.get("depth_frame_id")

    return observation


def _build_step_metadata(
    sample: Mapping[str, Any],
    cfg: CommunityDatasetIngestConfig,
    episode_id: str,
    step_idx: int,
    instruction: Optional[str],
    frame_id: str,
) -> MutableMapping[str, str]:
    metadata: MutableMapping[str, str] = {
        "source_dataset": cfg.dataset_id,
        "source_split": cfg.split,
        "source_episode": str(episode_id),
        "source_step": str(step_idx),
        "frame_id": frame_id,
    }
    if instruction:
        metadata["instruction"] = instruction
    if sample.get("image"):
        metadata["image_uri"] = str(sample["image"])
    if sample.get("image_path"):
        metadata["image_path"] = str(sample["image_path"])
    if sample.get("frame_id"):
        metadata["source_frame_id"] = str(sample["frame_id"])
    return metadata


def build_episode_payload(
    episode_id: str,
    samples: Sequence[Mapping[str, Any]],
    cfg: CommunityDatasetIngestConfig,
) -> Dict[str, Any]:
    """Construct an RLDS episode payload from community dataset samples."""

    steps = []
    instruction = _extract_instruction(samples[0]) if samples else None
    tags = [cfg.origin_tag, f"hf.split:{cfg.split}"]
    if instruction and cfg.add_instruction_tag:
        tags.append(f"instruction:{instruction}")

    metadata = {
        "xr_mode": cfg.xr_mode,
        "control_role": cfg.control_role,
        "environment_id": cfg.environment_id,
        "tags": tags,
        "software": {
            "xr_app": cfg.software_xr_app,
            "continuonbrain_os": cfg.software_runtime,
            "glove_firmware": cfg.software_glove,
        },
    }

    for idx, sample in enumerate(samples):
        if cfg.max_steps_per_episode and idx >= cfg.max_steps_per_episode:
            break
        frame_id = str(
            sample.get("frame_id")
            or sample.get("image_id")
            or sample.get("frame")
            or f"{cfg.default_frame_prefix}_{episode_id}_{idx:06d}"
        )
        timestamp_nanos = int(sample.get("timestamp_nanos") or sample.get("timestamp") or idx * 1_000_000)
        observation = _merge_observation_placeholders(sample, frame_id, timestamp_nanos)
        step_metadata = _build_step_metadata(sample, cfg, episode_id, idx, instruction, frame_id)
        action = {"command": _extract_action_command(sample), "source": cfg.default_action_source}

        steps.append({"observation": observation, "action": action, "step_metadata": step_metadata})

    return {"metadata": metadata, "steps": steps}


def write_episode(payload: Mapping[str, Any], output_dir: Path, episode_id: str) -> Path:
    episode_dir = output_dir / str(episode_id)
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    (episode_dir / "metadata.json").write_text(json.dumps(payload["metadata"], indent=2, sort_keys=True), "utf-8")

    steps_path = steps_dir / "000000.jsonl"
    with steps_path.open("w", encoding="utf-8") as handle:
        for step in payload.get("steps", []):
            handle.write(json.dumps(step, sort_keys=True))
            handle.write("\n")
    return episode_dir


def ingest_grouped_samples(
    grouped_samples: Mapping[str, Sequence[Mapping[str, Any]]],
    cfg: CommunityDatasetIngestConfig,
) -> List[Path]:
    cfg.ensure_output_dir()
    written: List[Path] = []
    for episode_idx, (episode_id, samples) in enumerate(grouped_samples.items()):
        if cfg.max_episodes and episode_idx >= cfg.max_episodes:
            break
        payload = build_episode_payload(episode_id, samples, cfg)
        report = validators.validate_episode(payload)
        if not report.ok:
            logger.error("Episode %s failed validation: %s", episode_id, "; ".join(report.errors))
            continue
        written.append(write_episode(payload, cfg.output_dir, episode_id))
    return written


def stream_huggingface_dataset(cfg: CommunityDatasetIngestConfig) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface-datasets is required to stream the community dataset. Install with `pip install datasets`."
        ) from exc

    ds = load_dataset(cfg.dataset_id, split=cfg.split, streaming=True)
    for sample in ds:
        yield sample


def ingest_huggingface_dataset(cfg: CommunityDatasetIngestConfig) -> List[Path]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for idx, sample in enumerate(stream_huggingface_dataset(cfg)):
        episode_id = _extract_episode_id(sample, len(grouped))
        if cfg.max_episodes and episode_id not in grouped and len(grouped) >= cfg.max_episodes:
            break
        grouped.setdefault(episode_id, []).append(sample)
    return ingest_grouped_samples(grouped, cfg)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HuggingFace VLA community dataset into RLDS episodes")
    parser.add_argument("--split", default="train", help="Dataset split to consume")
    parser.add_argument("--dataset-id", default="HuggingFaceVLA/community_dataset_v3", help="Dataset repository id")
    parser.add_argument("--output-dir", type=Path, default=Path("continuonbrain/rlds/episodes"), help="Target RLDS directory")
    parser.add_argument("--max-episodes", type=int, help="Optional cap on number of episodes to emit")
    parser.add_argument("--max-steps", type=int, help="Optional cap on steps per episode")
    parser.add_argument("--no-instruction-tag", action="store_true", help="Skip echoing instruction into metadata.tags")
    return parser.parse_args(argv)


def _build_config_from_args(args: argparse.Namespace) -> CommunityDatasetIngestConfig:
    cfg = CommunityDatasetIngestConfig(
        dataset_id=args.dataset_id,
        split=args.split,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        add_instruction_tag=not args.no_instruction_tag,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entry
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    cfg = _build_config_from_args(args)
    cfg.ensure_output_dir()
    written = ingest_huggingface_dataset(cfg)
    if not written:
        logger.warning("No episodes were written; check dataset availability or validation errors")
        return 1
    logger.info("Wrote %s episodes to %s", len(written), cfg.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
