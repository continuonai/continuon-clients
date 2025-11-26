"""
Adapter bridging WaveCore policy embeddings and RLDS-aligned metadata to VLA heads.

The adapter keeps the RLDS schema intact while letting callers swap routing
strategies (Wave-only, attention-only, or hybrid) without changing the input
contract. It preserves `frame_id` alignment and forwards the contextual signals
Fast SkillPolicies/LanguagePlanner heads depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional


RoutingMode = Literal["wave_only", "attention_only", "hybrid"]


@dataclass
class RldsFrameMetadata:
    """Parsed RLDS observation fields relevant to VLA heads."""

    frame_id: Optional[str]
    headset_pose: Optional[Dict[str, Any]]
    hand_poses: Dict[str, Dict[str, Any]]
    gaze: Optional[Dict[str, Any]]
    audio: Optional[Dict[str, Any]]
    glove: Optional[Dict[str, Any]]
    robot_state: Optional[Dict[str, Any]]
    ui_context: Optional[Dict[str, Any]]
    step_metadata: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]]


@dataclass
class PolicyAdapterSample:
    """Payload forwarded to a minimal VLA head."""

    frame_id: Optional[str]
    wave_embedding: Any
    attention_context: Dict[str, Any]
    diagnostics: Dict[str, Any]


@dataclass
class AdapterRoutingConfig:
    """Routing controls for Wave vs. attention inputs."""

    mode: RoutingMode = "hybrid"
    ensure_frame_alignment: bool = True
    attach_diagnostics: bool = True


def _extract_frame_id(obs: Dict[str, Any]) -> Optional[str]:
    video_frame = None
    if isinstance(obs.get("egocentric_video"), dict):
        video_frame = obs["egocentric_video"].get("frame_id")
    depth_frame = None
    if isinstance(obs.get("egocentric_depth"), dict):
        depth_frame = obs["egocentric_depth"].get("frame_id")
    audio_frame = None
    if isinstance(obs.get("audio"), dict):
        audio_frame = obs["audio"].get("frame_id")

    candidates = [val for val in (video_frame, depth_frame, audio_frame) if val]
    if not candidates:
        return None
    if len({*candidates}) > 1:
        raise ValueError(f"Frame_id mismatch across observation blocks: {candidates}")
    return candidates[0]


def map_rlds_observation(obs: Dict[str, Any]) -> RldsFrameMetadata:
    """
    Map RLDS observation fields into a structured bundle for the adapter.

    The mapping mirrors the schema in docs/rlds-schema.md so downstream heads
    receive consistent pose/gaze/audio/glove context alongside `frame_id`.
    """

    frame_id = _extract_frame_id(obs)
    hand_poses: Dict[str, Dict[str, Any]] = {}
    for key in ("xr_hand_left_pose", "xr_hand_right_pose"):
        if key in obs:
            hand_poses[key] = obs[key]

    glove_block = None
    if "glove" in obs:
        glove_block = {
            "flex": obs["glove"].get("flex"),
            "fsr": obs["glove"].get("fsr"),
            "orientation_quat": obs["glove"].get("orientation_quat"),
            "accel": obs["glove"].get("accel"),
            "valid": obs["glove"].get("valid"),
        }

    metadata = RldsFrameMetadata(
        frame_id=frame_id,
        headset_pose=obs.get("xr_headset_pose"),
        hand_poses=hand_poses,
        gaze=obs.get("gaze"),
        audio=obs.get("audio"),
        glove=glove_block,
        robot_state=obs.get("robot_state"),
        ui_context=obs.get("ui_context"),
        step_metadata=obs.get("step_metadata", {}),
        diagnostics=obs.get("diagnostics"),
    )
    return metadata


def build_policy_request(
    *,
    wavecore_output: Dict[str, Any],
    rlds_observation: Dict[str, Any],
    routing: AdapterRoutingConfig | None = None,
) -> PolicyAdapterSample:
    """
    Combine WaveCore policy embeddings with RLDS context for a VLA head.

    Args:
        wavecore_output: Result of WaveCore `encode_for_policy()`. Expected to
            carry a policy embedding and optional `frame_id`/extras.
        rlds_observation: Parsed RLDS observation dict for the current step.
        routing: Optional AdapterRoutingConfig controlling Wave vs attention
            routing and frame alignment.
    """

    cfg = routing or AdapterRoutingConfig()
    metadata = map_rlds_observation(rlds_observation)

    frame_id = metadata.frame_id or wavecore_output.get("frame_id")
    if cfg.ensure_frame_alignment and metadata.frame_id and wavecore_output.get("frame_id"):
        if metadata.frame_id != wavecore_output["frame_id"]:
            raise ValueError(
                f"frame_id mismatch between RLDS ({metadata.frame_id}) and WaveCore ({wavecore_output['frame_id']})"
            )

    wave_embedding = wavecore_output.get("policy_embedding", wavecore_output)

    attention_context: Dict[str, Any] = {
        "headset_pose": metadata.headset_pose,
        "hand_poses": metadata.hand_poses,
        "gaze": metadata.gaze,
        "audio": metadata.audio,
        "glove": metadata.glove,
        "robot_state": metadata.robot_state,
        "ui_context": metadata.ui_context,
        "step_metadata": metadata.step_metadata,
    }

    if cfg.mode == "wave_only":
        attention_context = {}
    elif cfg.mode == "attention_only":
        wave_embedding = None

    diagnostics = metadata.diagnostics or {}
    if cfg.attach_diagnostics:
        diagnostics = {"frame_id": frame_id, **diagnostics}

    return PolicyAdapterSample(
        frame_id=frame_id,
        wave_embedding=wave_embedding,
        attention_context=attention_context,
        diagnostics=diagnostics,
    )


def forward_to_vla_head(
    head_fn: Callable[..., Any],
    sample: PolicyAdapterSample,
) -> Any:
    """
    Forward the adapter sample into a minimal VLA head.

    `head_fn` can be a SkillPolicies or LanguagePlanner callable accepting
    keyword args for wave embeddings, attention context, and diagnostics.
    """

    return head_fn(
        wave_embedding=sample.wave_embedding,
        attention_context=sample.attention_context,
        frame_id=sample.frame_id,
        diagnostics=sample.diagnostics,
    )
