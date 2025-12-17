"""
Camera Episode Recorder (RGB + optional depth).

This is a *local* data capture tool for producing RLDS-compatible JSON episodes
that can be consumed by the current JAX WaveCore sanity trainer (which expects
`steps[*].observation` and `steps[*].action`).

Design constraints:
- Keep dependencies optional (Pi/Jetson imports must not break).
- Store media as files on disk and reference them by URI in the RLDS step.
- Include a compact numeric `observation.command` vector so the current trainer
  can ingest the episode without a vision model.

Depth support:
- If `pyrealsense2` is installed and a RealSense device is present, capture
  aligned depth frames.
- Otherwise capture RGB only via OpenCV.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from continuonbrain.synthetic.teachers import Teacher, HttpTeacher
except Exception:  # pragma: no cover
    Teacher = Any  # type: ignore
    HttpTeacher = None  # type: ignore


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _try_import_realsense():
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except Exception:
        return None


def _try_import_depthai():
    try:
        import depthai as dai  # type: ignore

        return dai
    except Exception:
        return None


def _try_import_hailo_feature_extractor():
    """
    Optional Hailo feature extractor hook.

    We keep this import guarded so Pi/Jetson environments without Hailo SDK can
    still import the recorder. Real implementation should live behind this hook.
    """
    try:
        from continuonbrain.runtime.hailo_feature_extractor import HailoFeatureExtractor  # noqa: WPS433

        return HailoFeatureExtractor
    except Exception:
        return None


def _now_ns() -> int:
    return time.time_ns()


def _downsample_grayscale_vector(rgb_bgr: np.ndarray, out_dim: int) -> List[float]:
    """
    Produce a compact numeric vector from an image without a heavy model.
    Strategy: grayscale -> resize to (w*h = out_dim) -> normalize to [0, 1].
    """
    cv2 = _try_import_cv2()
    if cv2 is None:
        # Fallback: naive numpy downsample by slicing.
        gray = rgb_bgr.mean(axis=2).astype(np.float32)
        h, w = gray.shape
        side = int(np.sqrt(out_dim))
        side = max(side, 1)
        ys = np.linspace(0, h - 1, side).astype(int)
        xs = np.linspace(0, w - 1, side).astype(int)
        small = gray[np.ix_(ys, xs)].flatten()
    else:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Choose a rectangle close to out_dim.
        w = int(np.ceil(np.sqrt(out_dim)))
        h = int(np.ceil(out_dim / w))
        small_img = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        small = small_img.flatten()[:out_dim]
    if small.size < out_dim:
        small = np.pad(small, (0, out_dim - small.size))
    # Normalize
    return (small / 255.0).clip(0.0, 1.0).astype(np.float32).tolist()


@dataclass
class CaptureConfig:
    out_dir: Path
    episode_id: str
    num_steps: int = 10
    step_interval_s: float = 0.5
    obs_dim: int = 128
    action_dim: int = 32
    camera_index: int = 0
    width: int = 640
    height: int = 480
    save_jpeg_quality: int = 90
    source: str = "auto"  # auto | opencv | realsense | depthai
    depth_mode: str = "auto"  # off | on | auto
    # If camera/RealSense isn't available (common in WSL), fall back to synthetic frames so
    # the training pipeline can still be exercised end-to-end.
    allow_synthetic_fallback: bool = True
    synthetic_seed: int = 0
    # Optional: when true and Hailo is available, use Hailo-derived embedding for observation.command.
    use_hailo_features: bool = False


class CameraEpisodeRecorder:
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir = self.cfg.out_dir / f"{self.cfg.episode_id}_blobs"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def _synthetic_frame(self, *, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rng = np.random.default_rng(int(self.cfg.synthetic_seed) + int(idx))
        h, w = int(self.cfg.height), int(self.cfg.width)
        # Gradient + noise so embeddings have signal and vary slightly per step.
        x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        base = 0.5 * x + 0.5 * y
        noise = rng.normal(0.0, 0.03, size=(h, w)).astype(np.float32)
        img = (base + noise).clip(0.0, 1.0)
        bgr = np.stack([img, np.roll(img, 15, axis=1), np.roll(img, 25, axis=0)], axis=-1)
        rgb_bgr = (bgr * 255.0).astype(np.uint8)
        return rgb_bgr, None

    def _capture_opencv(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        cv2 = _try_import_cv2()
        if cv2 is None:
            raise RuntimeError("OpenCV not installed. Install `opencv-python` to use webcam capture.")

        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {self.cfg.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from webcam")
        return frame, None

    def _capture_realsense(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rs = _try_import_realsense()
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed. Install it to use RealSense capture.")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.cfg.width, self.cfg.height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.cfg.width, self.cfg.height, rs.format.z16, 30)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        try:
            frames = pipeline.wait_for_frames(timeout_ms=4000)
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color:
                raise RuntimeError("No color frame from RealSense")
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data()) if depth else None
            return color_img, depth_img
        finally:
            pipeline.stop()

    def _capture_depthai(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Capture a single RGB frame and optional aligned depth frame from a Luxonis OAK device.

        - RGB is returned as BGR uint8 (OpenCV-compatible).
        - Depth is returned as uint16 in millimeters when available/aligned.
        """
        dai = _try_import_depthai()
        if dai is None:
            raise RuntimeError("depthai not installed. Install `depthai` to use OAK-D Lite capture.")

        depth_mode = (self.cfg.depth_mode or "auto").lower().strip()
        if depth_mode not in {"off", "on", "auto"}:
            raise RuntimeError(f"Invalid depth_mode={depth_mode}. Expected off|on|auto.")

        # Build pipeline
        pipeline = dai.Pipeline()

        # Color camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(int(self.cfg.width), int(self.cfg.height))
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # Optional stereo depth (OAK-D Lite: left/right mono sensors)
        want_depth = depth_mode in {"on", "auto"}
        if want_depth:
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            # Align to RGB for easy consumption in RLDS.
            try:
                stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            except Exception:
                pass
            stereo.setSubpixel(True)

            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        # Open device and fetch one frame
        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
            q_depth = device.getOutputQueue(name="depth", maxSize=1, blocking=True) if want_depth else None

            rgb_msg = q_rgb.get()
            rgb = rgb_msg.getCvFrame()

            depth = None
            if q_depth is not None:
                try:
                    depth_msg = q_depth.get()
                    depth = depth_msg.getFrame()
                except Exception:
                    depth = None

        if rgb is None:
            raise RuntimeError("Failed to read RGB frame from DepthAI device")
        if depth_mode == "on" and depth is None:
            raise RuntimeError("depth_mode=on but no depth frame available from DepthAI device")

        return rgb, depth

    def _capture(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.cfg.source == "depthai":
            return self._capture_depthai()
        if self.cfg.source == "opencv":
            return self._capture_opencv()
        if self.cfg.source == "realsense":
            return self._capture_realsense()

        # auto
        dai = _try_import_depthai()
        if dai is not None:
            try:
                return self._capture_depthai()
            except Exception:
                pass
        rs = _try_import_realsense()
        if rs is not None:
            try:
                return self._capture_realsense()
            except Exception:
                pass
        return self._capture_opencv()

    def record_episode(
        self,
        *,
        owner_display_name: Optional[str] = None,
        owner_preferred_name: Optional[str] = None,
        owner_id: Optional[str] = None,
        owner_roles: Optional[List[str]] = None,
        instruction: Optional[str] = None,
        privacy_local_only: bool = True,
        teacher: Optional["Teacher"] = None,
    ) -> Path:
        steps: List[Dict[str, Any]] = []

        # Minimal metadata required by validators + v1.1 optional blocks.
        metadata: Dict[str, Any] = {
            "schema_version": "1.1",
            "episode_id": self.cfg.episode_id,
            "xr_mode": "trainer",
            "control_role": "human_supervisor",
            "environment_id": "studio_local_camera",
            "tags": ["origin:studio:local_camera", "training:owner_identity"],
            "software": {
                "xr_app": "continuonbrain-studio-local",
                "continuonbrain_os": "continuonbrain-dev",
                "glove_firmware": "absent",
            },
            "capabilities": {
                "compute": {"device": "laptop", "ram_gb": 0, "gpu": "unknown", "dtype": "fp32"},
                "sensors": {"rgb": True, "depth": False, "audio": False, "imu": False, "glove": False},
                "actuators": {"drive": False, "arm": False, "gripper": False},
                "limits": {"allow_motion": False, "max_speed_mps": 0.0},
            },
            "safety": {
                "content_rating": {"audience": "general", "violence": "none", "language": "clean"},
                # Default conservative flags for owner recognition captures.
                "pii_attestation": {"pii_present": True, "faces_present": True, "name_present": bool(owner_display_name), "consent": True},
                "pii_cleared": False,
                "pii_redacted": False,
                "pending_review": True,
            },
            "share": {
                "public": False,
                "slug": "",
                "title": "",
                "license": "Proprietary",
                "tags": [],
            },
            "provenance": {
                "origin": "origin:studio:local_camera",
                "source_commit": "",
                "source_host": "",
                "notes": "Local owner identity capture; do not publish without PII review.",
            },
        }
        if self.cfg.source == "depthai":
            metadata["environment_id"] = "pi5_depthai_oakd"
            metadata["tags"] = ["origin:pi5:oakd", "training:owner_identity"]
            metadata["provenance"]["origin"] = "origin:pi5:oakd"

        if owner_display_name or owner_preferred_name or owner_id:
            metadata["owner"] = {
                "owner_id": owner_id or "",
                "display_name": owner_display_name or "",
                "preferred_name": owner_preferred_name or "",
                "roles": owner_roles or ["owner"],
                "preferences": {
                    "tone": "curious",
                    "ask_clarifying": "true",
                    "follow_owner_orders": "true",
                },
            }

        # Basic dialog scaffolding (HOPE-style): first turn declares identity + asks for guidance.
        if instruction:
            metadata["tags"].append(f"instruction:{instruction}")

        for idx in range(self.cfg.num_steps):
            t0 = _now_ns()
            try:
                rgb_bgr, depth = self._capture()
            except Exception:
                if not self.cfg.allow_synthetic_fallback:
                    raise
                rgb_bgr, depth = self._synthetic_frame(idx=idx)

            frame_id = f"{self.cfg.episode_id}_frame_{idx:06d}"
            rgb_path = self.blobs_dir / f"{frame_id}.jpg"
            depth_path = self.blobs_dir / f"{frame_id}_depth.npy"

            cv2 = _try_import_cv2()
            if cv2 is None:
                # Minimal fallback: write raw bytes as .npy (not ideal, but avoids extra deps).
                np.save(rgb_path.with_suffix(".npy"), rgb_bgr)
                rgb_uri = str(rgb_path.with_suffix(".npy"))
            else:
                cv2.imwrite(str(rgb_path), rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.save_jpeg_quality)])
                rgb_uri = str(rgb_path)

            depth_uri = None
            if depth is not None:
                np.save(depth_path, depth.astype(np.uint16))
                depth_uri = str(depth_path)
                metadata["capabilities"]["sensors"]["depth"] = True

            obs_vec = _downsample_grayscale_vector(rgb_bgr, self.cfg.obs_dim)
            if self.cfg.use_hailo_features:
                HailoExtractor = _try_import_hailo_feature_extractor()
                if HailoExtractor is not None:
                    try:
                        extractor = HailoExtractor(output_dim=self.cfg.obs_dim)
                        obs_vec = extractor.embed_rgb(rgb_bgr=rgb_bgr)
                    except Exception:
                        # Keep recorder robust; fall back to downsample vector on any failure.
                        obs_vec = obs_vec
            act_vec = [0.0] * self.cfg.action_dim

            # Minimal pose/robot/glove placeholders to satisfy strict validators.
            observation = {
                "headset_pose": {"position": [0.0, 0.0, 0.0], "orientation_quat": [0.0, 0.0, 0.0, 1.0], "valid": False},
                "right_hand_pose": {"position": [0.0, 0.0, 0.0], "orientation_quat": [0.0, 0.0, 0.0, 1.0], "valid": False},
                "left_hand_pose": {"position": [0.0, 0.0, 0.0], "orientation_quat": [0.0, 0.0, 0.0, 1.0], "valid": False},
                "gaze": {"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0], "confidence": 0.0},
                "robot_state": {
                    "timestamp_nanos": int(t0),
                    "joint_positions": [0.0, 0.0],
                    "joint_velocities": [0.0, 0.0],
                    "joint_efforts": [],
                    "end_effector_pose": {"position": [0.0, 0.0, 0.0], "orientation_quat": [0.0, 0.0, 0.0, 1.0], "valid": False},
                    "end_effector_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "gripper_open": True,
                    "frame_id": "camera_link",
                    "wall_time_millis": int(time.time() * 1000),
                },
                "glove": {
                    "timestamp_nanos": int(t0),
                    "flex": [0, 0, 0, 0, 0],
                    "fsr": [0, 0, 0, 0, 0, 0, 0, 0],
                    "orientation_quat": [0, 0, 0, 0],
                    "accel": [0, 0, 0],
                    "valid": False,
                },
                "diagnostics": {"latency_ms": 0.0, "glove_drops": 0, "ble_rssi": 0, "glove_sample_rate_hz": 0.0},
                "video_frame_id": frame_id,
                "depth_frame_id": frame_id,
                # JAX trainer-friendly vector.
                "command": obs_vec,
                # v1.1 media refs
                "media": {
                    "rgb": {"uri": rgb_uri, "frame_id": frame_id, "timestamp_ns": int(t0), "width": int(rgb_bgr.shape[1]), "height": int(rgb_bgr.shape[0]), "format": "jpeg" if rgb_uri.endswith(".jpg") else "npy"},
                    "depth": {"uri": depth_uri or "", "frame_id": frame_id, "timestamp_ns": int(t0), "width": int(rgb_bgr.shape[1]), "height": int(rgb_bgr.shape[0]), "format": "npy", "units": "mm"} if depth_uri else None,
                },
            }
            # Remove null depth block to keep JSON clean.
            if observation["media"].get("depth") is None:
                observation["media"].pop("depth", None)

            # Dialog: weave identity + curiosity about owner guidance.
            if idx == 0:
                user_text = (
                    f"I am {owner_display_name or 'the owner'}. "
                    f"My preferred name is {owner_preferred_name or (owner_display_name or 'Craig')}. "
                    "Please learn who I am and be curious about my guidance."
                )
            else:
                user_text = "What do you notice in the scene right now, and what question should you ask me to learn better?"

            assistant_text = (
                "Acknowledged. I'll treat you as the owner/operator for guidance and ask a concise clarifying question each time. "
                "Clarifying question: what task should we practice next?"
            )

            observation["dialog"] = {
                "speaker": "user",
                "text": user_text,
                "turn_id": f"t{idx:03d}u",
                "conversation_id": self.cfg.episode_id,
            }

            action = {
                "command": act_vec,
                "source": "human_supervisor",
                "dialog": {
                    "speaker": "assistant",
                    "text": assistant_text,
                    "turn_id": f"t{idx:03d}a",
                    "conversation_id": self.cfg.episode_id,
                },
            }

            # Optional helper teacher (tiny LLM/VLA) for better synthetic bootstrapping:
            # - can override observation.command with a learned embedding
            # - can provide caption/planner/tool traces for later distillation
            step_extra: Dict[str, str] = {}
            if teacher is not None:
                try:
                    teacher_prompt = (
                        "You are a small helper model used to enrich a robotics RLDS episode.\n"
                        "Return JSON with any of: embedding (len obs_dim), caption, planner, action_command (len action_dim), extra.\n"
                        "Be concise."
                    )
                    res = teacher.infer(
                        rgb_bgr=rgb_bgr,
                        depth=depth,
                        prompt=teacher_prompt,
                        obs_dim=self.cfg.obs_dim,
                        action_dim=self.cfg.action_dim,
                        context={"episode_id": self.cfg.episode_id, "step": idx},
                    )
                    if res.embedding is not None:
                        observation["command"] = res.embedding
                    if res.action_command is not None:
                        action["command"] = res.action_command
                    if res.caption:
                        step_extra["teacher.caption"] = res.caption[:500]
                    if res.planner:
                        action["planner"] = res.planner
                    # Tool calls: if teacher provides them in extra as JSON, stash in action.tool_calls for schema v1.1.
                    # (We keep this optional and tolerant of failures.)
                    if res.extra and "tool_calls" in res.extra:
                        try:
                            tool_calls = res.extra["tool_calls"]
                            if isinstance(tool_calls, list):
                                action["tool_calls"] = tool_calls
                        except Exception:
                            pass
                    if res.extra:
                        # Flatten extra into step_metadata strings.
                        for k, v in res.extra.items():
                            step_extra[f"teacher.{k}"] = str(v)[:500]
                except Exception:
                    # Teacher failures should never break recording.
                    pass

            steps.append(
                {
                    "observation": observation,
                    "action": action,
                    "is_terminal": idx == (self.cfg.num_steps - 1),
                    "step_metadata": {
                        "privacy_local_only": "true" if privacy_local_only else "false",
                        "training_objective": "owner_identity_and_curiosity",
                        **step_extra,
                    },
                }
            )

            if self.cfg.step_interval_s > 0:
                time.sleep(self.cfg.step_interval_s)

        episode = {"metadata": metadata, "steps": steps}
        out_path = self.cfg.out_dir / f"{self.cfg.episode_id}.json"
        out_path.write_text(json.dumps(episode, indent=2), encoding="utf-8")
        return out_path


