"""
Pose Estimation Backend

Hailo-accelerated YOLOv8-pose for human pose estimation.
Provides 17 COCO keypoints per detected person.
"""
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .backend import (
    VisionBackend,
    BackendType,
    BackendCapability,
    BackendResult,
    DetectionResult,
    PoseResult,
    KeypointResult,
)

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HEF_PATH = Path("/usr/share/hailo-models/yolov8s_pose_h8.hef")
WORKER_PATH = Path(__file__).parent.parent / "hailo_yolov8_pose_worker.py"

# Skeleton connections for visualization
SKELETON = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


class PoseBackend(VisionBackend):
    """
    Hailo-accelerated pose estimation backend.

    Uses YOLOv8-pose running on Hailo-8 NPU via subprocess worker
    for stability and isolation.
    """

    def __init__(
        self,
        hef_path: Optional[Path] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        timeout: float = 10.0,
    ):
        """
        Initialize pose backend.

        Args:
            hef_path: Path to YOLOv8-pose HEF file
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            timeout: Subprocess timeout in seconds
        """
        self._hef_path = hef_path or DEFAULT_HEF_PATH
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._timeout = timeout
        self._available = None
        self._last_poses: List[PoseResult] = []
        self._last_inference_ms = 0.0

    @property
    def backend_type(self) -> BackendType:
        return BackendType.HAILO_POSE

    @property
    def capabilities(self) -> List[BackendCapability]:
        return [BackendCapability.POSE, BackendCapability.DETECTION]

    def is_available(self) -> bool:
        """Check if pose model and worker are available."""
        if self._available is None:
            self._available = (
                self._hef_path.exists() and
                WORKER_PATH.exists()
            )
            if self._available:
                logger.info(f"Pose backend available: {self._hef_path}")
            else:
                logger.warning(f"Pose backend not available. HEF: {self._hef_path.exists()}, Worker: {WORKER_PATH.exists()}")
        return self._available

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """
        Run pose estimation as detection (returns person bboxes).

        For AINA integration, use detect_poses() instead.
        """
        poses = self.detect_poses(frame, conf_threshold)

        # Convert poses to detections (person bboxes)
        detections = [
            DetectionResult(
                label="person",
                confidence=pose.confidence,
                bbox=pose.bbox,
                class_id=0,
            )
            for pose in poses
        ]

        return BackendResult(
            ok=True,
            detections=detections,
            inference_time_ms=self._last_inference_ms,
            backend=self.backend_type.value,
            metadata={
                "num_poses": len(poses),
                "has_keypoints": True,
            },
        )

    def detect_poses(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[PoseResult]:
        """
        Run pose estimation and return full pose data.

        Args:
            frame: RGB numpy array (H, W, 3)
            conf_threshold: Override default confidence threshold

        Returns:
            List of PoseResult with keypoints
        """
        if not self.is_available():
            logger.warning("Pose backend not available")
            return []

        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available for image encoding")
            return []

        conf = conf_threshold or self._conf_threshold

        # Encode frame as JPEG
        success, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Failed to encode frame as JPEG")
            return []

        # Call worker subprocess
        try:
            result = subprocess.run(
                [
                    sys.executable, str(WORKER_PATH),
                    "--hef", str(self._hef_path),
                    "--conf", str(conf),
                    "--iou", str(self._iou_threshold),
                ],
                input=jpeg_bytes.tobytes(),
                capture_output=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Pose worker timed out")
            return []
        except Exception as e:
            logger.error(f"Pose worker failed: {e}")
            return []

        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')[:200]
            logger.error(f"Pose worker error: {stderr}")
            return []

        # Parse output
        try:
            output = result.stdout.decode('utf-8')
            data = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pose worker output: {e}")
            return []

        if not data.get("ok"):
            logger.warning(f"Pose worker returned error: {data.get('error')}")
            return []

        self._last_inference_ms = data.get("inference_time_ms", 0.0)

        # Convert to PoseResult objects
        poses = []
        for pose_data in data.get("poses", []):
            keypoints = [
                KeypointResult(
                    name=kp["name"],
                    x=kp["x"],
                    y=kp["y"],
                    confidence=kp["conf"],
                )
                for kp in pose_data.get("keypoints", [])
            ]

            bbox = pose_data.get("bbox", [0, 0, 0, 0])
            pose = PoseResult(
                person_id=pose_data.get("person_id", 0),
                confidence=pose_data.get("confidence", 0.0),
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                keypoints=keypoints,
            )
            poses.append(pose)

        self._last_poses = poses
        logger.debug(f"Pose detection found {len(poses)} people in {self._last_inference_ms:.1f}ms")

        return poses

    def get_wrist_positions(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Get wrist positions for AINA hand tracking.

        Args:
            frame: RGB numpy array
            min_confidence: Minimum keypoint confidence

        Returns:
            List of wrist positions with metadata
        """
        poses = self.detect_poses(frame)

        wrists = []
        for pose in poses:
            if pose.left_wrist and pose.left_wrist.confidence >= min_confidence:
                wrists.append({
                    "hand": "left",
                    "x": pose.left_wrist.x,
                    "y": pose.left_wrist.y,
                    "conf": pose.left_wrist.confidence,
                    "person_id": pose.person_id,
                })
            if pose.right_wrist and pose.right_wrist.confidence >= min_confidence:
                wrists.append({
                    "hand": "right",
                    "x": pose.right_wrist.x,
                    "y": pose.right_wrist.y,
                    "conf": pose.right_wrist.confidence,
                    "person_id": pose.person_id,
                })

        return wrists

    def draw_poses(
        self,
        frame: np.ndarray,
        poses: Optional[List[PoseResult]] = None,
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        keypoint_radius: int = 4,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw poses on frame for visualization.

        Args:
            frame: RGB numpy array to draw on
            poses: Poses to draw (uses last detection if None)
            draw_skeleton: Draw skeleton lines
            draw_keypoints: Draw keypoint circles
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines

        Returns:
            Frame with poses drawn
        """
        try:
            import cv2
        except ImportError:
            return frame

        if poses is None:
            poses = self._last_poses

        overlay = frame.copy()

        # Colors for different body parts
        colors = {
            "face": (0, 255, 255),  # Yellow
            "arm_left": (0, 255, 0),  # Green
            "arm_right": (255, 0, 0),  # Blue
            "body": (255, 255, 0),  # Cyan
            "leg_left": (0, 165, 255),  # Orange
            "leg_right": (255, 0, 255),  # Magenta
        }

        def get_color(kp_idx: int) -> Tuple[int, int, int]:
            if kp_idx <= 4:
                return colors["face"]
            elif kp_idx in [5, 7, 9]:
                return colors["arm_left"]
            elif kp_idx in [6, 8, 10]:
                return colors["arm_right"]
            elif kp_idx in [11, 12]:
                return colors["body"]
            elif kp_idx in [11, 13, 15]:
                return colors["leg_left"]
            else:
                return colors["leg_right"]

        for pose in poses:
            # Draw bounding box
            x1, y1, x2, y2 = pose.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence label
            label = f"Person {pose.person_id} ({pose.confidence:.0%})"
            cv2.putText(overlay, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Build keypoint positions
            kp_positions = {}
            for i, kp in enumerate(pose.keypoints):
                if kp.confidence > 0.3:
                    kp_positions[i] = (int(kp.x), int(kp.y))

            # Draw skeleton
            if draw_skeleton:
                for start_idx, end_idx in SKELETON:
                    if start_idx in kp_positions and end_idx in kp_positions:
                        pt1 = kp_positions[start_idx]
                        pt2 = kp_positions[end_idx]
                        color = get_color(start_idx)
                        cv2.line(overlay, pt1, pt2, color, line_thickness)

            # Draw keypoints
            if draw_keypoints:
                for i, (x, y) in kp_positions.items():
                    color = get_color(i)
                    cv2.circle(overlay, (x, y), keypoint_radius, color, -1)
                    cv2.circle(overlay, (x, y), keypoint_radius, (255, 255, 255), 1)

        return overlay

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "type": self.backend_type.value,
            "available": self.is_available(),
            "capabilities": [c.value for c in self.capabilities],
            "hef_path": str(self._hef_path),
            "hef_exists": self._hef_path.exists(),
            "worker_exists": WORKER_PATH.exists(),
            "last_inference_ms": self._last_inference_ms,
            "last_num_poses": len(self._last_poses),
        }

    def shutdown(self) -> None:
        """Release resources."""
        self._last_poses = []
        self._available = None
