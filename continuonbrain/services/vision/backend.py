"""
Vision Backend Interface

Unified interface for vision backends (Hailo, SAM, CPU fallback).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BackendType(Enum):
    """Vision backend types."""
    HAILO_PIPELINE = "hailo_pipeline"
    HAILO_SUBPROCESS = "hailo_subprocess"
    HAILO_POSE = "hailo_pose"
    SAM = "sam"
    CPU_YOLO = "cpu_yolo"
    CPU_OPENCV = "cpu_opencv"
    DEPTH_ENHANCED = "depth_enhanced"


class BackendCapability(Enum):
    """Capabilities that backends can provide."""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    POSE = "pose"


@dataclass
class KeypointResult:
    """Single keypoint from pose estimation."""
    name: str
    x: float
    y: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "x": self.x, "y": self.y, "conf": self.confidence}


@dataclass
class PoseResult:
    """Human pose estimation result."""
    person_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    keypoints: List[KeypointResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "keypoints": [kp.to_dict() for kp in self.keypoints],
        }

    def get_keypoint(self, name: str) -> Optional[KeypointResult]:
        """Get keypoint by name."""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    @property
    def left_wrist(self) -> Optional[KeypointResult]:
        return self.get_keypoint("left_wrist")

    @property
    def right_wrist(self) -> Optional[KeypointResult]:
        return self.get_keypoint("right_wrist")


@dataclass
class DetectionResult:
    """Standardized detection result."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: Optional[int] = None
    mask: Optional[np.ndarray] = None
    depth_mm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "has_mask": self.mask is not None,
            "depth_mm": self.depth_mm,
        }

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class BackendResult:
    """Result from a vision backend operation."""
    ok: bool
    detections: List[DetectionResult] = field(default_factory=list)
    inference_time_ms: float = 0.0
    backend: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ok": self.ok,
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": self.inference_time_ms,
            "backend": self.backend,
            "error": self.error,
            "metadata": self.metadata,
        }


class VisionBackend(ABC):
    """Abstract base class for vision backends."""

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[BackendCapability]:
        """Return supported capabilities."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and ready."""
        pass

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """
        Run object detection on a frame.

        Args:
            frame: RGB numpy array (H, W, 3)
            conf_threshold: Confidence threshold

        Returns:
            BackendResult with detections
        """
        pass

    def segment(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> BackendResult:
        """
        Run segmentation on a frame (optional).

        Default implementation returns empty result.
        """
        return BackendResult(
            ok=False,
            backend=self.backend_type.value,
            error="Segmentation not supported by this backend",
        )

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "type": self.backend_type.value,
            "available": self.is_available(),
            "capabilities": [c.value for c in self.capabilities],
        }

    def shutdown(self) -> None:
        """Release resources (optional)."""
        pass
