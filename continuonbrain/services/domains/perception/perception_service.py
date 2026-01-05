"""
Perception Service

Domain service for perception/vision functionality.
Wraps VisionManager and provides scene understanding.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from continuonbrain.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class PerceptionService:
    """
    Perception domain service implementing IPerceptionService.

    Wraps VisionManager and provides:
    - Object detection
    - Scene understanding
    - Depth processing
    - Visual segmentation
    """

    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        container: Optional["ServiceContainer"] = None,
        enable_hailo: bool = True,
        enable_sam: bool = True,
        **kwargs,
    ):
        """
        Initialize perception service.

        Args:
            config_dir: Configuration directory
            container: Service container for dependencies
            enable_hailo: Enable Hailo NPU backend
            enable_sam: Enable SAM segmentation backend
        """
        self.config_dir = Path(config_dir)
        self._container = container
        self._enable_hailo = enable_hailo
        self._enable_sam = enable_sam

        self._vision_manager = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of vision manager."""
        if self._initialized:
            return

        self._initialized = True

        try:
            from continuonbrain.services.vision import VisionManager

            self._vision_manager = VisionManager(
                enable_hailo=self._enable_hailo,
                enable_sam=self._enable_sam,
                enable_cpu_fallback=True,
                model_dir=str(self.config_dir / "model"),
            )
            logger.info("Perception service initialized with VisionManager")

        except ImportError as e:
            logger.warning(f"VisionManager not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize perception: {e}")

    def detect_objects(
        self,
        frame: Optional[np.ndarray] = None,
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Detect objects in frame."""
        self._ensure_initialized()

        if frame is None:
            # Try to capture from hardware service
            if self._container:
                try:
                    frame_data = self._container.hardware.capture_frame()
                    if frame_data:
                        frame = frame_data.get("rgb")
                except Exception:
                    pass

        if frame is None:
            return []

        if not self._vision_manager:
            return []

        result = self._vision_manager.detect(frame, conf_threshold)

        if not result.ok:
            return []

        return [det.to_dict() for det in result.detections]

    def describe_scene(
        self,
        frame: Optional[np.ndarray] = None,
    ) -> str:
        """Get natural language description of scene."""
        detections = self.detect_objects(frame)

        if not detections:
            return "No objects detected in the scene."

        # Build description from detections
        object_counts = {}
        for det in detections:
            label = det["label"]
            object_counts[label] = object_counts.get(label, 0) + 1

        parts = []
        for label, count in sorted(object_counts.items(), key=lambda x: -x[1]):
            if count == 1:
                parts.append(f"a {label}")
            else:
                parts.append(f"{count} {label}s")

        if len(parts) == 1:
            return f"I can see {parts[0]}."
        elif len(parts) == 2:
            return f"I can see {parts[0]} and {parts[1]}."
        else:
            return f"I can see {', '.join(parts[:-1])}, and {parts[-1]}."

    def get_scene_representation(
        self,
        frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Get full scene representation."""
        import time

        detections = self.detect_objects(frame)
        description = self.describe_scene(frame)

        return {
            "objects": detections,
            "depth_map": None,  # Would be populated if depth available
            "description": description,
            "statistics": {
                "num_objects": len(detections),
                "labels": list(set(d["label"] for d in detections)),
            },
            "timestamp": time.time(),
        }

    def segment(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Segment objects using SAM."""
        self._ensure_initialized()

        if not self._vision_manager:
            return {"masks": [], "scores": [], "boxes": []}

        result = self._vision_manager.segment(frame, prompt, points)

        if not result.ok:
            return {"masks": [], "scores": [], "boxes": [], "error": result.error}

        return {
            "masks": [det.mask for det in result.detections if det.mask is not None],
            "scores": [det.confidence for det in result.detections],
            "boxes": [list(det.bbox) for det in result.detections],
        }

    def get_capabilities(self) -> Dict[str, bool]:
        """Get perception capability flags."""
        self._ensure_initialized()

        if not self._vision_manager:
            return {
                "detection": False,
                "segmentation": False,
                "depth": False,
                "scene_description": False,
            }

        caps = self._vision_manager.get_capabilities()
        capabilities = caps.get("capabilities", [])

        return {
            "detection": "detection" in capabilities,
            "segmentation": "segmentation" in capabilities,
            "depth": "depth" in capabilities,
            "scene_description": True,
        }

    def is_available(self) -> bool:
        """Check if perception service is available."""
        self._ensure_initialized()
        return self._vision_manager is not None and self._vision_manager.is_available()

    def shutdown(self) -> None:
        """Shutdown perception service."""
        if self._vision_manager:
            self._vision_manager.shutdown()
        self._vision_manager = None
        self._initialized = False
