"""
SAM Vision Backend

Wraps SAMVisionService with VisionBackend interface for segmentation.
Supports SAM, SAM2, and SAM-HQ models.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .backend import (
    VisionBackend,
    BackendType,
    BackendCapability,
    BackendResult,
    DetectionResult,
)

logger = logging.getLogger(__name__)


class SAMBackend(VisionBackend):
    """
    SAM segmentation backend.

    Supports SAM2, SAM, SAM3, and SAM-HQ models for semantic segmentation.
    """

    def __init__(
        self,
        model_type: str = "auto",
        device: str = "auto",
    ):
        """
        Initialize SAM backend.

        Args:
            model_type: Model to use ("auto", "sam2", "sam", "sam_hq")
            device: Device to use ("auto", "cuda", "cpu")
        """
        self._service = None
        self._available = False
        self._model_type = model_type
        self._device = device
        self._init_attempted = False

    def _init_service(self) -> None:
        """Initialize the SAM service."""
        if self._init_attempted:
            return

        self._init_attempted = True

        try:
            from continuonbrain.services.sam3_vision import create_sam_service

            self._service = create_sam_service(
                device=self._device,
                model_type=self._model_type,
            )
            self._available = self._service.is_available()

            if self._available:
                logger.info(f"SAMBackend initialized: {self._service.get_available_models()}")

        except ImportError as e:
            logger.warning(f"SAMBackend: transformers/torch not available: {e}")
        except Exception as e:
            logger.error(f"SAMBackend initialization failed: {e}")

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SAM

    @property
    def capabilities(self) -> List[BackendCapability]:
        return [BackendCapability.SEGMENTATION, BackendCapability.DETECTION]

    def is_available(self) -> bool:
        if not self._init_attempted:
            self._init_service()
        return self._available and self._service is not None

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """
        Run detection via automatic mask generation.

        Note: This is slower than dedicated detection backends.
        Use for segmentation-based detection when Hailo is unavailable.
        """
        if not self.is_available():
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error="SAM service not available",
            )

        try:
            import time
            start = time.time()

            # Ensure initialized
            if not self._service._initialized:
                self._service.initialize()

            # Find objects using auto mask generation
            objects = self._service.find_objects(
                frame,
                min_area=100,
                max_objects=20,
            )
            inference_ms = (time.time() - start) * 1000

            detections = [
                DetectionResult(
                    label="object",
                    confidence=obj.get("score", 0.0),
                    bbox=tuple(int(v) for v in obj.get("box_xyxy", [0, 0, 0, 0])),
                    mask=obj.get("mask"),
                )
                for obj in objects
                if obj.get("score", 0) >= conf_threshold
            ]

            return BackendResult(
                ok=True,
                detections=detections,
                inference_time_ms=inference_ms,
                backend=self.backend_type.value,
            )

        except Exception as e:
            logger.error(f"SAM detection failed: {e}")
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def segment(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> BackendResult:
        """Run segmentation with text or point prompts."""
        if not self.is_available():
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error="SAM service not available",
            )

        try:
            import time
            start = time.time()

            if not self._service._initialized:
                self._service.initialize()

            if prompt:
                result = self._service.segment_text(frame, prompt)
            elif points:
                result = self._service.segment_points(
                    frame,
                    [[p[0], p[1]] for p in points],
                    [1] * len(points),
                )
            else:
                return BackendResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error="Either prompt or points required for segmentation",
                )

            inference_ms = (time.time() - start) * 1000

            if result is None:
                return BackendResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error="Segmentation returned no result",
                    inference_time_ms=inference_ms,
                )

            detections = []
            for i, mask in enumerate(result.masks):
                score = result.scores[i] if i < len(result.scores) else 0.5
                bbox = result.boxes_xyxy[i] if i < len(result.boxes_xyxy) else [0, 0, 0, 0]

                detections.append(DetectionResult(
                    label=prompt or "segment",
                    confidence=score,
                    bbox=tuple(int(v) for v in bbox),
                    mask=mask if isinstance(mask, np.ndarray) else None,
                ))

            return BackendResult(
                ok=True,
                detections=detections,
                inference_time_ms=inference_ms,
                backend=self.backend_type.value,
            )

        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get detailed backend status."""
        status = super().get_status()
        if self._service:
            try:
                status["models"] = self._service.get_available_models()
                status["current_model"] = self._service.current_model
            except Exception:
                pass
        return status

    def shutdown(self) -> None:
        """Unload SAM model."""
        if self._service:
            try:
                self._service.unload()
            except Exception as e:
                logger.warning(f"Error unloading SAM: {e}")
