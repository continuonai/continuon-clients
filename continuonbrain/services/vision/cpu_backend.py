"""
CPU Vision Backend

Fallback backend for when Hailo or SAM are not available.
Uses OpenCV for basic detection capabilities.
"""
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from .backend import (
    VisionBackend,
    BackendType,
    BackendCapability,
    BackendResult,
    DetectionResult,
)

logger = logging.getLogger(__name__)


class CPUBackend(VisionBackend):
    """
    CPU fallback backend using OpenCV.

    Provides basic detection capabilities when hardware accelerators
    are not available. Performance is significantly lower than Hailo.
    """

    def __init__(self):
        """Initialize CPU backend."""
        self._available = False
        self._cascade = None
        self._init_attempted = False

    def _init_detector(self) -> None:
        """Initialize the detector."""
        if self._init_attempted:
            return

        self._init_attempted = True

        try:
            import cv2

            # Try to load a pre-trained cascade for face/object detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._cascade = cv2.CascadeClassifier(cascade_path)

            if self._cascade.empty():
                logger.warning("CPUBackend: Failed to load cascade classifier")
                self._available = False
            else:
                self._available = True
                logger.info("CPUBackend initialized with Haar cascade")

        except ImportError:
            logger.warning("CPUBackend: OpenCV not available")
            self._available = False
        except Exception as e:
            logger.warning(f"CPUBackend initialization failed: {e}")
            self._available = False

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CPU_OPENCV

    @property
    def capabilities(self) -> List[BackendCapability]:
        return [BackendCapability.DETECTION]

    def is_available(self) -> bool:
        if not self._init_attempted:
            self._init_detector()
        return self._available

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """
        Run detection using OpenCV cascade classifier.

        Note: This is a very basic detector suitable only as a fallback.
        It primarily detects faces using Haar cascades.
        """
        if not self.is_available():
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error="CPU detector not available",
            )

        try:
            import cv2
            import time

            start = time.time()

            # Convert to grayscale for cascade
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Run detection
            detections_raw = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            inference_ms = (time.time() - start) * 1000

            # Convert to standard format
            detections = []
            for (x, y, w, h) in detections_raw:
                # Haar cascade doesn't provide confidence, use placeholder
                detections.append(DetectionResult(
                    label="face",
                    confidence=0.8,  # Placeholder confidence
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                    class_id=0,
                ))

            return BackendResult(
                ok=True,
                detections=detections,
                inference_time_ms=inference_ms,
                backend=self.backend_type.value,
                metadata={
                    "num_detections": len(detections),
                    "detector": "haar_cascade",
                },
            )

        except Exception as e:
            logger.error(f"CPU detection failed: {e}")
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        status = super().get_status()
        status["detector"] = "haar_cascade" if self._available else None
        return status
