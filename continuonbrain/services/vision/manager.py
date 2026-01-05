"""
Vision Manager

Unified vision system that manages multiple backends with automatic fallback.
Replaces the scattered VisionCore, SAM3Vision, and HailoVision implementations.
"""
import logging
import time
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


class VisionManager:
    """
    Unified vision manager with backend fallback.

    Provides:
    - Backend discovery and initialization
    - Automatic fallback when backends fail
    - Unified result format
    - Statistics and monitoring

    Usage:
        manager = VisionManager()
        result = manager.detect(frame, conf_threshold=0.3)
        if result.ok:
            for det in result.detections:
                print(f"{det.label}: {det.confidence:.2f}")
    """

    def __init__(
        self,
        enable_hailo: bool = True,
        enable_sam: bool = True,
        enable_cpu_fallback: bool = True,
        model_dir: Optional[str] = None,
        lazy_init: bool = True,
    ):
        """
        Initialize VisionManager.

        Args:
            enable_hailo: Enable Hailo NPU backend
            enable_sam: Enable SAM segmentation backend
            enable_cpu_fallback: Enable CPU fallback detection
            model_dir: Directory for model files
            lazy_init: Defer backend initialization until first use
        """
        self._backends: Dict[BackendType, VisionBackend] = {}
        self._backend_priority: List[BackendType] = []
        self._model_dir = model_dir

        # Configuration
        self._enable_hailo = enable_hailo
        self._enable_sam = enable_sam
        self._enable_cpu_fallback = enable_cpu_fallback
        self._lazy_init = lazy_init

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "fallback_count": 0,
            "backend_usage": {},
            "last_inference_ms": 0.0,
        }

        if not lazy_init:
            self._init_backends()

    def _init_backends(self) -> None:
        """Initialize all enabled backends."""
        if self._enable_hailo:
            self._init_hailo()

        if self._enable_sam:
            self._init_sam()

        if self._enable_cpu_fallback:
            self._init_cpu_fallback()

        logger.info(f"VisionManager initialized with backends: {list(self._backends.keys())}")

    def _ensure_initialized(self) -> None:
        """Ensure backends are initialized (lazy initialization)."""
        if not self._backends and self._lazy_init:
            self._init_backends()

    def _init_hailo(self) -> None:
        """Initialize Hailo backend."""
        try:
            from .hailo_backend import HailoBackend

            backend = HailoBackend(model_dir=self._model_dir)
            if backend.is_available():
                self._backends[BackendType.HAILO_PIPELINE] = backend
                self._backend_priority.append(BackendType.HAILO_PIPELINE)
                self._stats["backend_usage"][BackendType.HAILO_PIPELINE.value] = 0
                logger.info("Hailo backend available")
            else:
                logger.info("Hailo backend not available")
        except Exception as e:
            logger.warning(f"Failed to init Hailo backend: {e}")

    def _init_sam(self) -> None:
        """Initialize SAM backend."""
        try:
            from .sam_backend import SAMBackend

            backend = SAMBackend()
            if backend.is_available():
                self._backends[BackendType.SAM] = backend
                # SAM is lower priority for detection (slower)
                self._backend_priority.append(BackendType.SAM)
                self._stats["backend_usage"][BackendType.SAM.value] = 0
                logger.info("SAM backend available")
            else:
                logger.info("SAM backend not available")
        except Exception as e:
            logger.warning(f"Failed to init SAM backend: {e}")

    def _init_cpu_fallback(self) -> None:
        """Initialize CPU fallback backend."""
        try:
            from .cpu_backend import CPUBackend

            backend = CPUBackend()
            if backend.is_available():
                self._backends[BackendType.CPU_YOLO] = backend
                self._backend_priority.append(BackendType.CPU_YOLO)
                self._stats["backend_usage"][BackendType.CPU_YOLO.value] = 0
                logger.info("CPU fallback available")
        except ImportError:
            logger.debug("CPU fallback not available (missing dependencies)")
        except Exception as e:
            logger.debug(f"CPU fallback not available: {e}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        preferred_backend: Optional[BackendType] = None,
    ) -> BackendResult:
        """
        Run detection with automatic fallback.

        Args:
            frame: RGB numpy array (H, W, 3)
            conf_threshold: Confidence threshold
            preferred_backend: Override default backend priority

        Returns:
            BackendResult with detections from first successful backend
        """
        self._ensure_initialized()
        self._stats["total_requests"] += 1

        # Build priority list
        priority = []
        if preferred_backend and preferred_backend in self._backends:
            priority.append(preferred_backend)
        priority.extend([b for b in self._backend_priority if b not in priority])

        if not priority:
            return BackendResult(
                ok=False,
                backend="none",
                error="No backends available",
            )

        last_error = None
        for backend_type in priority:
            if backend_type not in self._backends:
                continue

            backend = self._backends[backend_type]
            if not backend.is_available():
                continue

            try:
                result = backend.detect(frame, conf_threshold)

                if result.ok:
                    self._stats["successful_requests"] += 1
                    self._stats["backend_usage"][backend_type.value] = \
                        self._stats["backend_usage"].get(backend_type.value, 0) + 1
                    self._stats["last_inference_ms"] = result.inference_time_ms

                    if backend_type != priority[0]:
                        self._stats["fallback_count"] += 1

                    return result
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Backend {backend_type.value} failed: {e}")

        return BackendResult(
            ok=False,
            backend="none",
            error=last_error or "All backends failed",
        )

    def segment(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        points: Optional[List[Tuple[int, int]]] = None,
    ) -> BackendResult:
        """
        Run segmentation using SAM backend.

        Args:
            frame: RGB numpy array
            prompt: Text prompt for segmentation
            points: List of (x, y) points to segment around

        Returns:
            BackendResult with segmentation masks
        """
        self._ensure_initialized()

        if BackendType.SAM not in self._backends:
            return BackendResult(
                ok=False,
                backend="none",
                error="SAM backend not available",
            )

        return self._backends[BackendType.SAM].segment(frame, prompt, points)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities across all backends."""
        self._ensure_initialized()

        capabilities = set()
        for backend in self._backends.values():
            capabilities.update(c.value for c in backend.capabilities)

        return {
            "backends": {
                bt.value: backend.get_status()
                for bt, backend in self._backends.items()
            },
            "capabilities": list(capabilities),
            "priority": [bt.value for bt in self._backend_priority],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = dict(self._stats)
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["fallback_rate"] = stats["fallback_count"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["fallback_rate"] = 0.0
        return stats

    def has_capability(self, capability: BackendCapability) -> bool:
        """Check if any backend has a capability."""
        self._ensure_initialized()

        for backend in self._backends.values():
            if capability in backend.capabilities:
                return True
        return False

    def is_available(self) -> bool:
        """Check if any backend is available."""
        self._ensure_initialized()
        return len(self._backends) > 0

    def shutdown(self) -> None:
        """Shutdown all backends."""
        for backend_type, backend in self._backends.items():
            try:
                backend.shutdown()
                logger.debug(f"Shutdown {backend_type.value}")
            except Exception as e:
                logger.warning(f"Backend {backend_type.value} shutdown error: {e}")
        self._backends.clear()
        self._backend_priority.clear()
