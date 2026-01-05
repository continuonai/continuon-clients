"""
Hailo Vision Backend

Wraps HailoPipeline with VisionBackend interface for unified access.
Provides ~47 FPS detection on Hailo-8.
"""
import asyncio
import logging
from pathlib import Path
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


class HailoBackend(VisionBackend):
    """
    Hailo NPU backend using persistent device pipeline.

    Wraps the existing HailoPipeline for high-throughput inference.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        auto_start: bool = False,
    ):
        """
        Initialize Hailo backend.

        Args:
            model_dir: Directory containing .hef model files
            auto_start: If True, start the pipeline immediately
        """
        self._pipeline = None
        self._available = False
        self._model_dir = model_dir or "/opt/continuonos/brain/model"
        self._init_attempted = False

        if auto_start:
            self._init_pipeline()

    def _init_pipeline(self) -> None:
        """Initialize the Hailo pipeline."""
        if self._init_attempted:
            return

        self._init_attempted = True

        try:
            from continuonbrain.services.hailo_pipeline import HailoPipeline

            model_dir = Path(self._model_dir)
            self._pipeline = HailoPipeline(model_dir=model_dir)
            self._available = self._pipeline.is_available()

            if self._available:
                logger.info("HailoBackend initialized successfully")
            else:
                logger.warning("HailoBackend: No Hailo models available")

        except ImportError as e:
            logger.warning(f"HailoBackend: hailo_platform not available: {e}")
            self._available = False
        except Exception as e:
            logger.error(f"HailoBackend initialization failed: {e}")
            self._available = False

    @property
    def backend_type(self) -> BackendType:
        return BackendType.HAILO_PIPELINE

    @property
    def capabilities(self) -> List[BackendCapability]:
        return [BackendCapability.DETECTION]

    def is_available(self) -> bool:
        if not self._init_attempted:
            self._init_pipeline()
        return self._available and self._pipeline is not None

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """Run detection using Hailo pipeline (synchronous)."""
        if not self.is_available():
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error="Hailo device not available",
            )

        try:
            import time
            start = time.time()

            # Use sync detection
            result = self._pipeline.detect_sync(frame, conf_threshold)
            inference_ms = (time.time() - start) * 1000

            if not result.ok:
                return BackendResult(
                    ok=False,
                    backend=self.backend_type.value,
                    error=result.error,
                    inference_time_ms=inference_ms,
                )

            # Convert to standard format
            detections = [
                DetectionResult(
                    label=d.get("label", "unknown"),
                    confidence=d.get("confidence", 0.0),
                    bbox=tuple(int(v) for v in d.get("bbox", [0, 0, 0, 0])),
                    class_id=d.get("class_id"),
                )
                for d in result.data.get("detections", [])
            ]

            return BackendResult(
                ok=True,
                detections=detections,
                inference_time_ms=inference_ms,
                backend=self.backend_type.value,
                metadata={"num_detections": len(detections)},
            )

        except Exception as e:
            logger.error(f"Hailo detection failed: {e}")
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    async def detect_async(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """Async detection using Hailo pipeline."""
        if not self.is_available():
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error="Hailo device not available",
            )

        try:
            import time
            start = time.time()

            # Ensure pipeline is started
            if not self._pipeline.is_running():
                await self._pipeline.start()

            result = await self._pipeline.detect(frame, conf_threshold)
            inference_ms = (time.time() - start) * 1000

            # Convert result
            detections = [
                DetectionResult(
                    label=d.get("label", "unknown"),
                    confidence=d.get("confidence", 0.0),
                    bbox=tuple(int(v) for v in d.get("bbox", [0, 0, 0, 0])),
                    class_id=d.get("class_id"),
                )
                for d in result.data.get("detections", [])
            ]

            return BackendResult(
                ok=result.ok,
                detections=detections,
                inference_time_ms=inference_ms,
                backend=self.backend_type.value,
                error=result.error,
            )

        except Exception as e:
            logger.error(f"Hailo async detection failed: {e}")
            return BackendResult(
                ok=False,
                backend=self.backend_type.value,
                error=str(e),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get detailed backend status."""
        status = super().get_status()
        if self._pipeline:
            try:
                status["stats"] = self._pipeline.get_stats()
                status["models"] = self._pipeline.get_available_models()
            except Exception:
                pass
        return status

    def shutdown(self) -> None:
        """Shutdown the Hailo pipeline."""
        if self._pipeline and self._pipeline.is_running():
            try:
                # Try to stop gracefully
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._pipeline.stop())
                else:
                    loop.run_until_complete(self._pipeline.stop())
            except RuntimeError:
                # No event loop - create one
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._pipeline.stop())
                loop.close()
            except Exception as e:
                logger.warning(f"Error stopping Hailo pipeline: {e}")
