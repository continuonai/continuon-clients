"""
Depth Enhancement Backend

Fuses OAK-D stereo depth with learned depth estimation (MiDaS) for
improved depth in occluded regions. Based on AINA FoundationStereo approach.

OAK-D provides ground truth stereo depth, while learned depth fills gaps
where stereo matching fails (occlusions, textureless surfaces, etc.).
"""
import logging
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
)

logger = logging.getLogger(__name__)


class DepthBackend(VisionBackend):
    """
    Enhanced depth estimation backend using stereo + learned depth fusion.

    Provides:
    - OAK-D stereo depth as primary source (accurate but with holes)
    - MiDaS/DepthAnything for gap filling (learned monocular depth)
    - Confidence-weighted fusion for best of both worlds

    For AINA integration, the enhanced depth improves:
    - Object point cloud extraction
    - Occlusion handling in demonstrations
    - Depth consistency across frames
    """

    def __init__(
        self,
        model_type: str = "midas_small",
        fill_threshold: float = 0.1,
        blend_sigma: float = 10.0,
    ):
        """
        Initialize depth enhancement backend.

        Args:
            model_type: Learned depth model ("midas_small", "midas_large", "depth_anything")
            fill_threshold: Coverage threshold to trigger gap filling (0-1)
            blend_sigma: Gaussian sigma for blending at depth boundaries
        """
        self._model_type = model_type
        self._fill_threshold = fill_threshold
        self._blend_sigma = blend_sigma
        self._available = None
        self._model = None
        self._transform = None
        self._device = "cpu"  # Default to CPU, can be upgraded to cuda/mps

        # Stats
        self._last_inference_ms = 0.0
        self._fill_count = 0

    @property
    def backend_type(self) -> BackendType:
        return BackendType.DEPTH_ENHANCED

    @property
    def capabilities(self) -> List[BackendCapability]:
        return [BackendCapability.DEPTH]

    def is_available(self) -> bool:
        """Check if depth enhancement is available."""
        if self._available is None:
            self._available = self._check_availability()
        return self._available

    def _check_availability(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import torch
            # Check for MiDaS or similar depth model availability
            # For now, we support basic gap filling without neural network
            # Full MiDaS support can be added when torch is available
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"DepthBackend available on {self._device}")
            return True
        except ImportError:
            logger.info("PyTorch not available, using statistical gap filling only")
            return True  # Still usable with statistical methods

    def _load_model(self) -> bool:
        """Lazy load the depth estimation model."""
        if self._model is not None:
            return True

        try:
            import torch

            if self._model_type == "midas_small":
                # Use MiDaS small for efficiency on Pi5
                self._model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "MiDaS_small",
                    pretrained=True,
                    trust_repo=True,
                )
                midas_transforms = torch.hub.load(
                    "intel-isl/MiDaS",
                    "transforms",
                    trust_repo=True,
                )
                self._transform = midas_transforms.small_transform

            elif self._model_type == "midas_large":
                self._model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "DPT_Large",
                    pretrained=True,
                    trust_repo=True,
                )
                midas_transforms = torch.hub.load(
                    "intel-isl/MiDaS",
                    "transforms",
                    trust_repo=True,
                )
                self._transform = midas_transforms.dpt_transform

            else:
                logger.warning(f"Unknown model type: {self._model_type}")
                return False

            self._model.to(self._device)
            self._model.eval()
            logger.info(f"Loaded {self._model_type} depth model on {self._device}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load MiDaS model: {e}")
            return False

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> BackendResult:
        """
        Depth backend doesn't do object detection.
        Use estimate_depth() instead.
        """
        return BackendResult(
            ok=False,
            backend=self.backend_type.value,
            error="DepthBackend does not support detection. Use estimate_depth().",
        )

    def estimate_depth(
        self,
        rgb_frame: np.ndarray,
        stereo_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Estimate enhanced depth from RGB with optional stereo fusion.

        Args:
            rgb_frame: RGB numpy array (H, W, 3)
            stereo_depth: Optional stereo depth in mm (H, W) uint16

        Returns:
            Dict with:
            - enhanced_depth: Fused depth map in mm
            - confidence: Per-pixel confidence (0-1)
            - fill_mask: Boolean mask of filled regions
            - inference_time_ms: Processing time
        """
        start_time = time.time()

        h, w = rgb_frame.shape[:2]

        # If no stereo depth, estimate from RGB alone
        if stereo_depth is None:
            learned_depth = self._estimate_from_rgb(rgb_frame)
            self._last_inference_ms = (time.time() - start_time) * 1000

            return {
                "enhanced_depth": learned_depth,
                "confidence": np.ones((h, w), dtype=np.float32),
                "fill_mask": np.ones((h, w), dtype=bool),
                "inference_time_ms": self._last_inference_ms,
                "method": "learned_only",
            }

        # Compute stereo depth coverage (valid pixels)
        valid_stereo = stereo_depth > 0
        coverage = np.sum(valid_stereo) / valid_stereo.size

        # If coverage is high enough, use stereo depth directly
        if coverage >= (1.0 - self._fill_threshold):
            # Simple hole filling with morphological operations
            enhanced = self._fill_small_holes(stereo_depth.astype(np.float32))
            self._last_inference_ms = (time.time() - start_time) * 1000

            return {
                "enhanced_depth": enhanced.astype(np.uint16),
                "confidence": valid_stereo.astype(np.float32),
                "fill_mask": ~valid_stereo,
                "inference_time_ms": self._last_inference_ms,
                "method": "stereo_with_fill",
                "stereo_coverage": coverage,
            }

        # Need significant gap filling - use learned depth
        self._fill_count += 1
        learned_depth = self._estimate_from_rgb(rgb_frame)

        if learned_depth is None:
            # Fallback to stereo with simple fill
            enhanced = self._fill_small_holes(stereo_depth.astype(np.float32))
            self._last_inference_ms = (time.time() - start_time) * 1000
            return {
                "enhanced_depth": enhanced.astype(np.uint16),
                "confidence": valid_stereo.astype(np.float32),
                "fill_mask": ~valid_stereo,
                "inference_time_ms": self._last_inference_ms,
                "method": "stereo_fallback",
                "stereo_coverage": coverage,
            }

        # Fuse stereo and learned depth
        enhanced, confidence = self._fuse_depths(
            stereo_depth.astype(np.float32),
            learned_depth,
            valid_stereo,
        )

        self._last_inference_ms = (time.time() - start_time) * 1000

        return {
            "enhanced_depth": enhanced.astype(np.uint16),
            "confidence": confidence,
            "fill_mask": ~valid_stereo,
            "inference_time_ms": self._last_inference_ms,
            "method": "stereo_learned_fusion",
            "stereo_coverage": coverage,
        }

    def _estimate_from_rgb(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate relative depth from RGB using learned model."""
        if not self._load_model():
            return None

        try:
            import torch

            h, w = rgb_frame.shape[:2]

            # Transform input
            input_batch = self._transform(rgb_frame).to(self._device)

            # Inference
            with torch.no_grad():
                prediction = self._model(input_batch)

            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Convert to numpy (relative depth, higher = closer)
            relative_depth = prediction.cpu().numpy()

            # Normalize to reasonable mm range (0-10000mm)
            # This is approximate - real scale needs calibration
            depth_min, depth_max = 100, 10000  # mm
            depth_range = relative_depth.max() - relative_depth.min()
            if depth_range > 0:
                # Invert (MiDaS outputs inverse depth)
                normalized = 1.0 - (relative_depth - relative_depth.min()) / depth_range
                depth_mm = depth_min + normalized * (depth_max - depth_min)
            else:
                depth_mm = np.full((h, w), 1000, dtype=np.float32)

            return depth_mm.astype(np.float32)

        except Exception as e:
            logger.error(f"Learned depth estimation failed: {e}")
            return None

    def _fill_small_holes(
        self,
        depth: np.ndarray,
        max_hole_size: int = 50,
    ) -> np.ndarray:
        """Fill small holes in depth map using morphological operations."""
        try:
            import cv2
        except ImportError:
            return depth

        filled = depth.copy()
        invalid = depth <= 0

        # Use inpainting for small holes
        if np.sum(invalid) > 0 and np.sum(invalid) < (depth.size * 0.3):
            mask = invalid.astype(np.uint8)
            # Scale depth for inpainting (works on uint8)
            depth_scaled = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_scaled.astype(np.uint8)

            inpainted = cv2.inpaint(depth_uint8, mask, 3, cv2.INPAINT_TELEA)

            # Scale back
            scale_factor = depth[~invalid].mean() / (depth_scaled[~invalid].mean() + 1e-6)
            filled[invalid] = inpainted[invalid] * scale_factor

        return filled

    def _fuse_depths(
        self,
        stereo_depth: np.ndarray,
        learned_depth: np.ndarray,
        valid_stereo: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse stereo and learned depth with confidence-weighted blending.

        Stereo depth is used where valid, learned depth fills gaps.
        Smooth blending at boundaries to avoid artifacts.
        """
        try:
            import cv2
        except ImportError:
            # Simple fusion without blending
            fused = stereo_depth.copy()
            fused[~valid_stereo] = learned_depth[~valid_stereo]
            confidence = valid_stereo.astype(np.float32)
            return fused, confidence

        # Scale learned depth to match stereo in valid regions
        if np.sum(valid_stereo) > 100:
            stereo_median = np.median(stereo_depth[valid_stereo])
            learned_median = np.median(learned_depth[valid_stereo])
            scale = stereo_median / (learned_median + 1e-6)
            learned_depth = learned_depth * scale

        # Create smooth transition mask
        distance_to_valid = cv2.distanceTransform(
            (~valid_stereo).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )

        # Sigmoid blend weight based on distance
        blend_weight = 1.0 / (1.0 + np.exp(-distance_to_valid / self._blend_sigma + 3))
        blend_weight = np.clip(blend_weight, 0, 1)

        # Fuse with blend
        fused = stereo_depth * (1 - blend_weight) + learned_depth * blend_weight

        # Use stereo where definitely valid
        fused[valid_stereo] = stereo_depth[valid_stereo]

        # Confidence: high for stereo, decreases with distance from valid
        confidence = np.clip(1.0 - blend_weight, 0.1, 1.0)
        confidence[valid_stereo] = 1.0

        return fused, confidence.astype(np.float32)

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "type": self.backend_type.value,
            "available": self.is_available(),
            "capabilities": [c.value for c in self.capabilities],
            "model_type": self._model_type,
            "model_loaded": self._model is not None,
            "device": self._device,
            "last_inference_ms": self._last_inference_ms,
            "fill_count": self._fill_count,
        }

    def shutdown(self) -> None:
        """Release resources."""
        self._model = None
        self._transform = None
        self._available = None


def test_depth_backend():
    """Test depth backend functionality."""
    import numpy as np

    backend = DepthBackend()
    print(f"DepthBackend available: {backend.is_available()}")
    print(f"Status: {backend.get_status()}")

    # Create test RGB frame
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create test stereo depth with holes
    depth = np.random.randint(500, 5000, (480, 640), dtype=np.uint16)
    # Create some holes
    depth[100:200, 200:400] = 0  # Large hole
    depth[300:320, 100:150] = 0  # Small hole

    print(f"\nStereo coverage: {np.sum(depth > 0) / depth.size:.1%}")

    # Test fusion
    result = backend.estimate_depth(rgb, depth)
    print(f"\nEnhanced depth result:")
    print(f"  Method: {result['method']}")
    print(f"  Fill mask sum: {np.sum(result['fill_mask'])}")
    print(f"  Inference time: {result['inference_time_ms']:.1f}ms")
    print(f"  Confidence mean: {result['confidence'].mean():.2f}")

    backend.shutdown()
    print("\nDepth backend test complete")


if __name__ == "__main__":
    test_depth_backend()
