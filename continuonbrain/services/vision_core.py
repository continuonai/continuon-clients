"""
VisionCore: Unified Perception Backbone

Fuses multiple vision sources into a coherent scene representation:
- OAK-D RGB + Depth camera
- Hailo-8 NPU object detection (26 TOPS)
- SAM3 semantic segmentation
- VPU features from Myriad X

This is the primary perception interface for the HOPE Agent.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Scene Representation
# =============================================================================

@dataclass
class DetectedObject:
    """Single detected object in the scene."""
    id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    depth_mm: Optional[float] = None  # Depth at object center
    mask: Optional[np.ndarray] = None  # Segmentation mask
    position_3d: Optional[Tuple[float, float, float]] = None  # x, y, z in meters
    source: str = "hailo"  # hailo, sam3, or fused
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "depth_mm": self.depth_mm,
            "has_mask": self.mask is not None,
            "position_3d": list(self.position_3d) if self.position_3d else None,
            "source": self.source,
        }


@dataclass
class SceneRepresentation:
    """Complete scene understanding from VisionCore."""
    timestamp: float
    objects: List[DetectedObject] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None  # Full depth image
    rgb_frame: Optional[np.ndarray] = None  # RGB image
    
    # Scene-level statistics
    nearest_object_mm: Optional[float] = None
    dominant_depth_mm: Optional[float] = None
    object_count: int = 0
    
    # Processing metadata
    hailo_inference_ms: float = 0.0
    sam3_inference_ms: float = 0.0
    total_ms: float = 0.0
    
    # Capability flags
    has_depth: bool = False
    has_detection: bool = False
    has_segmentation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "objects": [o.to_dict() for o in self.objects],
            "object_count": self.object_count,
            "nearest_object_mm": self.nearest_object_mm,
            "dominant_depth_mm": self.dominant_depth_mm,
            "has_depth": self.has_depth,
            "has_detection": self.has_detection,
            "has_segmentation": self.has_segmentation,
            "hailo_inference_ms": self.hailo_inference_ms,
            "sam3_inference_ms": self.sam3_inference_ms,
            "total_ms": self.total_ms,
        }
    
    def get_object_by_label(self, label: str) -> Optional[DetectedObject]:
        """Find object by label (case-insensitive partial match)."""
        label_lower = label.lower()
        for obj in self.objects:
            if label_lower in obj.label.lower():
                return obj
        return None
    
    def get_nearest_object(self) -> Optional[DetectedObject]:
        """Get the closest object to the camera."""
        if not self.objects:
            return None
        with_depth = [o for o in self.objects if o.depth_mm is not None]
        if not with_depth:
            return self.objects[0]  # Return first if no depth
        return min(with_depth, key=lambda o: o.depth_mm)


# =============================================================================
# VisionCore Main Class
# =============================================================================

class VisionCore:
    """
    Unified perception backbone for HOPE Agent.
    
    Combines:
    - OAK-D camera (RGB + Depth + VPU)
    - Hailo-8 NPU (object detection)
    - SAM3 (semantic segmentation)
    
    Gracefully degrades when components are missing.
    """
    
    def __init__(
        self,
        enable_hailo: bool = True,
        enable_sam3: bool = True,
        enable_depth: bool = True,
        hailo_model: str = "yolov8s",
        sam3_prompt: str = "object",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize VisionCore with available components.
        
        Args:
            enable_hailo: Attempt to use Hailo-8 NPU for detection
            enable_sam3: Attempt to use SAM3 for segmentation
            enable_depth: Attempt to use OAK-D depth
            hailo_model: Hailo model name to load
            sam3_prompt: Default prompt for SAM3 segmentation
            cache_dir: Directory for caching models
        """
        self.enable_hailo = enable_hailo
        self.enable_sam3 = enable_sam3
        self.enable_depth = enable_depth
        self.hailo_model = hailo_model
        self.sam3_prompt = sam3_prompt
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/visioncore")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Component references
        self._oak_camera = None
        self._hailo_vision = None
        self._sam3_service = None
        
        # Status tracking
        self._hailo_available = False
        self._sam3_available = False
        self._oak_available = False
        
        # Object ID counter
        self._object_id = 0
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize available vision components."""
        # OAK-D Camera
        if self.enable_depth:
            try:
                from continuonbrain.sensors.depth_sensor import OAKDepthCapture
                self._oak_camera = OAKDepthCapture()
                self._oak_camera.start()
                self._oak_available = True
                logger.info("✅ VisionCore: OAK-D camera initialized")
            except ImportError:
                # Try alternate import
                try:
                    import depthai as dai
                    # Direct depthai usage
                    self._oak_available = True
                    logger.info("✅ VisionCore: OAK-D available via depthai")
                except ImportError:
                    logger.warning("⚠️ VisionCore: OAK-D not available (depthai not installed)")
            except Exception as e:
                logger.warning(f"⚠️ VisionCore: OAK-D not available: {e}")
        
        # Hailo NPU
        if self.enable_hailo:
            try:
                from continuonbrain.services.hailo_vision import HailoVision
                self._hailo_vision = HailoVision()
                # Check if Hailo is available (handle missing method)
                if hasattr(self._hailo_vision, 'is_available'):
                    self._hailo_available = self._hailo_vision.is_available()
                elif hasattr(self._hailo_vision, 'available'):
                    self._hailo_available = self._hailo_vision.available
                else:
                    # Assume available if object created successfully
                    self._hailo_available = True
                
                if self._hailo_available:
                    logger.info("✅ VisionCore: Hailo-8 NPU initialized")
                else:
                    logger.warning("⚠️ VisionCore: Hailo-8 not available")
            except ImportError:
                logger.warning("⚠️ VisionCore: Hailo module not available")
            except Exception as e:
                logger.warning(f"⚠️ VisionCore: Hailo failed: {e}")
        
        # SAM3 Segmentation
        if self.enable_sam3:
            try:
                from continuonbrain.services.sam3_vision import create_sam_service
                self._sam3_service = create_sam_service()
                if self._sam3_service.is_available():
                    self._sam3_available = True
                    logger.info("✅ VisionCore: SAM3 initialized")
                else:
                    logger.warning("⚠️ VisionCore: SAM3 not available")
            except Exception as e:
                logger.warning(f"⚠️ VisionCore: SAM3 failed: {e}")
    
    def is_ready(self) -> bool:
        """Check if at least one vision component is ready."""
        return self._oak_available or self._hailo_available or self._sam3_available
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return available capabilities."""
        return {
            "oak_camera": self._oak_available,
            "hailo_detection": self._hailo_available,
            "sam3_segmentation": self._sam3_available,
            "depth": self._oak_available,
            "rgb": self._oak_available,
        }
    
    def perceive(
        self,
        rgb_frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
        run_detection: bool = True,
        run_segmentation: bool = False,
        segmentation_prompt: Optional[str] = None,
    ) -> SceneRepresentation:
        """
        Main perception function - fuse all vision sources.
        
        Args:
            rgb_frame: Optional RGB image (captures from OAK if None)
            depth_frame: Optional depth image (captures from OAK if None)
            run_detection: Run Hailo object detection
            run_segmentation: Run SAM3 segmentation
            segmentation_prompt: Text prompt for SAM3 (e.g., "a cup")
        
        Returns:
            SceneRepresentation with fused understanding
        """
        start_time = time.time()
        scene = SceneRepresentation(timestamp=start_time)
        
        # Step 1: Capture frames if not provided
        if rgb_frame is None or depth_frame is None:
            if self._oak_available and self._oak_camera:
                try:
                    captured = self._oak_camera.get_frame()
                    if captured:
                        if rgb_frame is None:
                            rgb_frame = captured.get("rgb")
                        if depth_frame is None:
                            depth_frame = captured.get("depth")
                except Exception as e:
                    logger.warning(f"OAK capture failed: {e}")
        
        scene.rgb_frame = rgb_frame
        scene.depth_map = depth_frame
        scene.has_depth = depth_frame is not None
        
        # Step 2: Compute depth statistics
        if depth_frame is not None:
            try:
                valid_depths = depth_frame[depth_frame > 0]
                if len(valid_depths) > 0:
                    scene.dominant_depth_mm = float(np.median(valid_depths))
                    scene.nearest_object_mm = float(np.min(valid_depths))
            except Exception as e:
                logger.debug(f"Depth stats failed: {e}")
        
        # Step 3: Run Hailo object detection
        hailo_objects = []
        if run_detection and self._hailo_available and self._hailo_vision and rgb_frame is not None:
            hailo_start = time.time()
            try:
                detections = self._hailo_vision.detect(rgb_frame)
                for det in detections:
                    self._object_id += 1
                    obj = DetectedObject(
                        id=self._object_id,
                        label=det.get("label", "unknown"),
                        confidence=det.get("confidence", 0.0),
                        bbox=tuple(det.get("bbox", [0, 0, 0, 0])),
                        source="hailo",
                    )
                    # Add depth at object center
                    if depth_frame is not None:
                        cx = (obj.bbox[0] + obj.bbox[2]) // 2
                        cy = (obj.bbox[1] + obj.bbox[3]) // 2
                        if 0 <= cy < depth_frame.shape[0] and 0 <= cx < depth_frame.shape[1]:
                            obj.depth_mm = float(depth_frame[cy, cx])
                    hailo_objects.append(obj)
                scene.has_detection = True
            except Exception as e:
                logger.warning(f"Hailo detection failed: {e}")
            scene.hailo_inference_ms = (time.time() - hailo_start) * 1000
        
        # Step 4: Run SAM3 segmentation
        sam3_objects = []
        if run_segmentation and self._sam3_available and self._sam3_service and rgb_frame is not None:
            sam_start = time.time()
            try:
                prompt = segmentation_prompt or self.sam3_prompt
                result = self._sam3_service.segment_text(rgb_frame, prompt)
                if result and result.get("masks"):
                    for i, mask in enumerate(result["masks"]):
                        self._object_id += 1
                        # Compute bbox from mask
                        if isinstance(mask, np.ndarray):
                            rows = np.any(mask, axis=1)
                            cols = np.any(mask, axis=0)
                            if rows.any() and cols.any():
                                y1, y2 = np.where(rows)[0][[0, -1]]
                                x1, x2 = np.where(cols)[0][[0, -1]]
                                bbox = (int(x1), int(y1), int(x2), int(y2))
                            else:
                                bbox = (0, 0, 0, 0)
                        else:
                            bbox = (0, 0, 0, 0)
                        
                        obj = DetectedObject(
                            id=self._object_id,
                            label=prompt,
                            confidence=result.get("scores", [0.8])[i] if i < len(result.get("scores", [])) else 0.8,
                            bbox=bbox,
                            mask=mask if isinstance(mask, np.ndarray) else None,
                            source="sam3",
                        )
                        # Add depth at mask center
                        if depth_frame is not None and bbox != (0, 0, 0, 0):
                            cx = (bbox[0] + bbox[2]) // 2
                            cy = (bbox[1] + bbox[3]) // 2
                            if 0 <= cy < depth_frame.shape[0] and 0 <= cx < depth_frame.shape[1]:
                                obj.depth_mm = float(depth_frame[cy, cx])
                        sam3_objects.append(obj)
                scene.has_segmentation = True
            except Exception as e:
                logger.warning(f"SAM3 segmentation failed: {e}")
            scene.sam3_inference_ms = (time.time() - sam_start) * 1000
        
        # Step 5: Fuse objects
        scene.objects = self._fuse_objects(hailo_objects, sam3_objects)
        scene.object_count = len(scene.objects)
        
        # Update nearest object
        if scene.objects:
            nearest = scene.get_nearest_object()
            if nearest and nearest.depth_mm:
                scene.nearest_object_mm = nearest.depth_mm
        
        scene.total_ms = (time.time() - start_time) * 1000
        return scene
    
    def _fuse_objects(
        self,
        hailo_objects: List[DetectedObject],
        sam3_objects: List[DetectedObject],
    ) -> List[DetectedObject]:
        """
        Fuse detection and segmentation results.
        
        If bboxes overlap significantly, merge into single object.
        """
        fused = []
        used_sam3 = set()
        
        for h_obj in hailo_objects:
            best_match = None
            best_iou = 0.0
            
            for i, s_obj in enumerate(sam3_objects):
                if i in used_sam3:
                    continue
                iou = self._compute_iou(h_obj.bbox, s_obj.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = (i, s_obj)
            
            if best_match:
                i, s_obj = best_match
                used_sam3.add(i)
                # Create fused object with Hailo label + SAM3 mask
                fused_obj = DetectedObject(
                    id=h_obj.id,
                    label=h_obj.label,
                    confidence=(h_obj.confidence + s_obj.confidence) / 2,
                    bbox=h_obj.bbox,
                    depth_mm=h_obj.depth_mm or s_obj.depth_mm,
                    mask=s_obj.mask,
                    source="fused",
                )
                fused.append(fused_obj)
            else:
                fused.append(h_obj)
        
        # Add remaining SAM3 objects
        for i, s_obj in enumerate(sam3_objects):
            if i not in used_sam3:
                fused.append(s_obj)
        
        # Sort by confidence
        fused.sort(key=lambda o: o.confidence, reverse=True)
        return fused
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union of two bboxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def describe_scene(self) -> str:
        """Get natural language description of current scene."""
        scene = self.perceive(run_detection=True, run_segmentation=False)
        
        if not scene.objects:
            if scene.has_depth and scene.dominant_depth_mm:
                return f"I don't see any distinct objects. The dominant depth is {scene.dominant_depth_mm:.0f}mm."
            return "I can't see anything right now."
        
        descriptions = []
        for obj in scene.objects[:5]:  # Top 5
            desc = f"{obj.label} ({obj.confidence:.0%} confidence)"
            if obj.depth_mm:
                desc += f" at {obj.depth_mm:.0f}mm"
            descriptions.append(desc)
        
        return f"I see {len(scene.objects)} objects: " + ", ".join(descriptions) + "."
    
    def find_object(self, query: str) -> Optional[DetectedObject]:
        """Find an object matching the query."""
        # First try detection
        scene = self.perceive(run_detection=True, run_segmentation=False)
        obj = scene.get_object_by_label(query)
        
        if obj:
            return obj
        
        # Try SAM3 segmentation with query as prompt
        if self._sam3_available:
            scene = self.perceive(
                run_detection=False,
                run_segmentation=True,
                segmentation_prompt=query,
            )
            if scene.objects:
                return scene.objects[0]
        
        return None
    
    def close(self):
        """Cleanup resources."""
        if self._oak_camera:
            try:
                self._oak_camera.stop()
            except Exception:
                pass
        logger.info("VisionCore closed")


# =============================================================================
# Factory Function
# =============================================================================

def create_vision_core(**kwargs) -> VisionCore:
    """Create VisionCore with auto-detection of available hardware."""
    return VisionCore(**kwargs)


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VisionCore Perception Test")
    parser.add_argument("--detect", action="store_true", help="Run object detection")
    parser.add_argument("--segment", type=str, help="Run SAM3 segmentation with prompt")
    parser.add_argument("--describe", action="store_true", help="Describe current scene")
    parser.add_argument("--find", type=str, help="Find specific object")
    parser.add_argument("--benchmark", type=int, default=0, help="Run N frames for benchmarking")
    args = parser.parse_args()
    
    print("=" * 60)
    print("VisionCore Perception Test")
    print("=" * 60)
    
    core = create_vision_core()
    print(f"Capabilities: {core.get_capabilities()}")
    
    if args.describe:
        print(f"\nScene: {core.describe_scene()}")
    
    if args.find:
        obj = core.find_object(args.find)
        if obj:
            print(f"\nFound: {obj.to_dict()}")
        else:
            print(f"\nCould not find: {args.find}")
    
    if args.detect:
        scene = core.perceive(run_detection=True)
        print(f"\nDetection Results ({scene.hailo_inference_ms:.1f}ms):")
        for obj in scene.objects:
            print(f"  - {obj.label}: {obj.confidence:.0%} at {obj.depth_mm}mm")
    
    if args.segment:
        scene = core.perceive(run_segmentation=True, segmentation_prompt=args.segment)
        print(f"\nSegmentation Results ({scene.sam3_inference_ms:.1f}ms):")
        for obj in scene.objects:
            print(f"  - {obj.label}: {obj.confidence:.0%}")
    
    if args.benchmark > 0:
        print(f"\nBenchmarking {args.benchmark} frames...")
        times = []
        for _ in range(args.benchmark):
            scene = core.perceive(run_detection=True)
            times.append(scene.total_ms)
        avg = sum(times) / len(times)
        fps = 1000 / avg if avg > 0 else 0
        print(f"Average: {avg:.1f}ms ({fps:.1f} FPS)")
    
    core.close()

