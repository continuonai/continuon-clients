"""
ContinuonBrain Vision System

Unified vision system with pluggable backends:
- HailoBackend: Hailo-8 NPU for fast object detection
- PoseBackend: Hailo-8 NPU for pose estimation
- SAMBackend: SAM/SAM2 for segmentation
- DepthBackend: Enhanced depth with MiDaS/OAK fusion
- CPUBackend: Fallback CPU-based detection

Usage:
    from continuonbrain.services.vision import VisionManager

    manager = VisionManager()
    result = manager.detect(frame)
    for detection in result.detections:
        print(f"{detection.label}: {detection.confidence:.2f}")

    # Pose estimation
    poses = manager.detect_poses(frame)
    for pose in poses:
        print(f"Person {pose.person_id}: {pose.confidence:.2f}")

    # Enhanced depth
    depth_result = manager.estimate_depth(rgb_frame, stereo_depth)
    enhanced = depth_result["enhanced_depth"]
"""
from .backend import (
    VisionBackend,
    BackendType,
    BackendCapability,
    BackendResult,
    DetectionResult,
    PoseResult,
    KeypointResult,
)
from .manager import VisionManager

__all__ = [
    'VisionBackend',
    'BackendType',
    'BackendCapability',
    'BackendResult',
    'DetectionResult',
    'PoseResult',
    'KeypointResult',
    'VisionManager',
]
