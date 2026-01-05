"""
ContinuonBrain Vision System

Unified vision system with pluggable backends:
- HailoBackend: Hailo-8 NPU for fast object detection
- SAMBackend: SAM/SAM2 for segmentation
- CPUBackend: Fallback CPU-based detection

Usage:
    from continuonbrain.services.vision import VisionManager

    manager = VisionManager()
    result = manager.detect(frame)
    for detection in result.detections:
        print(f"{detection.label}: {detection.confidence:.2f}")
"""
from .backend import (
    VisionBackend,
    BackendType,
    BackendCapability,
    BackendResult,
    DetectionResult,
)
from .manager import VisionManager

__all__ = [
    'VisionBackend',
    'BackendType',
    'BackendCapability',
    'BackendResult',
    'DetectionResult',
    'VisionManager',
]
