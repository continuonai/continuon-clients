"""Tests for vision system."""
import pytest
import numpy as np

from continuonbrain.services.vision.backend import (
    BackendType,
    BackendCapability,
    BackendResult,
    DetectionResult,
)


class TestDetectionResult:
    def test_create_detection(self):
        det = DetectionResult(
            label="person",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
            class_id=0,
        )
        assert det.label == "person"
        assert det.confidence == 0.95
        assert det.bbox == (100, 100, 200, 200)
        assert det.class_id == 0

    def test_to_dict(self):
        det = DetectionResult(
            label="car",
            confidence=0.8,
            bbox=(50, 50, 150, 100),
            class_id=2,
            depth_mm=5000.0,
        )
        d = det.to_dict()

        assert d["label"] == "car"
        assert d["confidence"] == 0.8
        assert d["bbox"] == [50, 50, 150, 100]
        assert d["class_id"] == 2
        assert d["depth_mm"] == 5000.0
        assert d["has_mask"] is False

    def test_center(self):
        det = DetectionResult(
            label="test",
            confidence=0.5,
            bbox=(100, 100, 200, 200),
        )
        assert det.center == (150, 150)

    def test_area(self):
        det = DetectionResult(
            label="test",
            confidence=0.5,
            bbox=(0, 0, 100, 50),
        )
        assert det.area == 5000


class TestBackendResult:
    def test_ok_result(self):
        result = BackendResult(
            ok=True,
            detections=[
                DetectionResult("person", 0.9, (0, 0, 100, 100)),
                DetectionResult("dog", 0.8, (200, 200, 300, 300)),
            ],
            inference_time_ms=25.0,
            backend="hailo",
        )

        assert result.ok is True
        assert len(result.detections) == 2
        assert result.inference_time_ms == 25.0
        assert result.backend == "hailo"
        assert result.error is None

    def test_error_result(self):
        result = BackendResult(
            ok=False,
            backend="hailo",
            error="Device not available",
        )

        assert result.ok is False
        assert len(result.detections) == 0
        assert result.error == "Device not available"

    def test_to_dict(self):
        result = BackendResult(
            ok=True,
            detections=[DetectionResult("person", 0.9, (0, 0, 100, 100))],
            inference_time_ms=30.0,
            backend="cpu",
            metadata={"num_frames": 1},
        )

        d = result.to_dict()

        assert d["ok"] is True
        assert len(d["detections"]) == 1
        assert d["detections"][0]["label"] == "person"
        assert d["inference_time_ms"] == 30.0
        assert d["backend"] == "cpu"
        assert d["metadata"]["num_frames"] == 1


class TestBackendType:
    def test_enum_values(self):
        assert BackendType.HAILO_PIPELINE.value == "hailo_pipeline"
        assert BackendType.SAM.value == "sam"
        assert BackendType.CPU_YOLO.value == "cpu_yolo"


class TestBackendCapability:
    def test_enum_values(self):
        assert BackendCapability.DETECTION.value == "detection"
        assert BackendCapability.SEGMENTATION.value == "segmentation"
        assert BackendCapability.DEPTH.value == "depth"
