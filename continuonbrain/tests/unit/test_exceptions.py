"""Tests for the exception hierarchy."""
import pytest

from continuonbrain.core.exceptions import (
    ContinuonError,
    ErrorContext,
    ErrorSeverity,
    ModelError,
    BatchDimensionError,
    VisionError,
    TrainingError,
    InsufficientDataError,
)


class TestErrorContext:
    def test_create_context(self):
        ctx = ErrorContext(
            component="VisionCore",
            operation="detect",
            inputs={"frame_shape": (480, 640, 3)},
        )
        assert ctx.component == "VisionCore"
        assert ctx.operation == "detect"
        assert ctx.inputs["frame_shape"] == (480, 640, 3)


class TestContinuonError:
    def test_basic_error(self):
        err = ContinuonError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.severity == ErrorSeverity.ERROR
        assert err.context is None
        assert err.cause is None

    def test_error_with_context(self):
        ctx = ErrorContext(component="Test", operation="test_op")
        err = ContinuonError("Test error", context=ctx)
        assert err.context.component == "Test"

    def test_to_dict(self):
        ctx = ErrorContext(component="Test", operation="test_op")
        err = ContinuonError("Test error", context=ctx, severity=ErrorSeverity.WARNING)

        d = err.to_dict()

        assert d["error_type"] == "ContinuonError"
        assert d["message"] == "Test error"
        assert d["severity"] == "warning"
        assert d["context"]["component"] == "Test"


class TestBatchDimensionError:
    def test_creates_descriptive_message(self):
        err = BatchDimensionError(
            expected_ndim=1,
            actual_ndim=3,
            tensor_name="x_obs",
        )

        assert "x_obs" in str(err)
        assert "3 dimensions" in str(err)
        assert "expected 1" in str(err)
        assert err.expected_ndim == 1
        assert err.actual_ndim == 3
        assert err.tensor_name == "x_obs"


class TestInsufficientDataError:
    def test_creates_message_with_counts(self):
        err = InsufficientDataError(required=100, available=42)

        assert "100" in str(err)
        assert "42" in str(err)
        assert err.severity == ErrorSeverity.WARNING
        assert err.required == 100
        assert err.available == 42
