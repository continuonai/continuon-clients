"""
ContinuonBrain Exception Hierarchy

Provides a unified exception system for consistent error handling across:
- JAX model operations
- Vision services
- Training pipeline
- Hardware interfaces
- Reasoning services
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    WARNING = "warning"      # Degraded but operational
    ERROR = "error"          # Operation failed, recoverable
    CRITICAL = "critical"    # System-level failure


@dataclass
class ErrorContext:
    """Structured context for error diagnostics."""
    component: str
    operation: str
    inputs: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None


class ContinuonError(Exception):
    """Base exception for all ContinuonBrain errors."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.context = context
        self.severity = severity
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "severity": self.severity.value,
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
            } if self.context else None,
            "cause": str(self.cause) if self.cause else None,
        }


# === Model Errors ===

class ModelError(ContinuonError):
    """Base exception for model-related errors."""
    pass


class BatchDimensionError(ModelError):
    """Raised when input tensor has incorrect batch dimensions."""

    def __init__(
        self,
        expected_ndim: int,
        actual_ndim: int,
        tensor_name: str,
        context: Optional[ErrorContext] = None,
    ):
        message = (
            f"Tensor '{tensor_name}' has {actual_ndim} dimensions, "
            f"expected {expected_ndim}. Use forward_batch() for batched inputs "
            f"or forward_single() for unbatched inputs."
        )
        super().__init__(message, context=context)
        self.expected_ndim = expected_ndim
        self.actual_ndim = actual_ndim
        self.tensor_name = tensor_name


class ModelNotInitializedError(ModelError):
    """Raised when attempting to use an uninitialized model."""

    def __init__(self, model_name: str, context: Optional[ErrorContext] = None):
        message = f"Model '{model_name}' has not been initialized. Call initialize() first."
        super().__init__(message, context=context)
        self.model_name = model_name


class InferenceError(ModelError):
    """Raised when inference fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, context=context, cause=cause)
        self.model_name = model_name


class CheckpointError(ModelError):
    """Raised when checkpoint operations fail."""

    def __init__(
        self,
        operation: str,
        path: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Checkpoint {operation} failed for path: {path}"
        super().__init__(message, context=context, cause=cause)
        self.operation = operation
        self.path = path


# === Vision Errors ===

class VisionError(ContinuonError):
    """Base exception for vision-related errors."""
    pass


class VisionBackendError(VisionError):
    """Raised when a vision backend fails."""

    def __init__(
        self,
        backend: str,
        operation: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Vision backend '{backend}' failed during {operation}"
        super().__init__(message, context=context, cause=cause)
        self.backend = backend
        self.operation = operation


class DeviceNotAvailableError(VisionError):
    """Raised when required hardware device is not available."""

    def __init__(
        self,
        device: str,
        fallback_available: bool = False,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Device '{device}' not available"
        if fallback_available:
            message += " - falling back to CPU"
        severity = ErrorSeverity.WARNING if fallback_available else ErrorSeverity.ERROR
        super().__init__(message, context=context, severity=severity)
        self.device = device
        self.fallback_available = fallback_available


class FrameCaptureError(VisionError):
    """Raised when frame capture fails."""

    def __init__(
        self,
        camera: str,
        reason: str,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Failed to capture frame from '{camera}': {reason}"
        super().__init__(message, context=context)
        self.camera = camera
        self.reason = reason


# === Training Errors ===

class TrainingError(ContinuonError):
    """Base exception for training-related errors."""
    pass


class InsufficientDataError(TrainingError):
    """Raised when there's not enough data to train."""

    def __init__(
        self,
        required: int,
        available: int,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Insufficient training data: need {required}, have {available}"
        super().__init__(message, context=context, severity=ErrorSeverity.WARNING)
        self.required = required
        self.available = available


class ResourceConstraintError(TrainingError):
    """Raised when training cannot proceed due to resource constraints."""

    def __init__(
        self,
        resource: str,
        current: float,
        limit: float,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Resource constraint violated: {resource} at {current:.1f}%, limit {limit:.1f}%"
        super().__init__(message, context=context, severity=ErrorSeverity.WARNING)
        self.resource = resource
        self.current = current
        self.limit = limit


class TrainingStepError(TrainingError):
    """Raised when a training step fails."""

    def __init__(
        self,
        step_name: str,
        reason: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Training step '{step_name}' failed: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.step_name = step_name
        self.reason = reason


# === Hardware Errors ===

class HardwareError(ContinuonError):
    """Base exception for hardware-related errors."""
    pass


class ActuatorError(HardwareError):
    """Raised when an actuator operation fails."""

    def __init__(
        self,
        actuator: str,
        operation: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Actuator '{actuator}' failed during {operation}"
        super().__init__(message, context=context, cause=cause)
        self.actuator = actuator
        self.operation = operation


class SafetyLimitError(HardwareError):
    """Raised when a safety limit is violated."""

    def __init__(
        self,
        limit_type: str,
        value: float,
        limit: float,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Safety limit violated: {limit_type} = {value:.3f}, limit = {limit:.3f}"
        super().__init__(message, context=context, severity=ErrorSeverity.CRITICAL)
        self.limit_type = limit_type
        self.value = value
        self.limit = limit


# === Service Errors ===

class ServiceError(ContinuonError):
    """Base exception for service-related errors."""
    pass


class ServiceNotAvailableError(ServiceError):
    """Raised when a required service is not available."""

    def __init__(
        self,
        service: str,
        reason: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Service '{service}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, context=context)
        self.service = service
        self.reason = reason


class ServiceInitializationError(ServiceError):
    """Raised when service initialization fails."""

    def __init__(
        self,
        service: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to initialize service '{service}'"
        super().__init__(message, context=context, cause=cause)
        self.service = service


# === Chat/Reasoning Errors ===

class ChatError(ContinuonError):
    """Base exception for chat-related errors."""
    pass


class ChatModelError(ChatError):
    """Raised when chat model fails."""

    def __init__(
        self,
        model: str,
        reason: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Chat model '{model}' failed: {reason}"
        super().__init__(message, context=context, cause=cause)
        self.model = model
        self.reason = reason


class ReasoningError(ContinuonError):
    """Base exception for reasoning-related errors."""
    pass


class ContextRetrievalError(ReasoningError):
    """Raised when context retrieval fails."""

    def __init__(
        self,
        query: str,
        reason: str,
        context: Optional[ErrorContext] = None,
    ):
        message = f"Context retrieval failed for query '{query}': {reason}"
        super().__init__(message, context=context)
        self.query = query
        self.reason = reason
