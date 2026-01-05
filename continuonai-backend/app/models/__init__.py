"""Pydantic models for ContinuonAI API."""

from app.models.robot import (
    Robot,
    RobotCreate,
    RobotUpdate,
    RobotCommand,
    RobotTelemetry,
    RobotStatus,
)
from app.models.model import (
    ModelInfo,
    ModelVersion,
    ModelCreate,
    ModelUploadResponse,
)
from app.models.training import (
    TrainingJob,
    TrainingConfig,
    TrainingStatus,
    TrainingMetrics,
)
from app.models.user import User, UserCreate, UserPlan

__all__ = [
    # Robot models
    "Robot",
    "RobotCreate",
    "RobotUpdate",
    "RobotCommand",
    "RobotTelemetry",
    "RobotStatus",
    # Model models
    "ModelInfo",
    "ModelVersion",
    "ModelCreate",
    "ModelUploadResponse",
    # Training models
    "TrainingJob",
    "TrainingConfig",
    "TrainingStatus",
    "TrainingMetrics",
    # User models
    "User",
    "UserCreate",
    "UserPlan",
]
