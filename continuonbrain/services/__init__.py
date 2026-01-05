"""
ContinuonBrain Services - Core service modules.

This package contains the core services for ContinuonBrain:
- autonomous_training_scheduler: Autonomous training orchestration
- model_validator: Model validation and regression testing
- model_distribution: GCP-based model distribution for continuonai.com
- cloud_training: Cloud training job management
- ota_updater: OTA update system
- update_scheduler: Background update checking
"""

from continuonbrain.services.autonomous_training_scheduler import (
    AutonomousTrainingScheduler,
    TrainingTriggerConfig,
    EpisodeQualityScorer,
    CapabilityGapDetector,
    EpisodeQualityScore,
    TrainingMode,
    SchedulerPhase,
    SchedulerStatus,
    create_autonomous_scheduler,
)

from continuonbrain.services.model_validator import (
    ModelValidator,
    ValidationConfig,
    ValidationLevel,
    ValidationStatus,
    ValidationResult,
    ValidationCheck,
    validate_model_quick,
)

from continuonbrain.services.model_distribution import (
    ModelDistributionService,
    DistributionConfig,
    ModelType,
    RobotStatus,
    UpdatePriority,
    ModelManifest,
    RegisteredRobot,
    UpdateNotification,
    upload_seed_model,
)

__all__ = [
    # Autonomous Training
    "AutonomousTrainingScheduler",
    "TrainingTriggerConfig",
    "EpisodeQualityScorer",
    "CapabilityGapDetector",
    "EpisodeQualityScore",
    "TrainingMode",
    "SchedulerPhase",
    "SchedulerStatus",
    "create_autonomous_scheduler",
    # Model Validation
    "ModelValidator",
    "ValidationConfig",
    "ValidationLevel",
    "ValidationStatus",
    "ValidationResult",
    "ValidationCheck",
    "validate_model_quick",
    # Model Distribution
    "ModelDistributionService",
    "DistributionConfig",
    "ModelType",
    "RobotStatus",
    "UpdatePriority",
    "ModelManifest",
    "RegisteredRobot",
    "UpdateNotification",
    "upload_seed_model",
]
