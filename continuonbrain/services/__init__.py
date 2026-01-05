"""
ContinuonBrain Services - Core service modules.

This package contains the core services for ContinuonBrain:
- autonomous_training_scheduler: Autonomous training orchestration
- model_validator: Model validation and regression testing
- model_distribution: GCP-based model distribution for continuonai.com
- cloud_training: Cloud training job management
- ota_updater: OTA update system
- update_scheduler: Background update checking
- world_model_integration: Unified sensory fusion and world state tracking
- multimodal_input_hub: Multi-modal input processing (vision, audio, text)
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

from continuonbrain.services.world_model_integration import (
    WorldModelIntegration,
    TeacherInterface,
    SensoryFrame,
    WorldState,
    create_world_model_integration,
)

from continuonbrain.services.multimodal_input_hub import (
    MultiModalInputHub,
    InputEvent,
    ProcessedInput,
    VisionInputProcessor,
    AudioInputProcessor,
    TextInputProcessor,
    create_multimodal_hub,
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
    # World Model Integration
    "WorldModelIntegration",
    "TeacherInterface",
    "SensoryFrame",
    "WorldState",
    "create_world_model_integration",
    # Multi-Modal Input Hub
    "MultiModalInputHub",
    "InputEvent",
    "ProcessedInput",
    "VisionInputProcessor",
    "AudioInputProcessor",
    "TextInputProcessor",
    "create_multimodal_hub",
]
