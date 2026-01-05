"""Training job Pydantic schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(str, Enum):
    """Type of training job."""

    FINE_TUNE = "fine_tune"
    IMITATION = "imitation"
    REINFORCEMENT = "reinforcement"
    DIFFUSION = "diffusion"
    EVALUATION = "evaluation"


class AcceleratorType(str, Enum):
    """GPU/TPU accelerator types."""

    NONE = "none"
    NVIDIA_TESLA_T4 = "NVIDIA_TESLA_T4"
    NVIDIA_TESLA_V100 = "NVIDIA_TESLA_V100"
    NVIDIA_TESLA_A100 = "NVIDIA_TESLA_A100"
    TPU_V2 = "TPU_V2"
    TPU_V3 = "TPU_V3"


class HyperParameters(BaseModel):
    """Training hyperparameters."""

    learning_rate: float = Field(default=1e-4, gt=0, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size")
    epochs: int = Field(default=10, ge=1, le=1000, description="Number of epochs")
    warmup_steps: int = Field(default=0, ge=0, description="Learning rate warmup steps")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    gradient_clip: Optional[float] = Field(None, gt=0, description="Gradient clipping threshold")
    dropout: float = Field(default=0.1, ge=0, le=1, description="Dropout rate")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    custom: Optional[Dict[str, Any]] = Field(None, description="Additional custom parameters")


class TrainingConfig(BaseModel):
    """Configuration for a training job."""

    name: str = Field(..., min_length=1, max_length=100, description="Job name")
    description: Optional[str] = Field(None, max_length=500)
    training_type: TrainingType = Field(default=TrainingType.FINE_TUNE)

    # Model configuration
    base_model_id: Optional[str] = Field(None, description="Base model to fine-tune")
    base_model_version: Optional[str] = Field(None, description="Base model version")
    output_model_name: Optional[str] = Field(None, description="Name for output model")

    # Data configuration
    robot_ids: List[str] = Field(
        default_factory=list, description="Robot IDs to use episodes from"
    )
    episode_ids: Optional[List[str]] = Field(
        None, description="Specific episode IDs to use"
    )
    data_split: Dict[str, float] = Field(
        default={"train": 0.8, "val": 0.1, "test": 0.1},
        description="Train/val/test split ratios",
    )

    # Compute configuration
    machine_type: str = Field(default="n1-standard-4", description="VM machine type")
    accelerator_type: AcceleratorType = Field(default=AcceleratorType.NONE)
    accelerator_count: int = Field(default=0, ge=0, le=8, description="Number of GPUs/TPUs")

    # Hyperparameters
    hyperparameters: HyperParameters = Field(default_factory=HyperParameters)

    # Checkpointing
    save_checkpoints: bool = Field(default=True)
    checkpoint_frequency: int = Field(default=1, ge=1, description="Save checkpoint every N epochs")
    keep_best_only: bool = Field(default=False, description="Only keep best checkpoint")


class TrainingMetrics(BaseModel):
    """Training metrics at a point in time."""

    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float
    throughput_samples_per_sec: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingJob(BaseModel):
    """Training job with full details."""

    id: str = Field(..., description="Unique job ID")
    user_id: str = Field(..., description="Owner user ID")
    config: TrainingConfig
    status: TrainingStatus = Field(default=TrainingStatus.PENDING)

    # Progress tracking
    current_epoch: int = Field(default=0, ge=0)
    total_epochs: int = Field(default=0, ge=0)
    progress_percent: float = Field(default=0, ge=0, le=100)

    # Results
    best_metrics: Optional[TrainingMetrics] = None
    final_metrics: Optional[TrainingMetrics] = None
    output_model_id: Optional[str] = Field(None, description="ID of trained model")
    output_model_version: Optional[str] = Field(None, description="Version of trained model")

    # Error handling
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Resource tracking
    vertex_ai_job_id: Optional[str] = Field(None, description="Vertex AI CustomJob ID")
    logs_uri: Optional[str] = Field(None, description="URI to training logs")

    model_config = {"from_attributes": True}


class TrainingJobCreate(BaseModel):
    """Schema for creating a training job."""

    config: TrainingConfig


class TrainingJobListResponse(BaseModel):
    """Paginated training job list response."""

    jobs: List[TrainingJob]
    total: int
    page: int = 1
    page_size: int = 20
