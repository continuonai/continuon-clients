"""Model registry Pydantic schemas."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Types of models supported."""

    POLICY = "policy"
    VISION = "vision"
    PERCEPTION = "perception"
    REASONING = "reasoning"
    ENSEMBLE = "ensemble"


class ModelFramework(str, Enum):
    """Model framework/format."""

    PYTORCH = "pytorch"
    JAX = "jax"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TFLITE = "tflite"


class ModelBase(BaseModel):
    """Base model schema."""

    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    description: Optional[str] = Field(None, max_length=1000, description="Model description")
    model_type: ModelType = Field(default=ModelType.POLICY, description="Type of model")
    framework: ModelFramework = Field(default=ModelFramework.JAX, description="Model framework")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class ModelCreate(ModelBase):
    """Schema for creating a new model entry."""

    is_public: bool = Field(default=False, description="Whether model is publicly available")


class ModelVersion(BaseModel):
    """Model version information."""

    version: str = Field(..., description="Semantic version string (e.g., '1.0.0')")
    model_id: str = Field(..., description="Parent model ID")
    checksum: str = Field(..., description="SHA256 checksum of model file")
    file_size_bytes: int = Field(..., ge=0, description="Model file size")
    download_url: Optional[str] = Field(None, description="Signed download URL")
    download_count: int = Field(default=0, ge=0, description="Number of downloads")
    release_notes: Optional[str] = Field(None, max_length=2000, description="Version release notes")
    metrics: Optional[Dict[str, float]] = Field(
        None, description="Performance metrics (accuracy, loss, etc.)"
    )
    is_latest: bool = Field(default=False, description="Whether this is the latest version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="User ID who uploaded this version")

    model_config = {"from_attributes": True}


class ModelInfo(ModelBase):
    """Full model information."""

    id: str = Field(..., description="Unique model ID")
    owner_id: str = Field(..., description="Owner user ID")
    is_public: bool = Field(default=False, description="Whether model is publicly available")
    latest_version: Optional[str] = Field(None, description="Latest version string")
    version_count: int = Field(default=0, ge=0, description="Total number of versions")
    total_downloads: int = Field(default=0, ge=0, description="Total downloads across all versions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class ModelUploadResponse(BaseModel):
    """Response after uploading a model."""

    model_id: str
    version: str
    upload_url: Optional[str] = Field(None, description="Signed upload URL for large files")
    download_url: str
    checksum: str
    message: str = "Model uploaded successfully"


class ModelDownloadResponse(BaseModel):
    """Response for model download request."""

    model_id: str
    version: str
    download_url: str
    expires_in: int = Field(default=3600, description="URL expiration in seconds")
    checksum: str
    file_size_bytes: int


class ModelListResponse(BaseModel):
    """Paginated model list response."""

    models: List[ModelInfo]
    total: int
    page: int = 1
    page_size: int = 20
