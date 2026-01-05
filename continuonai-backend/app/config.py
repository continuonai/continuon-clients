"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "ContinuonAI API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", description="deployment environment")

    # CORS
    cors_origins: List[str] = Field(
        default=["https://continuonai.com", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Firebase / GCP
    google_cloud_project: Optional[str] = Field(
        default=None, description="GCP project ID"
    )
    firebase_credentials_path: Optional[str] = Field(
        default=None, description="Path to Firebase service account JSON"
    )

    # GCS Storage
    gcs_bucket_name: str = Field(
        default="continuonai-models", description="GCS bucket for model storage"
    )
    gcs_episodes_bucket: str = Field(
        default="continuonai-episodes", description="GCS bucket for episode storage"
    )

    # Training
    vertex_ai_location: str = Field(
        default="us-central1", description="Vertex AI region"
    )
    training_machine_type: str = Field(
        default="n1-standard-4", description="Default training machine type"
    )
    training_accelerator_type: Optional[str] = Field(
        default=None, description="GPU accelerator type (e.g., NVIDIA_TESLA_T4)"
    )
    training_accelerator_count: int = Field(
        default=0, description="Number of GPUs for training"
    )

    # API Keys and secrets
    api_key_header: str = "X-API-Key"
    jwt_secret_key: Optional[str] = Field(
        default=None, description="Secret for JWT signing (fallback if Firebase unavailable)"
    )
    jwt_algorithm: str = "HS256"

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100, description="Max requests per minute"
    )
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
