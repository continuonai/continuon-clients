"""Google Cloud Storage service for ContinuonAI API."""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from fastapi import Depends, UploadFile
from google.cloud import storage
from google.cloud.exceptions import NotFound

from app.config import Settings, get_settings
from app.models.model import ModelFramework, ModelInfo, ModelType, ModelVersion

logger = logging.getLogger(__name__)

# GCS client singleton
_storage_client: Optional[storage.Client] = None


def get_storage_client(settings: Settings) -> storage.Client:
    """Get or create GCS client singleton."""
    global _storage_client

    if _storage_client is None:
        try:
            if settings.google_cloud_project:
                _storage_client = storage.Client(project=settings.google_cloud_project)
            else:
                _storage_client = storage.Client()
            logger.info("GCS client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise

    return _storage_client


class StorageService:
    """Service for Google Cloud Storage operations."""

    # Path prefixes in bucket
    MODELS_PREFIX = "models"
    EPISODES_PREFIX = "episodes"
    CHECKPOINTS_PREFIX = "checkpoints"
    LOGS_PREFIX = "logs"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = get_storage_client(settings)
        self.models_bucket = self.client.bucket(settings.gcs_bucket_name)
        self.episodes_bucket = self.client.bucket(settings.gcs_episodes_bucket)

    # ==================== Model Operations ====================

    def _model_path(self, model_id: str, version: Optional[str] = None) -> str:
        """Generate GCS path for a model."""
        if version:
            return f"{self.MODELS_PREFIX}/{model_id}/{version}/model.bin"
        return f"{self.MODELS_PREFIX}/{model_id}/"

    def _model_metadata_path(self, model_id: str) -> str:
        """Generate GCS path for model metadata."""
        return f"{self.MODELS_PREFIX}/{model_id}/metadata.json"

    def _version_metadata_path(self, model_id: str, version: str) -> str:
        """Generate GCS path for version metadata."""
        return f"{self.MODELS_PREFIX}/{model_id}/{version}/metadata.json"

    async def list_models(
        self, owner_id: Optional[str] = None, public_only: bool = False
    ) -> List[ModelInfo]:
        """List all available models."""
        models = []

        # List model directories
        blobs = self.client.list_blobs(
            self.models_bucket,
            prefix=f"{self.MODELS_PREFIX}/",
            delimiter="/",
        )

        # Get unique model IDs from prefixes
        model_ids = set()
        for page in blobs.pages:
            for prefix in page.prefixes:
                # Extract model ID from path like "models/model-123/"
                parts = prefix.rstrip("/").split("/")
                if len(parts) >= 2:
                    model_ids.add(parts[1])

        # Load metadata for each model
        for model_id in model_ids:
            try:
                metadata_blob = self.models_bucket.blob(self._model_metadata_path(model_id))
                if metadata_blob.exists():
                    import json
                    metadata = json.loads(metadata_blob.download_as_text())

                    # Filter by owner if specified
                    if owner_id and metadata.get("owner_id") != owner_id:
                        continue

                    # Filter by public if specified
                    if public_only and not metadata.get("is_public", False):
                        continue

                    models.append(ModelInfo(id=model_id, **metadata))
            except Exception as e:
                logger.warning(f"Failed to load model metadata for {model_id}: {e}")

        return models

    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        try:
            metadata_blob = self.models_bucket.blob(self._model_metadata_path(model_id))
            if not metadata_blob.exists():
                return None

            import json
            metadata = json.loads(metadata_blob.download_as_text())
            return ModelInfo(id=model_id, **metadata)
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None

    async def create_model(self, owner_id: str, name: str, **kwargs) -> ModelInfo:
        """Create a new model entry."""
        model_id = str(uuid4())

        model = ModelInfo(
            id=model_id,
            owner_id=owner_id,
            name=name,
            description=kwargs.get("description"),
            model_type=kwargs.get("model_type", ModelType.POLICY),
            framework=kwargs.get("framework", ModelFramework.JAX),
            tags=kwargs.get("tags", []),
            is_public=kwargs.get("is_public", False),
            version_count=0,
            total_downloads=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Save metadata
        import json
        metadata_blob = self.models_bucket.blob(self._model_metadata_path(model_id))
        metadata_blob.upload_from_string(
            json.dumps(model.model_dump(exclude={"id"}, mode="json")),
            content_type="application/json",
        )

        logger.info(f"Created model: {model_id}")
        return model

    async def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        versions = []

        # List version directories
        prefix = f"{self.MODELS_PREFIX}/{model_id}/"
        blobs = self.client.list_blobs(
            self.models_bucket,
            prefix=prefix,
            delimiter="/",
        )

        version_dirs = set()
        for page in blobs.pages:
            for blob_prefix in page.prefixes:
                # Extract version from path like "models/model-123/1.0.0/"
                parts = blob_prefix.rstrip("/").split("/")
                if len(parts) >= 3 and parts[2] != "metadata.json":
                    version_dirs.add(parts[2])

        # Load metadata for each version
        for version in sorted(version_dirs, reverse=True):
            try:
                metadata_blob = self.models_bucket.blob(
                    self._version_metadata_path(model_id, version)
                )
                if metadata_blob.exists():
                    import json
                    metadata = json.loads(metadata_blob.download_as_text())
                    versions.append(ModelVersion(model_id=model_id, version=version, **metadata))
            except Exception as e:
                logger.warning(f"Failed to load version metadata for {model_id}/{version}: {e}")

        return versions

    async def upload_model(
        self,
        model_id: str,
        version: str,
        file: UploadFile,
        user_id: str,
        release_notes: Optional[str] = None,
    ) -> str:
        """Upload a new model version."""
        # Read file content
        content = await file.read()

        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()

        # Upload model file
        model_blob = self.models_bucket.blob(self._model_path(model_id, version))
        model_blob.upload_from_string(content, content_type="application/octet-stream")

        # Create version metadata
        import json
        version_meta = ModelVersion(
            version=version,
            model_id=model_id,
            checksum=checksum,
            file_size_bytes=len(content),
            release_notes=release_notes,
            created_at=datetime.utcnow(),
            created_by=user_id,
        )

        metadata_blob = self.models_bucket.blob(
            self._version_metadata_path(model_id, version)
        )
        metadata_blob.upload_from_string(
            json.dumps(version_meta.model_dump(exclude={"download_url"}, mode="json")),
            content_type="application/json",
        )

        # Update model metadata with latest version
        model = await self.get_model(model_id)
        if model:
            model.latest_version = version
            model.version_count += 1
            model.updated_at = datetime.utcnow()

            model_meta_blob = self.models_bucket.blob(self._model_metadata_path(model_id))
            model_meta_blob.upload_from_string(
                json.dumps(model.model_dump(exclude={"id"}, mode="json")),
                content_type="application/json",
            )

        # Return download URL
        return await self.get_signed_url(model_id, version)

    async def get_signed_url(
        self, model_id: str, version: str, expiration_hours: int = 1
    ) -> str:
        """Generate a signed URL for model download."""
        blob = self.models_bucket.blob(self._model_path(model_id, version))

        if not blob.exists():
            raise NotFound(f"Model {model_id} version {version} not found")

        # Generate signed URL
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=expiration_hours),
            method="GET",
        )

        # Increment download count (best effort)
        try:
            import json
            metadata_blob = self.models_bucket.blob(
                self._version_metadata_path(model_id, version)
            )
            if metadata_blob.exists():
                metadata = json.loads(metadata_blob.download_as_text())
                metadata["download_count"] = metadata.get("download_count", 0) + 1
                metadata_blob.upload_from_string(
                    json.dumps(metadata),
                    content_type="application/json",
                )
        except Exception as e:
            logger.warning(f"Failed to increment download count: {e}")

        return url

    async def get_upload_signed_url(
        self, model_id: str, version: str, expiration_hours: int = 1
    ) -> str:
        """Generate a signed URL for model upload (for large files)."""
        blob = self.models_bucket.blob(self._model_path(model_id, version))

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=expiration_hours),
            method="PUT",
            content_type="application/octet-stream",
        )

        return url

    # ==================== Episode Operations ====================

    def _episode_path(self, robot_id: str, episode_id: str) -> str:
        """Generate GCS path for an episode."""
        return f"{self.EPISODES_PREFIX}/{robot_id}/{episode_id}/"

    async def upload_episode(
        self,
        robot_id: str,
        episode_id: str,
        data: bytes,
        metadata: dict,
    ) -> str:
        """Upload episode data."""
        import json

        # Upload episode data
        data_blob = self.episodes_bucket.blob(
            f"{self._episode_path(robot_id, episode_id)}data.npz"
        )
        data_blob.upload_from_string(data, content_type="application/octet-stream")

        # Upload metadata
        meta_blob = self.episodes_bucket.blob(
            f"{self._episode_path(robot_id, episode_id)}metadata.json"
        )
        meta_blob.upload_from_string(
            json.dumps(metadata),
            content_type="application/json",
        )

        logger.info(f"Uploaded episode {episode_id} for robot {robot_id}")
        return f"gs://{self.settings.gcs_episodes_bucket}/{self._episode_path(robot_id, episode_id)}"

    async def list_episodes(self, robot_id: str) -> List[dict]:
        """List episodes for a robot."""
        episodes = []

        prefix = f"{self.EPISODES_PREFIX}/{robot_id}/"
        blobs = self.client.list_blobs(
            self.episodes_bucket,
            prefix=prefix,
            delimiter="/",
        )

        episode_ids = set()
        for page in blobs.pages:
            for blob_prefix in page.prefixes:
                parts = blob_prefix.rstrip("/").split("/")
                if len(parts) >= 3:
                    episode_ids.add(parts[2])

        for episode_id in sorted(episode_ids):
            try:
                import json
                meta_blob = self.episodes_bucket.blob(
                    f"{self._episode_path(robot_id, episode_id)}metadata.json"
                )
                if meta_blob.exists():
                    metadata = json.loads(meta_blob.download_as_text())
                    metadata["episode_id"] = episode_id
                    episodes.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load episode metadata: {e}")

        return episodes

    async def get_episode_download_url(
        self, robot_id: str, episode_id: str, expiration_hours: int = 1
    ) -> str:
        """Get signed URL for episode download."""
        blob = self.episodes_bucket.blob(
            f"{self._episode_path(robot_id, episode_id)}data.npz"
        )

        if not blob.exists():
            raise NotFound(f"Episode {episode_id} not found for robot {robot_id}")

        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=expiration_hours),
            method="GET",
        )


async def get_storage_service(
    settings: Settings = Depends(get_settings),
) -> StorageService:
    """Dependency to get StorageService instance."""
    return StorageService(settings)
