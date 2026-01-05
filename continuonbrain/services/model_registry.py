"""
Cloud Model Registry - GCS/S3 Model Storage and Distribution

Centralized model registry for storing, versioning, and distributing trained models.
Supports both Google Cloud Storage (GCS) and Amazon S3, with GCS as the primary provider.

Usage:
    from continuonbrain.services.model_registry import ModelRegistry

    # Initialize with GCS (default)
    registry = ModelRegistry(bucket="continuon-models", provider="gcs")

    # List all models
    models = await registry.list_models()

    # Download a specific model
    path = await registry.download_model("seed", "v1.0.0", Path("/tmp/models"))

    # Upload a new model
    await registry.upload_model(
        local_path=Path("./model.npz"),
        model_id="seed",
        version="v1.0.1",
        metadata={"framework": "jax", "param_count": 172202}
    )
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union

logger = logging.getLogger("ModelRegistry")

# Try to import cloud storage libraries
GCS_AVAILABLE = False
S3_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    from google.cloud.exceptions import NotFound as GCSNotFound
    GCS_AVAILABLE = True
except ImportError:
    logger.debug("google-cloud-storage not installed. GCS support disabled.")

try:
    import boto3
    from botocore.exceptions import ClientError as S3ClientError
    S3_AVAILABLE = True
except ImportError:
    logger.debug("boto3 not installed. S3 support disabled.")


@dataclass
class ModelFile:
    """Represents a file in a model package."""
    name: str
    size: int
    checksum: str  # sha256:hex_digest


@dataclass
class TrainingInfo:
    """Training metadata for a model."""
    episodes_used: int = 0
    final_loss: Optional[float] = None
    source_robot: Optional[str] = None
    training_date: Optional[str] = None


@dataclass
class Compatibility:
    """Compatibility requirements for a model."""
    min_brain_version: str = "2.0.0"
    required_deps: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Complete model metadata."""
    model_id: str
    version: str
    model_type: str  # seed, adapter, release
    created_at: str
    framework: str = "jax"
    hardware_targets: List[str] = field(default_factory=lambda: ["arm64", "x86_64"])
    param_count: int = 0
    checksum: str = ""  # sha256:hex_digest of primary file
    files: List[ModelFile] = field(default_factory=list)
    training_info: Optional[TrainingInfo] = None
    compatibility: Optional[Compatibility] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert files to dicts
        data["files"] = [asdict(f) if isinstance(f, ModelFile) else f for f in self.files]
        if self.training_info:
            data["training_info"] = asdict(self.training_info) if isinstance(self.training_info, TrainingInfo) else self.training_info
        if self.compatibility:
            data["compatibility"] = asdict(self.compatibility) if isinstance(self.compatibility, Compatibility) else self.compatibility
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        files = [ModelFile(**f) if isinstance(f, dict) else f for f in data.get("files", [])]
        training_info = None
        if data.get("training_info"):
            ti = data["training_info"]
            training_info = TrainingInfo(**ti) if isinstance(ti, dict) else ti
        compatibility = None
        if data.get("compatibility"):
            c = data["compatibility"]
            compatibility = Compatibility(**c) if isinstance(c, dict) else c

        return cls(
            model_id=data["model_id"],
            version=data["version"],
            model_type=data["model_type"],
            created_at=data["created_at"],
            framework=data.get("framework", "jax"),
            hardware_targets=data.get("hardware_targets", ["arm64", "x86_64"]),
            param_count=data.get("param_count", 0),
            checksum=data.get("checksum", ""),
            files=files,
            training_info=training_info,
            compatibility=compatibility,
            description=data.get("description", ""),
        )


@dataclass
class RegistryIndex:
    """Registry index containing all models metadata."""
    updated_at: str
    models: Dict[str, Dict[str, Any]]  # model_id -> {latest, versions}

    def to_dict(self) -> Dict[str, Any]:
        return {"updated_at": self.updated_at, "models": self.models}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryIndex":
        return cls(
            updated_at=data.get("updated_at", ""),
            models=data.get("models", {}),
        )


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""
    pass


class ModelNotFoundError(ModelRegistryError):
    """Raised when a model is not found."""
    pass


class ChecksumError(ModelRegistryError):
    """Raised when checksum verification fails."""
    pass


class ProviderNotAvailableError(ModelRegistryError):
    """Raised when the requested provider is not available."""
    pass


class ModelRegistry:
    """
    Cloud model registry client.

    Supports GCS (primary) and S3 for model storage and distribution.
    Provides versioning, checksum verification, and model metadata management.
    """

    # Directory structure in bucket
    MANIFESTS_DIR = "manifests"
    REGISTRY_INDEX = "manifests/registry.json"
    SEED_DIR = "seed"
    ADAPTERS_DIR = "adapters"
    RELEASES_DIR = "releases"

    def __init__(
        self,
        bucket: str,
        provider: Literal["gcs", "s3"] = "gcs",
        credentials_path: Optional[str] = None,
        region: str = "us-central1",
    ):
        """
        Initialize the model registry.

        Args:
            bucket: Cloud storage bucket name
            provider: Storage provider ("gcs" or "s3")
            credentials_path: Path to credentials file (optional, uses default if not provided)
            region: Cloud region for S3 (default: us-central1)
        """
        self.bucket_name = bucket
        self.provider = provider
        self.credentials_path = credentials_path
        self.region = region

        self._client: Any = None
        self._bucket: Any = None

        self._validate_provider()
        self._init_client()

    def _validate_provider(self) -> None:
        """Validate that the requested provider is available."""
        if self.provider == "gcs" and not GCS_AVAILABLE:
            raise ProviderNotAvailableError(
                "GCS provider requested but google-cloud-storage is not installed. "
                "Install with: pip install google-cloud-storage"
            )
        if self.provider == "s3" and not S3_AVAILABLE:
            raise ProviderNotAvailableError(
                "S3 provider requested but boto3 is not installed. "
                "Install with: pip install boto3"
            )

    def _init_client(self) -> None:
        """Initialize the cloud storage client."""
        try:
            if self.provider == "gcs":
                self._init_gcs()
            elif self.provider == "s3":
                self._init_s3()
            logger.info(f"Initialized {self.provider.upper()} client for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise ModelRegistryError(f"Failed to initialize storage client: {e}")

    def _init_gcs(self) -> None:
        """Initialize GCS client."""
        if self.credentials_path:
            self._client = gcs_storage.Client.from_service_account_json(self.credentials_path)
        else:
            self._client = gcs_storage.Client()
        self._bucket = self._client.bucket(self.bucket_name)

    def _init_s3(self) -> None:
        """Initialize S3 client."""
        self._client = boto3.client("s3", region_name=self.region)
        self._bucket = self.bucket_name  # S3 uses bucket name directly

    def _get_model_dir(self, model_type: str) -> str:
        """Get directory path for model type."""
        if model_type == "seed":
            return self.SEED_DIR
        elif model_type == "adapter":
            return self.ADAPTERS_DIR
        elif model_type == "release":
            return self.RELEASES_DIR
        else:
            return model_type

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def verify_checksum(self, local_path: Path, expected: str) -> bool:
        """
        Verify checksum of a downloaded file.

        Args:
            local_path: Path to local file
            expected: Expected checksum in format "sha256:hexdigest"

        Returns:
            True if checksum matches, False otherwise
        """
        if not local_path.exists():
            return False

        actual = self._compute_checksum(local_path)
        return actual == expected

    async def _read_blob(self, blob_path: str) -> Optional[bytes]:
        """Read blob content from cloud storage."""
        try:
            if self.provider == "gcs":
                blob = self._bucket.blob(blob_path)
                return blob.download_as_bytes()
            else:  # s3
                response = self._client.get_object(Bucket=self._bucket, Key=blob_path)
                return response["Body"].read()
        except Exception as e:
            logger.debug(f"Failed to read blob {blob_path}: {e}")
            return None

    async def _write_blob(self, blob_path: str, content: bytes, content_type: str = "application/octet-stream") -> bool:
        """Write content to cloud storage."""
        try:
            if self.provider == "gcs":
                blob = self._bucket.blob(blob_path)
                blob.upload_from_string(content, content_type=content_type)
            else:  # s3
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=blob_path,
                    Body=content,
                    ContentType=content_type,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to write blob {blob_path}: {e}")
            return False

    async def _upload_file(self, local_path: Path, blob_path: str) -> bool:
        """Upload a local file to cloud storage."""
        try:
            if self.provider == "gcs":
                blob = self._bucket.blob(blob_path)
                blob.upload_from_filename(str(local_path))
            else:  # s3
                self._client.upload_file(str(local_path), self._bucket, blob_path)
            logger.debug(f"Uploaded {local_path} to {blob_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {blob_path}: {e}")
            return False

    async def _download_file(self, blob_path: str, local_path: Path) -> bool:
        """Download a file from cloud storage."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if self.provider == "gcs":
                blob = self._bucket.blob(blob_path)
                blob.download_to_filename(str(local_path))
            else:  # s3
                self._client.download_file(self._bucket, blob_path, str(local_path))
            logger.debug(f"Downloaded {blob_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {blob_path}: {e}")
            return False

    async def _list_blobs(self, prefix: str) -> List[str]:
        """List blobs with given prefix."""
        try:
            if self.provider == "gcs":
                blobs = self._client.list_blobs(self._bucket, prefix=prefix)
                return [blob.name for blob in blobs]
            else:  # s3
                response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
                return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            logger.error(f"Failed to list blobs with prefix {prefix}: {e}")
            return []

    async def _blob_exists(self, blob_path: str) -> bool:
        """Check if blob exists."""
        try:
            if self.provider == "gcs":
                blob = self._bucket.blob(blob_path)
                return blob.exists()
            else:  # s3
                try:
                    self._client.head_object(Bucket=self._bucket, Key=blob_path)
                    return True
                except S3ClientError:
                    return False
        except Exception:
            return False

    async def _get_registry_index(self) -> RegistryIndex:
        """Get or create registry index."""
        content = await self._read_blob(self.REGISTRY_INDEX)
        if content:
            try:
                data = json.loads(content.decode("utf-8"))
                return RegistryIndex.from_dict(data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse registry index, creating new one")

        # Return empty index
        return RegistryIndex(
            updated_at=datetime.now(timezone.utc).isoformat(),
            models={},
        )

    async def _save_registry_index(self, index: RegistryIndex) -> bool:
        """Save registry index."""
        index.updated_at = datetime.now(timezone.utc).isoformat()
        content = json.dumps(index.to_dict(), indent=2).encode("utf-8")
        return await self._write_blob(self.REGISTRY_INDEX, content, "application/json")

    async def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """
        List all models in the registry.

        Args:
            model_type: Filter by model type (seed, adapter, release)

        Returns:
            List of ModelInfo objects
        """
        index = await self._get_registry_index()
        models: List[ModelInfo] = []

        for model_id, model_data in index.models.items():
            # Get latest version info
            latest_version = model_data.get("latest")
            if not latest_version:
                continue

            # Determine model type from path
            if model_id.startswith("adapters/"):
                m_type = "adapter"
            elif model_id.startswith("releases/"):
                m_type = "release"
            else:
                m_type = "seed"

            # Filter by type if specified
            if model_type and m_type != model_type:
                continue

            # Try to load full manifest
            try:
                info = await self.get_model(model_id, latest_version)
                if info:
                    models.append(info)
            except ModelNotFoundError:
                # Create minimal info from index
                models.append(ModelInfo(
                    model_id=model_id,
                    version=latest_version,
                    model_type=m_type,
                    created_at="",
                ))

        return models

    async def get_model(self, model_id: str, version: Optional[str] = None) -> ModelInfo:
        """
        Get model information by ID and version.

        Args:
            model_id: Model identifier (e.g., "seed", "adapters/robot_001")
            version: Version string (e.g., "v1.0.0"). If None, returns latest.

        Returns:
            ModelInfo object

        Raises:
            ModelNotFoundError: If model not found
        """
        # Determine version
        if version is None:
            version = await self.get_latest_version(model_id)
            if not version:
                raise ModelNotFoundError(f"Model {model_id} not found in registry")

        # Determine model type and path
        if model_id.startswith("adapters/"):
            base_dir = self.ADAPTERS_DIR
            m_type = "adapter"
            model_name = model_id.replace("adapters/", "")
        elif model_id.startswith("releases/"):
            base_dir = self.RELEASES_DIR
            m_type = "release"
            model_name = model_id.replace("releases/", "")
        else:
            base_dir = self.SEED_DIR
            m_type = "seed"
            model_name = model_id

        # Read manifest
        manifest_path = f"{base_dir}/{model_name}/{version}/manifest.json"
        content = await self._read_blob(manifest_path)

        if not content:
            raise ModelNotFoundError(f"Model {model_id} version {version} not found")

        try:
            data = json.loads(content.decode("utf-8"))
            return ModelInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse manifest for {model_id}/{version}: {e}")
            raise ModelNotFoundError(f"Invalid manifest for {model_id}/{version}")

    async def download_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        dest: Path = None,
        verify: bool = True,
    ) -> Path:
        """
        Download model files to local path.

        Args:
            model_id: Model identifier
            version: Version string (None for latest)
            dest: Destination directory (default: temp dir)
            verify: Verify checksums after download

        Returns:
            Path to downloaded model directory

        Raises:
            ModelNotFoundError: If model not found
            ChecksumError: If verification fails
        """
        # Get model info
        info = await self.get_model(model_id, version)
        version = info.version

        # Determine paths
        if model_id.startswith("adapters/"):
            base_dir = self.ADAPTERS_DIR
            model_name = model_id.replace("adapters/", "")
        elif model_id.startswith("releases/"):
            base_dir = self.RELEASES_DIR
            model_name = model_id.replace("releases/", "")
        else:
            base_dir = self.SEED_DIR
            model_name = model_id

        remote_dir = f"{base_dir}/{model_name}/{version}"

        # Set destination
        if dest is None:
            dest = Path(tempfile.mkdtemp(prefix="model_"))
        else:
            dest = dest / model_name / version
            dest.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {model_id}/{version} to {dest}")

        # Download manifest
        manifest_path = f"{remote_dir}/manifest.json"
        if not await self._download_file(manifest_path, dest / "manifest.json"):
            raise ModelNotFoundError(f"Failed to download manifest for {model_id}/{version}")

        # Download all model files
        for file_info in info.files:
            file_path = f"{remote_dir}/{file_info.name}"
            local_path = dest / file_info.name

            if not await self._download_file(file_path, local_path):
                raise ModelNotFoundError(f"Failed to download {file_info.name}")

            # Verify checksum
            if verify and file_info.checksum:
                if not self.verify_checksum(local_path, file_info.checksum):
                    raise ChecksumError(f"Checksum verification failed for {file_info.name}")

        # Download checksum file if exists
        checksum_path = f"{remote_dir}/checksum.sha256"
        if await self._blob_exists(checksum_path):
            await self._download_file(checksum_path, dest / "checksum.sha256")

        logger.info(f"Successfully downloaded {model_id}/{version} to {dest}")
        return dest

    async def upload_model(
        self,
        local_path: Path,
        model_id: str,
        version: str,
        model_type: str = "seed",
        metadata: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
    ) -> str:
        """
        Upload model to cloud registry.

        Args:
            local_path: Local path to model file or directory
            model_id: Model identifier
            version: Version string
            model_type: Type of model (seed, adapter, release)
            metadata: Additional metadata to include in manifest
            files: List of files to upload (if local_path is directory)

        Returns:
            Cloud path to uploaded model

        Raises:
            ModelRegistryError: If upload fails
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise ModelRegistryError(f"Local path does not exist: {local_path}")

        # Determine remote directory
        base_dir = self._get_model_dir(model_type)
        if model_type == "adapter":
            # Keep adapter name from model_id
            remote_dir = f"{self.ADAPTERS_DIR}/{model_id}/{version}"
            full_model_id = f"adapters/{model_id}"
        elif model_type == "release":
            remote_dir = f"{self.RELEASES_DIR}/{model_id}/{version}"
            full_model_id = f"releases/{model_id}"
        else:
            remote_dir = f"{base_dir}/{version}"
            full_model_id = model_id

        logger.info(f"Uploading {model_id}/{version} to {remote_dir}")

        # Collect files to upload
        model_files: List[ModelFile] = []
        files_to_upload: List[tuple] = []

        if local_path.is_file():
            # Single file upload
            checksum = self._compute_checksum(local_path)
            model_files.append(ModelFile(
                name=local_path.name,
                size=local_path.stat().st_size,
                checksum=checksum,
            ))
            files_to_upload.append((local_path, f"{remote_dir}/{local_path.name}"))
        else:
            # Directory upload
            file_list = files or [f.name for f in local_path.iterdir() if f.is_file()]
            for file_name in file_list:
                file_path = local_path / file_name
                if file_path.exists() and file_path.is_file():
                    checksum = self._compute_checksum(file_path)
                    model_files.append(ModelFile(
                        name=file_name,
                        size=file_path.stat().st_size,
                        checksum=checksum,
                    ))
                    files_to_upload.append((file_path, f"{remote_dir}/{file_name}"))

        # Create manifest
        now = datetime.now(timezone.utc).isoformat()
        manifest = ModelInfo(
            model_id=full_model_id,
            version=version,
            model_type=model_type,
            created_at=now,
            files=model_files,
            checksum=model_files[0].checksum if model_files else "",
        )

        # Add additional metadata
        if metadata:
            if "framework" in metadata:
                manifest.framework = metadata["framework"]
            if "hardware_targets" in metadata:
                manifest.hardware_targets = metadata["hardware_targets"]
            if "param_count" in metadata:
                manifest.param_count = metadata["param_count"]
            if "description" in metadata:
                manifest.description = metadata["description"]
            if "training_info" in metadata:
                ti = metadata["training_info"]
                manifest.training_info = TrainingInfo(**ti) if isinstance(ti, dict) else ti
            if "compatibility" in metadata:
                c = metadata["compatibility"]
                manifest.compatibility = Compatibility(**c) if isinstance(c, dict) else c

        # Upload files
        for local, remote in files_to_upload:
            if not await self._upload_file(local, remote):
                raise ModelRegistryError(f"Failed to upload {local}")

        # Upload manifest
        manifest_content = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
        if not await self._write_blob(f"{remote_dir}/manifest.json", manifest_content, "application/json"):
            raise ModelRegistryError("Failed to upload manifest")

        # Create checksum file
        checksum_content = "\n".join(
            f"{f.checksum.split(':')[1]}  {f.name}" for f in model_files
        ).encode("utf-8")
        await self._write_blob(f"{remote_dir}/checksum.sha256", checksum_content, "text/plain")

        # Update registry index
        index = await self._get_registry_index()
        if full_model_id not in index.models:
            index.models[full_model_id] = {"latest": version, "versions": [version]}
        else:
            if version not in index.models[full_model_id]["versions"]:
                index.models[full_model_id]["versions"].append(version)
            index.models[full_model_id]["latest"] = version

        await self._save_registry_index(index)

        logger.info(f"Successfully uploaded {model_id}/{version}")
        return f"gs://{self.bucket_name}/{remote_dir}" if self.provider == "gcs" else f"s3://{self.bucket_name}/{remote_dir}"

    async def get_latest_version(self, model_id: str) -> Optional[str]:
        """
        Get the latest version of a model.

        Args:
            model_id: Model identifier

        Returns:
            Latest version string or None if not found
        """
        index = await self._get_registry_index()

        # Check various prefixes
        for check_id in [model_id, f"adapters/{model_id}", f"releases/{model_id}"]:
            if check_id in index.models:
                return index.models[check_id].get("latest")

        return None

    async def list_versions(self, model_id: str) -> List[str]:
        """
        List all versions of a model.

        Args:
            model_id: Model identifier

        Returns:
            List of version strings
        """
        index = await self._get_registry_index()

        # Check various prefixes
        for check_id in [model_id, f"adapters/{model_id}", f"releases/{model_id}"]:
            if check_id in index.models:
                return index.models[check_id].get("versions", [])

        return []

    async def delete_model(self, model_id: str, version: str) -> bool:
        """
        Delete a model version from the registry.

        Args:
            model_id: Model identifier
            version: Version to delete

        Returns:
            True if deleted successfully
        """
        try:
            # Determine paths
            if model_id.startswith("adapters/"):
                base_dir = self.ADAPTERS_DIR
                model_name = model_id.replace("adapters/", "")
            elif model_id.startswith("releases/"):
                base_dir = self.RELEASES_DIR
                model_name = model_id.replace("releases/", "")
            else:
                base_dir = self.SEED_DIR
                model_name = model_id

            remote_dir = f"{base_dir}/{model_name}/{version}"

            # List and delete all blobs in directory
            blobs = await self._list_blobs(remote_dir)
            for blob_path in blobs:
                try:
                    if self.provider == "gcs":
                        blob = self._bucket.blob(blob_path)
                        blob.delete()
                    else:
                        self._client.delete_object(Bucket=self._bucket, Key=blob_path)
                except Exception as e:
                    logger.warning(f"Failed to delete {blob_path}: {e}")

            # Update registry index
            index = await self._get_registry_index()
            if model_id in index.models:
                versions = index.models[model_id].get("versions", [])
                if version in versions:
                    versions.remove(version)
                if not versions:
                    del index.models[model_id]
                else:
                    index.models[model_id]["versions"] = versions
                    # Update latest if needed
                    if index.models[model_id].get("latest") == version:
                        index.models[model_id]["latest"] = versions[-1]
                await self._save_registry_index(index)

            logger.info(f"Deleted {model_id}/{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_id}/{version}: {e}")
            return False

    def is_available(self) -> bool:
        """Check if the registry is available."""
        return self._client is not None

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on registry.

        Returns:
            Health status dictionary
        """
        try:
            # Try to read registry index
            index = await self._get_registry_index()
            model_count = len(index.models)

            return {
                "status": "healthy",
                "provider": self.provider,
                "bucket": self.bucket_name,
                "model_count": model_count,
                "last_updated": index.updated_at,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "bucket": self.bucket_name,
                "error": str(e),
            }
