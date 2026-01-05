"""
Local Model Registry - Offline Fallback for Model Storage

Local filesystem-based model registry that mirrors the cloud registry interface.
Used when cloud connectivity is unavailable or for local development/testing.

Usage:
    from continuonbrain.services.model_registry_local import LocalModelRegistry

    # Initialize with local base directory
    registry = LocalModelRegistry(base_dir=Path("./models"))

    # Use same interface as ModelRegistry
    models = await registry.list_models()
    path = await registry.download_model("seed", "v1.0.0", dest)
"""

import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from continuonbrain.services.model_registry import (
    ModelInfo,
    ModelFile,
    RegistryIndex,
    TrainingInfo,
    Compatibility,
    ModelRegistryError,
    ModelNotFoundError,
    ChecksumError,
)

logger = logging.getLogger("LocalModelRegistry")


class LocalModelRegistry:
    """
    Local filesystem fallback when cloud is unavailable.

    Implements the same interface as ModelRegistry for seamless switching
    between cloud and local storage.

    Directory Structure:
        base_dir/
        ├── manifests/
        │   └── registry.json
        ├── seed/
        │   └── v1.0.0/
        │       ├── manifest.json
        │       ├── seed_model.npz
        │       └── checksum.sha256
        ├── adapters/
        │   └── robot_001_v1/
        │       ├── manifest.json
        │       └── lora_adapters.npz
        └── releases/
            └── prod_v2.1.0/
                ├── manifest.json
                ├── bundle.tar.gz
                └── signature.sig
    """

    MANIFESTS_DIR = "manifests"
    REGISTRY_INDEX = "manifests/registry.json"
    SEED_DIR = "seed"
    ADAPTERS_DIR = "adapters"
    RELEASES_DIR = "releases"

    def __init__(
        self,
        base_dir: Union[str, Path],
        create_if_missing: bool = True,
    ):
        """
        Initialize local model registry.

        Args:
            base_dir: Base directory for model storage
            create_if_missing: Create directory structure if it doesn't exist
        """
        self.base_dir = Path(base_dir)

        if create_if_missing:
            self._init_directory_structure()

    def _init_directory_structure(self) -> None:
        """Create required directory structure."""
        dirs = [
            self.base_dir / self.MANIFESTS_DIR,
            self.base_dir / self.SEED_DIR,
            self.base_dir / self.ADAPTERS_DIR,
            self.base_dir / self.RELEASES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize registry index if not exists
        index_path = self.base_dir / self.REGISTRY_INDEX
        if not index_path.exists():
            index = RegistryIndex(
                updated_at=datetime.now(timezone.utc).isoformat(),
                models={},
            )
            self._save_json(index_path, index.to_dict())

        logger.info(f"Local model registry initialized at {self.base_dir}")

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON file."""
        try:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {path}: {e}")
        return None

    def _save_json(self, path: Path, data: Dict[str, Any]) -> bool:
        """Save data to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {path}: {e}")
            return False

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def verify_checksum(self, local_path: Path, expected: str) -> bool:
        """
        Verify checksum of a file.

        Args:
            local_path: Path to file
            expected: Expected checksum in format "sha256:hexdigest"

        Returns:
            True if checksum matches
        """
        if not local_path.exists():
            return False
        actual = self._compute_checksum(local_path)
        return actual == expected

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

    def _get_model_path(self, model_id: str, version: str) -> Path:
        """Get full path to model version directory."""
        if model_id.startswith("adapters/"):
            base_dir = self.ADAPTERS_DIR
            model_name = model_id.replace("adapters/", "")
        elif model_id.startswith("releases/"):
            base_dir = self.RELEASES_DIR
            model_name = model_id.replace("releases/", "")
        else:
            base_dir = self.SEED_DIR
            model_name = model_id

        return self.base_dir / base_dir / model_name / version

    async def _get_registry_index(self) -> RegistryIndex:
        """Get registry index."""
        index_path = self.base_dir / self.REGISTRY_INDEX
        data = self._load_json(index_path)
        if data:
            return RegistryIndex.from_dict(data)
        return RegistryIndex(
            updated_at=datetime.now(timezone.utc).isoformat(),
            models={},
        )

    async def _save_registry_index(self, index: RegistryIndex) -> bool:
        """Save registry index."""
        index.updated_at = datetime.now(timezone.utc).isoformat()
        index_path = self.base_dir / self.REGISTRY_INDEX
        return self._save_json(index_path, index.to_dict())

    async def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """
        List all models in the local registry.

        Args:
            model_type: Filter by model type (seed, adapter, release)

        Returns:
            List of ModelInfo objects
        """
        index = await self._get_registry_index()
        models: List[ModelInfo] = []

        for model_id, model_data in index.models.items():
            latest_version = model_data.get("latest")
            if not latest_version:
                continue

            # Determine model type
            if model_id.startswith("adapters/"):
                m_type = "adapter"
            elif model_id.startswith("releases/"):
                m_type = "release"
            else:
                m_type = "seed"

            # Filter by type
            if model_type and m_type != model_type:
                continue

            try:
                info = await self.get_model(model_id, latest_version)
                if info:
                    models.append(info)
            except ModelNotFoundError:
                # Create minimal info
                models.append(ModelInfo(
                    model_id=model_id,
                    version=latest_version,
                    model_type=m_type,
                    created_at="",
                ))

        return models

    async def get_model(self, model_id: str, version: Optional[str] = None) -> ModelInfo:
        """
        Get model information.

        Args:
            model_id: Model identifier
            version: Version string (None for latest)

        Returns:
            ModelInfo object

        Raises:
            ModelNotFoundError: If model not found
        """
        if version is None:
            version = await self.get_latest_version(model_id)
            if not version:
                raise ModelNotFoundError(f"Model {model_id} not found")

        model_path = self._get_model_path(model_id, version)
        manifest_path = model_path / "manifest.json"

        if not manifest_path.exists():
            raise ModelNotFoundError(f"Model {model_id} version {version} not found")

        data = self._load_json(manifest_path)
        if not data:
            raise ModelNotFoundError(f"Invalid manifest for {model_id}/{version}")

        return ModelInfo.from_dict(data)

    async def download_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        dest: Optional[Path] = None,
        verify: bool = True,
    ) -> Path:
        """
        Copy model files to destination (or return source path).

        For local registry, this either copies files to dest or returns
        the existing local path.

        Args:
            model_id: Model identifier
            version: Version string (None for latest)
            dest: Destination directory (None to return source path)
            verify: Verify checksums

        Returns:
            Path to model directory

        Raises:
            ModelNotFoundError: If model not found
            ChecksumError: If verification fails
        """
        info = await self.get_model(model_id, version)
        version = info.version
        source_path = self._get_model_path(model_id, version)

        if not source_path.exists():
            raise ModelNotFoundError(f"Model files not found at {source_path}")

        # Verify checksums if requested
        if verify:
            for file_info in info.files:
                file_path = source_path / file_info.name
                if file_path.exists() and file_info.checksum:
                    if not self.verify_checksum(file_path, file_info.checksum):
                        raise ChecksumError(f"Checksum verification failed for {file_info.name}")

        # If no dest, return source path
        if dest is None:
            return source_path

        # Copy to destination
        dest = dest / model_id.replace("/", "_") / version
        dest.mkdir(parents=True, exist_ok=True)

        # Copy all files
        for item in source_path.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
            elif item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)

        logger.info(f"Copied {model_id}/{version} to {dest}")
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
        Add model to local registry.

        Args:
            local_path: Path to model file or directory
            model_id: Model identifier
            version: Version string
            model_type: Model type (seed, adapter, release)
            metadata: Additional metadata
            files: List of files to include (if local_path is directory)

        Returns:
            Local path to uploaded model
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise ModelRegistryError(f"Local path does not exist: {local_path}")

        # Determine destination
        base_dir = self._get_model_dir(model_type)
        if model_type == "adapter":
            dest_dir = self.base_dir / self.ADAPTERS_DIR / model_id / version
            full_model_id = f"adapters/{model_id}"
        elif model_type == "release":
            dest_dir = self.base_dir / self.RELEASES_DIR / model_id / version
            full_model_id = f"releases/{model_id}"
        else:
            dest_dir = self.base_dir / base_dir / version
            full_model_id = model_id

        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Adding {model_id}/{version} to local registry at {dest_dir}")

        # Collect and copy files
        model_files: List[ModelFile] = []

        if local_path.is_file():
            # Single file
            dest_file = dest_dir / local_path.name
            shutil.copy2(local_path, dest_file)
            checksum = self._compute_checksum(dest_file)
            model_files.append(ModelFile(
                name=local_path.name,
                size=dest_file.stat().st_size,
                checksum=checksum,
            ))
        else:
            # Directory
            file_list = files or [f.name for f in local_path.iterdir() if f.is_file()]
            for file_name in file_list:
                src_file = local_path / file_name
                if src_file.exists() and src_file.is_file():
                    dest_file = dest_dir / file_name
                    shutil.copy2(src_file, dest_file)
                    checksum = self._compute_checksum(dest_file)
                    model_files.append(ModelFile(
                        name=file_name,
                        size=dest_file.stat().st_size,
                        checksum=checksum,
                    ))

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

        # Add metadata
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

        # Save manifest
        self._save_json(dest_dir / "manifest.json", manifest.to_dict())

        # Create checksum file
        checksum_content = "\n".join(
            f"{f.checksum.split(':')[1]}  {f.name}" for f in model_files
        )
        with open(dest_dir / "checksum.sha256", "w") as f:
            f.write(checksum_content)

        # Update registry index
        index = await self._get_registry_index()
        if full_model_id not in index.models:
            index.models[full_model_id] = {"latest": version, "versions": [version]}
        else:
            if version not in index.models[full_model_id]["versions"]:
                index.models[full_model_id]["versions"].append(version)
            index.models[full_model_id]["latest"] = version

        await self._save_registry_index(index)

        logger.info(f"Added {model_id}/{version} to local registry")
        return str(dest_dir)

    async def get_latest_version(self, model_id: str) -> Optional[str]:
        """
        Get the latest version of a model.

        Args:
            model_id: Model identifier

        Returns:
            Latest version string or None
        """
        index = await self._get_registry_index()

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

        for check_id in [model_id, f"adapters/{model_id}", f"releases/{model_id}"]:
            if check_id in index.models:
                return index.models[check_id].get("versions", [])

        return []

    async def delete_model(self, model_id: str, version: str) -> bool:
        """
        Delete a model version from local registry.

        Args:
            model_id: Model identifier
            version: Version to delete

        Returns:
            True if deleted successfully
        """
        try:
            model_path = self._get_model_path(model_id, version)

            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"Deleted model directory: {model_path}")

            # Update registry index
            index = await self._get_registry_index()
            for check_id in [model_id, f"adapters/{model_id}", f"releases/{model_id}"]:
                if check_id in index.models:
                    versions = index.models[check_id].get("versions", [])
                    if version in versions:
                        versions.remove(version)
                    if not versions:
                        del index.models[check_id]
                    else:
                        index.models[check_id]["versions"] = versions
                        if index.models[check_id].get("latest") == version:
                            index.models[check_id]["latest"] = versions[-1]
                    break

            await self._save_registry_index(index)
            return True

        except Exception as e:
            logger.error(f"Failed to delete {model_id}/{version}: {e}")
            return False

    async def sync_from_cloud(
        self,
        cloud_registry: "ModelRegistry",  # type: ignore
        model_ids: Optional[List[str]] = None,
        latest_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Sync models from cloud registry to local.

        Args:
            cloud_registry: Cloud ModelRegistry instance
            model_ids: Specific models to sync (None for all)
            latest_only: Only sync latest version of each model

        Returns:
            Sync result summary
        """
        synced = []
        failed = []

        try:
            cloud_models = await cloud_registry.list_models()

            for model in cloud_models:
                if model_ids and model.model_id not in model_ids:
                    continue

                try:
                    versions = [model.version] if latest_only else await cloud_registry.list_versions(model.model_id)

                    for version in versions:
                        # Check if already exists locally
                        try:
                            local_info = await self.get_model(model.model_id, version)
                            if local_info:
                                logger.debug(f"Skipping {model.model_id}/{version} - already exists")
                                continue
                        except ModelNotFoundError:
                            pass

                        # Download from cloud
                        temp_path = await cloud_registry.download_model(
                            model.model_id, version, verify=True
                        )

                        # Import to local registry
                        await self.upload_model(
                            local_path=temp_path,
                            model_id=model.model_id.split("/")[-1],
                            version=version,
                            model_type=model.model_type,
                            metadata={
                                "framework": model.framework,
                                "hardware_targets": model.hardware_targets,
                                "param_count": model.param_count,
                                "description": model.description,
                            },
                        )

                        synced.append(f"{model.model_id}/{version}")
                        logger.info(f"Synced {model.model_id}/{version}")

                        # Cleanup temp
                        shutil.rmtree(temp_path, ignore_errors=True)

                except Exception as e:
                    failed.append({"model": model.model_id, "error": str(e)})
                    logger.error(f"Failed to sync {model.model_id}: {e}")

        except Exception as e:
            return {"success": False, "error": str(e), "synced": synced, "failed": failed}

        return {
            "success": True,
            "synced_count": len(synced),
            "synced": synced,
            "failed_count": len(failed),
            "failed": failed,
        }

    async def sync_to_cloud(
        self,
        cloud_registry: "ModelRegistry",  # type: ignore
        model_ids: Optional[List[str]] = None,
        latest_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Sync models from local registry to cloud.

        Args:
            cloud_registry: Cloud ModelRegistry instance
            model_ids: Specific models to sync (None for all)
            latest_only: Only sync latest version of each model

        Returns:
            Sync result summary
        """
        synced = []
        failed = []

        try:
            local_models = await self.list_models()

            for model in local_models:
                if model_ids and model.model_id not in model_ids:
                    continue

                try:
                    versions = [model.version] if latest_only else await self.list_versions(model.model_id)

                    for version in versions:
                        # Check if already exists in cloud
                        try:
                            cloud_info = await cloud_registry.get_model(model.model_id, version)
                            if cloud_info:
                                logger.debug(f"Skipping {model.model_id}/{version} - already in cloud")
                                continue
                        except ModelNotFoundError:
                            pass

                        # Get local path
                        local_path = self._get_model_path(model.model_id, version)

                        # Upload to cloud
                        await cloud_registry.upload_model(
                            local_path=local_path,
                            model_id=model.model_id.split("/")[-1],
                            version=version,
                            model_type=model.model_type,
                            metadata={
                                "framework": model.framework,
                                "hardware_targets": model.hardware_targets,
                                "param_count": model.param_count,
                                "description": model.description,
                            },
                        )

                        synced.append(f"{model.model_id}/{version}")
                        logger.info(f"Synced {model.model_id}/{version} to cloud")

                except Exception as e:
                    failed.append({"model": model.model_id, "error": str(e)})
                    logger.error(f"Failed to sync {model.model_id}: {e}")

        except Exception as e:
            return {"success": False, "error": str(e), "synced": synced, "failed": failed}

        return {
            "success": True,
            "synced_count": len(synced),
            "synced": synced,
            "failed_count": len(failed),
            "failed": failed,
        }

    def is_available(self) -> bool:
        """Check if local registry is available."""
        return self.base_dir.exists()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on local registry.

        Returns:
            Health status dictionary
        """
        try:
            if not self.base_dir.exists():
                return {
                    "status": "unhealthy",
                    "error": f"Base directory does not exist: {self.base_dir}",
                }

            index = await self._get_registry_index()
            model_count = len(index.models)

            # Calculate total size
            total_size = 0
            for d in [self.SEED_DIR, self.ADAPTERS_DIR, self.RELEASES_DIR]:
                dir_path = self.base_dir / d
                if dir_path.exists():
                    for f in dir_path.rglob("*"):
                        if f.is_file():
                            total_size += f.stat().st_size

            return {
                "status": "healthy",
                "base_dir": str(self.base_dir),
                "model_count": model_count,
                "last_updated": index.updated_at,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "base_dir": str(self.base_dir),
                "error": str(e),
            }


class HybridModelRegistry:
    """
    Hybrid registry that uses cloud when available, local as fallback.

    Automatically falls back to local registry when cloud is unavailable.
    Supports bidirectional sync between cloud and local.
    """

    def __init__(
        self,
        cloud_bucket: str,
        local_base_dir: Union[str, Path],
        cloud_provider: str = "gcs",
        cloud_credentials_path: Optional[str] = None,
        prefer_cloud: bool = True,
    ):
        """
        Initialize hybrid registry.

        Args:
            cloud_bucket: Cloud storage bucket name
            local_base_dir: Local base directory for fallback
            cloud_provider: Cloud provider (gcs or s3)
            cloud_credentials_path: Path to cloud credentials
            prefer_cloud: Prefer cloud over local when both available
        """
        self.prefer_cloud = prefer_cloud
        self._cloud: Optional["ModelRegistry"] = None
        self._local: LocalModelRegistry

        # Initialize local registry (always available)
        self._local = LocalModelRegistry(base_dir=local_base_dir)

        # Try to initialize cloud registry
        try:
            from continuonbrain.services.model_registry import ModelRegistry
            self._cloud = ModelRegistry(
                bucket=cloud_bucket,
                provider=cloud_provider,
                credentials_path=cloud_credentials_path,
            )
            logger.info("Hybrid registry initialized with cloud support")
        except Exception as e:
            logger.warning(f"Cloud registry unavailable, using local only: {e}")

    def _get_registry(self) -> Union["ModelRegistry", LocalModelRegistry]:
        """Get the appropriate registry based on availability and preference."""
        if self.prefer_cloud and self._cloud and self._cloud.is_available():
            return self._cloud
        return self._local

    async def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List all models."""
        return await self._get_registry().list_models(model_type)

    async def get_model(self, model_id: str, version: Optional[str] = None) -> ModelInfo:
        """Get model info."""
        return await self._get_registry().get_model(model_id, version)

    async def download_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        dest: Optional[Path] = None,
        verify: bool = True,
    ) -> Path:
        """Download model."""
        return await self._get_registry().download_model(model_id, version, dest, verify)

    async def upload_model(
        self,
        local_path: Path,
        model_id: str,
        version: str,
        model_type: str = "seed",
        metadata: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
    ) -> str:
        """Upload model."""
        return await self._get_registry().upload_model(
            local_path, model_id, version, model_type, metadata, files
        )

    async def get_latest_version(self, model_id: str) -> Optional[str]:
        """Get latest version."""
        return await self._get_registry().get_latest_version(model_id)

    def verify_checksum(self, local_path: Path, expected: str) -> bool:
        """Verify checksum."""
        return self._get_registry().verify_checksum(local_path, expected)

    async def sync(self, direction: str = "both") -> Dict[str, Any]:
        """
        Sync between cloud and local registries.

        Args:
            direction: "to_cloud", "from_cloud", or "both"

        Returns:
            Sync result summary
        """
        if not self._cloud:
            return {"success": False, "error": "Cloud registry not available"}

        results = {}

        if direction in ["from_cloud", "both"]:
            results["from_cloud"] = await self._local.sync_from_cloud(self._cloud)

        if direction in ["to_cloud", "both"]:
            results["to_cloud"] = await self._local.sync_to_cloud(self._cloud)

        return results

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on both registries."""
        local_health = await self._local.health_check()
        cloud_health = None

        if self._cloud:
            try:
                cloud_health = await self._cloud.health_check()
            except Exception as e:
                cloud_health = {"status": "unhealthy", "error": str(e)}

        return {
            "local": local_health,
            "cloud": cloud_health,
            "active": "cloud" if (self.prefer_cloud and cloud_health and cloud_health.get("status") == "healthy") else "local",
        }
