"""
Model Distribution Service - GCP-based seed model distribution for continuonai.com.

This service manages the complete model distribution pipeline:
1. Model Packaging - Bundle trained models with metadata
2. GCP Upload - Upload to Cloud Storage for hosting
3. Registry Management - Update model registry for continuonai.com
4. Robot Registration - Track registered robots and their model versions
5. Update Distribution - Push updates to registered robots

Architecture:
    Pi5 Robot → Train → Validate → Upload to GCS
                                      ↓
    GCS (gs://continuon-models/)  →  Registry (Firestore)
                                      ↓
    continuonai.com  ←  Robot Registration  →  OTA Updates

Usage:
    from continuonbrain.services.model_distribution import (
        ModelDistributionService,
        DistributionConfig,
    )

    service = ModelDistributionService(
        config=DistributionConfig(
            project_id="continuon-xr",
            bucket_name="continuon-models",
        )
    )

    # Upload a trained model
    result = await service.upload_model(
        model_path=Path("/opt/continuonos/brain/model/candidate"),
        version="1.0.0",
        release_notes="Initial seed model release",
    )

    # Register a new robot
    robot = await service.register_robot(
        device_id="robot_123",
        robot_model="donkeycar",
        owner_uid="user_abc",
    )

    # Push update to all robots
    await service.distribute_update(model_id="seed", version="1.0.0")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger("ModelDistributionService")


# Optional GCP dependencies
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    firestore = None


class ModelType(str, Enum):
    """Types of models for distribution."""
    SEED = "seed"                    # Base seed model for new robots
    ADAPTER = "adapter"              # Task-specific adapter
    FULL = "full"                    # Complete model replacement
    PATCH = "patch"                  # Incremental update patch


class RobotStatus(str, Enum):
    """Robot registration status."""
    PENDING = "pending"              # Awaiting approval
    ACTIVE = "active"                # Registered and active
    SUSPENDED = "suspended"          # Temporarily suspended
    DECOMMISSIONED = "decommissioned"


class UpdatePriority(str, Enum):
    """Update priority level."""
    CRITICAL = "critical"            # Security or safety update
    HIGH = "high"                    # Important feature
    NORMAL = "normal"                # Standard update
    LOW = "low"                      # Optional improvement


@dataclass
class DistributionConfig:
    """Configuration for model distribution service."""
    # GCP settings
    project_id: str = "continuon-xr"
    bucket_name: str = "continuon-models"
    region: str = "us-central1"
    service_account_path: Optional[str] = None

    # Model registry settings
    registry_collection: str = "model_registry"
    robots_collection: str = "registered_robots"
    updates_collection: str = "ota_updates"

    # Distribution settings
    enable_auto_distribution: bool = True
    rollout_percentage: float = 100.0  # Gradual rollout percentage
    require_robot_approval: bool = False

    # Model paths
    models_prefix: str = "models"
    bundles_prefix: str = "bundles"
    registry_file: str = "registry.json"

    # Website integration
    website_domain: str = "continuonai.com"
    api_base_url: str = "https://api.continuonai.com"

    @classmethod
    def from_file(cls, path: Path) -> "DistributionConfig":
        """Load config from JSON file."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return cls()


@dataclass
class ModelManifest:
    """Manifest for a distributed model."""
    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime

    # Model metadata
    name: str = ""
    description: str = ""
    release_notes: str = ""

    # Technical details
    architecture: str = "wavecore"
    framework: str = "jax"
    parameter_count: int = 0
    file_size_bytes: int = 0

    # Checksums
    checksum_sha256: str = ""
    signature: str = ""

    # Compatibility
    min_brain_version: str = "0.1.0"
    supported_hardware: List[str] = field(default_factory=lambda: ["pi5", "jetson", "generic"])

    # Distribution
    download_url: str = ""
    bundle_url: str = ""

    # Metrics
    training_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "name": self.name,
            "description": self.description,
            "release_notes": self.release_notes,
            "architecture": self.architecture,
            "framework": self.framework,
            "parameter_count": self.parameter_count,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "signature": self.signature,
            "min_brain_version": self.min_brain_version,
            "supported_hardware": self.supported_hardware,
            "download_url": self.download_url,
            "bundle_url": self.bundle_url,
            "training_loss": self.training_loss,
            "validation_accuracy": self.validation_accuracy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        """Create from dictionary."""
        return cls(
            model_id=data.get("model_id", "unknown"),
            model_type=ModelType(data.get("model_type", "seed")),
            version=data.get("version", "0.0.0"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            name=data.get("name", ""),
            description=data.get("description", ""),
            release_notes=data.get("release_notes", ""),
            architecture=data.get("architecture", "wavecore"),
            framework=data.get("framework", "jax"),
            parameter_count=data.get("parameter_count", 0),
            file_size_bytes=data.get("file_size_bytes", 0),
            checksum_sha256=data.get("checksum_sha256", ""),
            signature=data.get("signature", ""),
            min_brain_version=data.get("min_brain_version", "0.1.0"),
            supported_hardware=data.get("supported_hardware", ["pi5", "jetson", "generic"]),
            download_url=data.get("download_url", ""),
            bundle_url=data.get("bundle_url", ""),
            training_loss=data.get("training_loss"),
            validation_accuracy=data.get("validation_accuracy"),
        )


@dataclass
class RegisteredRobot:
    """Information about a registered robot."""
    device_id: str
    robot_model: str
    owner_uid: str
    status: RobotStatus
    registered_at: datetime

    # Device info
    hardware_profile: str = "pi5"
    brain_version: str = "0.1.0"
    current_model_version: str = "0.0.0"

    # Network info
    last_seen: Optional[datetime] = None
    ip_address: Optional[str] = None

    # Update preferences
    auto_update: bool = True
    update_channel: str = "stable"  # stable, beta, dev

    # Metadata
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "robot_model": self.robot_model,
            "owner_uid": self.owner_uid,
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "hardware_profile": self.hardware_profile,
            "brain_version": self.brain_version,
            "current_model_version": self.current_model_version,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "ip_address": self.ip_address,
            "auto_update": self.auto_update,
            "update_channel": self.update_channel,
            "name": self.name,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredRobot":
        """Create from dictionary."""
        return cls(
            device_id=data.get("device_id", ""),
            robot_model=data.get("robot_model", ""),
            owner_uid=data.get("owner_uid", ""),
            status=RobotStatus(data.get("status", "pending")),
            registered_at=datetime.fromisoformat(data["registered_at"]) if data.get("registered_at") else datetime.now(timezone.utc),
            hardware_profile=data.get("hardware_profile", "pi5"),
            brain_version=data.get("brain_version", "0.1.0"),
            current_model_version=data.get("current_model_version", "0.0.0"),
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
            ip_address=data.get("ip_address"),
            auto_update=data.get("auto_update", True),
            update_channel=data.get("update_channel", "stable"),
            name=data.get("name"),
            tags=data.get("tags", []),
        )


@dataclass
class UpdateNotification:
    """Notification of an available update."""
    update_id: str
    model_id: str
    version: str
    target_robots: List[str]  # device_ids, or ["all"]

    # Update info
    priority: UpdatePriority
    release_notes: str
    download_url: str
    file_size_bytes: int
    checksum: str

    # Rollout
    rollout_percentage: float
    created_at: datetime
    expires_at: Optional[datetime] = None

    # Status tracking
    robots_notified: List[str] = field(default_factory=list)
    robots_downloaded: List[str] = field(default_factory=list)
    robots_installed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "update_id": self.update_id,
            "model_id": self.model_id,
            "version": self.version,
            "target_robots": self.target_robots,
            "priority": self.priority.value,
            "release_notes": self.release_notes,
            "download_url": self.download_url,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "rollout_percentage": self.rollout_percentage,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "robots_notified": self.robots_notified,
            "robots_downloaded": self.robots_downloaded,
            "robots_installed": self.robots_installed,
        }


class ModelDistributionService:
    """
    Manages model distribution to robots via GCP and continuonai.com.

    Provides:
    - Model packaging and upload to GCS
    - Model registry management
    - Robot registration and tracking
    - OTA update distribution
    """

    def __init__(
        self,
        config: Optional[DistributionConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize ModelDistributionService.

        Args:
            config: Distribution configuration
            config_path: Path to config file
        """
        if config:
            self.config = config
        elif config_path:
            self.config = DistributionConfig.from_file(config_path)
        else:
            self.config = DistributionConfig()

        self._storage_client: Optional[Any] = None
        self._firestore_client: Optional[Any] = None
        self._initialized = False

        # Local cache
        self._registry_cache: Dict[str, Any] = {}
        self._cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minutes

    def _init_clients(self) -> bool:
        """Initialize GCP clients."""
        if self._initialized:
            return True

        if not GCS_AVAILABLE:
            logger.warning("google-cloud-storage not installed")
            return False

        if not FIRESTORE_AVAILABLE:
            logger.warning("google-cloud-firestore not installed")
            return False

        try:
            if self.config.service_account_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.service_account_path

            self._storage_client = storage.Client(project=self.config.project_id)
            self._firestore_client = firestore.Client(project=self.config.project_id)
            self._initialized = True

            logger.info(f"ModelDistributionService initialized for project: {self.config.project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return GCS_AVAILABLE and FIRESTORE_AVAILABLE

    # ========== Model Upload and Packaging ==========

    async def upload_model(
        self,
        model_path: Path,
        version: str,
        model_id: str = "seed",
        model_type: ModelType = ModelType.SEED,
        release_notes: str = "",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a trained model to GCS for distribution.

        Args:
            model_path: Path to model directory
            version: Version string (e.g., "1.0.0")
            model_id: Model identifier
            model_type: Type of model
            release_notes: Release notes for this version
            name: Human-readable name
            description: Model description

        Returns:
            Upload result with URLs and manifest
        """
        if not self._init_clients():
            return {"success": False, "error": "GCP clients not initialized"}

        model_path = Path(model_path)
        if not model_path.exists():
            return {"success": False, "error": f"Model path not found: {model_path}"}

        try:
            # Create manifest
            manifest = await self._create_manifest(
                model_path=model_path,
                model_id=model_id,
                model_type=model_type,
                version=version,
                name=name or f"{model_id} v{version}",
                description=description or "",
                release_notes=release_notes,
            )

            # Create bundle
            bundle_path = await self._create_bundle(model_path, manifest)

            # Upload to GCS
            upload_result = await self._upload_to_gcs(bundle_path, manifest)

            if not upload_result.get("success"):
                return upload_result

            # Update manifest with URLs
            manifest.download_url = upload_result["download_url"]
            manifest.bundle_url = upload_result["bundle_url"]

            # Update registry
            await self._update_registry(manifest)

            # Clean up temp bundle
            if bundle_path.exists():
                bundle_path.unlink()

            logger.info(f"Model {model_id} v{version} uploaded successfully")

            return {
                "success": True,
                "model_id": model_id,
                "version": version,
                "download_url": manifest.download_url,
                "bundle_url": manifest.bundle_url,
                "checksum": manifest.checksum_sha256,
                "manifest": manifest.to_dict(),
            }

        except Exception as e:
            logger.error(f"Model upload failed: {e}")
            return {"success": False, "error": str(e)}

    async def _create_manifest(
        self,
        model_path: Path,
        model_id: str,
        model_type: ModelType,
        version: str,
        name: str,
        description: str,
        release_notes: str,
    ) -> ModelManifest:
        """Create model manifest from directory."""
        # Read existing manifest if present
        existing_manifest_path = model_path / "manifest.json"
        existing_data = {}
        if existing_manifest_path.exists():
            try:
                existing_data = json.loads(existing_manifest_path.read_text())
            except Exception:
                pass

        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

        # Count parameters (if we can load the model)
        param_count = existing_data.get("parameter_count", 0)

        # Get training metrics
        training_loss = existing_data.get("training_loss") or existing_data.get("final_loss")
        validation_accuracy = existing_data.get("validation_accuracy")

        return ModelManifest(
            model_id=model_id,
            model_type=model_type,
            version=version,
            created_at=datetime.now(timezone.utc),
            name=name,
            description=description,
            release_notes=release_notes,
            architecture=existing_data.get("architecture", "wavecore"),
            framework=existing_data.get("framework", "jax"),
            parameter_count=param_count,
            file_size_bytes=total_size,
            min_brain_version=existing_data.get("min_brain_version", "0.1.0"),
            supported_hardware=existing_data.get("supported_hardware", ["pi5", "jetson", "generic"]),
            training_loss=training_loss,
            validation_accuracy=validation_accuracy,
        )

    async def _create_bundle(self, model_path: Path, manifest: ModelManifest) -> Path:
        """Create a distributable bundle from model directory."""
        loop = asyncio.get_event_loop()

        def _create():
            # Create temp file for bundle
            bundle_path = Path(tempfile.mktemp(suffix=".tar.gz"))

            with tarfile.open(bundle_path, "w:gz") as tar:
                # Add all model files
                for item in model_path.iterdir():
                    tar.add(item, arcname=item.name)

                # Write manifest
                manifest_content = json.dumps(manifest.to_dict(), indent=2)
                manifest_info = tarfile.TarInfo(name="manifest.json")
                manifest_info.size = len(manifest_content.encode())
                import io
                tar.addfile(manifest_info, io.BytesIO(manifest_content.encode()))

            # Compute checksum
            sha256 = hashlib.sha256()
            with open(bundle_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)

            manifest.checksum_sha256 = f"sha256:{sha256.hexdigest()}"
            manifest.file_size_bytes = bundle_path.stat().st_size

            return bundle_path

        return await loop.run_in_executor(None, _create)

    async def _upload_to_gcs(self, bundle_path: Path, manifest: ModelManifest) -> Dict[str, Any]:
        """Upload bundle to Google Cloud Storage."""
        loop = asyncio.get_event_loop()

        def _upload():
            bucket = self._storage_client.bucket(self.config.bucket_name)

            # Bundle path: bundles/{model_id}/{version}/bundle.tar.gz
            gcs_bundle_path = f"{self.config.bundles_prefix}/{manifest.model_id}/{manifest.version}/bundle.tar.gz"
            bundle_blob = bucket.blob(gcs_bundle_path)
            bundle_blob.upload_from_filename(str(bundle_path))

            # Make public (or use signed URLs)
            # For production, use signed URLs instead
            try:
                bundle_blob.make_public()
            except Exception:
                pass  # May not have permission

            # Manifest path: models/{model_id}/{version}/manifest.json
            gcs_manifest_path = f"{self.config.models_prefix}/{manifest.model_id}/{manifest.version}/manifest.json"
            manifest_blob = bucket.blob(gcs_manifest_path)
            manifest_blob.upload_from_string(
                json.dumps(manifest.to_dict(), indent=2),
                content_type="application/json"
            )

            return {
                "success": True,
                "bundle_url": f"gs://{self.config.bucket_name}/{gcs_bundle_path}",
                "download_url": f"https://storage.googleapis.com/{self.config.bucket_name}/{gcs_bundle_path}",
                "manifest_url": f"gs://{self.config.bucket_name}/{gcs_manifest_path}",
            }

        return await loop.run_in_executor(None, _upload)

    async def _update_registry(self, manifest: ModelManifest) -> None:
        """Update model registry with new version."""
        if not self._firestore_client:
            return

        try:
            # Update Firestore registry
            doc_ref = self._firestore_client.collection(self.config.registry_collection).document(manifest.model_id)

            # Get existing document
            doc = doc_ref.get()
            existing = doc.to_dict() if doc.exists else {}

            # Update versions
            versions = existing.get("versions", {})
            versions[manifest.version] = {
                "download_url": manifest.download_url,
                "checksum": manifest.checksum_sha256,
                "signature": manifest.signature,
                "size_bytes": manifest.file_size_bytes,
                "release_notes": manifest.release_notes,
                "min_brain_version": manifest.min_brain_version,
                "created_at": manifest.created_at.isoformat(),
            }

            # Update document
            doc_ref.set({
                "model_id": manifest.model_id,
                "model_type": manifest.model_type.value,
                "name": manifest.name,
                "description": manifest.description,
                "latest_version": manifest.version,
                "versions": versions,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }, merge=True)

            # Also update the public registry.json file
            await self._update_public_registry()

            logger.info(f"Registry updated for {manifest.model_id} v{manifest.version}")

        except Exception as e:
            logger.error(f"Failed to update registry: {e}")

    async def _update_public_registry(self) -> None:
        """Update public registry.json file in GCS."""
        loop = asyncio.get_event_loop()

        def _update():
            # Fetch all models from Firestore
            models = {}
            for doc in self._firestore_client.collection(self.config.registry_collection).stream():
                data = doc.to_dict()
                models[doc.id] = {
                    "model_id": data.get("model_id"),
                    "name": data.get("name"),
                    "latest_version": data.get("latest_version"),
                    "versions": data.get("versions", {}),
                }

            # Create registry file
            registry = {
                "models": models,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Upload to GCS
            bucket = self._storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(self.config.registry_file)
            blob.upload_from_string(
                json.dumps(registry, indent=2),
                content_type="application/json"
            )

            try:
                blob.make_public()
            except Exception:
                pass

        await loop.run_in_executor(None, _update)

    # ========== Robot Registration ==========

    async def register_robot(
        self,
        device_id: str,
        robot_model: str,
        owner_uid: str,
        hardware_profile: str = "pi5",
        name: Optional[str] = None,
        auto_update: bool = True,
    ) -> Dict[str, Any]:
        """
        Register a new robot for model distribution.

        Args:
            device_id: Unique device identifier
            robot_model: Model of robot (e.g., "donkeycar")
            owner_uid: Owner's user ID
            hardware_profile: Hardware profile
            name: Human-readable name
            auto_update: Enable automatic updates

        Returns:
            Registration result with robot info
        """
        if not self._init_clients():
            return {"success": False, "error": "GCP clients not initialized"}

        try:
            # Check if already registered
            doc_ref = self._firestore_client.collection(self.config.robots_collection).document(device_id)
            doc = doc_ref.get()

            if doc.exists:
                return {
                    "success": False,
                    "error": "Robot already registered",
                    "existing": doc.to_dict(),
                }

            # Create robot record
            robot = RegisteredRobot(
                device_id=device_id,
                robot_model=robot_model,
                owner_uid=owner_uid,
                status=RobotStatus.ACTIVE if not self.config.require_robot_approval else RobotStatus.PENDING,
                registered_at=datetime.now(timezone.utc),
                hardware_profile=hardware_profile,
                name=name,
                auto_update=auto_update,
            )

            # Save to Firestore
            doc_ref.set(robot.to_dict())

            logger.info(f"Robot registered: {device_id} (owner: {owner_uid})")

            return {
                "success": True,
                "device_id": device_id,
                "status": robot.status.value,
                "robot": robot.to_dict(),
            }

        except Exception as e:
            logger.error(f"Robot registration failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_robot(self, device_id: str) -> Optional[RegisteredRobot]:
        """Get robot information."""
        if not self._init_clients():
            return None

        try:
            doc_ref = self._firestore_client.collection(self.config.robots_collection).document(device_id)
            doc = doc_ref.get()

            if not doc.exists:
                return None

            return RegisteredRobot.from_dict(doc.to_dict())

        except Exception as e:
            logger.error(f"Failed to get robot: {e}")
            return None

    async def update_robot_status(
        self,
        device_id: str,
        status: Optional[RobotStatus] = None,
        brain_version: Optional[str] = None,
        current_model_version: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Update robot status (called by robot on heartbeat)."""
        if not self._init_clients():
            return False

        try:
            doc_ref = self._firestore_client.collection(self.config.robots_collection).document(device_id)

            updates = {
                "last_seen": firestore.SERVER_TIMESTAMP,
            }

            if status:
                updates["status"] = status.value
            if brain_version:
                updates["brain_version"] = brain_version
            if current_model_version:
                updates["current_model_version"] = current_model_version
            if ip_address:
                updates["ip_address"] = ip_address

            doc_ref.update(updates)
            return True

        except Exception as e:
            logger.error(f"Failed to update robot status: {e}")
            return False

    async def list_robots(
        self,
        owner_uid: Optional[str] = None,
        status: Optional[RobotStatus] = None,
        limit: int = 100,
    ) -> List[RegisteredRobot]:
        """List registered robots."""
        if not self._init_clients():
            return []

        try:
            query = self._firestore_client.collection(self.config.robots_collection)

            if owner_uid:
                query = query.where("owner_uid", "==", owner_uid)
            if status:
                query = query.where("status", "==", status.value)

            query = query.limit(limit)

            robots = []
            for doc in query.stream():
                robots.append(RegisteredRobot.from_dict(doc.to_dict()))

            return robots

        except Exception as e:
            logger.error(f"Failed to list robots: {e}")
            return []

    # ========== Update Distribution ==========

    async def distribute_update(
        self,
        model_id: str,
        version: str,
        target_robots: Optional[List[str]] = None,
        priority: UpdatePriority = UpdatePriority.NORMAL,
        rollout_percentage: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Distribute a model update to robots.

        Args:
            model_id: Model to distribute
            version: Version to distribute
            target_robots: List of device_ids (None = all eligible)
            priority: Update priority
            rollout_percentage: Percentage of robots to update

        Returns:
            Distribution result
        """
        if not self._init_clients():
            return {"success": False, "error": "GCP clients not initialized"}

        try:
            # Get model info
            model_doc = self._firestore_client.collection(self.config.registry_collection).document(model_id).get()

            if not model_doc.exists:
                return {"success": False, "error": f"Model not found: {model_id}"}

            model_data = model_doc.to_dict()
            version_info = model_data.get("versions", {}).get(version)

            if not version_info:
                return {"success": False, "error": f"Version not found: {version}"}

            # Determine target robots
            if target_robots is None:
                # Get all active robots with auto_update enabled
                robots = await self.list_robots(status=RobotStatus.ACTIVE)
                target_robots = [
                    r.device_id for r in robots
                    if r.auto_update and r.current_model_version != version
                ]

            # Apply rollout percentage
            import random
            if rollout_percentage < 100.0:
                num_to_update = int(len(target_robots) * (rollout_percentage / 100.0))
                target_robots = random.sample(target_robots, num_to_update)

            if not target_robots:
                return {
                    "success": True,
                    "message": "No robots need updating",
                    "robots_targeted": 0,
                }

            # Create update notification
            update_id = f"update_{uuid.uuid4().hex[:12]}"
            notification = UpdateNotification(
                update_id=update_id,
                model_id=model_id,
                version=version,
                target_robots=target_robots,
                priority=priority,
                release_notes=version_info.get("release_notes", ""),
                download_url=version_info.get("download_url", ""),
                file_size_bytes=version_info.get("size_bytes", 0),
                checksum=version_info.get("checksum", ""),
                rollout_percentage=rollout_percentage,
                created_at=datetime.now(timezone.utc),
            )

            # Save notification
            self._firestore_client.collection(self.config.updates_collection).document(update_id).set(
                notification.to_dict()
            )

            # Notify robots (write to their update queue)
            for device_id in target_robots:
                self._firestore_client.collection(self.config.robots_collection).document(device_id).collection("pending_updates").document(update_id).set({
                    "update_id": update_id,
                    "model_id": model_id,
                    "version": version,
                    "priority": priority.value,
                    "download_url": notification.download_url,
                    "checksum": notification.checksum,
                    "notified_at": firestore.SERVER_TIMESTAMP,
                })

            logger.info(f"Update {update_id} distributed to {len(target_robots)} robots")

            return {
                "success": True,
                "update_id": update_id,
                "model_id": model_id,
                "version": version,
                "robots_targeted": len(target_robots),
                "target_robots": target_robots,
            }

        except Exception as e:
            logger.error(f"Update distribution failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_pending_updates(self, device_id: str) -> List[Dict[str, Any]]:
        """Get pending updates for a robot (called by robot)."""
        if not self._init_clients():
            return []

        try:
            updates = []
            docs = self._firestore_client.collection(
                self.config.robots_collection
            ).document(device_id).collection("pending_updates").stream()

            for doc in docs:
                updates.append(doc.to_dict())

            return updates

        except Exception as e:
            logger.error(f"Failed to get pending updates: {e}")
            return []

    async def acknowledge_update(
        self,
        device_id: str,
        update_id: str,
        action: str,  # "downloaded", "installed", "rejected"
    ) -> bool:
        """Acknowledge an update (called by robot)."""
        if not self._init_clients():
            return False

        try:
            # Update notification tracking
            update_ref = self._firestore_client.collection(self.config.updates_collection).document(update_id)
            update_doc = update_ref.get()

            if update_doc.exists:
                update_data = update_doc.to_dict()

                if action == "downloaded":
                    if device_id not in update_data.get("robots_downloaded", []):
                        update_ref.update({
                            "robots_downloaded": firestore.ArrayUnion([device_id])
                        })
                elif action == "installed":
                    if device_id not in update_data.get("robots_installed", []):
                        update_ref.update({
                            "robots_installed": firestore.ArrayUnion([device_id])
                        })

            # Remove from pending updates
            self._firestore_client.collection(
                self.config.robots_collection
            ).document(device_id).collection("pending_updates").document(update_id).delete()

            # Update robot's current version if installed
            if action == "installed":
                update_data = update_doc.to_dict()
                await self.update_robot_status(
                    device_id=device_id,
                    current_model_version=update_data.get("version"),
                )

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge update: {e}")
            return False

    # ========== Registry Access ==========

    async def get_latest_version(self, model_id: str = "seed") -> Optional[str]:
        """Get latest version of a model."""
        if not self._init_clients():
            return None

        try:
            doc = self._firestore_client.collection(self.config.registry_collection).document(model_id).get()

            if not doc.exists:
                return None

            return doc.to_dict().get("latest_version")

        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None

    async def get_model_info(self, model_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model information from registry."""
        if not self._init_clients():
            return None

        try:
            doc = self._firestore_client.collection(self.config.registry_collection).document(model_id).get()

            if not doc.exists:
                return None

            data = doc.to_dict()

            if version is None:
                version = data.get("latest_version")

            version_info = data.get("versions", {}).get(version)

            if not version_info:
                return None

            return {
                "model_id": model_id,
                "version": version,
                **version_info,
            }

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "available": self.is_available,
            "initialized": self._initialized,
            "gcs_available": GCS_AVAILABLE,
            "firestore_available": FIRESTORE_AVAILABLE,
            "project_id": self.config.project_id,
            "bucket_name": self.config.bucket_name,
        }


# Convenience function
async def upload_seed_model(
    model_path: Path,
    version: str,
    release_notes: str = "",
) -> Dict[str, Any]:
    """
    Quick upload of a seed model.

    Args:
        model_path: Path to model directory
        version: Version string
        release_notes: Release notes

    Returns:
        Upload result
    """
    service = ModelDistributionService()
    return await service.upload_model(
        model_path=model_path,
        version=version,
        model_id="seed",
        model_type=ModelType.SEED,
        release_notes=release_notes,
        name=f"Seed Model v{version}",
        description="Universal seed model for ContinuonBrain robots",
    )
