"""
CloudTrainingService - Manages cloud training jobs for ContinuonBrain.

This service handles:
- Uploading RLDS episodes to Google Cloud Storage
- Triggering cloud training jobs via Cloud Functions or Vertex AI
- Tracking job status via Firestore
- Downloading trained model results
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
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


class JobStatus(str, Enum):
    """Training job status values."""
    PENDING = "pending"
    UPLOADING = "uploading"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CloudTrainingConfig:
    """Configuration for cloud training."""
    project_id: str = "continuon-xr"
    bucket_name: str = "continuon-training-data"
    region: str = "us-central1"
    episodes_prefix: str = "episodes"
    models_prefix: str = "models"
    service_account_path: Optional[str] = None
    robot_id: str = "robot_001"

    # Training defaults
    default_epochs: int = 100
    default_batch_size: int = 32
    default_learning_rate: float = 1e-4
    model_type: str = "wavecore"

    # Vertex AI settings (optional)
    use_vertex_ai: bool = False
    vertex_machine_type: str = "n1-standard-8"
    vertex_accelerator_type: str = "NVIDIA_TESLA_T4"
    vertex_accelerator_count: int = 1
    trainer_image_uri: str = "gcr.io/continuon/trainer:latest"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudTrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: Path) -> "CloudTrainingConfig":
        """Load config from YAML or JSON file."""
        if not path.exists():
            return cls()

        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                logger.warning("PyYAML not installed, cannot load YAML config")
                return cls()
        else:
            data = json.loads(content)

        return cls.from_dict(data or {})


@dataclass
class TrainingJobConfig:
    """Configuration for a specific training job."""
    model_type: str = "wavecore"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Model architecture
    obs_dim: int = 128
    action_dim: int = 32
    output_dim: int = 32
    arch_preset: str = "cloud"

    # Training options
    use_tpu: bool = False
    use_gpu: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    # Data options
    shuffle_buffer_size: int = 10000
    prefetch_buffer_size: int = 4

    # Optional extras
    sparsity_lambda: float = 0.0
    distillation_alpha: float = 0.0
    teacher_model: Optional[str] = None


@dataclass
class TrainingJobResult:
    """Result of a training job."""
    job_id: str
    status: JobStatus
    model_uri: Optional[str] = None
    final_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    epochs_completed: int = 0
    best_checkpoint_step: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class CloudTrainingService:
    """
    Manages cloud training jobs for ContinuonBrain.

    This service provides:
    - Episode upload to Google Cloud Storage
    - Training job triggering via Cloud Functions
    - Job status tracking via Firestore
    - Trained model download and installation
    """

    def __init__(
        self,
        config: Optional[CloudTrainingConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize CloudTrainingService.

        Args:
            config: CloudTrainingConfig instance
            config_path: Path to config file (YAML or JSON)
        """
        if config:
            self.config = config
        elif config_path:
            self.config = CloudTrainingConfig.from_file(config_path)
        else:
            self.config = CloudTrainingConfig()

        self._storage_client: Optional[Any] = None
        self._firestore_client: Optional[Any] = None
        self._initialized = False

        # Local paths
        self.local_episodes_dir = Path("/opt/continuonos/brain/rlds/episodes")
        self.local_models_dir = Path("/opt/continuonos/brain/model/adapters")

    def _init_clients(self) -> bool:
        """Initialize GCS and Firestore clients."""
        if self._initialized:
            return True

        if not GCS_AVAILABLE:
            logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            return False

        if not FIRESTORE_AVAILABLE:
            logger.error("google-cloud-firestore not installed. Install with: pip install google-cloud-firestore")
            return False

        try:
            # Set credentials if specified
            if self.config.service_account_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config.service_account_path

            self._storage_client = storage.Client(project=self.config.project_id)
            self._firestore_client = firestore.Client(project=self.config.project_id)
            self._initialized = True
            logger.info(f"CloudTrainingService initialized for project: {self.config.project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize cloud clients: {e}")
            return False

    @property
    def storage_client(self) -> Any:
        """Get storage client, initializing if needed."""
        if not self._initialized:
            self._init_clients()
        return self._storage_client

    @property
    def firestore_client(self) -> Any:
        """Get Firestore client, initializing if needed."""
        if not self._initialized:
            self._init_clients()
        return self._firestore_client

    async def upload_episodes(
        self,
        local_dir: Optional[Path] = None,
        episode_ids: Optional[List[str]] = None,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload RLDS episodes to Google Cloud Storage.

        Args:
            local_dir: Local directory containing episodes (default: /opt/continuonos/brain/rlds/episodes)
            episode_ids: Optional list of specific episode IDs to upload
            compress: Whether to compress episodes into a zip file

        Returns:
            Dict with upload result including GCS URI
        """
        if not self._init_clients():
            return {"success": False, "error": "Cloud clients not initialized"}

        local_dir = local_dir or self.local_episodes_dir

        if not local_dir.exists():
            return {"success": False, "error": f"Episodes directory not found: {local_dir}"}

        # Find episodes
        episode_files = list(local_dir.glob("**/*.json"))
        if not episode_files:
            return {"success": False, "error": "No episode files found"}

        # Filter by IDs if specified
        if episode_ids:
            episode_files = [f for f in episode_files if any(eid in f.stem for eid in episode_ids)]

        logger.info(f"Uploading {len(episode_files)} episode files from {local_dir}")

        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)

            # Create unique upload ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            upload_id = f"{self.config.robot_id}_{timestamp}"

            if compress:
                # Create zip archive
                gcs_path = await self._upload_compressed(
                    bucket, local_dir, episode_files, upload_id
                )
            else:
                # Upload files individually
                gcs_path = await self._upload_individual(
                    bucket, local_dir, episode_files, upload_id
                )

            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"

            result = {
                "success": True,
                "gcs_uri": gcs_uri,
                "upload_id": upload_id,
                "files_uploaded": len(episode_files),
                "compressed": compress,
            }

            logger.info(f"Episodes uploaded successfully: {gcs_uri}")
            return result

        except Exception as e:
            logger.error(f"Episode upload failed: {e}")
            return {"success": False, "error": str(e)}

    async def _upload_compressed(
        self,
        bucket: Any,
        local_dir: Path,
        episode_files: List[Path],
        upload_id: str,
    ) -> str:
        """Upload episodes as a compressed zip file."""
        loop = asyncio.get_event_loop()

        def _compress_and_upload():
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / f"{upload_id}.zip"

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for ep_file in episode_files:
                        arcname = ep_file.relative_to(local_dir)
                        zf.write(ep_file, arcname)

                gcs_path = f"{self.config.episodes_prefix}/{upload_id}.zip"
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(zip_path))

                return gcs_path

        return await loop.run_in_executor(None, _compress_and_upload)

    async def _upload_individual(
        self,
        bucket: Any,
        local_dir: Path,
        episode_files: List[Path],
        upload_id: str,
    ) -> str:
        """Upload episodes as individual files."""
        loop = asyncio.get_event_loop()

        def _upload_files():
            base_path = f"{self.config.episodes_prefix}/{upload_id}"

            for ep_file in episode_files:
                rel_path = ep_file.relative_to(local_dir)
                gcs_path = f"{base_path}/{rel_path}"
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(ep_file))

            return base_path

        return await loop.run_in_executor(None, _upload_files)

    async def trigger_training(
        self,
        episodes_uri: str,
        config: Optional[TrainingJobConfig] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger a cloud training job.

        Args:
            episodes_uri: GCS URI of uploaded episodes
            config: Training configuration
            job_id: Optional job ID (generated if not provided)

        Returns:
            Dict with job information including job_id
        """
        if not self._init_clients():
            return {"success": False, "error": "Cloud clients not initialized"}

        config = config or TrainingJobConfig()
        job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"

        # Create job document in Firestore
        job_doc = {
            "job_id": job_id,
            "robot_id": self.config.robot_id,
            "status": JobStatus.PENDING.value,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "episodes_uri": episodes_uri,
            "config": {
                "model_type": config.model_type,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "obs_dim": config.obs_dim,
                "action_dim": config.action_dim,
                "output_dim": config.output_dim,
                "arch_preset": config.arch_preset,
                "use_tpu": config.use_tpu,
                "use_gpu": config.use_gpu,
                "mixed_precision": config.mixed_precision,
                "sparsity_lambda": config.sparsity_lambda,
            },
            "result": None,
        }

        try:
            # Write job to Firestore
            doc_ref = self.firestore_client.collection("training_jobs").document(job_id)
            doc_ref.set(job_doc)

            logger.info(f"Training job created: {job_id}")

            # Trigger training via Cloud Function or Vertex AI
            if self.config.use_vertex_ai:
                trigger_result = await self._trigger_vertex_training(job_id, episodes_uri, config)
            else:
                trigger_result = await self._trigger_cloud_function(job_id, episodes_uri, config)

            if not trigger_result.get("success"):
                # Update job status to failed
                doc_ref.update({
                    "status": JobStatus.FAILED.value,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                    "result": {"error": trigger_result.get("error")},
                })
                return trigger_result

            # Update status to queued
            doc_ref.update({
                "status": JobStatus.QUEUED.value,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })

            return {
                "success": True,
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "episodes_uri": episodes_uri,
            }

        except Exception as e:
            logger.error(f"Failed to trigger training: {e}")
            return {"success": False, "error": str(e)}

    async def _trigger_cloud_function(
        self,
        job_id: str,
        episodes_uri: str,
        config: TrainingJobConfig,
    ) -> Dict[str, Any]:
        """Trigger training via Cloud Function."""
        try:
            # Write trigger document to Firestore (Cloud Function watches this)
            trigger_doc = {
                "job_id": job_id,
                "bucket": self.config.bucket_name,
                "name": episodes_uri.replace(f"gs://{self.config.bucket_name}/", ""),
                "config": asdict(config),
                "triggered_at": firestore.SERVER_TIMESTAMP,
            }

            trigger_ref = self.firestore_client.collection("training_triggers").document(job_id)
            trigger_ref.set(trigger_doc)

            logger.info(f"Cloud Function trigger created for job: {job_id}")
            return {"success": True}

        except Exception as e:
            logger.error(f"Cloud Function trigger failed: {e}")
            return {"success": False, "error": str(e)}

    async def _trigger_vertex_training(
        self,
        job_id: str,
        episodes_uri: str,
        config: TrainingJobConfig,
    ) -> Dict[str, Any]:
        """Trigger training via Vertex AI."""
        try:
            from google.cloud import aiplatform

            aiplatform.init(
                project=self.config.project_id,
                location=self.config.region,
            )

            # Prepare training args
            training_args = [
                "--job-id", job_id,
                "--episodes-uri", episodes_uri,
                "--config", json.dumps(asdict(config)),
            ]

            # Create custom job
            job = aiplatform.CustomJob(
                display_name=f"continuon-train-{job_id}",
                worker_pool_specs=[{
                    "machine_spec": {
                        "machine_type": self.config.vertex_machine_type,
                        "accelerator_type": self.config.vertex_accelerator_type,
                        "accelerator_count": self.config.vertex_accelerator_count,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": self.config.trainer_image_uri,
                        "args": training_args,
                    },
                }],
            )

            # Submit job (async)
            job.run(sync=False)

            # Update Firestore with Vertex job info
            doc_ref = self.firestore_client.collection("training_jobs").document(job_id)
            doc_ref.update({
                "vertex_job_name": job.resource_name,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })

            logger.info(f"Vertex AI job submitted: {job.resource_name}")
            return {"success": True, "vertex_job_name": job.resource_name}

        except ImportError:
            logger.error("google-cloud-aiplatform not installed")
            return {"success": False, "error": "Vertex AI SDK not installed"}
        except Exception as e:
            logger.error(f"Vertex AI trigger failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_job_status(self, job_id: str) -> TrainingJobResult:
        """
        Get the status of a training job.

        Args:
            job_id: The training job ID

        Returns:
            TrainingJobResult with current status and metrics
        """
        if not self._init_clients():
            return TrainingJobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message="Cloud clients not initialized",
            )

        try:
            doc_ref = self.firestore_client.collection("training_jobs").document(job_id)
            doc = doc_ref.get()

            if not doc.exists:
                return TrainingJobResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error_message=f"Job not found: {job_id}",
                )

            data = doc.to_dict()
            result_data = data.get("result") or {}

            return TrainingJobResult(
                job_id=job_id,
                status=JobStatus(data.get("status", "pending")),
                model_uri=result_data.get("model_uri"),
                final_loss=result_data.get("final_loss"),
                training_time_seconds=result_data.get("training_time_s"),
                epochs_completed=result_data.get("epochs_completed", 0),
                best_checkpoint_step=result_data.get("best_checkpoint_step"),
                metrics=result_data.get("metrics", {}),
                error_message=result_data.get("error"),
                created_at=str(data.get("created_at", "")),
                completed_at=str(data.get("completed_at", "")),
            )

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return TrainingJobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
            )

    async def list_jobs(
        self,
        limit: int = 20,
        status_filter: Optional[JobStatus] = None,
    ) -> List[TrainingJobResult]:
        """
        List training jobs for this robot.

        Args:
            limit: Maximum number of jobs to return
            status_filter: Optional status to filter by

        Returns:
            List of TrainingJobResult objects
        """
        if not self._init_clients():
            return []

        try:
            query = self.firestore_client.collection("training_jobs").where(
                "robot_id", "==", self.config.robot_id
            ).order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)

            if status_filter:
                query = query.where("status", "==", status_filter.value)

            jobs = []
            for doc in query.stream():
                data = doc.to_dict()
                result_data = data.get("result") or {}

                jobs.append(TrainingJobResult(
                    job_id=doc.id,
                    status=JobStatus(data.get("status", "pending")),
                    model_uri=result_data.get("model_uri"),
                    final_loss=result_data.get("final_loss"),
                    training_time_seconds=result_data.get("training_time_s"),
                    created_at=str(data.get("created_at", "")),
                    completed_at=str(data.get("completed_at", "")),
                ))

            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    async def download_result(
        self,
        job_id: str,
        dest_dir: Optional[Path] = None,
        install: bool = True,
    ) -> Dict[str, Any]:
        """
        Download trained model from a completed job.

        Args:
            job_id: The training job ID
            dest_dir: Destination directory for download
            install: Whether to install model to active adapters directory

        Returns:
            Dict with download result and local path
        """
        if not self._init_clients():
            return {"success": False, "error": "Cloud clients not initialized"}

        # Get job status
        job_result = await self.get_job_status(job_id)

        if job_result.status != JobStatus.COMPLETED:
            return {
                "success": False,
                "error": f"Job not completed. Current status: {job_result.status.value}",
            }

        if not job_result.model_uri:
            return {"success": False, "error": "No model URI in job result"}

        dest_dir = dest_dir or (self.local_models_dir / "cloud" / job_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)

            # Parse GCS path
            gcs_path = job_result.model_uri.replace(f"gs://{self.config.bucket_name}/", "")

            # Download all files in the model directory
            blobs = list(bucket.list_blobs(prefix=gcs_path))

            if not blobs:
                return {"success": False, "error": f"No files found at {job_result.model_uri}"}

            downloaded_files = []
            for blob in blobs:
                # Get relative path
                rel_path = blob.name.replace(gcs_path, "").lstrip("/")
                if not rel_path:
                    rel_path = Path(blob.name).name

                local_path = dest_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                blob.download_to_filename(str(local_path))
                downloaded_files.append(str(local_path))

            result = {
                "success": True,
                "job_id": job_id,
                "download_dir": str(dest_dir),
                "files_downloaded": len(downloaded_files),
                "model_uri": job_result.model_uri,
            }

            # Install to active adapters if requested
            if install:
                install_result = await self._install_model(dest_dir, job_id)
                result["installed"] = install_result.get("success", False)
                result["install_path"] = install_result.get("install_path")

            logger.info(f"Model downloaded successfully: {dest_dir}")
            return result

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return {"success": False, "error": str(e)}

    async def _install_model(self, source_dir: Path, job_id: str) -> Dict[str, Any]:
        """Install downloaded model to the active adapters directory."""
        try:
            # Target path for installed model
            install_path = self.local_models_dir / "candidate" / f"cloud_{job_id}"

            if install_path.exists():
                shutil.rmtree(install_path)

            shutil.copytree(source_dir, install_path)

            # Write installation manifest
            manifest = {
                "source": "cloud_training",
                "job_id": job_id,
                "installed_at": datetime.utcnow().isoformat(),
                "source_dir": str(source_dir),
            }

            manifest_path = install_path / "install_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            logger.info(f"Model installed to: {install_path}")
            return {"success": True, "install_path": str(install_path)}

        except Exception as e:
            logger.error(f"Model installation failed: {e}")
            return {"success": False, "error": str(e)}

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running or queued training job.

        Args:
            job_id: The training job ID to cancel

        Returns:
            Dict with cancellation result
        """
        if not self._init_clients():
            return {"success": False, "error": "Cloud clients not initialized"}

        try:
            doc_ref = self.firestore_client.collection("training_jobs").document(job_id)
            doc = doc_ref.get()

            if not doc.exists:
                return {"success": False, "error": f"Job not found: {job_id}"}

            data = doc.to_dict()
            current_status = JobStatus(data.get("status", "pending"))

            if current_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return {
                    "success": False,
                    "error": f"Cannot cancel job with status: {current_status.value}",
                }

            # Update status to cancelled
            doc_ref.update({
                "status": JobStatus.CANCELLED.value,
                "updated_at": firestore.SERVER_TIMESTAMP,
                "cancelled_at": firestore.SERVER_TIMESTAMP,
            })

            # If using Vertex AI, cancel the job there too
            if self.config.use_vertex_ai and data.get("vertex_job_name"):
                try:
                    from google.cloud import aiplatform

                    aiplatform.init(
                        project=self.config.project_id,
                        location=self.config.region,
                    )

                    job = aiplatform.CustomJob.get(data["vertex_job_name"])
                    job.cancel()

                except Exception as e:
                    logger.warning(f"Failed to cancel Vertex AI job: {e}")

            logger.info(f"Job cancelled: {job_id}")
            return {"success": True, "job_id": job_id, "status": JobStatus.CANCELLED.value}

        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return {"success": False, "error": str(e)}

    def is_available(self) -> bool:
        """Check if cloud training is available (dependencies installed)."""
        return GCS_AVAILABLE and FIRESTORE_AVAILABLE

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "available": self.is_available(),
            "initialized": self._initialized,
            "gcs_available": GCS_AVAILABLE,
            "firestore_available": FIRESTORE_AVAILABLE,
            "project_id": self.config.project_id,
            "bucket_name": self.config.bucket_name,
            "robot_id": self.config.robot_id,
            "use_vertex_ai": self.config.use_vertex_ai,
        }
