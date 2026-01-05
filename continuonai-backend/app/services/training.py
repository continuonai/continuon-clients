"""Training orchestration service for ContinuonAI API."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Depends

from app.config import Settings, get_settings
from app.models.training import (
    TrainingConfig,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from app.services.firestore import FirestoreService, get_firestore_service
from app.services.storage import StorageService, get_storage_service

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing training jobs and orchestrating Vertex AI."""

    def __init__(
        self,
        settings: Settings,
        firestore: FirestoreService,
        storage: StorageService,
    ):
        self.settings = settings
        self.firestore = firestore
        self.storage = storage
        self._vertex_client = None

    def _get_vertex_client(self):
        """Lazy initialization of Vertex AI client."""
        if self._vertex_client is None:
            try:
                from google.cloud import aiplatform

                aiplatform.init(
                    project=self.settings.google_cloud_project,
                    location=self.settings.vertex_ai_location,
                )
                self._vertex_client = aiplatform
                logger.info("Vertex AI client initialized")
            except ImportError:
                logger.warning("google-cloud-aiplatform not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {e}")

        return self._vertex_client

    async def create_job(self, user_id: str, config: TrainingConfig) -> TrainingJob:
        """Create a new training job."""
        # Validate user has quota
        # (In production, check user's training hours remaining)

        # Create job in Firestore
        job = await self.firestore.create_training_job(user_id, config)

        logger.info(f"Created training job {job.id} for user {user_id}")
        return job

    async def run_job(self, job_id: str) -> None:
        """
        Run a training job.

        This method is called as a background task.
        In production, this would submit a job to Vertex AI.
        """
        try:
            # Update status to running
            await self.firestore.update_training_job(
                job_id,
                {
                    "status": TrainingStatus.RUNNING.value,
                    "started_at": datetime.utcnow(),
                },
            )

            # Get job details
            job = await self.firestore.get_training_job(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return

            # In production, submit to Vertex AI
            vertex_job_id = await self._submit_to_vertex_ai(job)

            if vertex_job_id:
                await self.firestore.update_training_job(
                    job_id,
                    {"vertex_ai_job_id": vertex_job_id},
                )

            logger.info(f"Training job {job_id} started")

        except Exception as e:
            logger.error(f"Failed to run job {job_id}: {e}")
            await self.firestore.update_training_job(
                job_id,
                {
                    "status": TrainingStatus.FAILED.value,
                    "error_message": str(e),
                    "completed_at": datetime.utcnow(),
                },
            )

    async def _submit_to_vertex_ai(self, job: TrainingJob) -> Optional[str]:
        """Submit training job to Vertex AI."""
        vertex = self._get_vertex_client()

        if vertex is None:
            logger.warning("Vertex AI not available, running mock training")
            # For development/testing, simulate training
            await self._simulate_training(job)
            return None

        try:
            # Prepare training script location
            # In production, this would point to your training container
            training_image = f"gcr.io/{self.settings.google_cloud_project}/continuon-training:latest"

            # Prepare accelerator config
            accelerator_config = None
            if job.config.accelerator_count > 0:
                accelerator_config = {
                    "accelerator_type": job.config.accelerator_type.value,
                    "accelerator_count": job.config.accelerator_count,
                }

            # Create custom training job
            custom_job = vertex.CustomJob(
                display_name=f"continuon-{job.id}",
                worker_pool_specs=[
                    {
                        "machine_spec": {
                            "machine_type": job.config.machine_type,
                            **({"accelerator_type": accelerator_config["accelerator_type"],
                                "accelerator_count": accelerator_config["accelerator_count"]}
                               if accelerator_config else {}),
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": training_image,
                            "args": [
                                f"--job-id={job.id}",
                                f"--config={job.config.model_dump_json()}",
                            ],
                        },
                    }
                ],
            )

            # Submit job (non-blocking)
            custom_job.submit()

            return custom_job.resource_name

        except Exception as e:
            logger.error(f"Failed to submit to Vertex AI: {e}")
            raise

    async def _simulate_training(self, job: TrainingJob) -> None:
        """Simulate training for development/testing."""
        import asyncio

        total_epochs = job.config.hyperparameters.epochs

        for epoch in range(total_epochs):
            # Simulate epoch time
            await asyncio.sleep(0.5)

            # Update progress
            progress = ((epoch + 1) / total_epochs) * 100
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                step=(epoch + 1) * 100,
                train_loss=1.0 / (epoch + 1),
                val_loss=1.1 / (epoch + 1),
                learning_rate=job.config.hyperparameters.learning_rate,
            )

            await self.firestore.update_training_job(
                job.id,
                {
                    "current_epoch": epoch + 1,
                    "progress_percent": progress,
                    "best_metrics": metrics.model_dump(),
                },
            )

        # Mark as completed
        await self.firestore.update_training_job(
            job.id,
            {
                "status": TrainingStatus.COMPLETED.value,
                "completed_at": datetime.utcnow(),
                "final_metrics": metrics.model_dump(),
            },
        )

        logger.info(f"Training job {job.id} completed (simulated)")

    async def list_jobs(
        self, user_id: str, status: Optional[TrainingStatus] = None
    ) -> List[TrainingJob]:
        """List training jobs for a user."""
        return await self.firestore.list_training_jobs(user_id, status)

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID."""
        return await self.firestore.get_training_job(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        job = await self.firestore.get_training_job(job_id)

        if not job:
            return False

        if job.status not in [TrainingStatus.PENDING, TrainingStatus.QUEUED, TrainingStatus.RUNNING]:
            logger.warning(f"Cannot cancel job {job_id} with status {job.status}")
            return False

        # Cancel Vertex AI job if running
        if job.vertex_ai_job_id:
            try:
                vertex = self._get_vertex_client()
                if vertex:
                    custom_job = vertex.CustomJob(job.vertex_ai_job_id)
                    custom_job.cancel()
            except Exception as e:
                logger.error(f"Failed to cancel Vertex AI job: {e}")

        # Update status
        await self.firestore.update_training_job(
            job_id,
            {
                "status": TrainingStatus.CANCELLED.value,
                "completed_at": datetime.utcnow(),
            },
        )

        logger.info(f"Cancelled training job {job_id}")
        return True

    async def get_job_logs(self, job_id: str) -> Optional[str]:
        """Get logs for a training job."""
        job = await self.firestore.get_training_job(job_id)

        if not job or not job.logs_uri:
            return None

        # In production, fetch logs from Cloud Logging or GCS
        return f"Logs available at: {job.logs_uri}"


async def get_training_service(
    settings: Settings = Depends(get_settings),
    firestore: FirestoreService = Depends(get_firestore_service),
    storage: StorageService = Depends(get_storage_service),
) -> TrainingService:
    """Dependency to get TrainingService instance."""
    return TrainingService(settings, firestore, storage)
