"""Training job management API endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from app.auth.firebase import FirebaseUser, get_current_user
from app.models.training import (
    TrainingConfig,
    TrainingJob,
    TrainingJobCreate,
    TrainingJobListResponse,
    TrainingStatus,
)
from app.services.training import TrainingService, get_training_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/jobs", response_model=TrainingJob, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    user: FirebaseUser = Depends(get_current_user),
    training: TrainingService = Depends(get_training_service),
):
    """
    Create a new training job.

    Submits a training job to be executed. The job runs asynchronously
    and progress can be monitored via the job status endpoint.

    Training types:
    - fine_tune: Fine-tune an existing model
    - imitation: Train from demonstration episodes
    - reinforcement: Reinforcement learning
    - diffusion: Diffusion policy training
    - evaluation: Evaluate model performance
    """
    # Validate configuration
    config = job_data.config

    # Check if base model exists for fine-tuning
    if config.training_type.value == "fine_tune" and not config.base_model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="base_model_id required for fine-tuning",
        )

    # Check if robot_ids or episode_ids provided for data
    if not config.robot_ids and not config.episode_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify robot_ids or episode_ids for training data",
        )

    # Create the job
    job = await training.create_job(user.uid, config)

    # Start training in background
    background_tasks.add_task(training.run_job, job.id)

    logger.info(f"User {user.uid} created training job {job.id}")
    return job


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    status_filter: Optional[TrainingStatus] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: FirebaseUser = Depends(get_current_user),
    training: TrainingService = Depends(get_training_service),
):
    """
    List user's training jobs.

    Returns paginated list with optional status filtering.
    """
    jobs = await training.list_jobs(user.uid, status_filter)

    # Calculate pagination
    total = len(jobs)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_jobs = jobs[start:end]

    return TrainingJobListResponse(
        jobs=paginated_jobs,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(
    job_id: str,
    user: FirebaseUser = Depends(get_current_user),
    training: TrainingService = Depends(get_training_service),
):
    """
    Get training job details.

    Returns current status, progress, and metrics.
    """
    job = await training.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    if job.user_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this training job",
        )

    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: str,
    user: FirebaseUser = Depends(get_current_user),
    training: TrainingService = Depends(get_training_service),
):
    """
    Cancel a running or pending training job.

    Jobs that have already completed or failed cannot be cancelled.
    """
    job = await training.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    if job.user_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this training job",
        )

    if job.status not in [TrainingStatus.PENDING, TrainingStatus.QUEUED, TrainingStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status {job.status.value}",
        )

    success = await training.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel training job",
        )

    logger.info(f"User {user.uid} cancelled training job {job_id}")
    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/logs")
async def get_training_logs(
    job_id: str,
    user: FirebaseUser = Depends(get_current_user),
    training: TrainingService = Depends(get_training_service),
):
    """
    Get training job logs.

    Returns logs from the training process.
    Only available for running or completed jobs.
    """
    job = await training.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    if job.user_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this training job",
        )

    logs = await training.get_job_logs(job_id)

    if logs is None:
        return {"logs": None, "message": "Logs not yet available"}

    return {"logs": logs, "job_id": job_id}


@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(
    job_id: str,
    user: FirebaseUser = Depends(get_current_user),
    training: TrainingService = Depends(get_training_service),
):
    """
    Get detailed training metrics.

    Returns training and validation metrics over time.
    """
    job = await training.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    if job.user_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this training job",
        )

    return {
        "job_id": job_id,
        "status": job.status.value,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "progress_percent": job.progress_percent,
        "best_metrics": job.best_metrics.model_dump() if job.best_metrics else None,
        "final_metrics": job.final_metrics.model_dump() if job.final_metrics else None,
    }
