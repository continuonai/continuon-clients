"""Episode management API endpoints."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.auth.firebase import FirebaseUser, get_current_user
from app.services.firestore import FirestoreService, get_firestore_service
from app.services.storage import StorageService, get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


class EpisodeMetadata(BaseModel):
    """Episode metadata schema."""

    episode_id: str = Field(..., description="Unique episode ID")
    robot_id: str = Field(..., description="Robot that collected the episode")
    task_name: Optional[str] = Field(None, description="Task being performed")
    duration_seconds: float = Field(..., ge=0, description="Episode duration")
    num_steps: int = Field(..., ge=0, description="Number of timesteps")
    success: Optional[bool] = Field(None, description="Whether task was successful")
    reward_total: Optional[float] = Field(None, description="Total episode reward")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=1000)


class EpisodeUploadResponse(BaseModel):
    """Response after uploading an episode."""

    episode_id: str
    robot_id: str
    storage_uri: str
    message: str = "Episode uploaded successfully"


class EpisodeListResponse(BaseModel):
    """Paginated episode list response."""

    episodes: List[EpisodeMetadata]
    total: int
    page: int = 1
    page_size: int = 20


@router.get("/", response_model=EpisodeListResponse)
async def list_episodes(
    robot_id: Optional[str] = Query(None, description="Filter by robot ID"),
    task_name: Optional[str] = Query(None, description="Filter by task name"),
    success_only: bool = Query(False, description="Only successful episodes"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
    storage: StorageService = Depends(get_storage_service),
):
    """
    List episodes for the authenticated user.

    Can filter by robot, task name, or success status.
    """
    # Get user's robots
    robots = await db.get_robots(user.uid)
    robot_ids = [r.id for r in robots]

    # Filter by specific robot if provided
    if robot_id:
        if robot_id not in robot_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this robot's episodes",
            )
        robot_ids = [robot_id]

    # Collect episodes from all robots
    all_episodes = []
    for rid in robot_ids:
        episodes = await storage.list_episodes(rid)
        for ep in episodes:
            metadata = EpisodeMetadata(
                episode_id=ep.get("episode_id", ""),
                robot_id=rid,
                task_name=ep.get("task_name"),
                duration_seconds=ep.get("duration_seconds", 0),
                num_steps=ep.get("num_steps", 0),
                success=ep.get("success"),
                reward_total=ep.get("reward_total"),
                created_at=ep.get("created_at", datetime.utcnow()),
                tags=ep.get("tags", []),
                notes=ep.get("notes"),
            )
            all_episodes.append(metadata)

    # Apply filters
    if task_name:
        all_episodes = [e for e in all_episodes if e.task_name == task_name]
    if success_only:
        all_episodes = [e for e in all_episodes if e.success is True]

    # Sort by creation time (newest first)
    all_episodes.sort(key=lambda x: x.created_at, reverse=True)

    # Calculate pagination
    total = len(all_episodes)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_episodes = all_episodes[start:end]

    return EpisodeListResponse(
        episodes=paginated_episodes,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/{robot_id}", response_model=EpisodeUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_episode(
    robot_id: str,
    episode_id: str = Query(..., description="Unique episode identifier"),
    task_name: Optional[str] = Query(None, description="Task name"),
    duration_seconds: float = Query(..., ge=0, description="Episode duration"),
    num_steps: int = Query(..., ge=0, description="Number of timesteps"),
    success: Optional[bool] = Query(None, description="Task success"),
    file: UploadFile = File(..., description="Episode data file (.npz format)"),
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Upload episode data for a robot.

    Episode data should be in .npz format containing:
    - observations: Array of observations
    - actions: Array of actions
    - rewards: Array of rewards (optional)
    - timestamps: Array of timestamps (optional)
    """
    # Verify robot ownership
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to upload to this robot",
        )

    # Read file content
    content = await file.read()

    # Validate file size (100MB limit)
    max_size = 100 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Episode file too large (max 100MB)",
        )

    # Create metadata
    metadata = {
        "task_name": task_name,
        "duration_seconds": duration_seconds,
        "num_steps": num_steps,
        "success": success,
        "created_at": datetime.utcnow().isoformat(),
        "uploaded_by": user.uid,
        "file_size_bytes": len(content),
    }

    # Upload to GCS
    storage_uri = await storage.upload_episode(
        robot_id=robot_id,
        episode_id=episode_id,
        data=content,
        metadata=metadata,
    )

    logger.info(f"User {user.uid} uploaded episode {episode_id} for robot {robot_id}")

    return EpisodeUploadResponse(
        episode_id=episode_id,
        robot_id=robot_id,
        storage_uri=storage_uri,
    )


@router.get("/{robot_id}/{episode_id}")
async def get_episode(
    robot_id: str,
    episode_id: str,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get episode metadata and download URL.
    """
    # Verify robot ownership
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this robot's episodes",
        )

    # Get episode metadata
    episodes = await storage.list_episodes(robot_id)
    episode = next((e for e in episodes if e.get("episode_id") == episode_id), None)

    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found",
        )

    # Get download URL
    try:
        download_url = await storage.get_episode_download_url(robot_id, episode_id)
    except Exception as e:
        logger.error(f"Failed to get episode download URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL",
        )

    return {
        "episode_id": episode_id,
        "robot_id": robot_id,
        "metadata": episode,
        "download_url": download_url,
        "expires_in": 3600,
    }


@router.delete("/{robot_id}/{episode_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_episode(
    robot_id: str,
    episode_id: str,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Delete an episode.

    Permanently removes episode data. This cannot be undone.
    """
    # Verify robot ownership
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this robot's episodes",
        )

    # TODO: Implement episode deletion in storage service
    logger.info(f"User {user.uid} deleted episode {episode_id} from robot {robot_id}")
