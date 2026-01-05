"""Analytics and usage statistics API endpoints."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.auth.firebase import FirebaseUser, get_current_user
from app.models.robot import RobotStatus
from app.services.firestore import FirestoreService, get_firestore_service
from app.services.storage import StorageService, get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


class FleetSummary(BaseModel):
    """Summary of user's robot fleet."""

    total_robots: int = Field(default=0, description="Total number of robots")
    online_count: int = Field(default=0, description="Number of online robots")
    offline_count: int = Field(default=0, description="Number of offline robots")
    busy_count: int = Field(default=0, description="Number of busy robots")
    error_count: int = Field(default=0, description="Number of robots with errors")


class UsageStats(BaseModel):
    """Usage statistics for the user."""

    total_episodes: int = Field(default=0, description="Total episodes collected")
    total_training_jobs: int = Field(default=0, description="Total training jobs")
    completed_training_jobs: int = Field(default=0, description="Completed training jobs")
    total_models: int = Field(default=0, description="Total models created")
    storage_used_gb: float = Field(default=0, description="Storage used in GB")
    training_hours_used: float = Field(default=0, description="Training hours consumed")


class ActivityEvent(BaseModel):
    """Recent activity event."""

    event_type: str = Field(..., description="Type of event")
    description: str = Field(..., description="Event description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    robot_id: Optional[str] = None
    model_id: Optional[str] = None
    job_id: Optional[str] = None


class DashboardData(BaseModel):
    """Complete dashboard data for user."""

    fleet_summary: FleetSummary
    usage_stats: UsageStats
    recent_activity: List[ActivityEvent]
    robot_health: Dict[str, str]


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard(
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get dashboard summary data.

    Returns fleet status, usage statistics, and recent activity.
    """
    # Get fleet summary
    robots = await db.get_robots(user.uid)

    fleet_summary = FleetSummary(
        total_robots=len(robots),
        online_count=sum(1 for r in robots if r.status == RobotStatus.ONLINE),
        offline_count=sum(1 for r in robots if r.status == RobotStatus.OFFLINE),
        busy_count=sum(1 for r in robots if r.status == RobotStatus.BUSY),
        error_count=sum(1 for r in robots if r.status == RobotStatus.ERROR),
    )

    # Get usage stats
    training_jobs = await db.list_training_jobs(user.uid)
    models = await storage.list_models(owner_id=user.uid)

    # Count episodes across all robots
    total_episodes = 0
    for robot in robots:
        episodes = await storage.list_episodes(robot.id)
        total_episodes += len(episodes)

    usage_stats = UsageStats(
        total_episodes=total_episodes,
        total_training_jobs=len(training_jobs),
        completed_training_jobs=sum(1 for j in training_jobs if j.status.value == "completed"),
        total_models=len(models),
        storage_used_gb=0,  # TODO: Calculate actual storage usage
        training_hours_used=0,  # TODO: Calculate actual training hours
    )

    # Build recent activity from various sources
    recent_activity = []

    # Add recent training job events
    for job in training_jobs[:5]:
        recent_activity.append(
            ActivityEvent(
                event_type="training",
                description=f"Training job '{job.config.name}' - {job.status.value}",
                timestamp=job.created_at,
                job_id=job.id,
            )
        )

    # Add robot status changes
    for robot in robots:
        if robot.last_seen:
            recent_activity.append(
                ActivityEvent(
                    event_type="robot_status",
                    description=f"Robot '{robot.name}' - {robot.status.value}",
                    timestamp=robot.last_seen,
                    robot_id=robot.id,
                )
            )

    # Sort by timestamp and limit
    recent_activity.sort(key=lambda x: x.timestamp, reverse=True)
    recent_activity = recent_activity[:10]

    # Robot health map
    robot_health = {r.id: r.status.value for r in robots}

    return DashboardData(
        fleet_summary=fleet_summary,
        usage_stats=usage_stats,
        recent_activity=recent_activity,
        robot_health=robot_health,
    )


@router.get("/fleet/summary", response_model=FleetSummary)
async def get_fleet_summary(
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Get fleet status summary.

    Quick overview of robot fleet health.
    """
    robots = await db.get_robots(user.uid)

    return FleetSummary(
        total_robots=len(robots),
        online_count=sum(1 for r in robots if r.status == RobotStatus.ONLINE),
        offline_count=sum(1 for r in robots if r.status == RobotStatus.OFFLINE),
        busy_count=sum(1 for r in robots if r.status == RobotStatus.BUSY),
        error_count=sum(1 for r in robots if r.status == RobotStatus.ERROR),
    )


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Get detailed usage statistics.

    Returns storage, training, and episode statistics.
    """
    robots = await db.get_robots(user.uid)
    training_jobs = await db.list_training_jobs(user.uid)
    models = await storage.list_models(owner_id=user.uid)

    # Count episodes
    total_episodes = 0
    for robot in robots:
        episodes = await storage.list_episodes(robot.id)
        total_episodes += len(episodes)

    return UsageStats(
        total_episodes=total_episodes,
        total_training_jobs=len(training_jobs),
        completed_training_jobs=sum(1 for j in training_jobs if j.status.value == "completed"),
        total_models=len(models),
        storage_used_gb=0,
        training_hours_used=0,
    )


@router.get("/training/history")
async def get_training_history(
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Get training job history over time.

    Returns daily counts of training jobs by status.
    """
    training_jobs = await db.list_training_jobs(user.uid)

    # Filter to requested time range
    cutoff = datetime.utcnow() - timedelta(days=days)
    recent_jobs = [j for j in training_jobs if j.created_at >= cutoff]

    # Group by date
    daily_counts: Dict[str, Dict[str, int]] = {}
    for job in recent_jobs:
        date_str = job.created_at.strftime("%Y-%m-%d")
        if date_str not in daily_counts:
            daily_counts[date_str] = {"total": 0, "completed": 0, "failed": 0, "cancelled": 0}

        daily_counts[date_str]["total"] += 1
        if job.status.value == "completed":
            daily_counts[date_str]["completed"] += 1
        elif job.status.value == "failed":
            daily_counts[date_str]["failed"] += 1
        elif job.status.value == "cancelled":
            daily_counts[date_str]["cancelled"] += 1

    return {
        "period_days": days,
        "total_jobs": len(recent_jobs),
        "daily_counts": daily_counts,
    }


@router.get("/robots/{robot_id}/telemetry-history")
async def get_robot_telemetry_history(
    robot_id: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of history"),
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Get historical telemetry for a robot.

    Returns battery, temperature, and status history.
    """
    robot = await db.get_robot(robot_id)

    if not robot or robot.owner_id != user.uid:
        return {"error": "Robot not found or not authorized"}

    # TODO: Query telemetry subcollection for historical data
    # For now, return current telemetry only

    return {
        "robot_id": robot_id,
        "period_hours": hours,
        "current_telemetry": robot.latest_telemetry.model_dump() if robot.latest_telemetry else None,
        "history": [],  # TODO: Implement telemetry history query
    }
