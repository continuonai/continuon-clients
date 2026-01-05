"""Firestore database service for ContinuonAI API."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends
from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentReference, DocumentSnapshot

from app.config import Settings, get_settings
from app.models.robot import (
    Robot,
    RobotCommand,
    RobotCreate,
    RobotStatus,
    RobotTelemetry,
    RobotUpdate,
)
from app.models.training import TrainingConfig, TrainingJob, TrainingStatus
from app.models.user import User, UserCreate, UserPlan

logger = logging.getLogger(__name__)

# Firestore client singleton
_firestore_client: Optional[firestore.Client] = None


def get_firestore_client(settings: Settings) -> firestore.Client:
    """Get or create Firestore client singleton."""
    global _firestore_client

    if _firestore_client is None:
        try:
            if settings.google_cloud_project:
                _firestore_client = firestore.Client(project=settings.google_cloud_project)
            else:
                _firestore_client = firestore.Client()
            logger.info("Firestore client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            raise

    return _firestore_client


class FirestoreService:
    """Service for Firestore database operations."""

    # Collection names
    USERS = "users"
    ROBOTS = "robots"
    COMMANDS = "commands"
    TELEMETRY = "telemetry"
    TRAINING_JOBS = "training_jobs"
    MODELS = "models"
    EPISODES = "episodes"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = get_firestore_client(settings)

    # ==================== User Operations ====================

    async def get_user(self, uid: str) -> Optional[User]:
        """Get user by Firebase UID."""
        doc = self.db.collection(self.USERS).document(uid).get()
        if not doc.exists:
            return None
        return User(uid=uid, **doc.to_dict())

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user document."""
        doc_ref = self.db.collection(self.USERS).document(user_data.uid)

        user = User(
            uid=user_data.uid,
            email=user_data.email,
            name=user_data.name,
            plan=UserPlan.FREE,
            robot_limit=1,
            training_hours_remaining=0,
            storage_gb_limit=5,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        doc_ref.set(user.model_dump(exclude={"uid"}))
        logger.info(f"Created user: {user.uid}")
        return user

    async def update_user(self, uid: str, updates: Dict[str, Any]) -> Optional[User]:
        """Update user document."""
        doc_ref = self.db.collection(self.USERS).document(uid)
        updates["updated_at"] = datetime.utcnow()
        doc_ref.update(updates)
        return await self.get_user(uid)

    async def get_or_create_user(self, uid: str, email: str, name: Optional[str] = None) -> User:
        """Get existing user or create new one."""
        user = await self.get_user(uid)
        if user:
            # Update last login
            await self.update_user(uid, {"last_login": datetime.utcnow()})
            return user

        return await self.create_user(UserCreate(uid=uid, email=email, name=name))

    # ==================== Robot Operations ====================

    async def get_robots(self, owner_id: str) -> List[Robot]:
        """Get all robots owned by a user."""
        docs = (
            self.db.collection(self.ROBOTS)
            .where("owner_id", "==", owner_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .stream()
        )

        robots = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            robots.append(Robot(**data))

        return robots

    async def get_robot(self, robot_id: str) -> Optional[Robot]:
        """Get a robot by ID."""
        doc = self.db.collection(self.ROBOTS).document(robot_id).get()
        if not doc.exists:
            return None

        data = doc.to_dict()
        data["id"] = doc.id
        return Robot(**data)

    async def create_robot(self, owner_id: str, robot_data: RobotCreate) -> Robot:
        """Create a new robot."""
        robot_id = str(uuid4())
        doc_ref = self.db.collection(self.ROBOTS).document(robot_id)

        robot = Robot(
            id=robot_id,
            owner_id=owner_id,
            name=robot_data.name,
            device_id=robot_data.device_id,
            description=robot_data.description,
            model_type=robot_data.model_type,
            tags=robot_data.tags,
            status=RobotStatus.OFFLINE,
            model_version=robot_data.initial_model_version,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        doc_ref.set(robot.model_dump(exclude={"id"}))
        logger.info(f"Created robot: {robot_id} for user: {owner_id}")
        return robot

    async def update_robot(self, robot_id: str, updates: RobotUpdate) -> Optional[Robot]:
        """Update a robot."""
        doc_ref = self.db.collection(self.ROBOTS).document(robot_id)

        update_data = updates.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()

        doc_ref.update(update_data)
        return await self.get_robot(robot_id)

    async def delete_robot(self, robot_id: str) -> bool:
        """Delete a robot and its subcollections."""
        doc_ref = self.db.collection(self.ROBOTS).document(robot_id)

        # Delete subcollections
        for subcoll in [self.COMMANDS, self.TELEMETRY]:
            for doc in doc_ref.collection(subcoll).stream():
                doc.reference.delete()

        # Delete the robot document
        doc_ref.delete()
        logger.info(f"Deleted robot: {robot_id}")
        return True

    async def update_robot_status(
        self, robot_id: str, status: RobotStatus, telemetry: Optional[RobotTelemetry] = None
    ) -> None:
        """Update robot status and optionally add telemetry."""
        doc_ref = self.db.collection(self.ROBOTS).document(robot_id)

        updates = {
            "status": status.value,
            "last_seen": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        if telemetry:
            updates["latest_telemetry"] = telemetry.model_dump()

            # Also store in telemetry subcollection
            self.db.collection(self.ROBOTS).document(robot_id).collection(
                self.TELEMETRY
            ).add(telemetry.model_dump())

        doc_ref.update(updates)

    # ==================== Command Operations ====================

    async def add_command(self, robot_id: str, command: RobotCommand) -> str:
        """Add a command to robot's command queue."""
        command_id = str(uuid4())
        command.id = command_id
        command.created_at = datetime.utcnow()
        command.status = "pending"

        self.db.collection(self.ROBOTS).document(robot_id).collection(
            self.COMMANDS
        ).document(command_id).set(command.model_dump())

        logger.info(f"Added command {command_id} to robot {robot_id}")
        return command_id

    async def get_pending_commands(self, robot_id: str) -> List[RobotCommand]:
        """Get pending commands for a robot."""
        docs = (
            self.db.collection(self.ROBOTS)
            .document(robot_id)
            .collection(self.COMMANDS)
            .where("status", "==", "pending")
            .order_by("priority", direction=firestore.Query.DESCENDING)
            .order_by("created_at")
            .stream()
        )

        return [RobotCommand(**doc.to_dict()) for doc in docs]

    async def update_command_status(
        self, robot_id: str, command_id: str, status: str, result: Optional[Dict] = None
    ) -> None:
        """Update command status."""
        doc_ref = (
            self.db.collection(self.ROBOTS)
            .document(robot_id)
            .collection(self.COMMANDS)
            .document(command_id)
        )

        updates = {"status": status}
        if result:
            updates["result"] = result

        doc_ref.update(updates)

    # ==================== Training Job Operations ====================

    async def create_training_job(self, user_id: str, config: TrainingConfig) -> TrainingJob:
        """Create a new training job."""
        job_id = str(uuid4())
        doc_ref = self.db.collection(self.TRAINING_JOBS).document(job_id)

        job = TrainingJob(
            id=job_id,
            user_id=user_id,
            config=config,
            status=TrainingStatus.PENDING,
            total_epochs=config.hyperparameters.epochs,
            created_at=datetime.utcnow(),
        )

        doc_ref.set(job.model_dump(exclude={"id"}))
        logger.info(f"Created training job: {job_id} for user: {user_id}")
        return job

    async def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        doc = self.db.collection(self.TRAINING_JOBS).document(job_id).get()
        if not doc.exists:
            return None

        data = doc.to_dict()
        data["id"] = doc.id
        return TrainingJob(**data)

    async def list_training_jobs(
        self, user_id: str, status: Optional[TrainingStatus] = None, limit: int = 20
    ) -> List[TrainingJob]:
        """List training jobs for a user."""
        query = self.db.collection(self.TRAINING_JOBS).where("user_id", "==", user_id)

        if status:
            query = query.where("status", "==", status.value)

        query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)

        jobs = []
        for doc in query.stream():
            data = doc.to_dict()
            data["id"] = doc.id
            jobs.append(TrainingJob(**data))

        return jobs

    async def update_training_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[TrainingJob]:
        """Update a training job."""
        doc_ref = self.db.collection(self.TRAINING_JOBS).document(job_id)
        doc_ref.update(updates)
        return await self.get_training_job(job_id)

    async def update_training_progress(
        self,
        job_id: str,
        current_epoch: int,
        progress_percent: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update training job progress."""
        updates = {
            "current_epoch": current_epoch,
            "progress_percent": progress_percent,
        }

        if metrics:
            updates["best_metrics"] = metrics

        await self.update_training_job(job_id, updates)


async def get_firestore_service(
    settings: Settings = Depends(get_settings),
) -> FirestoreService:
    """Dependency to get FirestoreService instance."""
    return FirestoreService(settings)
