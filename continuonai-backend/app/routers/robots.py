"""Robot fleet management API endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth.firebase import FirebaseUser, get_current_user
from app.models.robot import (
    Robot,
    RobotCommand,
    RobotCommandResponse,
    RobotCreate,
    RobotListResponse,
    RobotStatus,
    RobotTelemetry,
    RobotUpdate,
)
from app.services.firestore import FirestoreService, get_firestore_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=RobotListResponse)
async def list_robots(
    status_filter: Optional[RobotStatus] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    List all robots owned by the authenticated user.

    Returns paginated list of robots with optional status filtering.
    """
    robots = await db.get_robots(user.uid)

    # Apply status filter
    if status_filter:
        robots = [r for r in robots if r.status == status_filter]

    # Calculate pagination
    total = len(robots)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_robots = robots[start:end]

    return RobotListResponse(
        robots=paginated_robots,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/", response_model=Robot, status_code=status.HTTP_201_CREATED)
async def create_robot(
    robot_data: RobotCreate,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Register a new robot.

    Creates a new robot entry associated with the authenticated user.
    The device_id must be unique.
    """
    # Check if device_id already exists
    existing_robots = await db.get_robots(user.uid)
    for robot in existing_robots:
        if robot.device_id == robot_data.device_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Robot with device_id '{robot_data.device_id}' already exists",
            )

    # TODO: Check user's robot quota
    # user_data = await db.get_user(user.uid)
    # if len(existing_robots) >= user_data.robot_limit:
    #     raise HTTPException(status_code=403, detail="Robot limit reached")

    robot = await db.create_robot(user.uid, robot_data)
    logger.info(f"User {user.uid} created robot {robot.id}")
    return robot


@router.get("/{robot_id}", response_model=Robot)
async def get_robot(
    robot_id: str,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Get robot details by ID.

    Returns full robot information including latest telemetry.
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    # Verify ownership
    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this robot",
        )

    return robot


@router.patch("/{robot_id}", response_model=Robot)
async def update_robot(
    robot_id: str,
    updates: RobotUpdate,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Update robot details.

    Allows updating name, description, tags, and model version.
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this robot",
        )

    updated_robot = await db.update_robot(robot_id, updates)
    logger.info(f"User {user.uid} updated robot {robot_id}")
    return updated_robot


@router.delete("/{robot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_robot(
    robot_id: str,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Delete a robot.

    Removes the robot and all associated data (commands, telemetry).
    This action cannot be undone.
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this robot",
        )

    await db.delete_robot(robot_id)
    logger.info(f"User {user.uid} deleted robot {robot_id}")


@router.post("/{robot_id}/command", response_model=RobotCommandResponse)
async def send_command(
    robot_id: str,
    command: RobotCommand,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Send a command to a robot.

    Commands are queued in Firestore and picked up by the robot.
    The robot must be online to receive commands.

    Command types:
    - move: Move to position
    - stop: Emergency stop
    - deploy_model: Deploy a new model version
    - reboot: Restart robot software
    - calibrate: Run calibration routine
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to command this robot",
        )

    command_id = await db.add_command(robot_id, command)
    logger.info(f"User {user.uid} sent command {command.type} to robot {robot_id}")

    return RobotCommandResponse(
        command_id=command_id,
        status="queued",
        message=f"Command '{command.type}' queued for robot",
    )


@router.get("/{robot_id}/commands", response_model=List[RobotCommand])
async def get_pending_commands(
    robot_id: str,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Get pending commands for a robot.

    Returns all commands that have not yet been executed.
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this robot",
        )

    return await db.get_pending_commands(robot_id)


@router.get("/{robot_id}/status", response_model=Robot)
async def get_robot_status(
    robot_id: str,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Get robot's current status.

    Returns current status, mode, and latest telemetry data.
    For real-time updates, use the WebSocket endpoint.
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this robot",
        )

    return robot


@router.post("/{robot_id}/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def update_telemetry(
    robot_id: str,
    telemetry: RobotTelemetry,
    user: FirebaseUser = Depends(get_current_user),
    db: FirestoreService = Depends(get_firestore_service),
):
    """
    Update robot telemetry data.

    This endpoint is called by the robot to report its current state.
    Telemetry is stored and the robot's status is updated.
    """
    robot = await db.get_robot(robot_id)

    if not robot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Robot not found",
        )

    if robot.owner_id != user.uid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this robot",
        )

    # Update robot status based on telemetry
    new_status = RobotStatus.ONLINE
    if telemetry.battery_level < 10:
        new_status = RobotStatus.ERROR  # Low battery warning

    await db.update_robot_status(robot_id, new_status, telemetry)

    return {"status": "accepted", "message": "Telemetry recorded"}
