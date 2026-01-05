"""Robot-related Pydantic schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RobotStatus(str, Enum):
    """Robot connection and operational status."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class RobotMode(str, Enum):
    """Robot operational mode."""

    IDLE = "idle"
    AUTONOMOUS = "autonomous"
    TELEOPERATION = "teleop"
    TRAINING = "training"
    CHARGING = "charging"


class RobotBase(BaseModel):
    """Base robot schema with common fields."""

    name: str = Field(..., min_length=1, max_length=100, description="Robot display name")
    device_id: str = Field(..., description="Unique hardware device identifier")
    description: Optional[str] = Field(None, max_length=500, description="Robot description")
    model_type: str = Field(default="continuon-v1", description="Robot model/type")
    tags: List[str] = Field(default_factory=list, description="Custom tags for organization")


class RobotCreate(RobotBase):
    """Schema for creating a new robot."""

    initial_model_version: Optional[str] = Field(
        None, description="Initial model version to deploy"
    )


class RobotUpdate(BaseModel):
    """Schema for updating robot details."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    model_version: Optional[str] = Field(None, description="Model version to deploy")


class RobotTelemetry(BaseModel):
    """Robot telemetry data."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    battery_level: float = Field(..., ge=0, le=100, description="Battery percentage")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    position: Optional[Dict[str, float]] = Field(
        None, description="Position data (x, y, z, roll, pitch, yaw)"
    )
    mode: RobotMode = Field(default=RobotMode.IDLE)
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, ge=0, le=100, description="Memory usage percentage")
    custom_data: Optional[Dict[str, Any]] = Field(None, description="Additional telemetry data")


class RobotCommand(BaseModel):
    """Command to send to a robot."""

    id: Optional[str] = Field(None, description="Command ID (auto-generated)")
    type: str = Field(..., description="Command type (e.g., 'move', 'stop', 'deploy_model')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Command payload")
    priority: int = Field(default=0, ge=0, le=10, description="Command priority (0-10)")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Command timeout")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="pending", description="Command status")


class Robot(RobotBase):
    """Full robot schema with all fields."""

    id: str = Field(..., description="Unique robot ID")
    owner_id: str = Field(..., description="Owner user ID")
    status: RobotStatus = Field(default=RobotStatus.OFFLINE)
    mode: RobotMode = Field(default=RobotMode.IDLE)
    model_version: Optional[str] = Field(None, description="Currently deployed model version")
    last_seen: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    latest_telemetry: Optional[RobotTelemetry] = Field(None, description="Most recent telemetry")

    model_config = {"from_attributes": True}


class RobotCommandResponse(BaseModel):
    """Response after sending a command."""

    command_id: str
    status: str = "queued"
    message: Optional[str] = None


class RobotListResponse(BaseModel):
    """Paginated robot list response."""

    robots: List[Robot]
    total: int
    page: int = 1
    page_size: int = 20
