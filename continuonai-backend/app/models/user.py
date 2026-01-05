"""User-related Pydantic schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserPlan(str, Enum):
    """User subscription plans."""

    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr = Field(..., description="User email address")
    name: Optional[str] = Field(None, max_length=100, description="Display name")
    organization: Optional[str] = Field(None, max_length=200, description="Organization name")


class UserCreate(UserBase):
    """Schema for creating a new user (internal use)."""

    uid: str = Field(..., description="Firebase UID")


class User(UserBase):
    """Full user schema."""

    uid: str = Field(..., description="Firebase UID")
    plan: UserPlan = Field(default=UserPlan.FREE, description="Subscription plan")
    robot_limit: int = Field(default=1, description="Max robots allowed")
    training_hours_remaining: float = Field(default=0, description="Training hours quota")
    storage_gb_limit: float = Field(default=5, description="Storage limit in GB")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    email_verified: bool = Field(default=False)
    is_active: bool = Field(default=True)

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    name: Optional[str] = Field(None, max_length=100)
    organization: Optional[str] = Field(None, max_length=200)


class UserQuota(BaseModel):
    """User quota/usage information."""

    uid: str
    plan: UserPlan
    robots_used: int
    robots_limit: int
    storage_used_gb: float
    storage_limit_gb: float
    training_hours_used: float
    training_hours_limit: float
    api_calls_today: int
    api_calls_limit: int


class TokenPayload(BaseModel):
    """JWT token payload."""

    uid: str
    email: Optional[str] = None
    email_verified: bool = False
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
