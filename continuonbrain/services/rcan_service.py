"""
RCAN (Robot Communication & Addressing Network) Service

Implements the RCAN protocol for robot discovery, authentication,
and multi-robot coordination.

See docs/rcan-protocol.md for full specification.
"""
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class UserRole(IntEnum):
    """User role hierarchy (higher = more privilege)."""
    UNKNOWN = 0
    GUEST = 1
    USER = 2
    LEASEE = 3
    OWNER = 4
    CREATOR = 5


class RobotCapability(IntEnum):
    """Robot capabilities that can be permission-gated."""
    VIEW_STATUS = 1
    CHAT = 2
    TELEOP_CONTROL = 3
    ARM_CONTROL = 4
    NAVIGATION = 5
    RECORD_EPISODES = 6
    TRAINING_CONTRIBUTE = 7
    INSTALL_SKILLS = 8
    OTA_UPDATES = 9
    SAFETY_CONFIG = 10
    USER_MANAGEMENT = 11
    MODEL_DEPLOYMENT = 12
    HARDWARE_DIAGNOSTICS = 13


# Capability matrix: role -> set of allowed capabilities
CAPABILITY_MATRIX: Dict[UserRole, set] = {
    UserRole.CREATOR: {cap for cap in RobotCapability},  # All capabilities
    UserRole.OWNER: {
        RobotCapability.VIEW_STATUS, RobotCapability.CHAT, RobotCapability.TELEOP_CONTROL,
        RobotCapability.ARM_CONTROL, RobotCapability.NAVIGATION, RobotCapability.RECORD_EPISODES,
        RobotCapability.TRAINING_CONTRIBUTE, RobotCapability.INSTALL_SKILLS, RobotCapability.OTA_UPDATES,
        RobotCapability.USER_MANAGEMENT, RobotCapability.HARDWARE_DIAGNOSTICS,
    },
    UserRole.LEASEE: {
        RobotCapability.VIEW_STATUS, RobotCapability.CHAT, RobotCapability.TELEOP_CONTROL,
        RobotCapability.ARM_CONTROL, RobotCapability.NAVIGATION, RobotCapability.RECORD_EPISODES,
    },
    UserRole.USER: {
        RobotCapability.VIEW_STATUS, RobotCapability.CHAT, RobotCapability.TELEOP_CONTROL,
        RobotCapability.NAVIGATION,
    },
    UserRole.GUEST: {
        RobotCapability.VIEW_STATUS, RobotCapability.CHAT,
    },
    UserRole.UNKNOWN: set(),
}


@dataclass
class RCANIdentity:
    """Robot identity for RCAN addressing."""
    registry: str = "continuon.cloud"
    manufacturer: str = "continuon"
    model: str = "companion-v1"
    device_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    @property
    def ruri(self) -> str:
        """Get the full Robot URI."""
        return f"rcan://{self.registry}/{self.manufacturer}/{self.model}/{self.device_id}"
    
    @classmethod
    def from_ruri(cls, ruri: str) -> "RCANIdentity":
        """Parse RURI string into identity."""
        # rcan://registry/manufacturer/model/device-id
        if not ruri.startswith("rcan://"):
            raise ValueError(f"Invalid RURI: {ruri}")
        parts = ruri[7:].split("/")
        if len(parts) < 4:
            raise ValueError(f"Invalid RURI format: {ruri}")
        return cls(
            registry=parts[0],
            manufacturer=parts[1],
            model=parts[2],
            device_id=parts[3].split(":")[0],  # Remove port if present
        )


@dataclass
class AuthSession:
    """Authenticated session for RCAN."""
    session_id: str
    user_id: str
    role: UserRole
    source_ruri: str
    target_ruri: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    capabilities: set = field(default_factory=set)
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def can_perform(self, capability: RobotCapability) -> bool:
        """Check if session allows a capability."""
        if self.is_expired():
            return False
        return capability in CAPABILITY_MATRIX.get(self.role, set())


@dataclass 
class RCANMessage:
    """RCAN protocol message."""
    version: str = "1.0.0"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_ruri: str = ""
    target_ruri: str = ""
    message_type: str = "STATUS"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    ttl_ms: int = 30000
    priority: str = "NORMAL"
    auth_token: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RCANMessage":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RCANService:
    """
    RCAN protocol service for robot communication and addressing.
    
    Handles:
    - Robot discovery (mDNS/broadcast)
    - Authentication and session management
    - Role-based access control
    - Multi-robot coordination
    """
    
    def __init__(self, config_dir: str, port: int = 8080):
        self.config_dir = Path(config_dir)
        self.port = port
        
        # Load or create identity
        self.identity = self._load_or_create_identity()
        
        # Active sessions
        self.sessions: Dict[str, AuthSession] = {}
        
        # Rate limiting for guests
        self.guest_requests: Dict[str, List[float]] = {}  # ip -> timestamps
        self.guest_rate_limit = 10  # requests per minute
        
        # Event callbacks
        self._on_claim: Optional[Callable] = None
        self._on_release: Optional[Callable] = None
        self._on_command: Optional[Callable] = None
        
        logger.info(f"RCAN Service initialized: {self.identity.ruri}")
    
    def _load_or_create_identity(self) -> RCANIdentity:
        """Load identity from config or create new."""
        identity_file = self.config_dir / "rcan_identity.json"
        
        if identity_file.exists():
            try:
                with open(identity_file) as f:
                    data = json.load(f)
                return RCANIdentity(**data)
            except Exception as e:
                logger.warning(f"Failed to load RCAN identity: {e}")
        
        # Create new identity
        identity = RCANIdentity(
            device_id=self._generate_device_id(),
        )
        
        # Persist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(identity_file, "w") as f:
            json.dump(asdict(identity), f, indent=2)
        
        return identity
    
    def _generate_device_id(self) -> str:
        """Generate a unique device ID."""
        # Try to use hardware identifiers
        try:
            # Raspberry Pi serial
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("Serial"):
                        return line.split(":")[1].strip()[-8:]
        except Exception:
            pass
        
        # Fall back to MAC address
        try:
            import uuid as uuid_mod
            mac = uuid_mod.getnode()
            return f"{mac:012x}"[-8:]
        except Exception:
            pass
        
        # Random UUID
        return str(uuid.uuid4())[:8]
    
    # =========================================================================
    # Discovery
    # =========================================================================
    
    def get_discovery_info(self) -> dict:
        """Get robot info for mDNS/discovery."""
        return {
            "ruri": self.identity.ruri,
            "model": self.identity.model,
            "manufacturer": self.identity.manufacturer,
            "caps": ["arm", "vision", "chat", "teleop", "training"],
            "roles": ["owner", "user", "guest"],
            "version": "1.0.0",
            "port": self.port,
            "hostname": socket.gethostname(),
        }
    
    def handle_discover(self, message: RCANMessage) -> RCANMessage:
        """Handle discovery request."""
        return RCANMessage(
            source_ruri=self.identity.ruri,
            target_ruri=message.source_ruri,
            message_type="ANNOUNCE",
            payload=self.get_discovery_info(),
        )
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    def handle_claim(self, message: RCANMessage, user_id: str, role: UserRole) -> RCANMessage:
        """Handle control claim request."""
        
        # Check if already claimed by higher privilege
        for session in self.sessions.values():
            if not session.is_expired() and session.role > role:
                return RCANMessage(
                    source_ruri=self.identity.ruri,
                    target_ruri=message.source_ruri,
                    message_type="DENIED",
                    payload={
                        "reason": "Robot is controlled by higher-privilege user",
                        "current_role": session.role.name,
                    },
                )
        
        # Create session
        session = AuthSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            source_ruri=message.source_ruri,
            target_ruri=self.identity.ruri,
            expires_at=datetime.now() + timedelta(hours=1) if role <= UserRole.LEASEE else None,
        )
        
        self.sessions[session.session_id] = session
        
        if self._on_claim:
            self._on_claim(session)
        
        logger.info(f"RCAN: Control claimed by {user_id} as {role.name}")
        
        return RCANMessage(
            source_ruri=self.identity.ruri,
            target_ruri=message.source_ruri,
            message_type="GRANTED",
            payload={
                "session_id": session.session_id,
                "role": role.name,
                "capabilities": [cap.name for cap in CAPABILITY_MATRIX.get(role, set())],
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            },
        )
    
    def handle_release(self, session_id: str) -> RCANMessage:
        """Handle control release."""
        session = self.sessions.pop(session_id, None)
        
        if session and self._on_release:
            self._on_release(session)
        
        return RCANMessage(
            source_ruri=self.identity.ruri,
            message_type="RELEASED",
            payload={"session_id": session_id},
        )
    
    def validate_session(self, session_id: str, capability: RobotCapability) -> tuple[bool, Optional[str]]:
        """Validate session and capability."""
        session = self.sessions.get(session_id)
        
        if not session:
            return False, "Session not found"
        
        if session.is_expired():
            del self.sessions[session_id]
            return False, "Session expired"
        
        if not session.can_perform(capability):
            return False, f"Role {session.role.name} cannot perform {capability.name}"
        
        return True, None
    
    # =========================================================================
    # Commands
    # =========================================================================
    
    def handle_command(self, message: RCANMessage, session_id: str) -> RCANMessage:
        """Handle command request."""
        command = message.payload.get("command", "")
        capability = self._command_to_capability(command)
        
        valid, error = self.validate_session(session_id, capability)
        if not valid:
            return RCANMessage(
                source_ruri=self.identity.ruri,
                target_ruri=message.source_ruri,
                message_type="ERROR",
                payload={"error": error, "command": command},
            )
        
        # Execute command via callback
        result = {"status": "ok"}
        if self._on_command:
            try:
                result = self._on_command(command, message.payload)
            except Exception as e:
                result = {"status": "error", "error": str(e)}
        
        return RCANMessage(
            source_ruri=self.identity.ruri,
            target_ruri=message.source_ruri,
            message_type="ACK",
            payload={"command": command, "result": result},
        )
    
    def _command_to_capability(self, command: str) -> RobotCapability:
        """Map command to required capability."""
        command_map = {
            "status": RobotCapability.VIEW_STATUS,
            "chat": RobotCapability.CHAT,
            "teleop": RobotCapability.TELEOP_CONTROL,
            "move": RobotCapability.NAVIGATION,
            "arm": RobotCapability.ARM_CONTROL,
            "record": RobotCapability.RECORD_EPISODES,
            "train": RobotCapability.TRAINING_CONTRIBUTE,
            "install": RobotCapability.INSTALL_SKILLS,
            "update": RobotCapability.OTA_UPDATES,
            "safety": RobotCapability.SAFETY_CONFIG,
            "users": RobotCapability.USER_MANAGEMENT,
            "deploy": RobotCapability.MODEL_DEPLOYMENT,
            "diagnostics": RobotCapability.HARDWARE_DIAGNOSTICS,
        }
        
        for prefix, cap in command_map.items():
            if command.lower().startswith(prefix):
                return cap
        
        return RobotCapability.VIEW_STATUS  # Default
    
    # =========================================================================
    # Rate Limiting (Guest)
    # =========================================================================
    
    def check_guest_rate_limit(self, ip: str) -> bool:
        """Check if guest IP is within rate limit."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old entries
        if ip in self.guest_requests:
            self.guest_requests[ip] = [t for t in self.guest_requests[ip] if t > window_start]
        else:
            self.guest_requests[ip] = []
        
        if len(self.guest_requests[ip]) >= self.guest_rate_limit:
            return False
        
        self.guest_requests[ip].append(now)
        return True
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_status(self) -> dict:
        """Get RCAN service status."""
        active_sessions = [
            {
                "session_id": s.session_id[:8],
                "role": s.role.name,
                "user_id": s.user_id[:8] if s.user_id else "anonymous",
                "expires_in": (s.expires_at - datetime.now()).total_seconds() if s.expires_at else None,
            }
            for s in self.sessions.values()
            if not s.is_expired()
        ]
        
        return {
            "ruri": self.identity.ruri,
            "active_sessions": len(active_sessions),
            "sessions": active_sessions,
            "capabilities": [cap.name for cap in RobotCapability],
        }
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def on_claim(self, callback: Callable[[AuthSession], None]):
        """Register callback for claim events."""
        self._on_claim = callback
    
    def on_release(self, callback: Callable[[AuthSession], None]):
        """Register callback for release events."""
        self._on_release = callback
    
    def on_command(self, callback: Callable[[str, dict], dict]):
        """Register callback for command events."""
        self._on_command = callback

