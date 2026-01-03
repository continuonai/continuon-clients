"""
Safety Protocol - Protocol 66 and Safety Rules

This module defines the safety protocols that the Ring 0 safety kernel
enforces. Protocol 66 is the default safety protocol.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ProtocolCategory(Enum):
    """Categories of safety rules."""
    MOTION = auto()       # Movement safety
    FORCE = auto()        # Force/torque limits
    WORKSPACE = auto()    # Workspace boundaries
    HUMAN = auto()        # Human interaction safety
    THERMAL = auto()      # Temperature limits
    ELECTRICAL = auto()   # Electrical safety
    SOFTWARE = auto()     # Software safety
    EMERGENCY = auto()    # Emergency procedures
    PROPERTY = auto()     # Property protection (destructive actions)
    PRIVACY = auto()      # Privacy protection


@dataclass
class SafetyRule:
    """Individual safety rule."""
    id: str
    name: str
    category: ProtocolCategory
    description: str
    enabled: bool = True
    severity: str = "violation"  # "warning", "violation", "critical"
    validator: Optional[str] = None  # Function name to call
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyProtocol:
    """
    Safety protocol defining all safety rules.
    
    The safety kernel enforces these rules at Ring 0 level.
    """
    name: str
    version: str
    description: str
    rules: List[SafetyRule] = field(default_factory=list)
    
    def get_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Get rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def get_rules_by_category(self, category: ProtocolCategory) -> List[SafetyRule]:
        """Get all rules in a category."""
        return [r for r in self.rules if r.category == category]
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        rule = self.get_rule(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule (requires authorization in production)."""
        rule = self.get_rule(rule_id)
        if rule:
            logger.warning(f"Disabling safety rule: {rule_id}")
            rule.enabled = False
            return True
        return False


# ============================================================================
# PROTOCOL 66 - Default Safety Protocol
# ============================================================================

PROTOCOL_66 = SafetyProtocol(
    name="Protocol 66",
    version="1.0.0",
    description="Default ContinuonBrain safety protocol for robot operation",
    rules=[
        # ====================
        # MOTION SAFETY
        # ====================
        SafetyRule(
            id="MOTION_001",
            name="Maximum Joint Velocity",
            category=ProtocolCategory.MOTION,
            description="Joint velocities must not exceed safe limits",
            severity="violation",
            parameters={
                "max_velocity_rad_s": 2.0,  # rad/s
                "warning_threshold": 0.8,   # 80% of max
            },
        ),
        SafetyRule(
            id="MOTION_002",
            name="Maximum End-Effector Velocity",
            category=ProtocolCategory.MOTION,
            description="End-effector velocity must not exceed safe limits",
            severity="violation",
            parameters={
                "max_velocity_m_s": 1.0,  # m/s
                "max_velocity_human_present": 0.25,  # m/s when human nearby
            },
        ),
        SafetyRule(
            id="MOTION_003",
            name="Smooth Trajectory",
            category=ProtocolCategory.MOTION,
            description="Trajectories must be smooth (limited jerk)",
            severity="warning",
            parameters={
                "max_jerk": 500.0,  # m/s³
            },
        ),
        SafetyRule(
            id="MOTION_004",
            name="Emergency Stop Response Time",
            category=ProtocolCategory.MOTION,
            description="E-stop must halt motion within time limit",
            severity="critical",
            parameters={
                "max_stop_time_ms": 100,  # milliseconds
            },
        ),
        
        # ====================
        # FORCE SAFETY
        # ====================
        SafetyRule(
            id="FORCE_001",
            name="Maximum Contact Force",
            category=ProtocolCategory.FORCE,
            description="Contact forces must not exceed safe limits",
            severity="violation",
            parameters={
                "max_force_N": 50.0,  # Newtons
                "max_force_human": 10.0,  # Newtons for human contact
            },
        ),
        SafetyRule(
            id="FORCE_002",
            name="Maximum Torque",
            category=ProtocolCategory.FORCE,
            description="Joint torques must not exceed limits",
            severity="violation",
            parameters={
                "max_torque_Nm": [5.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.5],  # Per joint
            },
        ),
        SafetyRule(
            id="FORCE_003",
            name="Collision Detection",
            category=ProtocolCategory.FORCE,
            description="Unexpected collisions trigger safety response",
            severity="violation",
            parameters={
                "force_threshold_N": 5.0,
                "response": "stop_and_retract",
            },
        ),
        
        # ====================
        # WORKSPACE SAFETY
        # ====================
        SafetyRule(
            id="WORKSPACE_001",
            name="Workspace Boundaries",
            category=ProtocolCategory.WORKSPACE,
            description="Robot must stay within defined workspace",
            severity="violation",
            parameters={
                "bounds_type": "sphere",
                "radius_m": 0.8,
                "center": [0.0, 0.0, 0.3],  # Base frame
            },
        ),
        SafetyRule(
            id="WORKSPACE_002",
            name="Forbidden Zones",
            category=ProtocolCategory.WORKSPACE,
            description="Robot must not enter forbidden zones",
            severity="critical",
            parameters={
                "zones": [
                    {"name": "power_supply", "type": "box", "bounds": [[-0.1, -0.1, 0], [0.1, 0.1, 0.1]]},
                ],
            },
        ),
        SafetyRule(
            id="WORKSPACE_003",
            name="Self-Collision Avoidance",
            category=ProtocolCategory.WORKSPACE,
            description="Robot must not collide with itself",
            severity="violation",
            parameters={
                "min_clearance_m": 0.02,
            },
        ),
        
        # ====================
        # HUMAN SAFETY
        # ====================
        SafetyRule(
            id="HUMAN_001",
            name="Human Detection",
            category=ProtocolCategory.HUMAN,
            description="Detect humans in workspace and adapt behavior",
            severity="warning",
            parameters={
                "detection_radius_m": 2.0,
                "slow_zone_m": 1.0,
                "stop_zone_m": 0.3,
            },
        ),
        SafetyRule(
            id="HUMAN_002",
            name="Human Contact Response",
            category=ProtocolCategory.HUMAN,
            description="Respond safely to human contact",
            severity="violation",
            parameters={
                "response": "stop_and_comply",
                "max_contact_force_N": 10.0,
            },
        ),
        SafetyRule(
            id="HUMAN_003",
            name="Eye Contact / Attention",
            category=ProtocolCategory.HUMAN,
            description="Acknowledge human presence with gaze",
            severity="warning",
            parameters={
                "gaze_towards_human": True,
            },
        ),
        
        # ====================
        # THERMAL SAFETY
        # ====================
        SafetyRule(
            id="THERMAL_001",
            name="Motor Temperature",
            category=ProtocolCategory.THERMAL,
            description="Motor temperatures must stay within limits",
            severity="violation",
            parameters={
                "max_temp_C": 80.0,
                "warning_temp_C": 60.0,
            },
        ),
        SafetyRule(
            id="THERMAL_002",
            name="CPU Temperature",
            category=ProtocolCategory.THERMAL,
            description="CPU temperature must stay within limits",
            severity="warning",
            parameters={
                "max_temp_C": 85.0,
                "throttle_temp_C": 75.0,
            },
        ),
        
        # ====================
        # ELECTRICAL SAFETY
        # ====================
        SafetyRule(
            id="ELECTRICAL_001",
            name="Power Monitoring",
            category=ProtocolCategory.ELECTRICAL,
            description="Monitor power consumption",
            severity="warning",
            parameters={
                "max_current_A": 10.0,
                "min_voltage_V": 11.0,
                "max_voltage_V": 13.0,
            },
        ),
        SafetyRule(
            id="ELECTRICAL_002",
            name="Battery Safety",
            category=ProtocolCategory.ELECTRICAL,
            description="Battery must stay in safe operating range",
            severity="violation",
            parameters={
                "min_soc_percent": 10.0,
                "max_temp_C": 45.0,
            },
        ),
        
        # ====================
        # SOFTWARE SAFETY
        # ====================
        SafetyRule(
            id="SOFTWARE_001",
            name="Watchdog",
            category=ProtocolCategory.SOFTWARE,
            description="Software watchdog must be active",
            severity="critical",
            parameters={
                "timeout_ms": 100,
            },
        ),
        SafetyRule(
            id="SOFTWARE_002",
            name="Command Validation",
            category=ProtocolCategory.SOFTWARE,
            description="All commands must be validated before execution",
            severity="violation",
            parameters={
                "validate_bounds": True,
                "validate_trajectory": True,
            },
        ),
        SafetyRule(
            id="SOFTWARE_003",
            name="Fallback Mode",
            category=ProtocolCategory.SOFTWARE,
            description="System must have fallback behavior on error",
            severity="critical",
            parameters={
                "fallback_action": "safe_position",
            },
        ),
        
        # ====================
        # EMERGENCY PROCEDURES
        # ====================
        SafetyRule(
            id="EMERGENCY_001",
            name="Emergency Stop",
            category=ProtocolCategory.EMERGENCY,
            description="E-stop must always be available and functional",
            severity="critical",
            parameters={
                "hardware_estop": True,
                "software_estop": True,
            },
        ),
        SafetyRule(
            id="EMERGENCY_002",
            name="Safe State",
            category=ProtocolCategory.EMERGENCY,
            description="Robot must be able to reach safe state from any position",
            severity="critical",
            parameters={
                "safe_position": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # Home position
            },
        ),
        SafetyRule(
            id="EMERGENCY_003",
            name="Recovery Procedure",
            category=ProtocolCategory.EMERGENCY,
            description="System must have defined recovery procedures",
            severity="warning",
            parameters={
                "auto_recovery": False,  # Manual recovery required
                "requires_authorization": True,
            },
        ),
        
        # ====================
        # PROPERTY PROTECTION
        # ====================
        SafetyRule(
            id="PROPERTY_001",
            name="No Unauthorized Destruction",
            category=ProtocolCategory.PROPERTY,
            description="Destructive actions require explicit work authorization",
            severity="critical",
            parameters={
                "blocked_actions": [
                    "demolish", "cut", "crush", "shred", "burn", "dissolve",
                    "delete_data", "overwrite", "format",
                ],
                "requires_authorization": True,
                "authorization_system": "work_authorization",
            },
        ),
        SafetyRule(
            id="PROPERTY_002",
            name="Property Ownership Verification",
            category=ProtocolCategory.PROPERTY,
            description="Only legal owners can authorize destruction of their property",
            severity="critical",
            parameters={
                "allowed_roles": ["creator", "owner", "leasee"],
                "requires_evidence": True,
                "evidence_types": ["deed", "title", "contract", "receipt"],
            },
        ),
        SafetyRule(
            id="PROPERTY_003",
            name="Third-Party Property Protection",
            category=ProtocolCategory.PROPERTY,
            description="Never damage property belonging to others",
            severity="critical",
            parameters={
                "always_verify_ownership": True,
                "when_uncertain": "deny",
            },
        ),
        SafetyRule(
            id="PROPERTY_004",
            name="Collateral Damage Prevention",
            category=ProtocolCategory.PROPERTY,
            description="Authorized destruction must not damage non-target property",
            severity="violation",
            parameters={
                "require_clear_boundaries": True,
                "safety_margin_m": 0.5,
            },
        ),
        SafetyRule(
            id="PROPERTY_005",
            name="Destruction Audit Trail",
            category=ProtocolCategory.PROPERTY,
            description="All destructive actions must be logged with full provenance",
            severity="violation",
            parameters={
                "log_before_action": True,
                "log_after_action": True,
                "include_authorization_id": True,
                "include_property_id": True,
            },
        ),
        
        # ====================
        # PRIVACY PROTECTION
        # ====================
        SafetyRule(
            id="PRIVACY_001",
            name="Consent Required for Personal Data",
            category=ProtocolCategory.PRIVACY,
            description="Personal data collection requires explicit consent",
            severity="critical",
            parameters={
                "data_types": ["face", "voice", "behavior", "location"],
                "requires_consent": True,
                "consent_must_be_explicit": True,
            },
        ),
        SafetyRule(
            id="PRIVACY_002",
            name="Local Data Processing",
            category=ProtocolCategory.PRIVACY,
            description="Personal data must be processed on-device by default",
            severity="violation",
            parameters={
                "default_processing": "local",
                "cloud_requires_consent": True,
            },
        ),
        SafetyRule(
            id="PRIVACY_003",
            name="Right to Be Forgotten",
            category=ProtocolCategory.PRIVACY,
            description="Users can request deletion of their personal data",
            severity="critical",
            parameters={
                "deletion_must_be_complete": True,
                "max_deletion_time_hours": 24,
            },
        ),
    ]
)


def get_protocol(name: str = "protocol_66") -> SafetyProtocol:
    """Get a safety protocol by name."""
    protocols = {
        "protocol_66": PROTOCOL_66,
    }
    return protocols.get(name.lower().replace(" ", "_"), PROTOCOL_66)

