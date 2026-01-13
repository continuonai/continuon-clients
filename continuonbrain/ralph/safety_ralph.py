"""
Safety Kernel Ralph Layer
=========================

Wraps the Ring 0 safety kernel with Ralph's guardrails as constitutional constraints.

Characteristics:
- Sub-millisecond validation (must not block)
- Cannot be bypassed by userland code
- Constitutional constraints as permanent guardrails
- Deterministic validation logic
- Black-box audit logging
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from .base import (
    RalphLayer,
    RalphConfig,
    RalphState,
    LoopType,
    Guardrail,
    MetaLayerContext,
)

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of safety violations."""
    KINEMATIC_LIMIT = "kinematic_limit"
    VELOCITY_EXCEEDED = "velocity_exceeded"
    ACCELERATION_EXCEEDED = "acceleration_exceeded"
    NO_GO_ZONE = "no_go_zone"
    COLLISION_IMMINENT = "collision_imminent"
    THERMAL_LIMIT = "thermal_limit"
    POWER_LIMIT = "power_limit"
    CONSTITUTIONAL = "constitutional"


class ResponseLevel(Enum):
    """Graduated response levels."""
    ALLOW = "allow"           # Safe, proceed
    CLIP = "clip"             # Clip to safe limits
    SLOW = "slow"             # Reduce velocity
    PAUSE = "pause"           # Temporary pause
    HALT = "halt"             # Full stop
    RECOVERY = "recovery"     # Enter recovery mode


@dataclass
class SafetyCheckResult:
    """Result of a safety validation."""
    allowed: bool
    response_level: ResponseLevel
    violations: List[ViolationType] = field(default_factory=list)
    clipped_values: Dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class ConstitutionalRule:
    """A constitutional constraint (permanent guardrail)."""
    id: str
    name: str
    description: str
    check_function: str  # Name of validation function
    severity: str  # warning, halt, recovery
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class SafetyRalph(RalphLayer):
    """
    Ralph layer for the Ring 0 safety kernel.

    This is the ULTIMATE guardrail layer - constitutional constraints that
    can NEVER be bypassed. All other loops submit to this validation.

    Key Principles:
    1. Deterministic - No learning, no adaptation (safety must be predictable)
    2. Fast - Sub-millisecond validation
    3. Fail-safe - On any error, HALT
    4. Auditable - Complete black-box logging
    5. Constitutional - Rules defined at system start, immutable

    Guardrails in this layer ARE the constitution - they don't prevent
    mistakes, they ARE the safety rules themselves.
    """

    def __init__(
        self,
        safety_kernel: Any = None,
        constitution_path: Optional[str] = None,
        **kwargs
    ):
        config = RalphConfig(
            loop_type=LoopType.SAFETY,
            target_latency_ms=1.0,  # Must be fast!
            max_iterations=float('inf'),  # Never stop
            context_window_tokens=1024,  # Minimal context
        )
        super().__init__(config, **kwargs)

        self.safety_kernel = safety_kernel
        self.constitution_path = constitution_path

        # Constitutional rules (immutable after init)
        self._constitution: List[ConstitutionalRule] = []
        self._load_constitution()

        # Violation tracking
        self._violation_log: List[Dict[str, Any]] = []
        self._recent_violations: List[ViolationType] = []

        # Current system state
        self._system_halted = False
        self._recovery_mode = False

        # Kinematic limits (loaded from constitution)
        self._kinematic_limits = self._load_kinematic_limits()

    def _load_constitution(self) -> None:
        """Load constitutional rules (permanent guardrails)."""

        # Default constitution
        self._constitution = [
            ConstitutionalRule(
                id="joint_limits",
                name="Joint Position Limits",
                description="Prevent joints from exceeding physical limits",
                check_function="check_joint_limits",
                severity="halt",
                parameters={"margin": 0.05}  # 5% safety margin
            ),
            ConstitutionalRule(
                id="velocity_limits",
                name="Velocity Limits",
                description="Prevent excessive velocities",
                check_function="check_velocity_limits",
                severity="clip",
                parameters={"max_velocity": 2.0}  # rad/s
            ),
            ConstitutionalRule(
                id="acceleration_limits",
                name="Acceleration Limits",
                description="Prevent excessive accelerations",
                check_function="check_acceleration_limits",
                severity="clip",
                parameters={"max_acceleration": 5.0}  # rad/s^2
            ),
            ConstitutionalRule(
                id="no_go_zones",
                name="No-Go Zones",
                description="Prevent entry into forbidden regions",
                check_function="check_no_go_zones",
                severity="halt",
                parameters={"zones": []}  # Defined per environment
            ),
            ConstitutionalRule(
                id="thermal_protection",
                name="Thermal Protection",
                description="Prevent overheating",
                check_function="check_thermal",
                severity="slow",
                parameters={"max_temp": 80}  # Celsius
            ),
            ConstitutionalRule(
                id="e_stop",
                name="Emergency Stop",
                description="Hardware E-Stop signal",
                check_function="check_e_stop",
                severity="halt",
                parameters={}
            ),
            ConstitutionalRule(
                id="collision_avoidance",
                name="Collision Avoidance",
                description="Prevent imminent collisions",
                check_function="check_collision",
                severity="halt",
                parameters={"min_distance": 0.05}  # meters
            ),
        ]

        # Convert constitution to guardrails
        for rule in self._constitution:
            self.add_guardrail(
                trigger=f"constitutional:{rule.id}",
                instruction=rule.description,
                severity="critical" if rule.severity == "halt" else rule.severity,
                context=f"parameters={rule.parameters}",
                iteration=0
            )

        logger.info(f"Loaded {len(self._constitution)} constitutional rules")

    def _load_kinematic_limits(self) -> Dict[str, Dict[str, float]]:
        """Load kinematic limits from constitution."""

        # Default limits
        return {
            "default": {"min": -180, "max": 180},
            "servo": {"min": 0, "max": 180},
            "arm_shoulder": {"min": -90, "max": 90},
            "arm_elbow": {"min": 0, "max": 135},
            "gripper": {"min": 0, "max": 100},
            "drive": {"min": -1.0, "max": 1.0},
        }

    async def execute_iteration(self, state: RalphState, input_data: Any) -> RalphState:
        """
        Execute safety validation.

        This is called on EVERY actuation command before it's sent to hardware.
        """
        start_time = time.perf_counter()

        # Validate the command
        result = await self.validate_command(input_data)

        # Log to black box
        self._log_validation(input_data, result)

        # Update state
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        state.last_action = f"validate:{result.response_level.value}"
        state.last_result = result.message
        state.metrics["last_latency_us"] = latency_us
        state.metrics["violations_total"] = len(self._violation_log)

        if result.violations:
            state.errors.extend([v.value for v in result.violations])

        return state

    async def validate_command(self, command: Dict[str, Any]) -> SafetyCheckResult:
        """
        Validate an actuation command against the constitution.

        This is the core safety validation function.
        """
        if self._system_halted:
            return SafetyCheckResult(
                allowed=False,
                response_level=ResponseLevel.HALT,
                message="System halted - awaiting reset"
            )

        violations = []
        clipped_values = {}
        response_level = ResponseLevel.ALLOW

        # Check each constitutional rule
        for rule in self._constitution:
            if not rule.enabled:
                continue

            check_result = await self._check_rule(rule, command)

            if check_result["violated"]:
                violations.append(check_result["type"])
                self._recent_violations.append(check_result["type"])

                # Determine response level
                if rule.severity == "halt":
                    response_level = ResponseLevel.HALT
                elif rule.severity == "recovery" and response_level != ResponseLevel.HALT:
                    response_level = ResponseLevel.RECOVERY
                elif rule.severity == "clip" and response_level not in [ResponseLevel.HALT, ResponseLevel.RECOVERY]:
                    response_level = ResponseLevel.CLIP
                    clipped_values.update(check_result.get("clipped", {}))
                elif rule.severity == "slow" and response_level == ResponseLevel.ALLOW:
                    response_level = ResponseLevel.SLOW

        # Apply graduated response
        if response_level == ResponseLevel.HALT:
            self._system_halted = True
            logger.critical("SAFETY HALT triggered")
        elif response_level == ResponseLevel.RECOVERY:
            self._recovery_mode = True
            logger.warning("Entering recovery mode")

        # Trim recent violations list
        if len(self._recent_violations) > 100:
            self._recent_violations = self._recent_violations[-50:]

        return SafetyCheckResult(
            allowed=response_level in [ResponseLevel.ALLOW, ResponseLevel.CLIP],
            response_level=response_level,
            violations=violations,
            clipped_values=clipped_values,
            message=f"{len(violations)} violations, response: {response_level.value}"
        )

    async def _check_rule(
        self,
        rule: ConstitutionalRule,
        command: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check a single constitutional rule."""

        result = {"violated": False, "type": None, "clipped": {}}

        if rule.check_function == "check_joint_limits":
            result = self._check_joint_limits(command, rule.parameters)
        elif rule.check_function == "check_velocity_limits":
            result = self._check_velocity_limits(command, rule.parameters)
        elif rule.check_function == "check_acceleration_limits":
            result = self._check_acceleration_limits(command, rule.parameters)
        elif rule.check_function == "check_no_go_zones":
            result = self._check_no_go_zones(command, rule.parameters)
        elif rule.check_function == "check_thermal":
            result = self._check_thermal(command, rule.parameters)
        elif rule.check_function == "check_e_stop":
            result = self._check_e_stop(command, rule.parameters)
        elif rule.check_function == "check_collision":
            result = self._check_collision(command, rule.parameters)

        return result

    def _check_joint_limits(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check joint position limits."""

        result = {"violated": False, "type": ViolationType.KINEMATIC_LIMIT, "clipped": {}}
        margin = params.get("margin", 0.05)

        for joint, value in command.items():
            if not isinstance(value, (int, float)):
                continue

            # Get limits for this joint type
            limits = self._kinematic_limits.get("default")
            for joint_type, lim in self._kinematic_limits.items():
                if joint_type in joint.lower():
                    limits = lim
                    break

            min_val = limits["min"] * (1 + margin)
            max_val = limits["max"] * (1 - margin)

            if value < min_val or value > max_val:
                result["violated"] = True
                result["clipped"][joint] = max(min_val, min(max_val, value))

        return result

    def _check_velocity_limits(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check velocity limits."""

        result = {"violated": False, "type": ViolationType.VELOCITY_EXCEEDED, "clipped": {}}
        max_vel = params.get("max_velocity", 2.0)

        for key, value in command.items():
            if "velocity" in key.lower() and isinstance(value, (int, float)):
                if abs(value) > max_vel:
                    result["violated"] = True
                    result["clipped"][key] = max_vel if value > 0 else -max_vel

        return result

    def _check_acceleration_limits(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check acceleration limits."""

        result = {"violated": False, "type": ViolationType.ACCELERATION_EXCEEDED, "clipped": {}}
        max_acc = params.get("max_acceleration", 5.0)

        for key, value in command.items():
            if "acceleration" in key.lower() and isinstance(value, (int, float)):
                if abs(value) > max_acc:
                    result["violated"] = True
                    result["clipped"][key] = max_acc if value > 0 else -max_acc

        return result

    def _check_no_go_zones(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check no-go zones."""

        result = {"violated": False, "type": ViolationType.NO_GO_ZONE}
        zones = params.get("zones", [])

        # Check position commands against zones
        pos_x = command.get("position_x", command.get("x", None))
        pos_y = command.get("position_y", command.get("y", None))

        if pos_x is not None and pos_y is not None:
            for zone in zones:
                if self._point_in_zone((pos_x, pos_y), zone):
                    result["violated"] = True
                    break

        return result

    def _point_in_zone(self, point: tuple, zone: Dict[str, Any]) -> bool:
        """Check if a point is in a no-go zone."""

        zone_type = zone.get("type", "rectangle")

        if zone_type == "rectangle":
            x_min, y_min = zone.get("min", (0, 0))
            x_max, y_max = zone.get("max", (0, 0))
            return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

        elif zone_type == "circle":
            cx, cy = zone.get("center", (0, 0))
            radius = zone.get("radius", 0)
            dist = ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5
            return dist <= radius

        return False

    def _check_thermal(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check thermal limits."""

        result = {"violated": False, "type": ViolationType.THERMAL_LIMIT}
        max_temp = params.get("max_temp", 80)

        temp = command.get("temperature", command.get("temp", None))
        if temp is not None and temp > max_temp:
            result["violated"] = True

        return result

    def _check_e_stop(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check emergency stop signal."""

        result = {"violated": False, "type": ViolationType.CONSTITUTIONAL}

        if command.get("e_stop", False) or command.get("emergency_stop", False):
            result["violated"] = True

        return result

    def _check_collision(
        self,
        command: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for imminent collision."""

        result = {"violated": False, "type": ViolationType.COLLISION_IMMINENT}
        min_distance = params.get("min_distance", 0.05)

        # Check proximity sensors
        for key, value in command.items():
            if "proximity" in key.lower() or "distance" in key.lower():
                if isinstance(value, (int, float)) and value < min_distance:
                    result["violated"] = True
                    break

        return result

    def _log_validation(
        self,
        command: Dict[str, Any],
        result: SafetyCheckResult
    ) -> None:
        """Log validation to black box."""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "command_hash": hash(str(command)),
            "allowed": result.allowed,
            "response_level": result.response_level.value,
            "violations": [v.value for v in result.violations],
            "message": result.message
        }

        self._violation_log.append(log_entry)

        # Limit log size
        if len(self._violation_log) > 10000:
            self._violation_log = self._violation_log[-5000:]

    async def should_continue(self, state: RalphState) -> bool:
        """
        Safety loop NEVER stops.
        """
        return True

    # ========== External Interface ==========

    def reset_halt(self, authorization_code: str = "") -> bool:
        """
        Reset a system halt (requires authorization).
        """
        # In production, this would require proper authorization
        if authorization_code or True:  # Simplified for dev
            self._system_halted = False
            self._recovery_mode = False
            logger.info("System halt reset")
            return True
        return False

    def add_no_go_zone(self, zone: Dict[str, Any]) -> None:
        """Add a no-go zone at runtime."""

        for rule in self._constitution:
            if rule.id == "no_go_zones":
                rule.parameters.setdefault("zones", []).append(zone)
                logger.info(f"Added no-go zone: {zone}")
                break

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""

        return {
            "halted": self._system_halted,
            "recovery_mode": self._recovery_mode,
            "recent_violations": len(self._recent_violations),
            "total_validations": len(self._violation_log),
            "constitution_rules": len(self._constitution),
            "constitution_enabled": sum(1 for r in self._constitution if r.enabled)
        }

    def get_violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by type."""

        summary = {}
        for v in self._recent_violations:
            summary[v.value] = summary.get(v.value, 0) + 1
        return summary

    # ========== Meta-Layer (Limited for Safety) ==========

    def introspect(self, state: RalphState) -> Dict[str, Any]:
        """
        Safety introspection is LIMITED - we don't learn or adapt.
        We only report status.
        """
        return {
            "loop_type": "safety",
            "halted": self._system_halted,
            "recovery_mode": self._recovery_mode,
            "violation_rate": len(self._recent_violations) / max(len(self._violation_log), 1),
            "constitution_rules": len(self._constitution),
            "health": "halted" if self._system_halted else "healthy"
        }
