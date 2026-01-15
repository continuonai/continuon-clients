"""
Hardware Gate - Enforces sandbox rules on motor/sensor access.

This is the robot equivalent of Cowork's filesystem proxy.
All hardware access goes through this gate, which enforces
allow/deny lists and logs all access attempts.
"""

from typing import Callable, Any
import time

from .manager import Sandbox, SandboxViolation


class HardwareGate:
    """
    Gate that enforces sandbox rules on hardware access.

    Wraps motor controller and sensor interfaces to enforce
    allow/deny lists, resource limits, and audit logging.

    Usage:
        sandbox = Sandbox("agent_1", SandboxConfig())
        gate = HardwareGate(sandbox, motor_controller)

        # All motor access goes through the gate
        gate.set_motor("motor_left", 0.5)

        # All sensor access goes through the gate
        value = gate.read_sensor("camera")

        # Emergency stop always works
        gate.emergency_stop()
    """

    def __init__(
        self,
        sandbox: Sandbox,
        motor_controller: Any,
        sensor_manager: Any = None,
    ):
        self.sandbox = sandbox
        self.motors = motor_controller
        self.sensors = sensor_manager

        # Audit log
        self._audit_log: list[dict] = []

    def set_motor(self, name: str, speed: float) -> float:
        """
        Set motor speed through the gate.

        Returns the actual speed after clamping.
        """
        entry = {
            "timestamp": time.time(),
            "type": "motor_set",
            "name": name,
            "requested_speed": speed,
        }

        try:
            # Check sandbox permissions
            self.sandbox.check_actuator(name)
            self.sandbox.check_rate_limit()

            # Check if this counts as motion
            is_moving = abs(speed) > 0.01
            self.sandbox.check_continuous_motion(is_moving)

            # Clamp speed to limits
            clamped = self.sandbox.clamp_speed(speed)

            # Execute on actual hardware
            self._execute_motor(name, clamped)

            # Log success
            entry["actual_speed"] = clamped
            entry["allowed"] = True
            self._audit_log.append(entry)

            return clamped

        except SandboxViolation as e:
            # Log failure
            entry["allowed"] = False
            entry["violation"] = str(e)
            entry["violation_type"] = e.violation_type
            self._audit_log.append(entry)

            # Stop motors on violation
            self._safe_stop()
            raise

    def _execute_motor(self, name: str, speed: float):
        """Execute motor command on actual hardware."""
        if hasattr(self.motors, "set_motor"):
            self.motors.set_motor(name, speed)
        elif name == "motor_left" and hasattr(self.motors, "set_left"):
            self.motors.set_left(speed)
        elif name == "motor_right" and hasattr(self.motors, "set_right"):
            self.motors.set_right(speed)
        elif hasattr(self.motors, "set_speed"):
            # Generic interface
            if name == "motor_left":
                self.motors.set_speed(speed, None)
            elif name == "motor_right":
                self.motors.set_speed(None, speed)
        else:
            raise SandboxViolation(
                f"Motor controller doesn't support '{name}'",
                violation_type="unsupported_actuator",
            )

    def set_motors(self, left: float, right: float) -> tuple[float, float]:
        """Set both motors at once."""
        left_actual = self.set_motor("motor_left", left)
        right_actual = self.set_motor("motor_right", right)
        return left_actual, right_actual

    def read_sensor(self, name: str) -> Any:
        """
        Read sensor value through the gate.
        """
        entry = {
            "timestamp": time.time(),
            "type": "sensor_read",
            "name": name,
        }

        try:
            # Check sandbox permissions
            self.sandbox.check_sensor(name)

            # Execute on actual hardware
            if self.sensors is None:
                value = None
            elif hasattr(self.sensors, "read"):
                value = self.sensors.read(name)
            elif hasattr(self.sensors, f"get_{name}"):
                value = getattr(self.sensors, f"get_{name}")()
            else:
                value = None

            # Log success
            entry["allowed"] = True
            entry["has_value"] = value is not None
            self._audit_log.append(entry)

            return value

        except SandboxViolation as e:
            # Log failure
            entry["allowed"] = False
            entry["violation"] = str(e)
            entry["violation_type"] = e.violation_type
            self._audit_log.append(entry)
            raise

    def emergency_stop(self):
        """
        Emergency stop - ALWAYS allowed, bypasses sandbox.

        This is the one operation that cannot be blocked.
        """
        self._safe_stop()

        self._audit_log.append({
            "timestamp": time.time(),
            "type": "emergency_stop",
            "allowed": True,
            "note": "Emergency stop bypasses sandbox",
        })

    def _safe_stop(self):
        """Stop motors safely."""
        try:
            if hasattr(self.motors, "stop"):
                self.motors.stop()
            elif hasattr(self.motors, "set_speed"):
                self.motors.set_speed(0, 0)
        except Exception:
            pass  # Best effort

    def get_audit_log(self) -> list[dict]:
        """Get the hardware access audit log."""
        return self._audit_log.copy()

    def get_stats(self) -> dict:
        """Get hardware gate statistics."""
        total = len(self._audit_log)
        allowed = sum(1 for e in self._audit_log if e.get("allowed", False))
        denied = total - allowed

        return {
            "total_requests": total,
            "allowed": allowed,
            "denied": denied,
            "motor_requests": sum(1 for e in self._audit_log if e["type"] == "motor_set"),
            "sensor_requests": sum(1 for e in self._audit_log if e["type"] == "sensor_read"),
            "emergency_stops": sum(1 for e in self._audit_log if e["type"] == "emergency_stop"),
        }


def create_gated_executor(gate: HardwareGate) -> Callable[[dict], None]:
    """
    Create an executor function that routes through the hardware gate.

    This integrates with the actor_runtime's execute_action().

    Usage:
        gate = HardwareGate(sandbox, motors)
        executor = create_gated_executor(gate)
        runtime.execute_action({"type": "forward", "speed": 0.5}, executor)
    """

    def executor(action: dict):
        action_type = action.get("type", "")
        speed = action.get("speed", 0.5)

        if action_type == "forward":
            gate.set_motors(speed, speed)

        elif action_type == "backward":
            gate.set_motors(-speed, -speed)

        elif action_type == "left":
            gate.set_motors(-speed * 0.5, speed * 0.5)

        elif action_type == "right":
            gate.set_motors(speed * 0.5, -speed * 0.5)

        elif action_type == "stop":
            gate.set_motors(0, 0)

        elif action_type == "motor":
            # Direct motor control
            name = action.get("name", "")
            gate.set_motor(name, speed)

        elif action_type == "emergency_stop":
            gate.emergency_stop()

        else:
            # Unknown action type - stop for safety
            gate.set_motors(0, 0)

    return executor
