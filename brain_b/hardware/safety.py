"""
Safety Monitor - Basic safety checks for motor operations.

This is intentionally simple. It's not a full safety kernel like the
original ContinuonBrain - just enough to prevent obvious problems.
"""

from dataclasses import dataclass
import time

from .motors import BaseMotorController, MotorState


@dataclass
class SafetyLimits:
    """Safety limits for motor operations."""

    max_speed: float = 1.0  # Maximum allowed speed
    max_runtime_seconds: float = 30.0  # Auto-stop after this duration
    watchdog_timeout: float = 2.0  # Stop if no commands for this long


class SafetyViolation(Exception):
    """Raised when a safety check fails."""

    pass


class SafetyMonitor:
    """
    Simple safety monitor for motor operations.

    Features:
    - Maximum speed limiting
    - Automatic stop after timeout
    - Watchdog timer (stop if no commands)
    - Emergency stop
    """

    def __init__(
        self,
        controller: BaseMotorController,
        limits: SafetyLimits | None = None,
    ):
        self.controller = controller
        self.limits = limits or SafetyLimits()

        self._last_command_time = time.time()
        self._run_start_time: float | None = None
        self._emergency_stopped = False

    def check(self):
        """
        Run safety checks. Call this before every motor command.

        Raises SafetyViolation if a check fails.
        """
        now = time.time()

        # Check emergency stop
        if self._emergency_stopped:
            raise SafetyViolation("Emergency stop active. Call reset() to clear.")

        # Check watchdog
        if now - self._last_command_time > self.limits.watchdog_timeout:
            state = self.controller.get_state()
            if state.left_speed != 0 or state.right_speed != 0:
                self.controller.stop()
                print("[Safety] Watchdog timeout - motors stopped")

        # Check max runtime
        if self._run_start_time:
            runtime = now - self._run_start_time
            if runtime > self.limits.max_runtime_seconds:
                self.controller.stop()
                self._run_start_time = None
                raise SafetyViolation(f"Max runtime ({self.limits.max_runtime_seconds}s) exceeded")

        self._last_command_time = now

    def clamp_speed(self, speed: float) -> float:
        """Clamp speed to safety limits."""
        return max(-self.limits.max_speed, min(self.limits.max_speed, speed))

    def start_run(self):
        """Mark the start of a continuous run (for runtime limiting)."""
        self._run_start_time = time.time()

    def end_run(self):
        """Mark the end of a continuous run."""
        self._run_start_time = None

    def emergency_stop(self):
        """Trigger emergency stop."""
        self._emergency_stopped = True
        self.controller.stop()
        print("[Safety] EMERGENCY STOP")

    def reset(self):
        """Reset safety state (clear emergency stop)."""
        self._emergency_stopped = False
        self._run_start_time = None
        print("[Safety] Reset")

    def is_safe(self) -> bool:
        """Check if system is in a safe state."""
        try:
            self.check()
            return True
        except SafetyViolation:
            return False

    def wrap_executor(self, executor):
        """
        Wrap an executor function with safety checks.

        Usage:
            safe_executor = safety.wrap_executor(executor)
            runtime.execute_action(action, safe_executor)
        """

        def safe_execute(action: dict):
            try:
                self.check()

                # Clamp any speed values
                if "speed" in action:
                    action["speed"] = self.clamp_speed(action["speed"])

                executor(action)

            except SafetyViolation as e:
                print(f"[Safety] Blocked: {e}")
                self.controller.stop()

        return safe_execute
