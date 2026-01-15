"""
Sandbox Manager - Lifecycle management for isolated agent environments.

Based on Anthropic's Claude Cowork pattern:
- Structural isolation (not behavioral trust)
- Explicit allow/deny lists
- Resource budgets
- Audit logging
"""

from dataclasses import dataclass, field
from typing import Any
import time


class SandboxViolation(Exception):
    """Raised when sandbox rules are violated."""

    def __init__(self, message: str, violation_type: str = "unknown"):
        super().__init__(message)
        self.violation_type = violation_type
        self.timestamp = time.time()


@dataclass
class SandboxConfig:
    """
    Configuration for an agent sandbox.

    Hardware Access:
    - Actuators use ALLOW-list (default deny)
    - Sensors use DENY-list (default allow)

    Network Access:
    - All domains use ALLOW-list (default deny)

    This matches the Cowork pattern:
    - Write = allow-list (explicit permits)
    - Read = deny-list (block sensitive)
    """

    # Hardware: Actuators (allow-list)
    allowed_actuators: list[str] = field(
        default_factory=lambda: ["motor_left", "motor_right"]
    )
    denied_actuators: list[str] = field(default_factory=list)

    # Hardware: Sensors (deny-list)
    allowed_sensors: list[str] | None = None  # None = allow all except denied
    denied_sensors: list[str] = field(
        default_factory=lambda: ["microphone"]  # Privacy by default
    )

    # Network (allow-list)
    allowed_domains: list[str] = field(
        default_factory=lambda: [
            "api.anthropic.com",
            "api.openai.com",
            "generativelanguage.googleapis.com",
        ]
    )
    denied_domains: list[str] = field(default_factory=list)

    # Resource limits
    max_speed: float = 0.5
    max_acceleration: float = 0.2  # per second
    max_actions_per_second: int = 10
    timeout_seconds: float = 300.0
    max_api_calls: int = 100
    max_continuous_motion_seconds: float = 30.0


@dataclass
class SandboxStats:
    """Runtime statistics for a sandbox."""

    action_count: int = 0
    api_call_count: int = 0
    violation_count: int = 0
    uptime_seconds: float = 0.0


class Sandbox:
    """
    An isolated execution environment for an agent.

    All hardware and network access goes through gates
    that enforce the sandbox configuration.

    Usage:
        sandbox = Sandbox("agent_1", SandboxConfig())

        # Before any actuator access:
        sandbox.check_actuator("motor_left")

        # Before any sensor access:
        sandbox.check_sensor("camera")

        # Before any network access:
        sandbox.check_network("api.anthropic.com")

        # Rate limiting:
        sandbox.check_rate_limit()

        # Speed clamping:
        safe_speed = sandbox.clamp_speed(requested_speed)
    """

    def __init__(self, sandbox_id: str, config: SandboxConfig):
        self.id = sandbox_id
        self.config = config
        self.created_at = time.time()

        # Counters
        self._action_count = 0
        self._api_call_count = 0
        self._violation_count = 0

        # Rate limiting
        self._last_action_time = 0.0
        self._motion_start_time: float | None = None

        # State
        self._active = True
        self._last_speed = 0.0

        # Audit log
        self._audit_log: list[dict] = []

    # === Actuator Checks (Allow-list) ===

    def check_actuator(self, name: str) -> bool:
        """
        Check if actuator access is allowed.

        Actuators use ALLOW-list: must be explicitly permitted.
        """
        self._ensure_active()

        # Explicit deny takes precedence
        if name in self.config.denied_actuators:
            self._log_violation("actuator_denied", name)
            raise SandboxViolation(
                f"Actuator '{name}' is explicitly denied",
                violation_type="actuator_denied",
            )

        # Must be in allow-list
        if name not in self.config.allowed_actuators:
            self._log_violation("actuator_not_allowed", name)
            raise SandboxViolation(
                f"Actuator '{name}' is not in allow-list",
                violation_type="actuator_not_allowed",
            )

        return True

    # === Sensor Checks (Deny-list) ===

    def check_sensor(self, name: str) -> bool:
        """
        Check if sensor access is allowed.

        Sensors use DENY-list: allowed by default, block specific.
        """
        self._ensure_active()

        # Check explicit deny
        if name in self.config.denied_sensors:
            self._log_violation("sensor_denied", name)
            raise SandboxViolation(
                f"Sensor '{name}' is explicitly denied",
                violation_type="sensor_denied",
            )

        # If allow-list is specified, check it
        if self.config.allowed_sensors is not None:
            if name not in self.config.allowed_sensors:
                self._log_violation("sensor_not_allowed", name)
                raise SandboxViolation(
                    f"Sensor '{name}' is not in allow-list",
                    violation_type="sensor_not_allowed",
                )

        return True

    # === Network Checks (Allow-list) ===

    def check_network(self, domain: str) -> bool:
        """
        Check if network access is allowed.

        Network uses ALLOW-list: must be explicitly permitted.
        """
        self._ensure_active()

        # Explicit deny takes precedence
        if domain in self.config.denied_domains:
            self._log_violation("domain_denied", domain)
            raise SandboxViolation(
                f"Domain '{domain}' is explicitly denied",
                violation_type="domain_denied",
            )

        # Must be in allow-list
        if domain not in self.config.allowed_domains:
            self._log_violation("domain_not_allowed", domain)
            raise SandboxViolation(
                f"Domain '{domain}' is not in allow-list",
                violation_type="domain_not_allowed",
            )

        # Check API call budget
        self._api_call_count += 1
        if self._api_call_count > self.config.max_api_calls:
            self._log_violation("api_limit_exceeded", domain)
            raise SandboxViolation(
                f"API call limit ({self.config.max_api_calls}) exceeded",
                violation_type="api_limit_exceeded",
            )

        return True

    # === Rate Limiting ===

    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        self._ensure_active()
        now = time.time()

        # Check overall timeout
        elapsed = now - self.created_at
        if elapsed > self.config.timeout_seconds:
            self._active = False
            self._log_violation("timeout", f"{elapsed:.1f}s")
            raise SandboxViolation(
                f"Sandbox timeout ({self.config.timeout_seconds}s) exceeded",
                violation_type="timeout",
            )

        # Check action rate limit
        if self._last_action_time > 0:
            since_last = now - self._last_action_time
            min_interval = 1.0 / self.config.max_actions_per_second
            if since_last < min_interval:
                self._log_violation("rate_limit", f"{since_last:.3f}s")
                raise SandboxViolation(
                    f"Rate limit exceeded ({self.config.max_actions_per_second}/s)",
                    violation_type="rate_limit",
                )

        self._last_action_time = now
        self._action_count += 1
        return True

    def check_continuous_motion(self, is_moving: bool) -> bool:
        """Check continuous motion limits."""
        now = time.time()

        if is_moving:
            if self._motion_start_time is None:
                self._motion_start_time = now
            else:
                duration = now - self._motion_start_time
                if duration > self.config.max_continuous_motion_seconds:
                    self._log_violation("continuous_motion", f"{duration:.1f}s")
                    raise SandboxViolation(
                        f"Continuous motion limit ({self.config.max_continuous_motion_seconds}s) exceeded",
                        violation_type="continuous_motion",
                    )
        else:
            self._motion_start_time = None

        return True

    # === Value Clamping ===

    def clamp_speed(self, speed: float) -> float:
        """Clamp speed to sandbox limits."""
        clamped = max(-self.config.max_speed, min(self.config.max_speed, speed))

        # Check acceleration
        if self._last_speed != 0:
            delta = abs(clamped - self._last_speed)
            if delta > self.config.max_acceleration:
                # Limit acceleration
                direction = 1 if clamped > self._last_speed else -1
                clamped = self._last_speed + (direction * self.config.max_acceleration)
                clamped = max(-self.config.max_speed, min(self.config.max_speed, clamped))

        self._last_speed = clamped
        return clamped

    # === Lifecycle ===

    def _ensure_active(self):
        """Ensure sandbox is still active."""
        if not self._active:
            raise SandboxViolation(
                "Sandbox is not active",
                violation_type="inactive",
            )

    def destroy(self):
        """Deactivate the sandbox."""
        self._active = False
        self._audit_log.append({
            "timestamp": time.time(),
            "type": "lifecycle",
            "event": "destroyed",
        })

    @property
    def is_active(self) -> bool:
        """Check if sandbox is active."""
        return self._active

    # === Audit Logging ===

    def _log_violation(self, violation_type: str, target: str):
        """Log a violation."""
        self._violation_count += 1
        self._audit_log.append({
            "timestamp": time.time(),
            "type": "violation",
            "violation_type": violation_type,
            "target": target,
        })

    def get_audit_log(self) -> list[dict]:
        """Get the full audit log."""
        return self._audit_log.copy()

    def get_stats(self) -> SandboxStats:
        """Get runtime statistics."""
        return SandboxStats(
            action_count=self._action_count,
            api_call_count=self._api_call_count,
            violation_count=self._violation_count,
            uptime_seconds=time.time() - self.created_at,
        )


class SandboxManager:
    """
    Manages sandbox lifecycle.

    Usage:
        manager = SandboxManager()

        # Create with default config
        sandbox = manager.create()

        # Create with custom config
        sandbox = manager.create(SandboxConfig(
            allowed_actuators=["motor_left", "motor_right", "arm_gripper"],
            max_speed=0.3,
        ))

        # Destroy when done
        manager.destroy(sandbox.id)

        # Or destroy all
        manager.destroy_all()
    """

    def __init__(self):
        self.sandboxes: dict[str, Sandbox] = {}
        self._counter = 0

    def create(
        self,
        config: SandboxConfig | None = None,
        sandbox_id: str | None = None,
    ) -> Sandbox:
        """Create a new sandbox."""
        self._counter += 1

        if sandbox_id is None:
            sandbox_id = f"sandbox_{self._counter}_{int(time.time())}"

        sandbox = Sandbox(sandbox_id, config or SandboxConfig())
        self.sandboxes[sandbox_id] = sandbox

        return sandbox

    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox by ID."""
        return self.sandboxes.get(sandbox_id)

    def list(self) -> list[Sandbox]:
        """List all sandboxes."""
        return list(self.sandboxes.values())

    def destroy(self, sandbox_id: str) -> bool:
        """Destroy a sandbox."""
        if sandbox_id in self.sandboxes:
            self.sandboxes[sandbox_id].destroy()
            del self.sandboxes[sandbox_id]
            return True
        return False

    def destroy_all(self):
        """Destroy all sandboxes."""
        for sandbox in self.sandboxes.values():
            sandbox.destroy()
        self.sandboxes.clear()

    def get_stats(self) -> dict[str, SandboxStats]:
        """Get stats for all sandboxes."""
        return {
            sandbox_id: sandbox.get_stats()
            for sandbox_id, sandbox in self.sandboxes.items()
        }
