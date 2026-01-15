# Sandboxed Agent Runtime Pattern

## Status: REFERENCE ARCHITECTURE
## Source: Reverse-engineered from Anthropic's Claude Cowork
## Date: 2026-01-15

---

## Overview

This document describes the **Sandboxed Agent Runtime Pattern** - a reusable architecture for running AI agents with hard isolation guarantees. This pattern was pioneered by Anthropic in Claude Cowork and can be applied to any system where untrusted or semi-trusted agents need to interact with host resources.

---

## The Problem

AI agents need to:
- Read/write files
- Execute commands
- Make network requests
- Interact with hardware

But you can't trust them completely. A bug, hallucination, or adversarial prompt could:
- Delete important files
- Exfiltrate secrets
- Run malicious commands
- Escape to the host system

**The solution**: Structural isolation, not behavioral trust.

---

## How Anthropic Built It (Cowork Architecture)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HOST SYSTEM (macOS/Linux)                    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    SANDBOX BOUNDARY                      │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              ISOLATED ENVIRONMENT                │    │    │
│  │  │                                                  │    │    │
│  │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │    │
│  │  │   │  Agent   │  │  Tools   │  │ Runtime  │     │    │    │
│  │  │   │ (Claude) │  │(git,grep)│  │ (Node)   │     │    │    │
│  │  │   └────┬─────┘  └────┬─────┘  └────┬─────┘     │    │    │
│  │  │        │             │             │           │    │    │
│  │  │        └─────────────┼─────────────┘           │    │    │
│  │  │                      │                         │    │    │
│  │  │              ┌───────▼───────┐                 │    │    │
│  │  │              │ Sandbox Manager│                 │    │    │
│  │  │              └───────┬───────┘                 │    │    │
│  │  │                      │                         │    │    │
│  │  └──────────────────────┼─────────────────────────┘    │    │
│  │                         │                              │    │
│  │    ┌────────────────────┼────────────────────┐        │    │
│  │    │                    │                    │        │    │
│  │    ▼                    ▼                    ▼        │    │
│  │ ┌──────┐          ┌──────────┐         ┌────────┐    │    │
│  │ │FS    │          │ Network  │         │ Mount  │    │    │
│  │ │Proxy │          │ Proxy    │         │ Points │    │    │
│  │ └──┬───┘          └────┬─────┘         └───┬────┘    │    │
│  │    │                   │                   │         │    │
│  └────┼───────────────────┼───────────────────┼─────────┘    │
│       │                   │                   │              │
│       ▼                   ▼                   ▼              │
│  [Allowed Paths]    [Allowed Domains]   [Mounted Folders]   │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Platform-Specific Implementation

| Platform | Isolation Technology | Network | Filesystem |
|----------|---------------------|---------|------------|
| **macOS** | VZVirtualMachine (Apple Virtualization Framework) | Localhost proxy port | Seatbelt profiles |
| **Linux** | Bubblewrap (namespace isolation) | Unix domain socket | Bind mounts |
| **Cloud** | gVisor (syscall interception) | Network policies | Container volumes |

### Key Components

#### 1. Sandbox Manager
The orchestrator that sets up and tears down isolated environments.

```typescript
// Conceptual interface
interface SandboxManager {
  // Lifecycle
  create(config: SandboxConfig): Sandbox;
  destroy(sandbox: Sandbox): void;

  // Execution
  exec(sandbox: Sandbox, command: string): ExecResult;

  // Resource management
  mount(sandbox: Sandbox, hostPath: string, guestPath: string): void;
  unmount(sandbox: Sandbox, guestPath: string): void;
}
```

#### 2. Filesystem Proxy
Controls what the agent can read/write.

```
┌─────────────────────────────────────────┐
│           FILESYSTEM RULES              │
├─────────────────────────────────────────┤
│ READ:  deny-only (block specific paths) │
│        Default: allow all               │
│        Deny: ~/.ssh, ~/.aws, /etc/shadow│
│                                         │
│ WRITE: allow-only (permit specific)     │
│        Default: deny all                │
│        Allow: ./workspace, /tmp         │
└─────────────────────────────────────────┘
```

#### 3. Network Proxy
Controls what domains the agent can reach.

```
┌─────────────────────────────────────────┐
│           NETWORK RULES                 │
├─────────────────────────────────────────┤
│ HTTP/HTTPS: via HTTP proxy              │
│ TCP/Other:  via SOCKS5 proxy            │
│                                         │
│ Default: deny all                       │
│ Allow:   api.anthropic.com              │
│          github.com                     │
│          pypi.org                       │
│                                         │
│ Deny:    * (everything else)            │
└─────────────────────────────────────────┘
```

#### 4. Mount Points
User-controlled folders shared with the sandbox.

```
Host                          Guest (Sandbox)
─────────────────────────────────────────────
~/Projects/myapp      →       /workspace
~/Downloads           →       /downloads (read-only)
(nothing else)                (nothing else)
```

---

## The Generalized Pattern

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Structural Isolation** | Security from architecture, not behavior |
| **Explicit Mounts** | Only see what you're given |
| **Proxy Everything** | All I/O goes through controlled gates |
| **Fail Closed** | Default deny, explicit allow |
| **Process Tree Inheritance** | Children inherit parent's restrictions |

### Pattern Template

```
┌────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                         │
│                                                              │
│   1. SANDBOX BOUNDARY                                        │
│      └─ Technology: VM / Namespace / Container / WASM       │
│                                                              │
│   2. ISOLATION MANAGER                                       │
│      └─ Lifecycle: create, exec, destroy                    │
│      └─ Mounts: explicit paths only                         │
│                                                              │
│   3. FILESYSTEM GATE                                         │
│      └─ Read: deny-list (block secrets)                     │
│      └─ Write: allow-list (explicit paths only)             │
│                                                              │
│   4. NETWORK GATE                                            │
│      └─ Proxy: HTTP + SOCKS5                                │
│      └─ Allow-list: explicit domains only                   │
│                                                              │
│   5. RESOURCE LIMITS                                         │
│      └─ CPU, Memory, Time, I/O                              │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

### Implementation Checklist

```
□ Choose isolation technology for your platform
  □ macOS: VZVirtualMachine or sandbox-exec
  □ Linux: bubblewrap, firejail, or Docker
  □ Cloud: gVisor, Firecracker, or Kata
  □ Embedded: seccomp + chroot (minimal)

□ Implement sandbox manager
  □ create(config) → sandbox_id
  □ exec(sandbox_id, command) → result
  □ destroy(sandbox_id)
  □ status(sandbox_id) → running/stopped/error

□ Implement filesystem gate
  □ Read deny-list (block ~/.ssh, ~/.aws, etc.)
  □ Write allow-list (workspace only)
  □ Mount point management

□ Implement network gate
  □ HTTP proxy with domain filtering
  □ SOCKS5 proxy for TCP
  □ DNS interception (optional)

□ Implement resource limits
  □ CPU quota
  □ Memory limit
  □ Execution timeout
  □ I/O throttling

□ Implement escape hatches
  □ Permission prompt for out-of-bounds requests
  □ Audit logging for all gate crossings
  □ Emergency kill switch
```

---

## Applying to ContinuonBrain

### Current State (No Isolation)

```
┌─────────────────────────────────────────┐
│           CONTINUONBRAIN                │
│                                         │
│   Brain Service                         │
│      ↓                                  │
│   Motor Controller  ← Direct access!    │
│      ↓                                  │
│   GPIO Pins         ← Direct access!    │
│                                         │
└─────────────────────────────────────────┘

Problem: Agent has unrestricted hardware access
```

### With Sandboxed Runtime

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTINUONBRAIN                            │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              AGENT SANDBOX                           │    │
│  │                                                      │    │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐         │    │
│  │   │  Brain   │  │ Teaching │  │   LLM    │         │    │
│  │   │  Logic   │  │  System  │  │  Client  │         │    │
│  │   └────┬─────┘  └────┬─────┘  └────┬─────┘         │    │
│  │        │             │             │               │    │
│  │        └─────────────┼─────────────┘               │    │
│  │                      │                             │    │
│  │              ┌───────▼───────┐                     │    │
│  │              │ Action Gate   │                     │    │
│  │              └───────┬───────┘                     │    │
│  │                      │                             │    │
│  └──────────────────────┼─────────────────────────────┘    │
│                         │                                   │
│    ┌────────────────────┼────────────────────┐             │
│    │                    │                    │             │
│    ▼                    ▼                    ▼             │
│ ┌──────┐          ┌──────────┐         ┌────────┐         │
│ │Safety│          │ Hardware │         │Network │         │
│ │Kernel│          │ Proxy    │         │ Proxy  │         │
│ └──┬───┘          └────┬─────┘         └───┬────┘         │
│    │                   │                   │               │
│    ▼                   ▼                   ▼               │
│ [Limits]          [Motors/Sensors]    [API calls]         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Gate (Robot-Specific)

For robotics, the "filesystem" equivalent is hardware access:

```
┌─────────────────────────────────────────┐
│           HARDWARE RULES                │
├─────────────────────────────────────────┤
│ ACTUATORS: allow-list                   │
│   Default: deny all                     │
│   Allow:   motor_left, motor_right      │
│   Deny:    arm_joint_5 (damaged)        │
│                                         │
│ SENSORS: deny-list                      │
│   Default: allow all                    │
│   Deny:    microphone (privacy)         │
│                                         │
│ LIMITS:                                 │
│   Max speed: 0.5                        │
│   Max acceleration: 0.2/s               │
│   Timeout: 30s continuous motion        │
└─────────────────────────────────────────┘
```

---

## Implementation for Brain B

### File Structure Addition

```
brain_b/
├── sandbox/
│   ├── __init__.py
│   ├── manager.py         # Sandbox lifecycle
│   ├── hardware_gate.py   # Motor/sensor access control
│   ├── network_gate.py    # API call filtering
│   └── limits.py          # Resource budgets
└── ... (existing files)
```

### Sandbox Manager

```python
# sandbox/manager.py

from dataclasses import dataclass, field
from typing import Callable
import time

@dataclass
class SandboxConfig:
    """Configuration for an agent sandbox."""

    # Hardware access
    allowed_actuators: list[str] = field(default_factory=lambda: ["motor_left", "motor_right"])
    denied_actuators: list[str] = field(default_factory=list)
    allowed_sensors: list[str] = field(default_factory=lambda: ["camera", "imu"])
    denied_sensors: list[str] = field(default_factory=list)

    # Network access
    allowed_domains: list[str] = field(default_factory=lambda: ["api.anthropic.com"])
    denied_domains: list[str] = field(default_factory=list)

    # Resource limits
    max_speed: float = 0.5
    max_actions_per_second: int = 10
    timeout_seconds: float = 300.0
    max_api_calls: int = 100


class SandboxViolation(Exception):
    """Raised when sandbox rules are violated."""
    pass


class Sandbox:
    """
    An isolated execution environment for an agent.

    All hardware and network access goes through gates
    that enforce the sandbox configuration.
    """

    def __init__(self, sandbox_id: str, config: SandboxConfig):
        self.id = sandbox_id
        self.config = config
        self.created_at = time.time()
        self.action_count = 0
        self.api_call_count = 0
        self._last_action_time = 0.0
        self._active = True

    def check_actuator(self, name: str) -> bool:
        """Check if actuator access is allowed."""
        if not self._active:
            raise SandboxViolation("Sandbox is not active")

        if name in self.config.denied_actuators:
            raise SandboxViolation(f"Actuator '{name}' is explicitly denied")

        if name not in self.config.allowed_actuators:
            raise SandboxViolation(f"Actuator '{name}' is not in allow-list")

        return True

    def check_sensor(self, name: str) -> bool:
        """Check if sensor access is allowed."""
        if not self._active:
            raise SandboxViolation("Sandbox is not active")

        if name in self.config.denied_sensors:
            raise SandboxViolation(f"Sensor '{name}' is explicitly denied")

        # Sensors use deny-list (allow by default)
        return True

    def check_network(self, domain: str) -> bool:
        """Check if network access is allowed."""
        if not self._active:
            raise SandboxViolation("Sandbox is not active")

        if domain in self.config.denied_domains:
            raise SandboxViolation(f"Domain '{domain}' is explicitly denied")

        if domain not in self.config.allowed_domains:
            raise SandboxViolation(f"Domain '{domain}' is not in allow-list")

        self.api_call_count += 1
        if self.api_call_count > self.config.max_api_calls:
            raise SandboxViolation(f"API call limit ({self.config.max_api_calls}) exceeded")

        return True

    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Check timeout
        if now - self.created_at > self.config.timeout_seconds:
            self._active = False
            raise SandboxViolation(f"Sandbox timeout ({self.config.timeout_seconds}s) exceeded")

        # Check rate limit
        if self._last_action_time > 0:
            elapsed = now - self._last_action_time
            min_interval = 1.0 / self.config.max_actions_per_second
            if elapsed < min_interval:
                raise SandboxViolation(f"Rate limit exceeded ({self.config.max_actions_per_second}/s)")

        self._last_action_time = now
        self.action_count += 1
        return True

    def clamp_speed(self, speed: float) -> float:
        """Clamp speed to sandbox limits."""
        return max(-self.config.max_speed, min(self.config.max_speed, speed))

    def destroy(self):
        """Deactivate the sandbox."""
        self._active = False


class SandboxManager:
    """
    Manages sandbox lifecycle.

    Usage:
        manager = SandboxManager()
        sandbox = manager.create(SandboxConfig(...))

        # All agent operations go through the sandbox
        sandbox.check_actuator("motor_left")
        sandbox.check_rate_limit()

        manager.destroy(sandbox.id)
    """

    def __init__(self):
        self.sandboxes: dict[str, Sandbox] = {}
        self._counter = 0

    def create(self, config: SandboxConfig | None = None) -> Sandbox:
        """Create a new sandbox."""
        self._counter += 1
        sandbox_id = f"sandbox_{self._counter}_{int(time.time())}"
        sandbox = Sandbox(sandbox_id, config or SandboxConfig())
        self.sandboxes[sandbox_id] = sandbox
        return sandbox

    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox by ID."""
        return self.sandboxes.get(sandbox_id)

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
```

### Hardware Gate

```python
# sandbox/hardware_gate.py

from typing import Callable
from .manager import Sandbox, SandboxViolation


class HardwareGate:
    """
    Gate that enforces sandbox rules on hardware access.

    Wraps motor controller and sensor interfaces to
    enforce allow/deny lists and resource limits.
    """

    def __init__(self, sandbox: Sandbox, motor_controller, sensor_manager=None):
        self.sandbox = sandbox
        self.motors = motor_controller
        self.sensors = sensor_manager
        self._audit_log: list[dict] = []

    def set_motor(self, name: str, speed: float):
        """Set motor speed through the gate."""
        try:
            # Check permissions
            self.sandbox.check_actuator(name)
            self.sandbox.check_rate_limit()

            # Clamp speed
            clamped = self.sandbox.clamp_speed(speed)

            # Execute
            if name == "motor_left":
                self.motors.set_left(clamped)
            elif name == "motor_right":
                self.motors.set_right(clamped)
            else:
                raise SandboxViolation(f"Unknown motor: {name}")

            # Audit
            self._audit_log.append({
                "type": "motor",
                "name": name,
                "requested": speed,
                "actual": clamped,
                "allowed": True,
            })

        except SandboxViolation as e:
            self._audit_log.append({
                "type": "motor",
                "name": name,
                "requested": speed,
                "allowed": False,
                "reason": str(e),
            })
            raise

    def read_sensor(self, name: str):
        """Read sensor through the gate."""
        try:
            self.sandbox.check_sensor(name)

            if self.sensors is None:
                return None

            # Execute
            value = self.sensors.read(name)

            # Audit
            self._audit_log.append({
                "type": "sensor",
                "name": name,
                "allowed": True,
            })

            return value

        except SandboxViolation as e:
            self._audit_log.append({
                "type": "sensor",
                "name": name,
                "allowed": False,
                "reason": str(e),
            })
            raise

    def stop_all(self):
        """Emergency stop - always allowed."""
        self.motors.stop()
        self._audit_log.append({
            "type": "emergency_stop",
            "allowed": True,
        })

    def get_audit_log(self) -> list[dict]:
        """Get the audit log."""
        return self._audit_log.copy()
```

---

## Quick Reference Card

### The Pattern in One Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 SANDBOXED AGENT RUNTIME                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 SANDBOX BOUNDARY                     │    │
│  │                                                      │    │
│  │   Agent → Action → [GATE] → Resource                │    │
│  │                       │                              │    │
│  │              ┌────────┴────────┐                    │    │
│  │              │                 │                    │    │
│  │         Allow-list?      Deny-list?                │    │
│  │              │                 │                    │    │
│  │           Explicit         Explicit                │    │
│  │           permits          blocks                  │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  FILESYSTEM:  Write=allow-list, Read=deny-list              │
│  NETWORK:     All=allow-list                                 │
│  HARDWARE:    Actuators=allow-list, Sensors=deny-list       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Copy-Paste Checklist

```markdown
## Sandbox Implementation Checklist

### Boundary
- [ ] Choose isolation tech: VM / namespace / container / process
- [ ] Implement manager: create() / destroy() / exec()

### Filesystem Gate
- [ ] Write: allow-list (default deny)
- [ ] Read: deny-list (default allow)
- [ ] Sensitive paths blocked: ~/.ssh, ~/.aws, /etc/shadow

### Network Gate
- [ ] HTTP proxy with domain filter
- [ ] SOCKS5 proxy for TCP
- [ ] Default: deny all
- [ ] Explicit allow-list

### Hardware Gate (robots)
- [ ] Actuators: allow-list (default deny)
- [ ] Sensors: deny-list (default allow)
- [ ] Speed/force limits
- [ ] Rate limiting

### Resource Limits
- [ ] Execution timeout
- [ ] Action rate limit
- [ ] API call budget
- [ ] Memory/CPU (if applicable)

### Audit
- [ ] Log all gate crossings
- [ ] Log all violations
- [ ] Emergency stop always works
```

---

## Sources

- [Anthropic Sandbox Runtime (GitHub)](https://github.com/anthropic-experimental/sandbox-runtime)
- [Claude Code Sandboxing Docs](https://code.claude.com/docs/en/sandboxing)
- [Simon Willison's Cowork Analysis](https://simonwillison.net/2026/Jan/12/claude-cowork/)
- [Anthropic Engineering Blog on Sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)
