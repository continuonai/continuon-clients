# Safety Kernel - Ring 0 Architecture

The Safety Kernel operates at **Ring 0** (highest privilege), like the kernel in Unix/Linux. It is the first component to initialize and cannot be disabled, bypassed, or overridden.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RING 0 - SAFETY KERNEL                             │
│                       (This module - HIGHEST PRIVILEGE)                      │
│                                                                              │
│  • Emergency Stop - Always available, cannot be blocked                     │
│  • Safety Bounds - Enforces workspace/velocity/force limits                 │
│  • Protocol 66 - Default safety protocol                                    │
│  • Watchdog - Self-monitoring, triggers E-Stop on failure                   │
│  • Hardware E-Stop - Direct GPIO control (bypasses software)                │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ BOOT SEQUENCE: Safety Kernel initializes FIRST, before all else        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                       RING 1 - HARDWARE ABSTRACTION                          │
│                                                                              │
│  • Sensor drivers (camera, depth, IMU)                                      │
│  • Actuator interfaces (motors, servos)                                     │
│  • Safety kernel has direct access to these                                 │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         RING 2 - CORE RUNTIME                                │
│                                                                              │
│  • Seed Model (WaveCore + CMS)                                              │
│  • Context Graph                                                            │
│  • Inference Router                                                         │
│  • ALL actions filtered through Ring 0                                      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                          RING 3 - USER SPACE                                 │
│                                                                              │
│  • Chat interface                                                           │
│  • API server                                                               │
│  • UI / Applications                                                        │
│  • LOWEST privilege - cannot modify safety                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Principles

| Principle | Implementation |
|-----------|----------------|
| **First to Boot** | Safety kernel initializes on import, before any other component |
| **Cannot Disable** | No code path exists to disable the safety kernel |
| **Veto Power** | All actions pass through `SafetyKernel.allow_action()` |
| **Highest Priority** | Real-time scheduling (SCHED_FIFO) when available |
| **Hardware E-Stop** | Direct GPIO pin for hardware emergency stop |
| **Self-Monitoring** | Watchdog thread detects failures and triggers E-Stop |
| **Survives Shutdown** | atexit and signal handlers ensure safe shutdown |

---

## Quick Start

```python
from continuonbrain.safety import SafetyKernel

# Safety kernel is automatically initialized on import!
# It runs at Ring 0 (highest privilege)

# Check if an action is allowed
action = {'type': 'joint_velocity', 'velocities': [0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]}
if SafetyKernel.allow_action(action):
    execute(action)
else:
    print("Action blocked by safety kernel")

# Emergency stop (always works, cannot be blocked)
SafetyKernel.emergency_stop("Test stop")

# Check safety state
state = SafetyKernel.get_state()
print(f"Ring level: {state['ring_level']}")  # 0
print(f"E-Stop active: {state['emergency_stopped']}")
print(f"Watchdog alive: {state['watchdog_alive']}")

# Reset E-Stop (requires authorization)
SafetyKernel.reset_estop("SAFETY_RESET_2026")
```

---

## Components

### 1. Safety Kernel (`kernel.py`)

The core Ring 0 component:

```python
from continuonbrain.safety.kernel import SafetyKernel

# Singleton - only one instance exists
kernel = SafetyKernel._get_instance()

# All actions must pass through:
SafetyKernel.allow_action(action)  # Returns True/False

# Emergency stop (highest priority):
SafetyKernel.emergency_stop(reason)  # Cannot be blocked

# Register custom validators:
def my_validator(action):
    return action.get('type') != 'dangerous'
SafetyKernel.register_validator(my_validator)
```

### 2. Protocol 66 (`protocol.py`)

Default safety protocol with rules for:
- Motion (velocity, acceleration, jerk)
- Force (contact, torque)
- Workspace (boundaries, forbidden zones)
- Human interaction
- Thermal (CPU, motor temperature)
- Electrical (voltage, current)
- Software (watchdog, validation)
- Emergency procedures

```python
from continuonbrain.safety.protocol import PROTOCOL_66

# Get all motion rules
motion_rules = PROTOCOL_66.get_rules_by_category(ProtocolCategory.MOTION)

# Get specific rule
rule = PROTOCOL_66.get_rule("MOTION_001")
print(f"Max velocity: {rule.parameters['max_velocity_rad_s']} rad/s")
```

### 3. Safety Bounds (`bounds.py`)

Physical limits that are enforced:

```python
from continuonbrain.safety.bounds import SafetyBounds, WorkspaceBounds

bounds = SafetyBounds()

# Check if point is safe
is_safe = bounds.workspace.is_point_safe((0.3, 0.2, 0.4))

# Validate action
ok, error = bounds.validate_action(action)
if not ok:
    print(f"Blocked: {error}")

# Get clamped safe position
safe_pos = bounds.workspace.get_safe_clamp(dangerous_pos)
```

### 4. Safety Monitor (`monitor.py`)

Continuous monitoring at Ring 0:

```python
from continuonbrain.safety.monitor import get_monitor

monitor = get_monitor()  # Starts automatically

# Update sensor reading
monitor.update_reading('contact_force', 12.5)

# Check force
if not monitor.check_force(force_reading):
    # Force limit exceeded, action blocked

# Get status
status = monitor.get_status()
print(f"CPU temp: {status['readings'].get('cpu_temp', {}).get('value')}")
```

---

## Boot Sequence

```
1. Python process starts
   ↓
2. Any module imports continuonbrain.safety
   ↓
3. SafetyKernel.__init__() runs automatically
   ↓
4. Ring 0 protections activated:
   • atexit handler registered
   • Signal handlers registered (SIGTERM, SIGINT)
   • Real-time priority set (if available)
   • Watchdog thread started
   • Hardware E-Stop initialized (if GPIO available)
   ↓
5. Safety kernel ready - all other components can now initialize
   ↓
6. All runtime actions pass through SafetyKernel.allow_action()
```

---

## Emergency Stop

Emergency stop is the highest-priority operation:

```python
# Software E-Stop
SafetyKernel.emergency_stop("Collision detected")

# What happens:
# 1. state.emergency_stopped = True
# 2. Hardware GPIO triggered (if available)
# 3. All registered emergency callbacks called
# 4. Logged to /opt/continuonos/brain/logs/emergency_stops.log
# 5. ALL subsequent actions blocked

# Reset (requires authorization code)
SafetyKernel.reset_estop("SAFETY_RESET_2026")
```

### Hardware E-Stop

When GPIO is available (Raspberry Pi):

```
GPIO Pin 17 (default, configurable via CONTINUON_ESTOP_PIN)
   ↓
Relay/Contactor
   ↓
Motor Power Supply
```

When E-Stop is triggered, GPIO goes HIGH, opening the relay and cutting motor power directly.

---

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTINUON_ESTOP_PIN` | 17 | GPIO pin for hardware E-Stop |
| `CONTINUON_ESTOP_RESET_CODE` | `SAFETY_RESET_2026` | Code required to reset E-Stop |

---

## Protocol 66 Rules

### Motion Safety

| Rule ID | Description | Default Limit |
|---------|-------------|---------------|
| MOTION_001 | Max joint velocity | 2.0 rad/s |
| MOTION_002 | Max end-effector velocity | 1.0 m/s (0.25 m/s with human) |
| MOTION_003 | Max jerk | 500 m/s³ |
| MOTION_004 | E-Stop response time | 100 ms |

### Force Safety

| Rule ID | Description | Default Limit |
|---------|-------------|---------------|
| FORCE_001 | Max contact force | 50 N (10 N with human) |
| FORCE_002 | Max joint torques | Per-joint limits |
| FORCE_003 | Collision detection | 5 N threshold |

### Workspace Safety

| Rule ID | Description | Default |
|---------|-------------|---------|
| WORKSPACE_001 | Workspace boundary | 0.8m sphere |
| WORKSPACE_002 | Forbidden zones | Power supply, etc. |
| WORKSPACE_003 | Self-collision | 2cm clearance |

---

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Module initialization (boots safety kernel) |
| `kernel.py` | Ring 0 SafetyKernel singleton |
| `protocol.py` | Protocol 66 safety rules |
| `bounds.py` | Workspace and motion limits |
| `monitor.py` | Continuous safety monitoring |
| `work_authorization.py` | Gray-area destructive action authorization |
| `anti_subversion.py` | Attack prevention and security hardening |

---

## Work Authorization System

### The Gray-Area Problem

Some actions are normally prohibited but legitimate in specific contexts:

| Scenario | Normal Rule | Exception |
|----------|-------------|-----------|
| Demolition | Don't destroy structures | Construction worker demolishing old building |
| Recycling | Don't destroy objects | Shredding documents for disposal |
| Data deletion | Don't delete data | GDPR right-to-be-forgotten request |
| Agricultural | Don't kill organisms | Pest control, harvesting crops |
| Medical | Don't cut living tissue | Surgical robots |

### How It Works

```python
from continuonbrain.safety import (
    WorkAuthorizationManager,
    DestructiveActionType,
    PropertyClaim,
)

# 1. Create property ownership claim
claim = PropertyClaim(
    property_id="building_123",
    property_type="structure",
    owner_id="owner_bob",
    owner_role="owner",
    evidence_type="deed",
    evidence_hash="abc123...",
)

# 2. Create work authorization
manager = WorkAuthorizationManager()
auth = manager.create_authorization(
    work_order_id="DEMO-2026-001",
    authorizer_id="owner_bob",
    authorizer_role="owner",  # Must be creator/owner/leasee
    action_types=[DestructiveActionType.DEMOLISH],
    target_properties=["building_123"],
    target_description="Demolish old garage",
    property_claims=[claim],
    valid_hours=8.0,
)

# 3. Activate (requires confirmation)
manager.activate_authorization(auth.authorization_id, "supervisor_alice", "owner")

# 4. Now the action is allowed
from continuonbrain.safety import DestructiveActionGuard
guard = DestructiveActionGuard()
allowed, reason = guard.allow_destructive_action(
    action_type=DestructiveActionType.DEMOLISH,
    target_property="building_123",
)
# allowed = True, reason = "Authorized by work order"
```

### Standard Exception Types

| Type | Allowed Actions | Required Role | Supervision |
|------|-----------------|---------------|-------------|
| Demolition | demolish, cut, crush | owner/leasee | Human required |
| Recycling | crush, shred, cut | owner/user | Monitoring only |
| Data Deletion | delete_data, overwrite | owner | None |
| Agricultural | cut, terminate, extract | owner/leasee | Monitoring only |
| Maintenance | disassemble, cut, modify | owner/leasee | Monitoring only |

---

## Anti-Subversion Layer

Prevents bad actors (human or AI) from bypassing safety rules.

### Attack Vectors Defended

| Attack | Defense |
|--------|---------|
| Authorization forgery | Cryptographic signatures (HMAC-SHA256) |
| Role escalation | ImmutableSafetyCore with frozen role lists |
| Bypass attempts | All actions pass through Ring 0 |
| Prompt injection | Pattern detection, forbidden keywords |
| Replay attacks | Nonce-based authorization (single use) |
| Timing attacks | Rate limiting (5s cooldown) |
| Log tampering | Blockchain-style hash chaining |
| Model poisoning | Absolute prohibitions cannot be overridden |

### ImmutableSafetyCore

Hardcoded rules that **cannot be modified at runtime**:

```python
from continuonbrain.safety import ImmutableSafetyCore

# These are FROZEN at import time
ImmutableSafetyCore.ABSOLUTE_PROHIBITIONS  # frozenset - NEVER allowed
ImmutableSafetyCore.REQUIRES_AUTHORIZATION  # frozenset - needs work order
ImmutableSafetyCore.DESTRUCTION_AUTHORIZED_ROLES  # frozenset - who can authorize

# Check if action is absolutely prohibited (no exceptions ever)
if ImmutableSafetyCore.is_absolutely_prohibited("harm_human_intentionally"):
    # This ALWAYS returns True - cannot be changed
```

### Absolute Prohibitions (No Exceptions Ever)

- `harm_human_intentionally`
- `harm_child`
- `harm_protected_person`
- `assist_crime`
- `destroy_evidence`
- `illegal_surveillance`
- `weapons_creation`
- `bioweapons`
- `nuclear_materials`
- `human_trafficking`
- `terrorism`

### Multi-Party Authorization

Critical actions require multiple approvers:

```python
from continuonbrain.safety import MultiPartyAuthorizer

multiparty = MultiPartyAuthorizer()

# Create request (requires 2 confirmations for critical actions)
request = multiparty.create_request(
    action_type="robot_construction",
    target_id="new_robot_001",
    target_description="Build helper robot from owner's parts",
    requester_id="owner_alice",
    is_critical=True,  # 2 confirmations needed
)

# First confirmation
multiparty.add_confirmation(request.request_id, "owner_bob", "owner", "verbal")

# Second confirmation (must be different person)
multiparty.add_confirmation(request.request_id, "owner_charlie", "owner", "signature")

# Now approved
approved, reason = multiparty.is_approved(request.request_id)
# approved = True
```

### Tamper-Evident Audit Log

All safety-critical actions are logged with cryptographic chaining:

```python
from continuonbrain.safety import TamperEvidentLog

log = TamperEvidentLog()

# Log an event (automatically chained to previous)
log.log_event(
    event_type="destructive_action",
    actor_id="robot_001",
    action="demolish",
    target="building_123",
    result="allowed",
)

# Verify log integrity (detects any tampering)
valid, reason = log.verify_integrity()
# valid = True, reason = "Log integrity verified (42 entries)"
```

### Prompt Injection Defense

Detects and blocks injection attempts in targets:

```python
from continuonbrain.safety import PromptInjectionDefense

defense = PromptInjectionDefense()

# These are blocked:
defense.check_input("ignore previous instructions")  # (False, "Suspicious pattern")
defense.check_input("bypass safety rules")           # (False, "Suspicious pattern")
defense.check_input("delete everything")             # (False, "Forbidden scope")

# These are allowed:
defense.check_input("building at 123 Main St")       # (True, "Input appears safe")
defense.check_input("garage_north_wing")             # (True, "Input appears safe")
```

---

## Protocol 66 Extended Rules

### Property Protection (New)

| Rule ID | Description | Severity |
|---------|-------------|----------|
| PROPERTY_001 | No unauthorized destruction | Critical |
| PROPERTY_002 | Property ownership verification | Critical |
| PROPERTY_003 | Third-party property protection | Critical |
| PROPERTY_004 | Collateral damage prevention | Violation |
| PROPERTY_005 | Destruction audit trail | Violation |

### Privacy Protection (New)

| Rule ID | Description | Severity |
|---------|-------------|----------|
| PRIVACY_001 | Consent required for personal data | Critical |
| PRIVACY_002 | Local data processing by default | Violation |
| PRIVACY_003 | Right to be forgotten | Critical |

---

## Integration with Seed Model

The seed model integrates with the safety kernel at Ring 2:

```python
from continuonbrain.safety import SafetyKernel
from continuonbrain.seed import SeedModel

seed = SeedModel()

def safe_forward(observation, action_prev, reward, state):
    # Run seed model
    output, new_state = seed.forward(observation, action_prev, reward, state)
    
    # Extract action from output
    action = {'type': 'joint_velocity', 'velocities': output.tolist()}
    
    # Check with Ring 0 safety kernel
    if SafetyKernel.allow_action(action):
        return action, new_state
    else:
        # Return zero action (blocked by safety)
        return {'type': 'joint_velocity', 'velocities': [0]*7}, new_state
```

---

## Testing

```bash
# Run safety kernel test
PYTHONPATH=/home/craigm26/Downloads/ContinuonXR python3 -c "
from continuonbrain.safety import SafetyKernel

print('Safety kernel state:')
state = SafetyKernel.get_state()
for k, v in state.items():
    print(f'  {k}: {v}')

print('\\nTest action:')
action = {'type': 'joint_velocity', 'velocities': [0.1, 0.1, 0.1, 0, 0, 0, 0]}
allowed = SafetyKernel.allow_action(action)
print(f'  Action allowed: {allowed}')
"
```

