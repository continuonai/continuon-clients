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

