# Swarm Intelligence Module

Enables robots to build other robots, clone their seed image, and coordinate as a swarm.

## Overview

The swarm module provides:

1. **Robot Builder** - Plan and execute robot construction from available parts
2. **Seed Replicator** - Clone the ContinuonBrain seed image to new hardware
3. **Swarm Coordination** - Multi-robot discovery, task delegation, and experience sharing
4. **Authorized Builder** - Safety-integrated construction with owner authorization

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RING 0 - SAFETY LAYER                                 │
│  AntiSubversionGuard │ WorkAuthorizationManager │ MultiPartyAuthorizer      │
├─────────────────────────────────────────────────────────────────────────────┤
│                        SWARM INTELLIGENCE                                    │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ AuthorizedBuilder│───▶│  RobotBuilder   │───▶│  SeedReplicator │         │
│  │ (safety gate)   │    │ (planning)      │    │ (cloning)       │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                              │                  │
│           ▼                                              ▼                  │
│  ┌─────────────────┐                           ┌─────────────────┐         │
│  │ SwarmCoordinator│◀─────────────────────────▶│   New Robot     │         │
│  │ (communication) │                           │                 │         │
│  └─────────────────┘                           └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Safety Requirements

All robot construction requires:

| Requirement | Description |
|-------------|-------------|
| **Owner Authorization** | Only `creator`, `owner`, or `leasee` roles can authorize |
| **Parts Ownership** | All parts must be owned by the authorizing party |
| **Multi-Party Approval** | Critical builds require 2+ different approvers |
| **Signed Work Orders** | Cryptographically signed authorization |
| **Audit Trail** | All actions logged in tamper-evident log |

---

## Robot Builder

Plans and manages robot construction from available parts.

### Part Categories

| Category | Examples |
|----------|----------|
| `compute_sbc` | Raspberry Pi 5, Jetson Nano |
| `compute_mcu` | Arduino, ESP32 |
| `npu` | Hailo-8, Coral TPU |
| `power_battery` | LiPo battery |
| `power_supply` | USB-C PSU |
| `actuator_servo` | STS3215 servo |
| `actuator_motor` | DC motor, brushless |
| `sensor_camera` | OAK-D, Pi Camera |
| `sensor_depth` | OAK-D Lite depth |
| `structure_lego` | Lego Technic parts |
| `structure_3dprint` | 3D printed parts |
| `storage_sd` | microSD card |

### Robot Archetypes

| Archetype | Description | Required Parts |
|-----------|-------------|----------------|
| `stationary_assistant` | Desktop robot, no mobility | Compute, power, structure |
| `wheeled_rover` | Mobile platform on wheels | + motors, wheels |
| `arm_manipulator` | Fixed robotic arm | + 4-6 servos |
| `mobile_manipulator` | Arm on mobile base | + motors + servos |

### Usage

```python
from continuonbrain.swarm import RobotBuilder, Part, PartCategory, RobotArchetype

builder = RobotBuilder()

# Register owner's parts inventory
parts = [
    Part("pi5", "Raspberry Pi 5 8GB", PartCategory.COMPUTE_SBC.value),
    Part("sd64", "64GB microSD", PartCategory.STORAGE_SD.value),
    Part("psu27", "27W USB-C PSU", PartCategory.POWER_SUPPLY.value),
    Part("lego", "Lego Technic Box", PartCategory.STRUCTURE_LEGO.value, quantity=200),
]
inventory = builder.register_inventory("owner_alice", parts)

# Analyze what can be built
analysis = builder.analyze_parts("owner_alice")
print(f"Can build: {analysis['can_build_robot']}")
print(f"Suggested: {analysis['suggested_archetypes']}")
print(f"Capabilities: {analysis['capabilities']}")

# Generate build plan
plan = builder.generate_build_plan(
    owner_id="owner_alice",
    archetype=RobotArchetype.STATIONARY_ASSISTANT,
    name="HelperBot",
)

# Plan contains step-by-step instructions
for step in plan.steps:
    print(f"{step.order}. {step.description}")
    print(f"   Time: {step.estimated_time_minutes} min")
    print(f"   Human required: {step.requires_human}")
```

---

## Seed Replicator

Clones the ContinuonBrain seed image to new storage devices.

### Features

- **Device Detection** - Auto-detects removable storage (SD, SSD)
- **Image Cloning** - Copies seed image with verification
- **Unique Identity** - Each new robot gets unique ID
- **Lineage Tracking** - Tracks parent→child relationships

### Usage

```python
from continuonbrain.swarm import SeedReplicator

replicator = SeedReplicator()

# Detect available storage devices
devices = replicator.detect_storage_devices()
for dev in devices:
    print(f"{dev.device_path}: {dev.device_name} ({dev.capacity_gb:.1f}GB)")

# Create clone job
job = replicator.create_clone_job(
    target_device="/dev/sdb",
    parent_robot_id="continuon-abc123",
    owner_id="owner_alice",
)

# Start cloning (with work authorization)
success, message = replicator.start_clone_job(job.job_id)
print(f"Result: {message}")
print(f"New robot ID: {job.target_robot_id}")

# Track lineage
lineage = replicator.get_lineage(job.target_robot_id)
print(f"Lineage: {' → '.join(lineage)}")
```

---

## Swarm Coordination

Enables multi-robot communication and collaboration.

### Features

| Feature | Description |
|---------|-------------|
| **Discovery** | Multicast discovery on local network |
| **Task Delegation** | Offer tasks to swarm, accept/decline |
| **Conflict Avoidance** | Announce intent, yield/proceed |
| **Experience Sharing** | Share learned skills (not personal data) |
| **Same-Owner Only** | Only robots with same owner communicate |

### Message Types

| Type | Purpose |
|------|---------|
| `ANNOUNCE` | Robot announcing presence |
| `DISCOVER` | Request nearby robots |
| `TASK_OFFER` | Offer task to swarm |
| `TASK_ACCEPT` | Accept offered task |
| `INTENT` | Announce intended action |
| `YIELD` | Yield to another robot |
| `EXPERIENCE_OFFER` | Share learned experience |
| `HEARTBEAT` | Alive signal (every 30s) |

### Usage

```python
from continuonbrain.swarm import SwarmCoordinator

# Initialize coordinator
coordinator = SwarmCoordinator(
    robot_id="continuon-abc123",
    owner_id="owner_alice",
    robot_name="HelperBot",
    capabilities=["vision", "manipulation"],
)

# Start listening and announcing
coordinator.start()

# Discover other robots
coordinator.discover()

# Get swarm members
members = coordinator.get_online_members()
for robot in members:
    print(f"{robot.name}: {robot.capabilities}")

# Offer a task to the swarm
coordinator.offer_task(
    task_id="sort_001",
    description="Sort items in bin A",
    required_capabilities=["vision", "manipulation"],
)

# Share an experience
coordinator.share_experience(
    experience_type="skill",
    description="Learned efficient cube stacking",
    data={"efficiency_gain": 0.15},
)

# Announce intent (for conflict avoidance)
coordinator.announce_intent(
    action="pick_up",
    target="red_cube",
    priority=5,
)

# Stop when done
coordinator.stop()
```

---

## Authorized Builder

Integrates robot construction with the safety system.

### Flow

1. **Register Parts** - Owner registers their parts inventory
2. **Request Build** - Create multi-party authorization request
3. **Confirm Build** - Second person confirms (for critical builds)
4. **Start Build** - Generate plan with work authorization
5. **Clone Seed** - Clone image to new storage
6. **Pair Robot** - New robot paired to same owner

### Usage

```python
from continuonbrain.swarm import AuthorizedRobotBuilder, Part, PartCategory, RobotArchetype

builder = AuthorizedRobotBuilder()

# 1. Register parts (owner role required)
parts = [
    Part("pi5", "Raspberry Pi 5", PartCategory.COMPUTE_SBC.value),
    Part("sd64", "64GB SD Card", PartCategory.STORAGE_SD.value),
    Part("psu", "27W PSU", PartCategory.POWER_SUPPLY.value),
]
success, msg, inventory = builder.register_parts("owner_alice", "owner", parts)

# 2. Request build (creates multi-party authorization)
success, msg, request_id = builder.request_build(
    owner_id="owner_alice",
    owner_role="owner",
    archetype=RobotArchetype.STATIONARY_ASSISTANT,
    robot_name="HelperBot",
    description="Build a desktop assistant robot",
)
print(f"Request: {msg}")  # "Needs 2 confirmations"

# 3. Second person confirms
success, msg = builder.confirm_build(
    request_id=request_id,
    confirmer_id="owner_bob",
    confirmer_role="owner",
    confirmation_method="verbal",
)

# 4. Start build (after approval)
success, msg, plan = builder.start_build(
    request_id=request_id,
    owner_id="owner_alice",
    owner_role="owner",
    archetype=RobotArchetype.STATIONARY_ASSISTANT,
    robot_name="HelperBot",
)

# 5. Clone seed to new storage
success, msg, job = builder.clone_for_new_robot(
    owner_id="owner_alice",
    owner_role="owner",
    parent_robot_id="continuon-parent",
    target_device="/dev/sdb",
)
print(f"New robot: {job.target_robot_id}")
```

---

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `builder.py` | Robot construction planner |
| `replicator.py` | Seed image cloning |
| `coordination.py` | Multi-robot communication |
| `authorized_builder.py` | Safety-integrated builder |

---

## Privacy and Security

### What IS Shared Between Robots

- Learned skills (e.g., efficient grasping)
- Obstacle locations
- Route maps
- Task completions

### What is NOT Shared

- Face embeddings
- Personal conversations
- Owner personal data
- Private RLDS episodes

### Owner Controls

- Swarm mode is opt-in
- Owner can disable sharing anytime
- All shared data is anonymized
- Robots only communicate with same-owner robots

---

## Testing

```bash
# Test swarm module imports
python3 -c "
from continuonbrain.swarm import (
    RobotBuilder, Part, PartCategory, RobotArchetype,
    SeedReplicator, SwarmCoordinator, AuthorizedRobotBuilder,
)
print('All swarm modules imported successfully')
"
```

