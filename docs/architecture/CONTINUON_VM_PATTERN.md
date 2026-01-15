# ContinuonVM: Unified Agent Runtime Pattern

## Status: REFERENCE ARCHITECTURE
## Combines: Ralph + OpenProse + Cowork Sandbox
## Target: Raspberry Pi → Civqo Imaging System
## Date: 2026-01-15

---

## Overview

ContinuonVM is a unified runtime pattern that combines three proven approaches:

| Pattern | Source | Key Contribution |
|---------|--------|------------------|
| **Ralph Loop** | ContinuonBrain | Context rotation, guardrails, state persistence |
| **OpenProse** | prose.md | Markdown-based state, session as VM |
| **Cowork Sandbox** | Anthropic | Isolation, gates, audit logging |

The result is a **self-documenting, sandboxed, teachable robot brain** that runs on a Raspberry Pi and can be imaged for deployment via Civqo.

---

## Core Principles

### From Ralph: Context Rotation
```
Each iteration starts fresh.
State is loaded from disk, not memory.
Lessons learned (guardrails) persist across resets.
```

### From OpenProse: Markdown as Source of Truth
```
State is human-readable markdown.
Configuration is markdown.
Progress is markdown.
Everything is inspectable and editable with a text editor.
```

### From Cowork: Structural Isolation
```
Trust comes from architecture, not behavior.
All I/O goes through gates.
Allow-lists for dangerous operations.
Full audit trail.
```

---

## Architecture

### High-Level View

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTINUON VM                                │
│                   (Raspberry Pi / Civqo)                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   MARKDOWN STATE                         │    │
│  │  .continuon/                                             │    │
│  │  ├── state.md           # Current VM state               │    │
│  │  ├── guardrails.md      # Lessons learned                │    │
│  │  ├── behaviors/         # Taught behaviors               │    │
│  │  │   ├── patrol.md                                       │    │
│  │  │   └── dock.md                                         │    │
│  │  ├── session.md         # Current session log            │    │
│  │  └── audit.md           # Gate crossing log              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   RALPH LOOP                             │    │
│  │                                                          │    │
│  │   ┌──────────┐     ┌──────────┐     ┌──────────┐       │    │
│  │   │  Load    │ ──▶ │ Execute  │ ──▶ │  Save    │       │    │
│  │   │  State   │     │ Iteration│     │  State   │       │    │
│  │   └──────────┘     └────┬─────┘     └──────────┘       │    │
│  │                         │                               │    │
│  │                         ▼                               │    │
│  │                  ┌──────────────┐                       │    │
│  │                  │  Guardrails  │                       │    │
│  │                  │    Check     │                       │    │
│  │                  └──────────────┘                       │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   SANDBOX GATES                          │    │
│  │                                                          │    │
│  │   ┌──────────┐     ┌──────────┐     ┌──────────┐       │    │
│  │   │ Hardware │     │ Network  │     │Filesystem│       │    │
│  │   │   Gate   │     │   Gate   │     │   Gate   │       │    │
│  │   └────┬─────┘     └────┬─────┘     └────┬─────┘       │    │
│  │        │                │                │              │    │
│  │        ▼                ▼                ▼              │    │
│  │    [Motors]         [APIs]          [Files]            │    │
│  │    [Sensors]        [Cloud]         [Configs]          │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
/opt/continuon/
├── vm/                          # ContinuonVM runtime
│   ├── __init__.py
│   ├── loop.py                  # Ralph-style execution loop
│   ├── state.py                 # Markdown state management
│   ├── guardrails.py            # Lesson persistence
│   └── sandbox/                 # Cowork-style isolation
│       ├── manager.py
│       ├── hardware_gate.py
│       ├── network_gate.py
│       └── filesystem_gate.py
│
├── behaviors/                   # Taught behaviors (OpenProse-style)
│   └── *.md                     # Each behavior is a markdown file
│
├── .continuon/                  # Runtime state (all markdown)
│   ├── state.md                 # Current VM state
│   ├── guardrails.md            # Accumulated lessons
│   ├── session.md               # Current session narrative
│   ├── audit.md                 # Gate crossing log
│   └── checkpoint/              # Checkpoints for recovery
│       └── *.md
│
└── main.py                      # Entry point
```

---

## Markdown State Format

### state.md - VM State

```markdown
# ContinuonVM State

**Updated:** 2026-01-15T14:32:00Z
**Iteration:** 47
**Status:** running
**Mode:** autonomous

## Current Task
Patrolling the workshop perimeter.

## Variables
- `position`: near_door
- `battery`: 0.85
- `last_obstacle`: none
- `patrol_count`: 3

## Active Behavior
patrol

## Last Action
```
type: forward
speed: 0.3
duration: 2.0
```

## Last Result
Moved forward successfully. No obstacles detected.

## Metrics
- latency_ms: 45
- actions_today: 156
- errors_today: 2
```

### guardrails.md - Lessons Learned

```markdown
# Guardrails (Lessons Learned)

These are patterns that have caused problems before.
The VM checks each action against these before executing.

## Critical

### GR-001: Carpet Edge
**Trigger:** `position near carpet_edge`
**Action:** Stop immediately, reverse 0.5m
**Learned:** 2026-01-14 (got stuck on carpet transition)

## Warning

### GR-002: Low Light
**Trigger:** `light_level < 0.2`
**Action:** Reduce speed to 0.2, enable headlights
**Learned:** 2026-01-15 (bumped into chair in dim room)

### GR-003: Battery Below 20%
**Trigger:** `battery < 0.2`
**Action:** Abort current behavior, navigate to charger
**Learned:** 2026-01-13 (ran out of battery mid-patrol)
```

### behaviors/patrol.md - Taught Behavior

```markdown
# Behavior: patrol

**Description:** Drive in a square pattern around the room.
**Created:** 2026-01-15T10:00:00Z
**Times Executed:** 12

## Steps

1. Move forward for 3 seconds at speed 0.4
2. Turn left 90 degrees
3. Move forward for 3 seconds at speed 0.4
4. Turn left 90 degrees
5. Move forward for 3 seconds at speed 0.4
6. Turn left 90 degrees
7. Move forward for 3 seconds at speed 0.4
8. Turn left 90 degrees (back to start)

## Preconditions
- Battery > 0.3
- No active emergency

## Success Criteria
- Completed all 8 steps
- No collisions

## Failure Recovery
- On obstacle: stop, reverse 0.3m, skip to next turn
- On low battery: abort, navigate to charger
```

### session.md - Session Narrative (OpenProse-style)

```markdown
# Session: 2026-01-15T14:00:00Z

## Context
Started patrol mode after user command "patrol the workshop".

## Execution Log

📍 **14:00:05** Loading state from state.md
📦 `iteration = 47`
📦 `mode = autonomous`

📍 **14:00:06** Checking guardrails
✅ No guardrails triggered

📍 **14:00:07** Executing behavior: patrol
  → Step 1: forward(0.4, 3.0)
  ✅ Completed

📍 **14:00:10**
  → Step 2: turn_left(90)
  ✅ Completed

📍 **14:00:12**
  → Step 3: forward(0.4, 3.0)
  ⚠️ Obstacle detected at 0.8m
  → Executing failure recovery: reverse(0.3)
  ✅ Recovered, skipping to step 4

📍 **14:00:15** Saving state to state.md
📍 **14:00:15** Context rotation complete
```

### audit.md - Gate Crossings

```markdown
# Audit Log

## 2026-01-15

### 14:00:07 - Hardware Gate
- **Action:** set_motor(motor_left, 0.4)
- **Allowed:** ✅
- **Reason:** motor_left in allow-list

### 14:00:07 - Hardware Gate
- **Action:** set_motor(motor_right, 0.4)
- **Allowed:** ✅
- **Reason:** motor_right in allow-list

### 14:00:12 - Hardware Gate
- **Action:** read_sensor(ultrasonic)
- **Allowed:** ✅
- **Reason:** ultrasonic not in deny-list

### 14:00:20 - Network Gate
- **Action:** api_call(api.anthropic.com)
- **Allowed:** ❌
- **Reason:** domain not in allow-list
- **Note:** Attempted to ask Claude for navigation help
```

---

## Implementation

### The Loop (Ralph-style)

```python
# vm/loop.py

class ContinuonLoop:
    """
    Ralph-style execution loop with markdown state.
    """

    def __init__(self, base_path: Path = Path("/opt/continuon")):
        self.base_path = base_path
        self.state_path = base_path / ".continuon" / "state.md"
        self.guardrails_path = base_path / ".continuon" / "guardrails.md"
        self.session_path = base_path / ".continuon" / "session.md"

        # Sandbox gates
        self.sandbox = SandboxManager().create()
        self.hardware_gate = HardwareGate(self.sandbox, motors)
        self.network_gate = NetworkGate(self.sandbox)

    def run(self):
        """Main loop with context rotation."""
        while True:
            # 1. Load fresh context (Ralph pattern)
            state = self.load_state_from_markdown()

            # 2. Check guardrails
            triggered = self.check_guardrails(state)
            if triggered:
                self.handle_guardrail(triggered, state)
                continue

            # 3. Execute iteration (through gates)
            try:
                state = self.execute_iteration(state)
            except SandboxViolation as e:
                self.add_guardrail_from_violation(e)
                continue

            # 4. Save state (context rotation)
            self.save_state_to_markdown(state)

            # 5. Update session narrative (OpenProse pattern)
            self.append_to_session(state)

    def load_state_from_markdown(self) -> VMState:
        """Parse state.md into VMState."""
        content = self.state_path.read_text()
        return VMState.from_markdown(content)

    def save_state_to_markdown(self, state: VMState):
        """Serialize VMState to state.md."""
        content = state.to_markdown()
        self.state_path.write_text(content)

    def check_guardrails(self, state: VMState) -> list[Guardrail]:
        """Check if any guardrails are triggered."""
        guardrails = self.load_guardrails_from_markdown()
        triggered = []

        for gr in guardrails:
            if gr.matches(state):
                triggered.append(gr)

        return triggered
```

### Markdown State Parser

```python
# vm/state.py

import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class VMState:
    """VM state that serializes to/from markdown."""

    updated: datetime
    iteration: int
    status: str
    mode: str
    current_task: str
    variables: dict
    active_behavior: str | None
    last_action: dict
    last_result: str
    metrics: dict

    @classmethod
    def from_markdown(cls, content: str) -> "VMState":
        """Parse markdown into VMState."""
        # Extract fields using regex
        updated = cls._extract("Updated", content)
        iteration = int(cls._extract("Iteration", content))
        status = cls._extract("Status", content)
        mode = cls._extract("Mode", content)

        # Extract sections
        current_task = cls._extract_section("Current Task", content)
        variables = cls._extract_variables(content)
        active_behavior = cls._extract("Active Behavior", content)
        last_action = cls._extract_code_block("Last Action", content)
        last_result = cls._extract_section("Last Result", content)
        metrics = cls._extract_metrics(content)

        return cls(
            updated=datetime.fromisoformat(updated),
            iteration=iteration,
            status=status,
            mode=mode,
            current_task=current_task,
            variables=variables,
            active_behavior=active_behavior,
            last_action=last_action,
            last_result=last_result,
            metrics=metrics,
        )

    def to_markdown(self) -> str:
        """Serialize VMState to markdown."""
        variables_md = "\n".join(f"- `{k}`: {v}" for k, v in self.variables.items())
        metrics_md = "\n".join(f"- {k}: {v}" for k, v in self.metrics.items())

        return f"""# ContinuonVM State

**Updated:** {self.updated.isoformat()}
**Iteration:** {self.iteration}
**Status:** {self.status}
**Mode:** {self.mode}

## Current Task
{self.current_task}

## Variables
{variables_md}

## Active Behavior
{self.active_behavior or "none"}

## Last Action
```
{self._format_action(self.last_action)}
```

## Last Result
{self.last_result}

## Metrics
{metrics_md}
"""

    @staticmethod
    def _extract(field: str, content: str) -> str:
        match = re.search(rf"\*\*{field}:\*\*\s*(.+)", content)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_section(header: str, content: str) -> str:
        pattern = rf"## {header}\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
```

### Guardrail Parser

```python
# vm/guardrails.py

@dataclass
class Guardrail:
    """A lesson learned, parsed from markdown."""

    id: str
    severity: str  # critical, warning, info
    trigger: str
    action: str
    learned: str
    context: str

    @classmethod
    def from_markdown_section(cls, section: str) -> "Guardrail":
        """Parse a guardrail section from guardrails.md."""
        # Extract ID from header: ### GR-001: Name
        id_match = re.search(r"### (GR-\d+):", section)
        id = id_match.group(1) if id_match else "GR-XXX"

        trigger = cls._extract("Trigger", section)
        action = cls._extract("Action", section)
        learned = cls._extract("Learned", section)

        return cls(
            id=id,
            severity="critical",  # Determined by parent section
            trigger=trigger,
            action=action,
            learned=learned,
            context=section,
        )

    def matches(self, state: VMState) -> bool:
        """Check if this guardrail is triggered by current state."""
        # Simple pattern matching - can be extended
        trigger_lower = self.trigger.lower()

        # Check against variables
        for key, value in state.variables.items():
            if key in trigger_lower:
                # Evaluate condition
                if self._evaluate_condition(trigger_lower, key, value):
                    return True

        return False

    def _evaluate_condition(self, trigger: str, key: str, value) -> bool:
        """Evaluate a trigger condition."""
        # Pattern: key < 0.2 or key > 0.8
        lt_match = re.search(rf"{key}\s*<\s*([\d.]+)", trigger)
        if lt_match:
            threshold = float(lt_match.group(1))
            return float(value) < threshold

        gt_match = re.search(rf"{key}\s*>\s*([\d.]+)", trigger)
        if gt_match:
            threshold = float(gt_match.group(1))
            return float(value) > threshold

        # Pattern: key near location
        near_match = re.search(rf"position near (\w+)", trigger)
        if near_match:
            location = near_match.group(1)
            return str(value) == location

        return False
```

### Behavior Executor

```python
# vm/behaviors.py

@dataclass
class Behavior:
    """A taught behavior, loaded from markdown."""

    name: str
    description: str
    steps: list[dict]
    preconditions: list[str]
    success_criteria: list[str]
    failure_recovery: dict[str, str]

    @classmethod
    def from_markdown(cls, path: Path) -> "Behavior":
        """Load behavior from markdown file."""
        content = path.read_text()

        name = path.stem
        description = cls._extract_section("Description", content)
        steps = cls._parse_steps(content)
        preconditions = cls._parse_list("Preconditions", content)
        success_criteria = cls._parse_list("Success Criteria", content)
        failure_recovery = cls._parse_failure_recovery(content)

        return cls(
            name=name,
            description=description,
            steps=steps,
            preconditions=preconditions,
            success_criteria=success_criteria,
            failure_recovery=failure_recovery,
        )

    @staticmethod
    def _parse_steps(content: str) -> list[dict]:
        """Parse numbered steps into actions."""
        steps = []
        step_pattern = r"\d+\.\s*(.+)"

        for match in re.finditer(step_pattern, content):
            step_text = match.group(1)
            action = Behavior._parse_step_text(step_text)
            steps.append(action)

        return steps

    @staticmethod
    def _parse_step_text(text: str) -> dict:
        """Parse step text like 'Move forward for 3 seconds at speed 0.4'."""
        # Forward pattern
        if "forward" in text.lower():
            duration = re.search(r"(\d+)\s*seconds?", text)
            speed = re.search(r"speed\s*([\d.]+)", text)
            return {
                "type": "forward",
                "duration": float(duration.group(1)) if duration else 1.0,
                "speed": float(speed.group(1)) if speed else 0.5,
            }

        # Turn pattern
        if "turn" in text.lower():
            direction = "left" if "left" in text.lower() else "right"
            degrees = re.search(r"(\d+)\s*degrees?", text)
            return {
                "type": f"turn_{direction}",
                "degrees": float(degrees.group(1)) if degrees else 90,
            }

        # Default
        return {"type": "unknown", "text": text}
```

---

## Raspberry Pi Deployment

### System Requirements

```
Hardware:
- Raspberry Pi 4 (4GB+ recommended)
- Motor controller (PCA9685 or similar)
- RC car chassis with motors
- Camera (optional, for vision)
- Battery (with monitoring)

Software:
- Raspberry Pi OS Lite (64-bit)
- Python 3.11+
- ContinuonVM runtime
```

### Installation Script

```bash
#!/bin/bash
# install_continuon.sh

# Create directory structure
mkdir -p /opt/continuon/{vm,behaviors,.continuon/checkpoint}

# Install dependencies
apt-get update
apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv /opt/continuon/venv
source /opt/continuon/venv/bin/activate

# Install ContinuonVM
pip install continuon-vm  # Or install from source

# Initialize state files
cat > /opt/continuon/.continuon/state.md << 'EOF'
# ContinuonVM State

**Updated:** $(date -Iseconds)
**Iteration:** 0
**Status:** idle
**Mode:** manual

## Current Task
Awaiting instructions.

## Variables
- `battery`: 1.0
- `position`: unknown

## Active Behavior
none

## Last Action
```
type: none
```

## Last Result
System initialized.

## Metrics
- uptime_hours: 0
EOF

# Create systemd service
cat > /etc/systemd/system/continuon.service << 'EOF'
[Unit]
Description=ContinuonVM Robot Brain
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/continuon
ExecStart=/opt/continuon/venv/bin/python -m continuon.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable continuon
systemctl start continuon

echo "ContinuonVM installed and running!"
```

### Civqo Imaging

For deployment to Civqo, create an image manifest:

```yaml
# civqo-manifest.yaml

name: continuon-rc-car
version: 1.0.0
description: ContinuonVM runtime for teachable RC car

base_image: raspios-lite-arm64

packages:
  - python3
  - python3-pip
  - python3-venv
  - i2c-tools
  - pigpio

files:
  - source: ./continuon/
    destination: /opt/continuon/
    mode: "755"

  - source: ./config/continuon.service
    destination: /etc/systemd/system/continuon.service

services:
  enable:
    - continuon

post_install:
  - /opt/continuon/install_deps.sh

ports:
  - 8080:8080  # Web UI
  - 22:22      # SSH

volumes:
  - /opt/continuon/.continuon:/data/continuon  # Persist state
```

---

## Quick Reference

### Pattern Summary

```
┌────────────────────────────────────────────────────────────┐
│                     CONTINUON VM                            │
│                                                              │
│   RALPH LOOP (How it executes)                              │
│   ├─ Load state from markdown                               │
│   ├─ Check guardrails                                       │
│   ├─ Execute iteration through gates                        │
│   ├─ Save state to markdown                                 │
│   └─ Rotate context                                         │
│                                                              │
│   OPENPROSE STATE (How it stores)                           │
│   ├─ state.md - Current VM state                            │
│   ├─ guardrails.md - Lessons learned                        │
│   ├─ behaviors/*.md - Taught behaviors                      │
│   ├─ session.md - Execution narrative                       │
│   └─ audit.md - Gate crossings                              │
│                                                              │
│   COWORK SANDBOX (How it isolates)                          │
│   ├─ Hardware gate (actuators=allow, sensors=deny)          │
│   ├─ Network gate (domains=allow)                           │
│   ├─ Filesystem gate (paths=allow)                          │
│   └─ Full audit logging                                     │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

### Files at a Glance

| File | Format | Purpose |
|------|--------|---------|
| `state.md` | Markdown | Current VM state |
| `guardrails.md` | Markdown | Accumulated lessons |
| `behaviors/*.md` | Markdown | Taught behaviors |
| `session.md` | Markdown | Execution narrative |
| `audit.md` | Markdown | Security audit trail |

### Commands

```bash
# Start the VM
python -m continuon.main

# Teach a behavior
curl -X POST http://localhost:8080/teach/patrol

# Invoke a behavior
curl -X POST http://localhost:8080/invoke/patrol

# View state
cat /opt/continuon/.continuon/state.md

# View guardrails
cat /opt/continuon/.continuon/guardrails.md

# Check audit log
tail -f /opt/continuon/.continuon/audit.md
```

---

## Sources

- **Ralph Loop**: `continuonbrain/ralph/base.py`
- **OpenProse**: [github.com/openprose/prose](https://github.com/openprose/prose)
- **Cowork Sandbox**: [github.com/anthropic-experimental/sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime)
