# ContinuonBrain Actor Runtime Design

## Status: DRAFT
## Author: Architecture Review
## Date: 2026-01-15

---

## Executive Summary

This document proposes a **streamlined actor runtime** for ContinuonBrain, designed around a single use case: **talking to your robot and teaching it things**.

The current system is overbuilt. This design cuts 80% of the complexity while preserving the 20% that matters.

---

## Problem Statement

### Current State: Too Much
```
Current ContinuonBrain
├── 6 Domain Services (Chat, Hardware, Perception, Learning, Audio, Reasoning)
├── 4 Ralph Layers (Fast, Mid, Slow, Safety + Meta orchestrator)
├── 3-Loop HOPE Architecture (10ms, 100ms, 1s+)
├── Constitutional Safety Kernel with graduated response
├── 12.8M parameter WaveCore Mamba SSM
├── RLDS pipeline with cloud training
├── Context graph with NetworkX reasoning
├── Multiple LLM fallbacks (Claude, Gemini, GPT-4, Ollama)
└── Hardware abstraction for cameras, NPUs, servos, drivetrains
```

**Reality check**: For an RC car you want to talk to and teach, you need maybe 10% of this.

### What You Actually Want
```
What you need
├── Talk to the robot naturally
├── Tell it what to do ("drive to the red cone")
├── Teach it new things ("when I say 'patrol', drive in a square")
├── Have it remember what you taught it
├── Pick up where you left off after restart
└── Not have it crash or go haywire
```

---

## Proposed Architecture: Event-Sourced Actor Runtime

### Design Philosophy

**Cowork-style interaction**: Like Claude's Cowork mode, the robot should feel like a capable assistant that:
- Listens and responds naturally
- Executes actions when asked
- Learns from corrections
- Maintains context across sessions
- Can be interrupted and redirected

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (You)                               │
│                    Voice / Text / Gestures                       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONVERSATION LAYER                           │
│                                                                   │
│  "Drive forward"  →  Intent: MOVE   →  Action: motors(0.5, 0.5) │
│  "Stop"           →  Intent: HALT   →  Action: motors(0, 0)     │
│  "Learn: patrol"  →  Intent: TEACH  →  Record sequence          │
│                                                                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ACTOR RUNTIME                               │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Agent     │  │    Agent     │  │    Agent     │          │
│  │   (Drive)    │  │   (Vision)   │  │  (Learning)  │          │
│  │              │  │              │  │              │          │
│  │ State: {...} │  │ State: {...} │  │ State: {...} │          │
│  │ Mailbox: []  │  │ Mailbox: []  │  │ Mailbox: []  │          │
│  │ Budget: 100  │  │ Budget: 50   │  │ Budget: 200  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    EVENT LOG                             │    │
│  │  [t1: spawn(drive)] [t2: move(0.5)] [t3: checkpoint()]  │    │
│  │  [t4: teach("patrol", seq)] [t5: invoke("patrol")]      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HARDWARE LAYER                              │
│                                                                   │
│     Motors          Sensors          Safety Stop                 │
│   (PWM/GPIO)      (Camera/IMU)      (Hardware kill)             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Primitives

### 1. Agent

The fundamental unit of execution. Each agent is isolated with its own state.

```python
@dataclass
class Agent:
    id: str
    state: dict                    # Agent's working memory
    mailbox: deque                 # Incoming messages
    budget: ResourceBudget         # Execution limits
    event_log: EventLog            # Append-only history

@dataclass
class ResourceBudget:
    max_tokens: int = 1000         # LLM token limit per action
    max_actions: int = 100         # Action limit before checkpoint
    timeout_ms: int = 30000        # Wall-clock limit
```

### 2. Runtime Primitives

| Primitive | Purpose | Example |
|-----------|---------|---------|
| `spawn(agent_type, config)` | Create isolated agent | `spawn("driver", {max_speed: 0.5})` |
| `send(agent_id, message)` | Async message passing | `send("driver", {cmd: "forward"})` |
| `checkpoint()` | Snapshot state for recovery | Auto-called every N actions |
| `suspend()` / `resume()` | Pause/continue execution | Battery low → suspend → charge → resume |
| `teach(name, sequence)` | Record action sequence | `teach("patrol", [fwd, left, fwd, left...])` |
| `invoke(name)` | Execute learned behavior | `invoke("patrol")` |

### 3. Event Log

Every action is recorded. This is the killer feature.

```python
@dataclass
class Event:
    timestamp: float
    event_type: str               # "action", "observation", "checkpoint", "teach"
    agent_id: str
    payload: dict

class EventLog:
    def append(self, event: Event): ...
    def replay(self, from_checkpoint: str) -> Iterator[Event]: ...
    def checkpoint(self) -> str: ...  # Returns checkpoint ID
```

**Why this matters:**
- **Crash recovery**: Robot dies mid-action? Replay from last checkpoint.
- **Teaching**: Record what you did, replay it later.
- **Debugging**: See exactly what happened and when.
- **Determinism**: Same events + same seed = same behavior.

---

## Conversation Interface

### Natural Teaching Flow

```
You: "Hey robot"
Bot: "Listening."

You: "Drive forward slowly"
Bot: "Moving forward at 30% speed."
     [Action: motors(0.3, 0.3)]

You: "Stop"
Bot: "Stopped."
     [Action: motors(0, 0)]

You: "I want to teach you something called 'patrol'"
Bot: "Ready to learn 'patrol'. Show me what to do."
     [Mode: RECORDING]

You: "Go forward"
Bot: [Recording: forward]

You: "Turn left"
Bot: [Recording: turn_left]

You: "Go forward"
Bot: [Recording: forward]

You: "Turn left"
Bot: [Recording: turn_left]

You: "Done teaching"
Bot: "Got it. 'patrol' is: forward → left → forward → left.
      Say 'patrol' anytime to run it."
     [Saved: teach("patrol", [fwd, left, fwd, left])]

You: "patrol"
Bot: "Running patrol."
     [Action: invoke("patrol")]
```

### Intent Classification (Simple)

```python
INTENTS = {
    # Movement
    "forward|go|drive": Intent.MOVE_FORWARD,
    "back|reverse": Intent.MOVE_BACKWARD,
    "left|turn left": Intent.TURN_LEFT,
    "right|turn right": Intent.TURN_RIGHT,
    "stop|halt|freeze": Intent.STOP,

    # Teaching
    "teach|learn|remember": Intent.START_TEACHING,
    "done|finished|that's it": Intent.STOP_TEACHING,

    # Recall
    "do|run|execute": Intent.INVOKE_BEHAVIOR,

    # Meta
    "what can you do": Intent.LIST_CAPABILITIES,
    "what did you learn": Intent.LIST_BEHAVIORS,
    "forget": Intent.DELETE_BEHAVIOR,
}
```

For complex intents, use a small LLM (Gemma 2B local, or Claude API).

---

## What to Keep from Current System

### Keep (Essential)

| Component | Why | Location |
|-----------|-----|----------|
| Safety kernel basics | Prevent runaway motors | `kernel/safety_kernel.py` |
| Hardware abstraction | Motor/sensor interface | `actuators/`, `sensors/` |
| Session store | Conversation history | `core/session_store.py` |
| System event logger | Debugging | `system_events.py` |

### Simplify (Reduce Scope)

| Component | Current | Proposed |
|-----------|---------|----------|
| Ralph layers | 4 layers + meta | 1 agent with checkpointing |
| HOPE loops | 3 frequency tiers | 1 main loop (50-100ms) |
| Domain services | 6 services | 2 services (Chat, Hardware) |
| Neural network | 12.8M params | Optional, start with rules |
| LLM providers | 4 fallbacks | 1 primary (Claude or local) |

### Cut (Remove Entirely)

| Component | Why Cut |
|-----------|---------|
| Context graph with NetworkX | Overkill for RC car |
| RLDS cloud training pipeline | Not needed for teaching |
| VQ-GAN/VQ-VAE tokenization | Not needed yet |
| Production agent pipeline | Premature |
| Graduated response system | Simple stop is enough |
| Meta-layer introspection | Complexity without value |

---

## Implementation: Minimal Viable Runtime

### File Structure

```
continuonbrain/
├── actor_runtime/
│   ├── __init__.py
│   ├── agent.py              # Agent dataclass, lifecycle
│   ├── runtime.py            # spawn, send, checkpoint, suspend/resume
│   ├── event_log.py          # Append-only log, replay, checkpoints
│   ├── teaching.py           # teach(), invoke(), behavior storage
│   └── budgets.py            # Resource limits, enforcement
├── conversation/
│   ├── __init__.py
│   ├── listener.py           # Voice/text input
│   ├── intents.py            # Intent classification
│   ├── responder.py          # Natural language output
│   └── session.py            # Conversation state
├── hardware/
│   ├── __init__.py
│   ├── motors.py             # Motor control (PWM/GPIO)
│   ├── sensors.py            # Camera, IMU, etc.
│   └── safety.py             # Hardware-level kill switch
└── main.py                   # Entry point
```

### Minimal Agent Implementation

```python
# actor_runtime/agent.py

from dataclasses import dataclass, field
from collections import deque
from typing import Any
import time
import json

@dataclass
class ResourceBudget:
    max_actions: int = 100
    timeout_ms: int = 30000
    actions_used: int = 0
    start_time: float = field(default_factory=time.time)

    def check(self) -> bool:
        elapsed = (time.time() - self.start_time) * 1000
        return self.actions_used < self.max_actions and elapsed < self.timeout_ms

    def consume(self):
        self.actions_used += 1


@dataclass
class Agent:
    id: str
    agent_type: str
    state: dict = field(default_factory=dict)
    mailbox: deque = field(default_factory=deque)
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    suspended: bool = False

    def receive(self, message: dict):
        self.mailbox.append(message)

    def process_next(self) -> dict | None:
        if self.suspended or not self.mailbox or not self.budget.check():
            return None

        message = self.mailbox.popleft()
        self.budget.consume()
        return message

    def suspend(self):
        self.suspended = True

    def resume(self):
        self.suspended = False
        self.budget = ResourceBudget()  # Fresh budget on resume

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "state": self.state,
            "mailbox": list(self.mailbox),
            "suspended": self.suspended,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        agent = cls(
            id=data["id"],
            agent_type=data["agent_type"],
            state=data["state"],
            suspended=data["suspended"],
        )
        agent.mailbox = deque(data["mailbox"])
        return agent
```

### Event Log Implementation

```python
# actor_runtime/event_log.py

from dataclasses import dataclass
from typing import Iterator
import json
import time
import os

@dataclass
class Event:
    timestamp: float
    event_type: str
    agent_id: str
    payload: dict
    checkpoint_id: str | None = None


class EventLog:
    def __init__(self, path: str = "/opt/continuonos/brain/events"):
        self.path = path
        self.log_file = os.path.join(path, "events.jsonl")
        self.checkpoint_dir = os.path.join(path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def append(self, event: Event):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "agent_id": event.agent_id,
                "payload": event.payload,
                "checkpoint_id": event.checkpoint_id,
            }) + "\n")

    def checkpoint(self, agents: dict[str, "Agent"]) -> str:
        checkpoint_id = f"ckpt_{int(time.time() * 1000)}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        with open(checkpoint_path, "w") as f:
            json.dump({
                "checkpoint_id": checkpoint_id,
                "timestamp": time.time(),
                "agents": {aid: agent.to_dict() for aid, agent in agents.items()},
            }, f, indent=2)

        # Log the checkpoint event
        self.append(Event(
            timestamp=time.time(),
            event_type="checkpoint",
            agent_id="runtime",
            payload={"checkpoint_id": checkpoint_id},
            checkpoint_id=checkpoint_id,
        ))

        return checkpoint_id

    def restore(self, checkpoint_id: str) -> dict[str, "Agent"]:
        from .agent import Agent

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(checkpoint_path) as f:
            data = json.load(f)

        return {aid: Agent.from_dict(adata) for aid, adata in data["agents"].items()}

    def replay(self, from_checkpoint: str | None = None) -> Iterator[Event]:
        start_replaying = from_checkpoint is None

        with open(self.log_file) as f:
            for line in f:
                event_data = json.loads(line)
                event = Event(**event_data)

                if not start_replaying:
                    if event.checkpoint_id == from_checkpoint:
                        start_replaying = True
                    continue

                yield event

    def latest_checkpoint(self) -> str | None:
        checkpoints = sorted(os.listdir(self.checkpoint_dir))
        if checkpoints:
            return checkpoints[-1].replace(".json", "")
        return None
```

### Teaching System

```python
# actor_runtime/teaching.py

from dataclasses import dataclass, field
import json
import os
from typing import Callable

@dataclass
class Behavior:
    name: str
    actions: list[dict]
    created_at: float
    description: str = ""


class TeachingSystem:
    def __init__(self, path: str = "/opt/continuonos/brain/behaviors"):
        self.path = path
        self.behaviors_file = os.path.join(path, "behaviors.json")
        os.makedirs(path, exist_ok=True)
        self.behaviors: dict[str, Behavior] = {}
        self._recording: list[dict] | None = None
        self._recording_name: str | None = None
        self._load()

    def _load(self):
        if os.path.exists(self.behaviors_file):
            with open(self.behaviors_file) as f:
                data = json.load(f)
                self.behaviors = {
                    name: Behavior(**bdata)
                    for name, bdata in data.items()
                }

    def _save(self):
        with open(self.behaviors_file, "w") as f:
            json.dump({
                name: {
                    "name": b.name,
                    "actions": b.actions,
                    "created_at": b.created_at,
                    "description": b.description,
                }
                for name, b in self.behaviors.items()
            }, f, indent=2)

    def start_recording(self, name: str) -> str:
        self._recording = []
        self._recording_name = name
        return f"Ready to learn '{name}'. Show me what to do."

    def record_action(self, action: dict):
        if self._recording is not None:
            self._recording.append(action)

    def stop_recording(self) -> str:
        if self._recording is None or self._recording_name is None:
            return "Not currently recording."

        import time
        behavior = Behavior(
            name=self._recording_name,
            actions=self._recording.copy(),
            created_at=time.time(),
        )
        self.behaviors[self._recording_name] = behavior
        self._save()

        action_summary = " -> ".join(a.get("type", "?") for a in self._recording[:5])
        if len(self._recording) > 5:
            action_summary += f" ... ({len(self._recording)} total)"

        result = f"Learned '{self._recording_name}': {action_summary}"
        self._recording = None
        self._recording_name = None
        return result

    @property
    def is_recording(self) -> bool:
        return self._recording is not None

    def invoke(self, name: str, executor: Callable[[dict], None]) -> str:
        if name not in self.behaviors:
            return f"I don't know '{name}'. Teach me first."

        behavior = self.behaviors[name]
        for action in behavior.actions:
            executor(action)

        return f"Completed '{name}' ({len(behavior.actions)} actions)"

    def list_behaviors(self) -> list[str]:
        return list(self.behaviors.keys())

    def forget(self, name: str) -> str:
        if name in self.behaviors:
            del self.behaviors[name]
            self._save()
            return f"Forgot '{name}'."
        return f"I don't know '{name}'."
```

### Main Runtime

```python
# actor_runtime/runtime.py

from dataclasses import dataclass, field
from typing import Callable
import time

from .agent import Agent, ResourceBudget
from .event_log import EventLog, Event
from .teaching import TeachingSystem


class ActorRuntime:
    def __init__(self, event_path: str = "/opt/continuonos/brain/events"):
        self.agents: dict[str, Agent] = {}
        self.event_log = EventLog(event_path)
        self.teaching = TeachingSystem()
        self._action_count = 0
        self._checkpoint_interval = 50  # Checkpoint every 50 actions

    def spawn(self, agent_type: str, config: dict | None = None) -> Agent:
        agent_id = f"{agent_type}_{int(time.time() * 1000)}"
        agent = Agent(
            id=agent_id,
            agent_type=agent_type,
            state=config or {},
        )
        self.agents[agent_id] = agent

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="spawn",
            agent_id=agent_id,
            payload={"agent_type": agent_type, "config": config},
        ))

        return agent

    def send(self, agent_id: str, message: dict):
        if agent_id in self.agents:
            self.agents[agent_id].receive(message)

            self.event_log.append(Event(
                timestamp=time.time(),
                event_type="message",
                agent_id=agent_id,
                payload=message,
            ))

    def execute_action(self, action: dict, executor: Callable[[dict], None]):
        """Execute an action, recording it if teaching."""
        executor(action)

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="action",
            agent_id="runtime",
            payload=action,
        ))

        # Record if teaching
        if self.teaching.is_recording:
            self.teaching.record_action(action)

        # Auto-checkpoint
        self._action_count += 1
        if self._action_count >= self._checkpoint_interval:
            self.checkpoint()
            self._action_count = 0

    def checkpoint(self) -> str:
        return self.event_log.checkpoint(self.agents)

    def suspend(self, agent_id: str):
        if agent_id in self.agents:
            self.agents[agent_id].suspend()
            self.event_log.append(Event(
                timestamp=time.time(),
                event_type="suspend",
                agent_id=agent_id,
                payload={},
            ))

    def resume(self, agent_id: str):
        if agent_id in self.agents:
            self.agents[agent_id].resume()
            self.event_log.append(Event(
                timestamp=time.time(),
                event_type="resume",
                agent_id=agent_id,
                payload={},
            ))

    def restore_from_checkpoint(self, checkpoint_id: str | None = None):
        """Restore runtime state from checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = self.event_log.latest_checkpoint()

        if checkpoint_id:
            self.agents = self.event_log.restore(checkpoint_id)
            return f"Restored from {checkpoint_id}"
        return "No checkpoint found"

    def shutdown(self):
        """Clean shutdown with final checkpoint."""
        self.checkpoint()
        for agent in self.agents.values():
            agent.suspend()
```

---

## Example: Complete RC Car Session

```python
# main.py

from actor_runtime import ActorRuntime
from conversation import ConversationHandler
from hardware import MotorController, SafetyMonitor

def main():
    # Initialize
    runtime = ActorRuntime()
    motors = MotorController()
    safety = SafetyMonitor(motors)

    # Restore from last session if available
    runtime.restore_from_checkpoint()

    # Spawn the driver agent
    driver = runtime.spawn("driver", {"max_speed": 0.5})

    # Action executor
    def execute(action: dict):
        safety.check()  # Always check safety first

        if action["type"] == "move":
            motors.set_speed(action["left"], action["right"])
        elif action["type"] == "stop":
            motors.stop()

    # Conversation handler
    conversation = ConversationHandler(runtime, execute)

    print("Robot ready. Say something or type 'quit' to exit.")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                break

            response = conversation.handle(user_input)
            print(f"Bot: {response}")

    finally:
        motors.stop()
        runtime.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
```

---

## Migration Path

### Phase 1: Standalone Prototype (This Week)
- Implement minimal runtime in `actor_runtime/`
- Test with RC car hardware
- Validate teaching flow works

### Phase 2: Integrate with Existing (Next)
- Keep existing `actuators/` and `sensors/`
- Replace Ralph layers with single ActorRuntime
- Simplify BrainService to use new runtime

### Phase 3: Optional Enhancements (Later, If Needed)
- Add LLM for complex intent understanding
- Add vision-based behaviors
- Add multi-robot coordination

---

## Key Insight

The current system tries to be a **general-purpose robotics brain**.

What you need is a **teachable RC car companion**.

Build the second thing first. The first thing can grow from it naturally.

---

## Comparison: Claude Cowork vs This Design

| Aspect | Claude Cowork | ContinuonBrain Actor Runtime |
|--------|---------------|------------------------------|
| Interaction | Chat + actions | Chat + actions |
| State | Conversation history | Event log + checkpoints |
| Learning | In-context examples | Recorded behaviors |
| Recovery | Resume conversation | Replay from checkpoint |
| Execution | Tool calls | Action primitives |
| Boundaries | Token limits | Resource budgets |

The mental model is the same: **a capable assistant that listens, acts, and learns**.

---

## Next Steps

1. Review this design
2. Decide what to prototype first
3. Build minimal `actor_runtime/` module
4. Test with actual RC car hardware
5. Iterate based on what works

The goal is a robot you can talk to and teach, not a distributed robotics operating system.
