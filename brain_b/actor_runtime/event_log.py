"""
Event Log - Append-only log for deterministic replay and crash recovery.

This is the killer feature. Every action is recorded, so you can:
- Recover from crashes by replaying from the last checkpoint
- Debug by seeing exactly what happened
- Reproduce any execution with the same events
"""

from dataclasses import dataclass, field
from typing import Iterator, TYPE_CHECKING
import json
import time
import os

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class Event:
    """A single recorded event in the log."""

    timestamp: float
    event_type: str  # spawn, message, action, checkpoint, teach, invoke, suspend, resume
    agent_id: str
    payload: dict
    checkpoint_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "checkpoint_id": self.checkpoint_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            agent_id=data["agent_id"],
            payload=data["payload"],
            checkpoint_id=data.get("checkpoint_id"),
        )


class EventLog:
    """
    Append-only event log with checkpointing.

    Events are written to a JSONL file. Checkpoints save full agent state
    to separate files for fast recovery.
    """

    def __init__(self, path: str = "./brain_b_data"):
        self.path = path
        self.log_file = os.path.join(path, "events.jsonl")
        self.checkpoint_dir = os.path.join(path, "checkpoints")
        self.behaviors_dir = os.path.join(path, "behaviors")

        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.behaviors_dir, exist_ok=True)

        # In-memory recent events for quick access
        self._recent: list[Event] = []
        self._max_recent = 100

    def append(self, event: Event):
        """Append an event to the log."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

        self._recent.append(event)
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)

    def checkpoint(self, agents: dict[str, "Agent"], metadata: dict | None = None) -> str:
        """
        Create a checkpoint of all agent states.

        Returns the checkpoint ID for later restoration.
        """
        checkpoint_id = f"ckpt_{int(time.time() * 1000)}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
            "agents": {aid: agent.to_dict() for aid, agent in agents.items()},
            "metadata": metadata or {},
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Log the checkpoint event
        self.append(Event(
            timestamp=time.time(),
            event_type="checkpoint",
            agent_id="runtime",
            payload={"checkpoint_id": checkpoint_id},
            checkpoint_id=checkpoint_id,
        ))

        return checkpoint_id

    def restore(self, checkpoint_id: str) -> tuple[dict[str, "Agent"], dict]:
        """
        Restore agents from a checkpoint.

        Returns (agents_dict, metadata).
        """
        from .agent import Agent

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(checkpoint_path) as f:
            data = json.load(f)

        agents = {
            aid: Agent.from_dict(adata)
            for aid, adata in data.get("agents", {}).items()
        }

        return agents, data.get("metadata", {})

    def latest_checkpoint(self) -> str | None:
        """Get the most recent checkpoint ID."""
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoints = sorted([
            f.replace(".json", "")
            for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".json")
        ])

        return checkpoints[-1] if checkpoints else None

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            return []

        return sorted([
            f.replace(".json", "")
            for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".json")
        ])

    def replay(self, from_checkpoint: str | None = None) -> Iterator[Event]:
        """
        Replay events from the log.

        If from_checkpoint is provided, starts replaying after that checkpoint.
        """
        if not os.path.exists(self.log_file):
            return

        start_replaying = from_checkpoint is None

        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                event = Event.from_dict(json.loads(line))

                if not start_replaying:
                    if event.checkpoint_id == from_checkpoint:
                        start_replaying = True
                    continue

                yield event

    def recent_events(self, count: int = 10) -> list[Event]:
        """Get recent events from memory."""
        return self._recent[-count:]

    def clear(self):
        """Clear all logs and checkpoints. Use with caution."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        for f in os.listdir(self.checkpoint_dir):
            os.remove(os.path.join(self.checkpoint_dir, f))

        self._recent.clear()
