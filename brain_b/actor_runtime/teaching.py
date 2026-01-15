"""
Teaching System - Record and replay behaviors.

This is how you teach the robot:
1. Say "teach patrol"
2. Show it what to do (forward, left, forward, left)
3. Say "done"
4. Later, say "patrol" to replay
"""

from dataclasses import dataclass, field
from typing import Callable
import json
import time
import os


@dataclass
class Behavior:
    """A learned behavior - a named sequence of actions."""

    name: str
    actions: list[dict]
    created_at: float
    description: str = ""
    times_invoked: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "actions": self.actions,
            "created_at": self.created_at,
            "description": self.description,
            "times_invoked": self.times_invoked,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Behavior":
        return cls(
            name=data["name"],
            actions=data["actions"],
            created_at=data["created_at"],
            description=data.get("description", ""),
            times_invoked=data.get("times_invoked", 0),
        )


class TeachingSystem:
    """
    Records action sequences and replays them as named behaviors.

    Usage:
        teaching = TeachingSystem()
        teaching.start_recording("patrol")
        teaching.record_action({"type": "forward", "speed": 0.5})
        teaching.record_action({"type": "left", "degrees": 90})
        teaching.stop_recording()

        # Later...
        teaching.invoke("patrol", executor_func)
    """

    def __init__(self, path: str = "./brain_b_data/behaviors"):
        self.path = path
        self.behaviors_file = os.path.join(path, "behaviors.json")
        os.makedirs(path, exist_ok=True)

        self.behaviors: dict[str, Behavior] = {}
        self._recording: list[dict] | None = None
        self._recording_name: str | None = None

        self._load()

    def _load(self):
        """Load behaviors from disk."""
        if os.path.exists(self.behaviors_file):
            try:
                with open(self.behaviors_file) as f:
                    data = json.load(f)
                    self.behaviors = {
                        name: Behavior.from_dict(bdata)
                        for name, bdata in data.items()
                    }
            except (json.JSONDecodeError, KeyError):
                self.behaviors = {}

    def _save(self):
        """Save behaviors to disk."""
        with open(self.behaviors_file, "w") as f:
            json.dump(
                {name: b.to_dict() for name, b in self.behaviors.items()},
                f,
                indent=2,
            )

    def start_recording(self, name: str) -> str:
        """Start recording a new behavior."""
        if self._recording is not None:
            return f"Already recording '{self._recording_name}'. Say 'done' first."

        self._recording = []
        self._recording_name = name
        return f"Ready to learn '{name}'. Show me what to do."

    def record_action(self, action: dict) -> bool:
        """Record an action if currently teaching."""
        if self._recording is not None:
            self._recording.append(action)
            return True
        return False

    def stop_recording(self) -> str:
        """Stop recording and save the behavior."""
        if self._recording is None or self._recording_name is None:
            return "Not currently recording."

        if not self._recording:
            self._recording = None
            self._recording_name = None
            return "Nothing recorded. Cancelled."

        behavior = Behavior(
            name=self._recording_name,
            actions=self._recording.copy(),
            created_at=time.time(),
        )
        self.behaviors[self._recording_name] = behavior
        self._save()

        # Create summary
        action_types = [a.get("type", "?") for a in self._recording]
        if len(action_types) > 5:
            summary = " -> ".join(action_types[:5]) + f" ... ({len(action_types)} total)"
        else:
            summary = " -> ".join(action_types)

        result = f"Learned '{self._recording_name}': {summary}"

        self._recording = None
        self._recording_name = None

        return result

    def cancel_recording(self) -> str:
        """Cancel current recording without saving."""
        if self._recording is None:
            return "Not currently recording."

        name = self._recording_name
        self._recording = None
        self._recording_name = None
        return f"Cancelled recording '{name}'."

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording is not None

    @property
    def recording_name(self) -> str | None:
        """Get name of behavior being recorded."""
        return self._recording_name

    @property
    def recording_length(self) -> int:
        """Get number of actions recorded so far."""
        return len(self._recording) if self._recording else 0

    def invoke(
        self,
        name: str,
        executor: Callable[[dict], None],
        on_step: Callable[[int, int, dict], None] | None = None,
    ) -> str:
        """
        Execute a learned behavior.

        Args:
            name: Behavior name
            executor: Function to execute each action
            on_step: Optional callback(step_num, total_steps, action) for progress
        """
        if name not in self.behaviors:
            similar = self._find_similar(name)
            if similar:
                return f"I don't know '{name}'. Did you mean '{similar}'?"
            return f"I don't know '{name}'. Teach me first."

        behavior = self.behaviors[name]
        total = len(behavior.actions)

        for i, action in enumerate(behavior.actions):
            if on_step:
                on_step(i + 1, total, action)
            executor(action)

        behavior.times_invoked += 1
        self._save()

        return f"Completed '{name}' ({total} actions)"

    def _find_similar(self, name: str) -> str | None:
        """Find a similar behavior name (simple prefix match)."""
        name_lower = name.lower()
        for known in self.behaviors:
            if known.lower().startswith(name_lower) or name_lower.startswith(known.lower()):
                return known
        return None

    def list_behaviors(self) -> list[str]:
        """List all known behavior names."""
        return list(self.behaviors.keys())

    def get_behavior(self, name: str) -> Behavior | None:
        """Get a behavior by name."""
        return self.behaviors.get(name)

    def forget(self, name: str) -> str:
        """Delete a learned behavior."""
        if name not in self.behaviors:
            return f"I don't know '{name}'."

        del self.behaviors[name]
        self._save()
        return f"Forgot '{name}'."

    def forget_all(self) -> str:
        """Delete all learned behaviors."""
        count = len(self.behaviors)
        self.behaviors.clear()
        self._save()
        return f"Forgot {count} behaviors."

    def describe(self, name: str) -> str:
        """Get a description of a behavior."""
        if name not in self.behaviors:
            return f"I don't know '{name}'."

        b = self.behaviors[name]
        action_types = [a.get("type", "?") for a in b.actions]
        summary = " -> ".join(action_types[:10])
        if len(action_types) > 10:
            summary += " ..."

        return f"'{name}': {summary} ({len(b.actions)} actions, invoked {b.times_invoked}x)"
