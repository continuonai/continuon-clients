"""
Agent - The fundamental unit of execution in Brain B.

Each agent is isolated with its own state, mailbox, and resource budget.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Any
import time


@dataclass
class ResourceBudget:
    """Resource limits for an agent to prevent runaway execution."""

    max_actions: int = 100
    timeout_ms: int = 30_000
    actions_used: int = 0
    start_time: float = field(default_factory=time.time)

    def check(self) -> bool:
        """Check if agent is within budget."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        return self.actions_used < self.max_actions and elapsed_ms < self.timeout_ms

    def consume(self, count: int = 1):
        """Consume actions from budget."""
        self.actions_used += count

    def reset(self):
        """Reset budget for new execution cycle."""
        self.actions_used = 0
        self.start_time = time.time()

    def remaining(self) -> dict:
        """Get remaining budget info."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        return {
            "actions_remaining": self.max_actions - self.actions_used,
            "time_remaining_ms": max(0, self.timeout_ms - elapsed_ms),
        }


@dataclass
class Agent:
    """
    An isolated execution unit with its own state and message queue.

    Agents communicate via mailboxes (async message passing) and have
    resource budgets that prevent runaway execution.
    """

    id: str
    agent_type: str
    state: dict = field(default_factory=dict)
    mailbox: deque = field(default_factory=deque)
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    suspended: bool = False

    def receive(self, message: dict):
        """Add a message to the agent's mailbox."""
        self.mailbox.append({
            "payload": message,
            "received_at": time.time(),
        })

    def process_next(self) -> dict | None:
        """
        Process the next message from the mailbox.

        Returns None if suspended, mailbox empty, or budget exhausted.
        """
        if self.suspended:
            return None
        if not self.mailbox:
            return None
        if not self.budget.check():
            return None

        message = self.mailbox.popleft()
        self.budget.consume()
        return message["payload"]

    def has_messages(self) -> bool:
        """Check if there are pending messages."""
        return len(self.mailbox) > 0

    def suspend(self):
        """Pause agent execution."""
        self.suspended = True

    def resume(self):
        """Resume agent execution with fresh budget."""
        self.suspended = False
        self.budget.reset()

    def update_state(self, key: str, value: Any):
        """Update agent state."""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from agent state."""
        return self.state.get(key, default)

    def to_dict(self) -> dict:
        """Serialize agent for checkpointing."""
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "state": self.state,
            "mailbox": list(self.mailbox),
            "suspended": self.suspended,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        """Restore agent from checkpoint."""
        agent = cls(
            id=data["id"],
            agent_type=data["agent_type"],
            state=data.get("state", {}),
            suspended=data.get("suspended", False),
        )
        agent.mailbox = deque(data.get("mailbox", []))
        return agent

    def __repr__(self) -> str:
        return f"Agent({self.id}, type={self.agent_type}, msgs={len(self.mailbox)}, suspended={self.suspended})"
