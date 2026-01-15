"""
Actor Runtime - The core orchestrator for Brain B.

Manages agent lifecycle, event logging, and teaching integration.
"""

from typing import Callable
import time

from .agent import Agent, ResourceBudget
from .event_log import EventLog, Event
from .teaching import TeachingSystem


class ActorRuntime:
    """
    The main runtime that coordinates everything.

    Responsibilities:
    - Spawn and manage agents
    - Route messages between agents
    - Checkpoint and restore state
    - Integrate with teaching system
    - Enforce resource budgets
    """

    def __init__(
        self,
        data_path: str = "./brain_b_data",
        checkpoint_interval: int = 50,
        auto_restore: bool = True,
    ):
        self.data_path = data_path
        self.agents: dict[str, Agent] = {}
        self.event_log = EventLog(data_path)
        self.teaching = TeachingSystem(f"{data_path}/behaviors")

        self._action_count = 0
        self._checkpoint_interval = checkpoint_interval
        self._started_at = time.time()

        # Auto-restore from last checkpoint
        if auto_restore:
            self._try_restore()

    def _try_restore(self):
        """Try to restore from the latest checkpoint."""
        latest = self.event_log.latest_checkpoint()
        if latest:
            try:
                self.agents, metadata = self.event_log.restore(latest)
                print(f"[Runtime] Restored from {latest} ({len(self.agents)} agents)")
            except Exception as e:
                print(f"[Runtime] Could not restore: {e}")

    # === Agent Lifecycle ===

    def spawn(
        self,
        agent_type: str,
        agent_id: str | None = None,
        config: dict | None = None,
        budget: ResourceBudget | None = None,
    ) -> Agent:
        """
        Spawn a new agent.

        Args:
            agent_type: Type identifier (e.g., "driver", "vision")
            agent_id: Optional custom ID (auto-generated if not provided)
            config: Initial state for the agent
            budget: Resource limits (uses defaults if not provided)
        """
        if agent_id is None:
            agent_id = f"{agent_type}_{int(time.time() * 1000)}"

        agent = Agent(
            id=agent_id,
            agent_type=agent_type,
            state=config or {},
            budget=budget or ResourceBudget(),
        )
        self.agents[agent_id] = agent

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="spawn",
            agent_id=agent_id,
            payload={"agent_type": agent_type, "config": config},
        ))

        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        """List all agents."""
        return list(self.agents.values())

    def kill(self, agent_id: str) -> bool:
        """Remove an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.event_log.append(Event(
                timestamp=time.time(),
                event_type="kill",
                agent_id=agent_id,
                payload={},
            ))
            return True
        return False

    # === Message Passing ===

    def send(self, agent_id: str, message: dict) -> bool:
        """Send a message to an agent's mailbox."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].receive(message)

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="message",
            agent_id=agent_id,
            payload=message,
        ))

        return True

    def broadcast(self, message: dict, agent_type: str | None = None):
        """Send a message to all agents (optionally filtered by type)."""
        for agent in self.agents.values():
            if agent_type is None or agent.agent_type == agent_type:
                agent.receive(message)

    # === Action Execution ===

    def execute_action(
        self,
        action: dict,
        executor: Callable[[dict], None],
        agent_id: str = "runtime",
    ):
        """
        Execute an action, recording it for teaching and logging.

        Args:
            action: The action to execute
            executor: Function that actually performs the action
            agent_id: Which agent is performing this (for logging)
        """
        # Execute the action
        executor(action)

        # Log it
        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="action",
            agent_id=agent_id,
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

    # === Checkpointing ===

    def checkpoint(self, metadata: dict | None = None) -> str:
        """Create a checkpoint of current state."""
        checkpoint_id = self.event_log.checkpoint(self.agents, metadata)
        self._action_count = 0
        return checkpoint_id

    def restore(self, checkpoint_id: str | None = None) -> str:
        """Restore from a checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = self.event_log.latest_checkpoint()

        if checkpoint_id is None:
            return "No checkpoint found."

        try:
            self.agents, metadata = self.event_log.restore(checkpoint_id)
            return f"Restored from {checkpoint_id} ({len(self.agents)} agents)"
        except FileNotFoundError:
            return f"Checkpoint not found: {checkpoint_id}"

    # === Suspend / Resume ===

    def suspend(self, agent_id: str) -> bool:
        """Suspend an agent."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].suspend()

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="suspend",
            agent_id=agent_id,
            payload={},
        ))

        return True

    def resume(self, agent_id: str) -> bool:
        """Resume a suspended agent."""
        if agent_id not in self.agents:
            return False

        self.agents[agent_id].resume()

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="resume",
            agent_id=agent_id,
            payload={},
        ))

        return True

    def suspend_all(self):
        """Suspend all agents."""
        for agent_id in self.agents:
            self.suspend(agent_id)

    # === Teaching Integration ===

    def teach(self, name: str) -> str:
        """Start recording a new behavior."""
        result = self.teaching.start_recording(name)

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="teach_start",
            agent_id="runtime",
            payload={"behavior_name": name},
        ))

        return result

    def done_teaching(self) -> str:
        """Stop recording and save the behavior."""
        result = self.teaching.stop_recording()

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="teach_done",
            agent_id="runtime",
            payload={"result": result},
        ))

        return result

    def invoke(
        self,
        name: str,
        executor: Callable[[dict], None],
        on_step: Callable[[int, int, dict], None] | None = None,
    ) -> str:
        """Execute a learned behavior."""
        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="invoke_start",
            agent_id="runtime",
            payload={"behavior_name": name},
        ))

        result = self.teaching.invoke(name, executor, on_step)

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="invoke_done",
            agent_id="runtime",
            payload={"behavior_name": name, "result": result},
        ))

        return result

    # === Lifecycle ===

    def shutdown(self):
        """Clean shutdown with final checkpoint."""
        self.checkpoint(metadata={"reason": "shutdown"})
        self.suspend_all()

        self.event_log.append(Event(
            timestamp=time.time(),
            event_type="shutdown",
            agent_id="runtime",
            payload={"uptime_s": time.time() - self._started_at},
        ))

    def status(self) -> dict:
        """Get runtime status."""
        return {
            "agents": len(self.agents),
            "agents_suspended": sum(1 for a in self.agents.values() if a.suspended),
            "behaviors": len(self.teaching.list_behaviors()),
            "is_recording": self.teaching.is_recording,
            "recording_name": self.teaching.recording_name,
            "checkpoints": len(self.event_log.list_checkpoints()),
            "uptime_s": time.time() - self._started_at,
        }
