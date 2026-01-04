"""
State Manager

Persists and restores orchestrator state for recovery.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .task import Task, TaskStatus
from .workflow import Workflow, WorkflowStatus

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    """Snapshot of orchestrator state."""

    timestamp: datetime = field(default_factory=datetime.now)
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    workflows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tasks": self.tasks,
            "workflows": self.workflows,
            "metrics": self.metrics,
            "config": self.config,
            "custom_data": self.custom_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorState":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            tasks=data.get("tasks", {}),
            workflows=data.get("workflows", {}),
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            custom_data=data.get("custom_data", {}),
        )


class StateManager:
    """
    Manages orchestrator state persistence.

    Features:
    - Periodic auto-save
    - Manual save/restore
    - State versioning
    - Crash recovery
    """

    def __init__(
        self,
        state_file: Path = None,
        auto_save_interval_sec: int = 60,
        max_history: int = 10,
    ):
        self._state_file = state_file or Path("/tmp/orchestrator_state.json")
        self._auto_save_interval = auto_save_interval_sec
        self._max_history = max_history
        self._current_state = OrchestratorState()
        self._lock = threading.RLock()
        self._running = False
        self._save_thread: Optional[threading.Thread] = None
        self._dirty = False
        self._state_providers: Dict[str, callable] = {}

    def register_provider(
        self,
        name: str,
        provider: callable,
    ) -> None:
        """
        Register a state provider.

        Providers are called during save to collect current state.
        """
        self._state_providers[name] = provider

    def start(self) -> None:
        """Start the auto-save background thread."""
        if self._running:
            return

        self._running = True
        self._save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True,
            name="StateAutoSave",
        )
        self._save_thread.start()
        logger.info(f"State manager started (auto-save every {self._auto_save_interval}s)")

    def stop(self) -> None:
        """Stop auto-save and perform final save."""
        self._running = False
        if self._save_thread:
            self._save_thread.join(timeout=5)

        # Final save
        self.save()
        logger.info("State manager stopped")

    def _auto_save_loop(self) -> None:
        """Background loop for periodic saves."""
        elapsed = 0
        while self._running:
            time.sleep(1)  # Check every second for quick shutdown
            elapsed += 1
            if elapsed >= self._auto_save_interval:
                elapsed = 0
                if self._dirty:
                    try:
                        self.save()
                    except Exception as e:
                        logger.error(f"Auto-save failed: {e}")

    def mark_dirty(self) -> None:
        """Mark state as needing save."""
        self._dirty = True

    def save(self, force: bool = False) -> bool:
        """
        Save current state to file.

        Returns True if save was successful.
        """
        if not force and not self._dirty:
            return True

        with self._lock:
            try:
                # Collect state from providers
                self._collect_state()

                # Update timestamp
                self._current_state.timestamp = datetime.now()

                # Ensure directory exists
                self._state_file.parent.mkdir(parents=True, exist_ok=True)

                # Write to temp file first
                temp_file = self._state_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(self._current_state.to_dict(), f, indent=2, default=str)

                # Rotate old state files
                self._rotate_history()

                # Atomic rename
                temp_file.rename(self._state_file)

                self._dirty = False
                logger.debug(f"State saved to {self._state_file}")
                return True

            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                return False

    def _collect_state(self) -> None:
        """Collect state from registered providers."""
        for name, provider in self._state_providers.items():
            try:
                data = provider()
                if name == "tasks":
                    self._current_state.tasks = data
                elif name == "workflows":
                    self._current_state.workflows = data
                elif name == "metrics":
                    self._current_state.metrics = data
                elif name == "config":
                    self._current_state.config = data
                else:
                    self._current_state.custom_data[name] = data
            except Exception as e:
                logger.error(f"State provider '{name}' error: {e}")

    def _rotate_history(self) -> None:
        """Rotate old state files."""
        if not self._state_file.exists():
            return

        # Rename existing state files
        for i in range(self._max_history - 1, 0, -1):
            old_file = self._state_file.with_suffix(f".{i}.json")
            new_file = self._state_file.with_suffix(f".{i + 1}.json")
            if old_file.exists():
                if i + 1 >= self._max_history:
                    old_file.unlink()
                else:
                    old_file.rename(new_file)

        # Move current to .1
        backup = self._state_file.with_suffix(".1.json")
        if self._state_file.exists():
            self._state_file.rename(backup)

    def load(self) -> bool:
        """
        Load state from file.

        Returns True if load was successful.
        """
        with self._lock:
            try:
                if not self._state_file.exists():
                    logger.info("No state file found, starting fresh")
                    return False

                with open(self._state_file) as f:
                    data = json.load(f)

                self._current_state = OrchestratorState.from_dict(data)
                logger.info(f"State loaded from {self._state_file}")
                return True

            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return False

    def get_state(self) -> OrchestratorState:
        """Get current state."""
        with self._lock:
            return self._current_state

    def set_custom_data(self, key: str, value: Any) -> None:
        """Set custom data in state."""
        with self._lock:
            self._current_state.custom_data[key] = value
            self._dirty = True

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """Get custom data from state."""
        with self._lock:
            return self._current_state.custom_data.get(key, default)

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get saved task state."""
        return self._current_state.tasks.get(task_id)

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get saved workflow state."""
        return self._current_state.workflows.get(workflow_id)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get list of available state history files."""
        history = []
        if self._state_file.exists():
            stat = self._state_file.stat()
            history.append({
                "file": str(self._state_file),
                "version": 0,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        for i in range(1, self._max_history + 1):
            backup = self._state_file.with_suffix(f".{i}.json")
            if backup.exists():
                stat = backup.stat()
                history.append({
                    "file": str(backup),
                    "version": i,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        return history

    def restore_from_history(self, version: int) -> bool:
        """Restore state from a history version."""
        if version == 0:
            return self.load()

        backup = self._state_file.with_suffix(f".{version}.json")
        if not backup.exists():
            logger.error(f"History version {version} not found")
            return False

        with self._lock:
            try:
                with open(backup) as f:
                    data = json.load(f)
                self._current_state = OrchestratorState.from_dict(data)
                self._dirty = True
                logger.info(f"State restored from version {version}")
                return True
            except Exception as e:
                logger.error(f"Failed to restore from version {version}: {e}")
                return False

    def clear(self) -> None:
        """Clear current state."""
        with self._lock:
            self._current_state = OrchestratorState()
            self._dirty = True

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        history = self.get_history()
        return {
            "state_file": str(self._state_file),
            "auto_save_interval_sec": self._auto_save_interval,
            "dirty": self._dirty,
            "running": self._running,
            "num_tasks": len(self._current_state.tasks),
            "num_workflows": len(self._current_state.workflows),
            "history_versions": len(history),
            "last_save": self._current_state.timestamp.isoformat(),
        }
