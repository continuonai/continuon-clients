"""
Task System

Defines tasks and the task queue for the orchestrator.
"""

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class TaskPriority(Enum):
    """Priority levels for tasks."""

    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


class TaskType(Enum):
    """Built-in task types."""

    # Training tasks
    TRAIN = "train"
    TRAIN_STEP = "train_step"
    VALIDATE = "validate"
    CHECKPOINT = "checkpoint"

    # Inference tasks
    INFERENCE = "inference"
    BATCH_INFERENCE = "batch_inference"

    # Data tasks
    LOAD_DATA = "load_data"
    PREPROCESS = "preprocess"
    GENERATE_EPISODES = "generate_episodes"

    # Benchmark tasks
    BENCHMARK = "benchmark"
    INFERENCE_TEST = "inference_test"

    # System tasks
    HEALTH_CHECK = "health_check"
    CLEANUP = "cleanup"
    EXPORT_MODEL = "export_model"

    # Custom
    CUSTOM = "custom"


@dataclass
class TaskResult:
    """Result of a completed task."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_sec: float = 0.0


@dataclass
class Task:
    """
    A unit of work to be executed by the orchestrator.

    Tasks can be submitted to the queue and executed by workers.
    """

    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[TaskResult] = None
    timeout_sec: Optional[int] = None
    retries: int = 0
    max_retries: int = 3
    worker_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Task") -> bool:
        """Compare tasks by priority for heap queue."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "params": self.params,
            "priority": self.priority.name,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": {
                "success": self.result.success,
                "data": self.result.data,
                "error": self.result.error,
                "duration_sec": self.result.duration_sec,
            } if self.result else None,
            "timeout_sec": self.timeout_sec,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "worker_id": self.worker_id,
            "parent_task_id": self.parent_task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        task = cls(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            params=data.get("params", {}),
            priority=TaskPriority[data.get("priority", "NORMAL")],
            status=TaskStatus[data.get("status", "PENDING")],
            created_at=datetime.fromisoformat(data["created_at"]),
            timeout_sec=data.get("timeout_sec"),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
            worker_id=data.get("worker_id"),
            parent_task_id=data.get("parent_task_id"),
            metadata=data.get("metadata", {}),
        )

        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        if data.get("result"):
            task.result = TaskResult(**data["result"])

        return task

    def mark_started(self, worker_id: str) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.worker_id = worker_id

    def mark_completed(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.result = TaskResult(success=False, error=error)

    def mark_cancelled(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retries < self.max_retries

    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


class TaskQueue:
    """
    Priority queue for tasks.

    Features:
    - Priority-based ordering
    - Thread-safe operations
    - Task lookup by ID
    - Queue statistics
    """

    def __init__(self, max_size: int = 1000):
        self._heap: List[Task] = []
        self._task_map: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._not_empty = threading.Condition(self._lock)

    def put(self, task: Task) -> bool:
        """
        Add a task to the queue.

        Returns False if queue is full.
        """
        with self._lock:
            if len(self._heap) >= self._max_size:
                logger.warning(f"Queue full, rejecting task {task.task_id}")
                return False

            task.status = TaskStatus.QUEUED
            heapq.heappush(self._heap, task)
            self._task_map[task.task_id] = task
            self._not_empty.notify()
            logger.debug(f"Task {task.task_id} added to queue")
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Get the highest priority task from the queue.

        Blocks until a task is available or timeout expires.
        """
        with self._not_empty:
            if timeout is not None:
                end_time = time.time() + timeout
                while not self._heap:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(remaining)
            else:
                while not self._heap:
                    self._not_empty.wait()

            if not self._heap:
                return None

            task = heapq.heappop(self._heap)
            return task

    def get_nowait(self) -> Optional[Task]:
        """Get task without blocking."""
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def peek(self) -> Optional[Task]:
        """Peek at the next task without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0]

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        with self._lock:
            return self._task_map.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task."""
        with self._lock:
            task = self._task_map.get(task_id)
            if task and task.status == TaskStatus.QUEUED:
                task.mark_cancelled()
                # Remove from heap (expensive but rare)
                self._heap = [t for t in self._heap if t.task_id != task_id]
                heapq.heapify(self._heap)
                return True
            return False

    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def clear(self) -> int:
        """Clear all queued tasks. Returns count of cleared tasks."""
        with self._lock:
            count = len(self._heap)
            for task in self._heap:
                task.mark_cancelled()
            self._heap.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            by_priority = {}
            by_type = {}
            for task in self._heap:
                pname = task.priority.name
                tname = task.task_type.value
                by_priority[pname] = by_priority.get(pname, 0) + 1
                by_type[tname] = by_type.get(tname, 0) + 1

            return {
                "size": len(self._heap),
                "max_size": self._max_size,
                "by_priority": by_priority,
                "by_type": by_type,
                "total_tracked": len(self._task_map),
            }

    def get_all_tasks(self, include_completed: bool = False) -> List[Task]:
        """Get all tracked tasks."""
        with self._lock:
            if include_completed:
                return list(self._task_map.values())
            return [
                t for t in self._task_map.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING)
            ]
