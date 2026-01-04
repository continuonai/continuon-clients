"""
Worker Pool System

Manages worker threads that execute tasks from the queue.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import logging
import uuid

from .task import Task, TaskResult, TaskStatus, TaskType, TaskQueue
from .events import EventBus, EventType, Event

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of a worker."""

    IDLE = auto()
    BUSY = auto()
    STOPPED = auto()


@dataclass
class Worker:
    """Represents a single worker in the pool."""

    worker_id: str = field(default_factory=lambda: f"worker_{uuid.uuid4().hex[:8]}")
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[Task] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "status": self.status.name,
            "current_task": self.current_task.task_id if self.current_task else None,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "started_at": self.started_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


# Type alias for task handlers
TaskHandler = Callable[[Task], TaskResult]


class WorkerPool:
    """
    Pool of workers that execute tasks.

    Features:
    - Configurable number of workers
    - Task handler registration
    - Automatic task dispatch
    - Worker health monitoring
    """

    def __init__(
        self,
        num_workers: int = 4,
        task_queue: Optional[TaskQueue] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self._num_workers = num_workers
        self._task_queue = task_queue or TaskQueue()
        self._event_bus = event_bus or EventBus()
        self._workers: Dict[str, Worker] = {}
        self._handlers: Dict[TaskType, TaskHandler] = {}
        self._default_handler: Optional[TaskHandler] = None
        self._running = False
        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_threads: List[threading.Thread] = []
        self._futures: Dict[str, Future] = {}

    def register_handler(
        self,
        task_type: TaskType,
        handler: TaskHandler,
    ) -> None:
        """Register a handler for a task type."""
        with self._lock:
            self._handlers[task_type] = handler
            logger.info(f"Registered handler for {task_type.value}")

    def set_default_handler(self, handler: TaskHandler) -> None:
        """Set the default handler for unregistered task types."""
        self._default_handler = handler

    def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            logger.warning("Worker pool already running")
            return

        self._running = True
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_workers,
            thread_name_prefix="OrchestratorWorker",
        )

        # Create worker metadata
        for i in range(self._num_workers):
            worker = Worker(worker_id=f"worker_{i}")
            self._workers[worker.worker_id] = worker

        # Start worker threads
        for worker_id in self._workers:
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True,
                name=f"WorkerLoop-{worker_id}",
            )
            self._worker_threads.append(thread)
            thread.start()

        self._event_bus.emit(
            EventType.WORKER_STARTED,
            {"num_workers": self._num_workers},
            source="worker_pool",
        )
        logger.info(f"Worker pool started with {self._num_workers} workers")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker pool."""
        if not self._running:
            return

        logger.info("Stopping worker pool...")
        self._running = False

        # Wait for worker threads
        for thread in self._worker_threads:
            thread.join(timeout=timeout / self._num_workers)

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        # Update worker statuses
        for worker in self._workers.values():
            worker.status = WorkerStatus.STOPPED

        self._event_bus.emit(
            EventType.WORKER_STOPPED,
            {"num_workers": self._num_workers},
            source="worker_pool",
        )
        logger.info("Worker pool stopped")

    def _worker_loop(self, worker_id: str) -> None:
        """Main loop for a worker thread."""
        worker = self._workers[worker_id]
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get next task (blocks with timeout)
                task = self._task_queue.get(timeout=1.0)
                if task is None:
                    continue

                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    continue

                # Execute task
                self._execute_task(worker, task)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)

        logger.debug(f"Worker {worker_id} stopped")

    def _execute_task(self, worker: Worker, task: Task) -> None:
        """Execute a single task."""
        worker.status = WorkerStatus.BUSY
        worker.current_task = task
        worker.last_active = datetime.now()
        task.mark_started(worker.worker_id)

        self._event_bus.emit(
            EventType.TASK_STARTED,
            {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "worker_id": worker.worker_id,
            },
            source="worker_pool",
        )

        start_time = time.time()
        try:
            # Get handler
            handler = self._handlers.get(task.task_type, self._default_handler)
            if handler is None:
                raise ValueError(f"No handler for task type: {task.task_type.value}")

            # Execute with timeout
            if task.timeout_sec:
                future = self._executor.submit(handler, task)
                result = future.result(timeout=task.timeout_sec)
            else:
                result = handler(task)

            duration = time.time() - start_time
            result.duration_sec = duration
            task.mark_completed(result)
            worker.tasks_completed += 1

            self._event_bus.emit(
                EventType.TASK_COMPLETED,
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "success": result.success,
                    "duration_sec": duration,
                    "data": result.data,
                },
                source="worker_pool",
            )
            logger.info(f"Task {task.task_id} completed in {duration:.2f}s")

        except TimeoutError:
            duration = time.time() - start_time
            task.status = TaskStatus.TIMEOUT
            task.completed_at = datetime.now()
            task.result = TaskResult(
                success=False,
                error=f"Task timed out after {task.timeout_sec}s",
                duration_sec=duration,
            )
            worker.tasks_failed += 1
            self._handle_task_failure(task, "Timeout")

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            task.mark_failed(error_msg)
            task.result.duration_sec = duration
            worker.tasks_failed += 1
            self._handle_task_failure(task, error_msg)

        finally:
            worker.status = WorkerStatus.IDLE
            worker.current_task = None
            worker.last_active = datetime.now()

    def _handle_task_failure(self, task: Task, error: str) -> None:
        """Handle a failed task, possibly retrying."""
        self._event_bus.emit(
            EventType.TASK_FAILED,
            {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "error": error,
                "retries": task.retries,
                "max_retries": task.max_retries,
            },
            source="worker_pool",
        )

        if task.can_retry():
            task.retries += 1
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.completed_at = None
            task.result = None
            self._task_queue.put(task)
            logger.warning(f"Task {task.task_id} failed, retry {task.retries}/{task.max_retries}")
        else:
            logger.error(f"Task {task.task_id} failed permanently: {error}")

    def submit(self, task: Task) -> bool:
        """Submit a task to the queue."""
        return self._task_queue.put(task)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        return self._task_queue.cancel(task_id)

    def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get a worker by ID."""
        return self._workers.get(worker_id)

    def get_workers(self) -> List[Worker]:
        """Get all workers."""
        return list(self._workers.values())

    def get_idle_workers(self) -> List[Worker]:
        """Get idle workers."""
        return [w for w in self._workers.values() if w.status == WorkerStatus.IDLE]

    def get_busy_workers(self) -> List[Worker]:
        """Get busy workers."""
        return [w for w in self._workers.values() if w.status == WorkerStatus.BUSY]

    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            workers = list(self._workers.values())
            return {
                "num_workers": self._num_workers,
                "running": self._running,
                "idle_workers": len([w for w in workers if w.status == WorkerStatus.IDLE]),
                "busy_workers": len([w for w in workers if w.status == WorkerStatus.BUSY]),
                "total_completed": sum(w.tasks_completed for w in workers),
                "total_failed": sum(w.tasks_failed for w in workers),
                "queue_size": self._task_queue.size(),
                "registered_handlers": list(self._handlers.keys()),
            }
