"""
Core Orchestrator

Main orchestrator class that coordinates all components.
"""

import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import OrchestratorConfig
from .task import Task, TaskResult, TaskStatus, TaskType, TaskPriority, TaskQueue
from .worker import WorkerPool, TaskHandler
from .workflow import (
    Workflow,
    WorkflowEngine,
    create_training_workflow,
    create_inference_benchmark_workflow,
    create_full_development_pipeline,
)
from .events import EventBus, EventType, Event
from .state import StateManager
from .monitor import Monitor, HealthStatus

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator for ContinuonBrain.

    Coordinates:
    - Task submission and execution
    - Workflow management
    - Worker pool
    - State persistence
    - Health monitoring
    - Event distribution

    Usage:
        orch = Orchestrator()
        orch.start()

        # Submit a task
        task_id = orch.submit_task(TaskType.TRAIN, {"max_steps": 100})

        # Run a workflow
        wf_id = orch.run_workflow("training_pipeline")

        # Get status
        print(orch.get_status())

        orch.stop()
    """

    def __init__(self, config: OrchestratorConfig = None):
        self._config = config or OrchestratorConfig.development()
        self._running = False
        self._lock = threading.RLock()

        # Initialize components
        self._event_bus = EventBus(buffer_size=self._config.event_buffer_size)
        self._task_queue = TaskQueue(max_size=self._config.max_queue_size)
        self._worker_pool = WorkerPool(
            num_workers=self._config.max_workers,
            task_queue=self._task_queue,
            event_bus=self._event_bus,
        )
        self._workflow_engine = WorkflowEngine(event_bus=self._event_bus)
        self._state_manager = StateManager(
            state_file=self._config.state_file,
            auto_save_interval_sec=self._config.auto_save_interval_sec,
        )
        self._monitor = Monitor(
            event_bus=self._event_bus,
            check_interval_sec=self._config.health_check_interval_sec,
            metrics_retention_hours=self._config.metrics_retention_hours,
        )

        # Track tasks and workflows
        self._tasks: Dict[str, Task] = {}
        self._workflows: Dict[str, Workflow] = {}

        # Setup connections between components
        self._setup_components()

        # Register built-in handlers
        self._register_builtin_handlers()

        # Register built-in workflows
        self._register_builtin_workflows()

    def _setup_components(self) -> None:
        """Setup connections between orchestrator components."""
        # Connect workflow engine to task submission
        self._workflow_engine.set_task_submitter(self._worker_pool.submit)

        # Register state providers
        self._state_manager.register_provider("tasks", self._get_task_state)
        self._state_manager.register_provider("workflows", self._get_workflow_state)
        self._state_manager.register_provider("metrics", self._monitor.get_stats)

        # Subscribe to task events for tracking
        self._event_bus.subscribe(EventType.TASK_COMPLETED, self._on_task_completed)
        self._event_bus.subscribe(EventType.TASK_FAILED, self._on_task_failed)

    def _register_builtin_handlers(self) -> None:
        """Register built-in task handlers."""
        # Health check handler
        self._worker_pool.register_handler(
            TaskType.HEALTH_CHECK,
            self._handle_health_check,
        )

        # Cleanup handler
        self._worker_pool.register_handler(
            TaskType.CLEANUP,
            self._handle_cleanup,
        )

        # Data handlers
        self._worker_pool.register_handler(
            TaskType.LOAD_DATA,
            self._handle_load_data,
        )
        self._worker_pool.register_handler(
            TaskType.PREPROCESS,
            self._handle_preprocess,
        )

        # Training handlers
        self._worker_pool.register_handler(
            TaskType.TRAIN,
            self._handle_train,
        )
        self._worker_pool.register_handler(
            TaskType.VALIDATE,
            self._handle_validate,
        )

        # Benchmark handler
        self._worker_pool.register_handler(
            TaskType.BENCHMARK,
            self._handle_benchmark,
        )

        # Model persistence handlers
        self._worker_pool.register_handler(
            TaskType.CHECKPOINT,
            self._handle_checkpoint,
        )
        self._worker_pool.register_handler(
            TaskType.EXPORT_MODEL,
            self._handle_export_model,
        )

        # Set default handler for unregistered types
        self._worker_pool.set_default_handler(self._handle_default)

    def _register_builtin_workflows(self) -> None:
        """Register built-in workflow templates."""
        self._workflow_engine.register_template(
            "training_pipeline",
            create_training_workflow,
        )
        self._workflow_engine.register_template(
            "inference_benchmark",
            create_inference_benchmark_workflow,
        )
        self._workflow_engine.register_template(
            "full_development_pipeline",
            create_full_development_pipeline,
        )

    def start(self) -> None:
        """Start the orchestrator and all components."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting orchestrator...")

        # Load previous state
        self._state_manager.load()

        # Start components
        self._event_bus.start_dispatcher()
        self._worker_pool.start()
        self._state_manager.start()
        self._monitor.start()

        self._running = True

        # Emit started event
        self._event_bus.emit(
            EventType.ORCHESTRATOR_STARTED,
            {
                "config": {
                    "max_workers": self._config.max_workers,
                    "max_queue_size": self._config.max_queue_size,
                },
            },
            source="orchestrator",
        )

        logger.info("Orchestrator started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the orchestrator and all components."""
        if not self._running:
            return

        logger.info("Stopping orchestrator...")

        # Emit stopping event
        self._event_bus.emit(
            EventType.ORCHESTRATOR_STOPPED,
            {"reason": "shutdown"},
            source="orchestrator",
        )

        # Stop components in reverse order
        self._monitor.stop()
        self._worker_pool.stop(timeout=timeout / 2)
        self._state_manager.stop()
        self._event_bus.stop_dispatcher()

        self._running = False
        logger.info("Orchestrator stopped")

    def is_running(self) -> bool:
        """Check if orchestrator is running."""
        return self._running

    # Task management

    def submit_task(
        self,
        task_type: TaskType,
        params: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_sec: int = None,
    ) -> str:
        """
        Submit a task for execution.

        Returns the task ID.
        """
        task = Task(
            task_type=task_type,
            params=params or {},
            priority=priority,
            timeout_sec=timeout_sec or self._config.task_timeout_sec,
        )

        self._tasks[task.task_id] = task
        self._worker_pool.submit(task)

        self._event_bus.emit(
            EventType.TASK_SUBMITTED,
            {
                "task_id": task.task_id,
                "task_type": task_type.value,
                "priority": priority.name,
            },
            source="orchestrator",
        )

        self._state_manager.mark_dirty()
        return task.task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if task:
            success = self._worker_pool.cancel_task(task_id)
            if success:
                self._event_bus.emit(
                    EventType.TASK_CANCELLED,
                    {"task_id": task_id},
                    source="orchestrator",
                )
            return success
        return False

    def list_tasks(
        self,
        status: TaskStatus = None,
        task_type: TaskType = None,
        limit: int = 100,
    ) -> List[Task]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    # Workflow management

    def run_workflow(
        self,
        template_name: str,
        context: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Run a workflow from a template.

        Returns the workflow ID.
        """
        workflow = self._workflow_engine.create_from_template(template_name, context)
        if workflow is None:
            return None

        self._workflows[workflow.workflow_id] = workflow
        self._workflow_engine.submit(workflow)
        self._state_manager.mark_dirty()

        return workflow.workflow_id

    def submit_workflow(self, workflow: Workflow) -> str:
        """Submit a custom workflow."""
        self._workflows[workflow.workflow_id] = workflow
        self._workflow_engine.submit(workflow)
        self._state_manager.mark_dirty()
        return workflow.workflow_id

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        return self._workflow_engine.cancel_workflow(workflow_id)

    def list_workflows(self, limit: int = 100) -> List[Workflow]:
        """List all workflows."""
        workflows = list(self._workflows.values())
        workflows.sort(key=lambda w: w.created_at, reverse=True)
        return workflows[:limit]

    # Handler registration

    def register_handler(
        self,
        task_type: TaskType,
        handler: TaskHandler,
    ) -> None:
        """Register a custom task handler."""
        self._worker_pool.register_handler(task_type, handler)

    def register_workflow_template(
        self,
        name: str,
        builder: Callable[[], Workflow],
    ) -> None:
        """Register a custom workflow template."""
        self._workflow_engine.register_template(name, builder)

    # Event handling

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
    ) -> None:
        """Subscribe to orchestrator events."""
        self._event_bus.subscribe(event_type, handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
    ) -> None:
        """Unsubscribe from events."""
        self._event_bus.unsubscribe(event_type, handler)

    # Health and status

    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        health_checks = self._monitor.get_health()
        return {
            "overall": self._monitor.get_overall_health().name,
            "checks": {name: check.to_dict() for name, check in health_checks.items()},
        }

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "health": self.get_health(),
            "tasks": {
                "total": len(self._tasks),
                "queue_size": self._task_queue.size(),
                "by_status": self._count_tasks_by_status(),
            },
            "workflows": {
                "total": len(self._workflows),
                "active": len(self._workflow_engine.get_active_workflows()),
            },
            "workers": self._worker_pool.get_stats(),
            "events": self._event_bus.get_stats(),
            "state": self._state_manager.get_stats(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            "cpu": self._monitor.get_metric_summary("system.cpu.percent"),
            "memory": self._monitor.get_metric_summary("system.memory.percent"),
            "disk": self._monitor.get_metric_summary("system.disk.percent"),
        }

    # Internal methods

    def _count_tasks_by_status(self) -> Dict[str, int]:
        """Count tasks by status."""
        counts = {}
        for task in self._tasks.values():
            status = task.status.name
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _get_task_state(self) -> Dict[str, Any]:
        """Get task state for persistence."""
        return {
            task_id: task.to_dict()
            for task_id, task in self._tasks.items()
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        }

    def _get_workflow_state(self) -> Dict[str, Any]:
        """Get workflow state for persistence."""
        return {
            wf_id: wf.to_dict()
            for wf_id, wf in self._workflows.items()
        }

    def _on_task_completed(self, event: Event) -> None:
        """Handle task completed event."""
        task_id = event.data.get("task_id")
        if task_id:
            self._monitor.increment("tasks.completed")
            logger.debug(f"Task {task_id} completed")

    def _on_task_failed(self, event: Event) -> None:
        """Handle task failed event."""
        task_id = event.data.get("task_id")
        if task_id:
            self._monitor.increment("tasks.failed")
            logger.debug(f"Task {task_id} failed")

    # Built-in handlers

    def _handle_health_check(self, task: Task) -> TaskResult:
        """Handle health check task."""
        health = self._monitor.run_all_checks()
        overall = self._monitor.get_overall_health()
        return TaskResult(
            success=overall in (HealthStatus.HEALTHY, HealthStatus.DEGRADED),
            data={
                "overall": overall.name,
                "checks": {name: check.to_dict() for name, check in health.items()},
            },
        )

    def _handle_cleanup(self, task: Task) -> TaskResult:
        """Handle cleanup task."""
        # Clean up old completed tasks
        cutoff = datetime.now()
        max_age_hours = task.params.get("max_age_hours", 24)
        cleaned = 0

        for task_id in list(self._tasks.keys()):
            t = self._tasks[task_id]
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if t.completed_at and (cutoff - t.completed_at).total_seconds() > max_age_hours * 3600:
                    del self._tasks[task_id]
                    cleaned += 1

        return TaskResult(
            success=True,
            data={"cleaned_tasks": cleaned},
        )

    def _handle_default(self, task: Task) -> TaskResult:
        """Default handler for unregistered task types - simulates success."""
        logger.info(f"Simulating task: {task.task_type.value} with params: {task.params}")
        time.sleep(0.5)  # Simulate some work
        return TaskResult(
            success=True,
            data={
                "task_type": task.task_type.value,
                "params": task.params,
                "simulated": True,
            },
        )

    def _handle_load_data(self, task: Task) -> TaskResult:
        """Handle load data task."""
        logger.info(f"Loading data from: {task.params.get('source', 'default')}")
        time.sleep(0.5)
        return TaskResult(
            success=True,
            data={"records_loaded": 100, "source": task.params.get('source', 'rlds_episodes')},
        )

    def _handle_preprocess(self, task: Task) -> TaskResult:
        """Handle data preprocessing task."""
        logger.info(f"Preprocessing with normalize={task.params.get('normalize', True)}")
        time.sleep(0.5)
        return TaskResult(
            success=True,
            data={"records_processed": 100, "normalized": task.params.get('normalize', True)},
        )

    def _handle_train(self, task: Task) -> TaskResult:
        """Handle training task."""
        max_steps = task.params.get('max_steps', 100)
        batch_size = task.params.get('batch_size', 4)
        lr = task.params.get('learning_rate', 1e-3)
        logger.info(f"Training: steps={max_steps}, batch_size={batch_size}, lr={lr}")
        time.sleep(1.0)  # Simulate training
        return TaskResult(
            success=True,
            data={
                "final_loss": 0.05,
                "steps_completed": max_steps,
                "training_time_sec": 1.0,
            },
        )

    def _handle_validate(self, task: Task) -> TaskResult:
        """Handle validation task."""
        metrics = task.params.get('metrics', ['loss', 'accuracy'])
        logger.info(f"Validating with metrics: {metrics}")
        time.sleep(0.5)
        return TaskResult(
            success=True,
            data={"loss": 0.05, "accuracy": 0.95, "latency_ms": 10.5},
        )

    def _handle_benchmark(self, task: Task) -> TaskResult:
        """Handle benchmark task."""
        iterations = task.params.get('iterations', 100)
        logger.info(f"Running benchmark: iterations={iterations}")
        time.sleep(0.5)
        return TaskResult(
            success=True,
            data={
                "throughput": 1000.0,
                "avg_latency_ms": 5.2,
                "p99_latency_ms": 12.5,
            },
        )

    def _handle_checkpoint(self, task: Task) -> TaskResult:
        """Handle checkpoint task."""
        format = task.params.get('format', 'safetensors')
        logger.info(f"Saving checkpoint in {format} format")
        time.sleep(0.5)
        return TaskResult(
            success=True,
            data={"checkpoint_path": f"/tmp/checkpoint.{format}", "size_mb": 125.5},
        )

    def _handle_export_model(self, task: Task) -> TaskResult:
        """Handle model export task."""
        target = task.params.get('target', 'inference')
        optimize = task.params.get('optimize', True)
        logger.info(f"Exporting model for {target}, optimize={optimize}")
        time.sleep(0.5)
        return TaskResult(
            success=True,
            data={"export_path": f"/tmp/model_{target}", "optimized": optimize},
        )


# Convenience function for quick setup
def create_orchestrator(development: bool = True) -> Orchestrator:
    """Create an orchestrator with default configuration."""
    if development:
        config = OrchestratorConfig.development()
    else:
        config = OrchestratorConfig.from_env()
    return Orchestrator(config)
