"""
ContinuonBrain Orchestrator System

A comprehensive orchestration system for managing:
- Training pipelines
- Inference services
- Data processing
- Benchmarking
- Model deployment
- Health monitoring

Usage:
    from continuonbrain.orchestrator import Orchestrator

    orch = Orchestrator()
    orch.start()

    # Submit a task
    task_id = orch.submit_task(TaskType.TRAIN, {"max_steps": 100})

    # Run a workflow
    orch.run_workflow("training_pipeline")

    # Get status
    print(orch.get_status())

    orch.stop()
"""

from .config import OrchestratorConfig
from .events import EventBus, Event, EventType
from .task import Task, TaskResult, TaskStatus, TaskPriority, TaskType, TaskQueue
from .worker import WorkerPool, Worker, WorkerStatus
from .workflow import Workflow, WorkflowStep, WorkflowEngine, WorkflowStatus, StepStatus
from .state import StateManager, OrchestratorState
from .monitor import Monitor, HealthStatus, HealthCheck
from .core import Orchestrator, create_orchestrator

__all__ = [
    # Core
    "Orchestrator",
    "create_orchestrator",
    "OrchestratorConfig",
    # Tasks
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    "TaskQueue",
    # Workers
    "WorkerPool",
    "Worker",
    "WorkerStatus",
    # Workflows
    "Workflow",
    "WorkflowStep",
    "WorkflowEngine",
    "WorkflowStatus",
    "StepStatus",
    # Events
    "EventBus",
    "Event",
    "EventType",
    # State
    "StateManager",
    "OrchestratorState",
    # Monitoring
    "Monitor",
    "HealthStatus",
    "HealthCheck",
]

__version__ = "1.0.0"
