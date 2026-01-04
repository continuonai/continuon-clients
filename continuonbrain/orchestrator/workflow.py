"""
Workflow Engine

Defines and executes multi-step workflows/pipelines.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import logging

from .task import Task, TaskResult, TaskStatus, TaskType, TaskPriority
from .events import EventBus, EventType

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of a workflow."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    PAUSED = auto()


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class WorkflowStep:
    """
    A single step in a workflow.

    Steps can have dependencies on other steps and conditions for execution.
    """

    name: str
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    timeout_sec: Optional[int] = None
    retry_on_failure: bool = True
    max_retries: int = 3
    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    status: StepStatus = StepStatus.PENDING
    task: Optional[Task] = None
    result: Optional[TaskResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "task_type": self.task_type.value,
            "params": self.params,
            "dependencies": self.dependencies,
            "timeout_sec": self.timeout_sec,
            "status": self.status.name,
            "task_id": self.task.task_id if self.task else None,
            "result": {
                "success": self.result.success,
                "data": self.result.data,
                "error": self.result.error,
            } if self.result else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def can_run(self, completed_steps: Set[str], context: Dict[str, Any]) -> bool:
        """Check if this step can run based on dependencies and conditions."""
        # Check dependencies
        for dep in self.dependencies:
            if dep not in completed_steps:
                return False

        # Check condition
        if self.condition is not None:
            try:
                return self.condition(context)
            except Exception as e:
                logger.error(f"Step {self.name} condition error: {e}")
                return False

        return True


@dataclass
class Workflow:
    """
    A workflow is a sequence of steps that execute in order.

    Features:
    - Step dependencies (DAG execution)
    - Conditional execution
    - Parallel step execution when possible
    - Context passing between steps
    """

    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:12]}")
    status: WorkflowStatus = WorkflowStatus.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def add_step(
        self,
        name: str,
        task_type: TaskType,
        params: Dict[str, Any] = None,
        dependencies: List[str] = None,
        condition: Callable[[Dict[str, Any]], bool] = None,
        timeout_sec: int = None,
    ) -> WorkflowStep:
        """Add a step to the workflow."""
        step = WorkflowStep(
            name=name,
            task_type=task_type,
            params=params or {},
            dependencies=dependencies or [],
            condition=condition,
            timeout_sec=timeout_sec,
        )
        self.steps.append(step)
        return step

    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_step_by_id(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_runnable_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to run."""
        completed = {
            s.name for s in self.steps
            if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        }
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING and s.can_run(completed, self.context)
        ]

    def get_progress(self) -> Dict[str, Any]:
        """Get workflow progress."""
        total = len(self.steps)
        completed = len([s for s in self.steps if s.status == StepStatus.COMPLETED])
        failed = len([s for s in self.steps if s.status == StepStatus.FAILED])
        running = len([s for s in self.steps if s.status == StepStatus.RUNNING])
        skipped = len([s for s in self.steps if s.status == StepStatus.SKIPPED])
        pending = len([s for s in self.steps if s.status == StepStatus.PENDING])

        return {
            "total_steps": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "skipped": skipped,
            "pending": pending,
            "progress_pct": (completed + skipped) / total * 100 if total > 0 else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status.name,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "progress": self.get_progress(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class WorkflowEngine:
    """
    Engine for executing workflows.

    Features:
    - Workflow registration
    - Concurrent workflow execution
    - Step dependency resolution
    - Event emission
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._event_bus = event_bus or EventBus()
        self._workflows: Dict[str, Workflow] = {}
        self._templates: Dict[str, Callable[[], Workflow]] = {}
        self._lock = threading.RLock()
        self._task_submitter: Optional[Callable[[Task], bool]] = None
        self._active_workflows: Dict[str, threading.Thread] = {}

    def set_task_submitter(self, submitter: Callable[[Task], bool]) -> None:
        """Set the function to submit tasks for execution."""
        self._task_submitter = submitter

    def register_template(
        self,
        name: str,
        builder: Callable[[], Workflow],
    ) -> None:
        """Register a workflow template."""
        self._templates[name] = builder
        logger.info(f"Registered workflow template: {name}")

    def create_from_template(
        self,
        template_name: str,
        context: Dict[str, Any] = None,
    ) -> Optional[Workflow]:
        """Create a workflow from a template."""
        builder = self._templates.get(template_name)
        if builder is None:
            logger.error(f"Unknown workflow template: {template_name}")
            return None

        workflow = builder()
        if context:
            workflow.context.update(context)
        self._workflows[workflow.workflow_id] = workflow
        return workflow

    def submit(self, workflow: Workflow) -> str:
        """Submit a workflow for execution."""
        with self._lock:
            self._workflows[workflow.workflow_id] = workflow

        # Start execution in background
        thread = threading.Thread(
            target=self._execute_workflow,
            args=(workflow,),
            daemon=True,
            name=f"Workflow-{workflow.workflow_id}",
        )
        self._active_workflows[workflow.workflow_id] = thread
        thread.start()

        return workflow.workflow_id

    def _execute_workflow(self, workflow: Workflow) -> None:
        """Execute a workflow."""
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()

        self._event_bus.emit(
            EventType.WORKFLOW_STARTED,
            {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "num_steps": len(workflow.steps),
            },
            source="workflow_engine",
        )

        try:
            while workflow.status == WorkflowStatus.RUNNING:
                # Get runnable steps
                runnable = workflow.get_runnable_steps()

                if not runnable:
                    # Check if all steps are done
                    pending = [s for s in workflow.steps if s.status == StepStatus.PENDING]
                    running = [s for s in workflow.steps if s.status == StepStatus.RUNNING]

                    if not pending and not running:
                        # All steps completed
                        failed = [s for s in workflow.steps if s.status == StepStatus.FAILED]
                        if failed:
                            workflow.status = WorkflowStatus.FAILED
                            workflow.error = f"{len(failed)} step(s) failed"
                        else:
                            workflow.status = WorkflowStatus.COMPLETED
                        break
                    elif running:
                        # Wait for running steps
                        time.sleep(0.1)
                        continue
                    else:
                        # Deadlock - pending steps can't run
                        workflow.status = WorkflowStatus.FAILED
                        workflow.error = "Workflow deadlock: pending steps cannot run"
                        break

                # Execute runnable steps
                for step in runnable:
                    self._execute_step(workflow, step)

                # Brief pause to prevent tight loop
                time.sleep(0.05)

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            logger.error(f"Workflow {workflow.workflow_id} error: {e}")

        finally:
            workflow.completed_at = datetime.now()
            event_type = (
                EventType.WORKFLOW_COMPLETED
                if workflow.status == WorkflowStatus.COMPLETED
                else EventType.WORKFLOW_FAILED
            )
            self._event_bus.emit(
                event_type,
                {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "status": workflow.status.name,
                    "progress": workflow.get_progress(),
                    "error": workflow.error,
                },
                source="workflow_engine",
            )

    def _execute_step(self, workflow: Workflow, step: WorkflowStep) -> None:
        """Execute a workflow step."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()

        # Merge workflow context into step params
        params = {**workflow.context, **step.params}

        # Create task
        task = Task(
            task_type=step.task_type,
            params=params,
            timeout_sec=step.timeout_sec,
            max_retries=step.max_retries if step.retry_on_failure else 0,
            parent_task_id=workflow.workflow_id,
            metadata={"workflow_id": workflow.workflow_id, "step_name": step.name},
        )
        step.task = task

        # Submit task
        if self._task_submitter:
            self._task_submitter(task)
        else:
            logger.error("No task submitter configured")
            step.status = StepStatus.FAILED
            step.result = TaskResult(success=False, error="No task submitter")
            return

        # Wait for task completion
        while task.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING):
            time.sleep(0.1)

        # Update step status
        step.completed_at = datetime.now()
        step.result = task.result

        if task.status == TaskStatus.COMPLETED and task.result and task.result.success:
            step.status = StepStatus.COMPLETED
            # Update workflow context with step results
            if task.result.data:
                workflow.context[f"{step.name}_result"] = task.result.data
        else:
            step.status = StepStatus.FAILED

        self._event_bus.emit(
            EventType.WORKFLOW_STEP_COMPLETED,
            {
                "workflow_id": workflow.workflow_id,
                "step_name": step.name,
                "step_status": step.status.name,
                "task_id": task.task_id,
            },
            source="workflow_engine",
        )

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        workflow = self._workflows.get(workflow_id)
        if workflow and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            return True
        return False

    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        workflow = self._workflows.get(workflow_id)
        if workflow and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            return True
        return False

    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        workflow = self._workflows.get(workflow_id)
        if workflow and workflow.status == WorkflowStatus.PAUSED:
            workflow.status = WorkflowStatus.RUNNING
            return True
        return False

    def get_active_workflows(self) -> List[Workflow]:
        """Get all active workflows."""
        return [
            w for w in self._workflows.values()
            if w.status in (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get workflow engine statistics."""
        workflows = list(self._workflows.values())
        return {
            "total_workflows": len(workflows),
            "running": len([w for w in workflows if w.status == WorkflowStatus.RUNNING]),
            "completed": len([w for w in workflows if w.status == WorkflowStatus.COMPLETED]),
            "failed": len([w for w in workflows if w.status == WorkflowStatus.FAILED]),
            "templates": list(self._templates.keys()),
        }


# Built-in workflow templates

def create_training_workflow() -> Workflow:
    """Create a standard training workflow."""
    workflow = Workflow(name="training_pipeline")

    workflow.add_step(
        name="load_data",
        task_type=TaskType.LOAD_DATA,
        params={},
    )

    workflow.add_step(
        name="preprocess",
        task_type=TaskType.PREPROCESS,
        params={},
        dependencies=["load_data"],
    )

    workflow.add_step(
        name="train",
        task_type=TaskType.TRAIN,
        params={},
        dependencies=["preprocess"],
    )

    workflow.add_step(
        name="validate",
        task_type=TaskType.VALIDATE,
        params={},
        dependencies=["train"],
    )

    workflow.add_step(
        name="checkpoint",
        task_type=TaskType.CHECKPOINT,
        params={},
        dependencies=["validate"],
    )

    return workflow


def create_inference_benchmark_workflow() -> Workflow:
    """Create an inference benchmark workflow."""
    workflow = Workflow(name="inference_benchmark")

    workflow.add_step(
        name="health_check",
        task_type=TaskType.HEALTH_CHECK,
        params={},
    )

    workflow.add_step(
        name="benchmark",
        task_type=TaskType.BENCHMARK,
        params={},
        dependencies=["health_check"],
    )

    workflow.add_step(
        name="inference_test",
        task_type=TaskType.INFERENCE_TEST,
        params={},
        dependencies=["benchmark"],
    )

    return workflow
