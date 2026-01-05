"""
Training Orchestrator Base Class

Provides unified training lifecycle management with:
- Step-based execution
- Resource-aware scheduling
- Status tracking and reporting
- Checkpoint management

Usage:
    class MyTrainer(TrainerOrchestrator):
        def get_training_steps(self):
            return ["prepare", "train", "evaluate", "export"]

        def execute_step(self, step_name, context):
            if step_name == "train":
                # Training logic here
                return StepResult(name="train", status=StepStatus.COMPLETED, ...)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import threading
import time

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phases."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    """Result of a training step."""
    name: str
    status: StepStatus
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class TrainingStatus:
    """Current training status."""
    phase: TrainingPhase
    current_step: Optional[str]
    steps_completed: int
    total_steps: int
    current_loss: Optional[float]
    best_loss: Optional[float]
    started_at: Optional[str]
    updated_at: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


class ResourceChecker:
    """Check system resources before training."""

    def __init__(
        self,
        max_memory_percent: float = 80.0,
        max_temp_c: float = 75.0,
        min_battery_percent: float = 20.0,
    ):
        self.max_memory_percent = max_memory_percent
        self.max_temp_c = max_temp_c
        self.min_battery_percent = min_battery_percent

    def check(self) -> tuple[bool, Optional[str]]:
        """
        Check if resources allow training.

        Returns:
            (ok, reason) - ok is True if training can proceed
        """
        try:
            import psutil

            # Memory check
            mem = psutil.virtual_memory()
            if mem.percent > self.max_memory_percent:
                return False, f"Memory usage {mem.percent:.1f}% > {self.max_memory_percent}%"

            # Temperature check (Linux)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > self.max_temp_c:
                                return False, f"CPU temp {entry.current:.1f}C > {self.max_temp_c}C"
            except (AttributeError, RuntimeError):
                pass  # Not all platforms support this

            # Battery check (if applicable)
            try:
                battery = psutil.sensors_battery()
                if battery and not battery.power_plugged:
                    if battery.percent < self.min_battery_percent:
                        return False, f"Battery {battery.percent:.0f}% < {self.min_battery_percent}%"
            except (AttributeError, RuntimeError):
                pass

            return True, None

        except ImportError:
            return True, None  # psutil not available, assume OK


class TrainerOrchestrator(ABC):
    """
    Base class for training orchestrators.

    Subclasses implement:
    - get_training_steps(): Define training pipeline steps
    - execute_step(): Execute a specific step
    """

    def __init__(
        self,
        status_path: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        resource_checker: Optional[ResourceChecker] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize the orchestrator.

        Args:
            status_path: Path to write status JSON
            checkpoint_dir: Directory for checkpoints
            resource_checker: Resource checker instance
            max_retries: Max retries per step on failure
            retry_delay: Delay between retries (seconds)
        """
        self.status_path = status_path or Path("/tmp/training_status.json")
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/checkpoints")
        self.resource_checker = resource_checker or ResourceChecker()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # State
        self._phase = TrainingPhase.IDLE
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Tracking
        self._steps_completed = 0
        self._current_step: Optional[str] = None
        self._current_loss: Optional[float] = None
        self._best_loss: Optional[float] = None
        self._started_at: Optional[str] = None
        self._step_results: List[StepResult] = []
        self._error: Optional[str] = None

        # Callbacks
        self._on_step_complete: Optional[Callable[[StepResult], None]] = None
        self._on_training_complete: Optional[Callable[[bool], None]] = None

    @abstractmethod
    def get_training_steps(self) -> List[str]:
        """Return ordered list of training step names."""
        pass

    @abstractmethod
    def execute_step(self, step_name: str, context: Dict[str, Any]) -> StepResult:
        """
        Execute a training step.

        Args:
            step_name: Name of the step to execute
            context: Shared context between steps (mutable)

        Returns:
            StepResult with status and details
        """
        pass

    def should_skip_step(self, step_name: str, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if step should be skipped. Override in subclass."""
        return False, None

    def on_step_complete(self, result: StepResult, context: Dict[str, Any]) -> None:
        """Called after each step completes. Override for custom behavior."""
        pass

    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Called when training starts. Override for setup logic."""
        pass

    def on_training_complete(self, success: bool, context: Dict[str, Any]) -> None:
        """Called when training completes. Override for cleanup logic."""
        pass

    def set_callbacks(
        self,
        on_step_complete: Optional[Callable[[StepResult], None]] = None,
        on_training_complete: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Set external callbacks."""
        self._on_step_complete = on_step_complete
        self._on_training_complete = on_training_complete

    def start(self) -> bool:
        """
        Start training in background thread.

        Returns:
            True if training was started
        """
        if self._running:
            logger.warning("Training already running")
            return False

        self._running = True
        self._stop_event.clear()
        self._started_at = datetime.utcnow().isoformat()
        self._step_results = []
        self._steps_completed = 0
        self._error = None

        self._thread = threading.Thread(
            target=self._run_training,
            daemon=True,
            name=f"Trainer-{self.__class__.__name__}",
        )
        self._thread.start()
        logger.info(f"{self.__class__.__name__} started")
        return True

    def stop(self) -> None:
        """Stop training gracefully."""
        if not self._running:
            return

        logger.info(f"Stopping {self.__class__.__name__}...")
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=10.0)

        self._phase = TrainingPhase.STOPPED
        self._write_status()
        logger.info(f"{self.__class__.__name__} stopped")

    def is_running(self) -> bool:
        """Check if training is running."""
        return self._running

    def _run_training(self) -> None:
        """Main training loop."""
        self._phase = TrainingPhase.PREPARING
        self._write_status()

        steps = self.get_training_steps()
        context: Dict[str, Any] = {}
        success = False

        try:
            self.on_training_start(context)

            for step_name in steps:
                if not self._running:
                    logger.info("Training stopped by request")
                    break

                # Check resources
                ok, reason = self.resource_checker.check()
                if not ok:
                    logger.warning(f"Resource constraint: {reason}")
                    time.sleep(30)
                    # Retry resource check
                    ok, reason = self.resource_checker.check()
                    if not ok:
                        self._error = f"Resource constraint: {reason}"
                        self._phase = TrainingPhase.ERROR
                        break

                # Check if should skip
                skip, skip_reason = self.should_skip_step(step_name, context)
                if skip:
                    result = StepResult(
                        name=step_name,
                        status=StepStatus.SKIPPED,
                        duration_ms=0,
                        details={"reason": skip_reason},
                    )
                    self._step_results.append(result)
                    self._steps_completed += 1
                    continue

                # Execute step with retries
                result = self._execute_step_with_retry(step_name, context)

                self._step_results.append(result)
                self._steps_completed += 1

                if result.status == StepStatus.COMPLETED:
                    self.on_step_complete(result, context)
                    if self._on_step_complete:
                        self._on_step_complete(result)
                elif result.status == StepStatus.FAILED:
                    self._error = result.error
                    self._phase = TrainingPhase.ERROR
                    break

                self._write_status()

            if self._phase != TrainingPhase.ERROR and self._running:
                self._phase = TrainingPhase.COMPLETED
                success = True

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            self._error = str(e)
            self._phase = TrainingPhase.ERROR

        finally:
            self._running = False
            self.on_training_complete(success, context)
            if self._on_training_complete:
                self._on_training_complete(success)
            self._write_status()

    def _execute_step_with_retry(
        self,
        step_name: str,
        context: Dict[str, Any],
    ) -> StepResult:
        """Execute a step with retry logic."""
        self._current_step = step_name
        self._phase = TrainingPhase.TRAINING
        self._write_status()

        last_result = None
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                result = self.execute_step(step_name, context)
                if result.status == StepStatus.COMPLETED:
                    return result
                last_result = result
            except Exception as e:
                logger.warning(f"Step {step_name} attempt {attempt + 1} failed: {e}")
                last_result = StepResult(
                    name=step_name,
                    status=StepStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    error=str(e),
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return last_result or StepResult(
            name=step_name,
            status=StepStatus.FAILED,
            duration_ms=0,
            error="Max retries exceeded",
        )

    def get_status(self) -> TrainingStatus:
        """Get current training status."""
        return TrainingStatus(
            phase=self._phase,
            current_step=self._current_step,
            steps_completed=self._steps_completed,
            total_steps=len(self.get_training_steps()),
            current_loss=self._current_loss,
            best_loss=self._best_loss,
            started_at=self._started_at,
            updated_at=datetime.utcnow().isoformat(),
            error=self._error,
        )

    def get_step_results(self) -> List[Dict[str, Any]]:
        """Get results of completed steps."""
        return [r.to_dict() for r in self._step_results]

    def set_current_loss(self, loss: float) -> None:
        """Update current loss (call from execute_step)."""
        self._current_loss = loss
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = loss

    def _write_status(self) -> None:
        """Persist status to file."""
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            status = self.get_status().to_dict()
            status["step_results"] = self.get_step_results()
            self.status_path.write_text(json.dumps(status, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write status: {e}")
