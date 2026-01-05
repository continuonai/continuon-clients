"""
WaveCore Training Orchestrator

Implements the training pipeline for JAX CoreModel using the
TrainerOrchestrator base class.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time

from .orchestrator import (
    TrainerOrchestrator,
    StepResult,
    StepStatus,
    ResourceChecker,
)

logger = logging.getLogger(__name__)


class WaveCoreOrchestrator(TrainerOrchestrator):
    """
    Orchestrator for WaveCore JAX model training.

    Training Pipeline:
    1. Data preparation (load RLDS episodes)
    2. Fast loop training (quick adaptation, high LR)
    3. Mid loop training (balanced)
    4. Slow loop training (thorough, low LR)
    5. Evaluation
    6. Checkpoint export

    Usage:
        orchestrator = WaveCoreOrchestrator(
            rlds_dir=Path("/opt/continuonos/brain/rlds/episodes"),
        )
        orchestrator.start()

        while orchestrator.is_running():
            status = orchestrator.get_status()
            print(f"Phase: {status.phase}, Step: {status.current_step}")
            time.sleep(5)
    """

    def __init__(
        self,
        rlds_dir: Path = Path("/opt/continuonos/brain/rlds/episodes"),
        checkpoint_dir: Path = Path("/opt/continuonos/brain/checkpoints"),
        export_dir: Path = Path("/opt/continuonos/brain/model/adapters/candidate"),
        fast_loop_steps: int = 12,
        mid_loop_steps: int = 24,
        slow_loop_steps: int = 32,
        batch_size: int = 4,
        status_path: Optional[Path] = None,
        min_episodes: int = 4,
    ):
        """
        Initialize WaveCore orchestrator.

        Args:
            rlds_dir: Directory containing RLDS episodes
            checkpoint_dir: Directory for checkpoints
            export_dir: Directory for exported models
            fast_loop_steps: Steps for fast loop
            mid_loop_steps: Steps for mid loop
            slow_loop_steps: Steps for slow loop
            batch_size: Training batch size
            status_path: Path for status file
            min_episodes: Minimum episodes required to train
        """
        super().__init__(
            status_path=status_path or Path("/opt/continuonos/brain/trainer/status.json"),
            checkpoint_dir=checkpoint_dir,
            resource_checker=ResourceChecker(
                max_memory_percent=70.0,
                max_temp_c=75.0,
            ),
        )

        self.rlds_dir = rlds_dir
        self.export_dir = export_dir
        self.fast_loop_steps = fast_loop_steps
        self.mid_loop_steps = mid_loop_steps
        self.slow_loop_steps = slow_loop_steps
        self.batch_size = batch_size
        self.min_episodes = min_episodes

        # Lazy loaded trainer
        self._trainer = None

    def _ensure_trainer(self):
        """Lazy load the underlying trainer."""
        if self._trainer is None:
            try:
                from continuonbrain.services.wavecore_trainer import WavecoreTrainer

                self._trainer = WavecoreTrainer(
                    default_rlds_dir=self.rlds_dir,
                    checkpoint_dir=self.checkpoint_dir,
                    export_dir=self.export_dir,
                )
            except ImportError as e:
                logger.warning(f"WavecoreTrainer not available: {e}")
                raise

    def get_training_steps(self) -> List[str]:
        """Return the training pipeline steps."""
        return [
            "prepare_data",
            "fast_loop",
            "mid_loop",
            "slow_loop",
            "evaluate",
            "export",
        ]

    def should_skip_step(
        self,
        step_name: str,
        context: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Check if a step should be skipped."""
        # Skip export if training didn't produce checkpoint
        if step_name == "export":
            if "slow_checkpoint" not in context:
                return True, "No checkpoint to export"
            if context.get("slow_loss", float('inf')) > 10.0:
                return True, "Training loss too high to export"

        # Skip evaluation if no checkpoint
        if step_name == "evaluate":
            if "slow_checkpoint" not in context:
                return True, "No checkpoint to evaluate"

        return False, None

    def execute_step(self, step_name: str, context: Dict[str, Any]) -> StepResult:
        """Execute a training step."""
        start = time.time()

        try:
            if step_name == "prepare_data":
                return self._step_prepare_data(context)
            elif step_name == "fast_loop":
                return self._step_fast_loop(context)
            elif step_name == "mid_loop":
                return self._step_mid_loop(context)
            elif step_name == "slow_loop":
                return self._step_slow_loop(context)
            elif step_name == "evaluate":
                return self._step_evaluate(context)
            elif step_name == "export":
                return self._step_export(context)
            else:
                return StepResult(
                    name=step_name,
                    status=StepStatus.FAILED,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"Unknown step: {step_name}",
                )
        except Exception as e:
            logger.exception(f"Step {step_name} failed: {e}")
            return StepResult(
                name=step_name,
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    def _step_prepare_data(self, context: Dict[str, Any]) -> StepResult:
        """Prepare training data."""
        start = time.time()

        # Find episodes
        episodes = list(self.rlds_dir.glob("*.json")) + list(self.rlds_dir.glob("*.jsonl"))

        if len(episodes) < self.min_episodes:
            return StepResult(
                name="prepare_data",
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                error=f"Not enough episodes: {len(episodes)} < {self.min_episodes}",
            )

        context["episode_count"] = len(episodes)
        context["episode_paths"] = [str(p) for p in episodes]

        return StepResult(
            name="prepare_data",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start) * 1000,
            details={"episode_count": len(episodes)},
        )

    def _step_fast_loop(self, context: Dict[str, Any]) -> StepResult:
        """Run fast adaptation loop."""
        return self._run_loop(
            "fast",
            self.fast_loop_steps,
            learning_rate=1e-3,
            context=context,
        )

    def _step_mid_loop(self, context: Dict[str, Any]) -> StepResult:
        """Run mid training loop."""
        return self._run_loop(
            "mid",
            self.mid_loop_steps,
            learning_rate=5e-4,
            context=context,
        )

    def _step_slow_loop(self, context: Dict[str, Any]) -> StepResult:
        """Run slow training loop."""
        result = self._run_loop(
            "slow",
            self.slow_loop_steps,
            learning_rate=2e-4,
            context=context,
        )

        # Store checkpoint path for export
        if result.status == StepStatus.COMPLETED:
            context["slow_checkpoint"] = result.details.get("checkpoint_dir")
            context["slow_loss"] = result.details.get("final_loss")
            self.set_current_loss(result.details.get("final_loss", 0.0))

        return result

    def _run_loop(
        self,
        loop_name: str,
        max_steps: int,
        learning_rate: float,
        context: Dict[str, Any],
    ) -> StepResult:
        """Run a training loop using WavecoreTrainer."""
        import asyncio

        start = time.time()
        self._ensure_trainer()

        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._trainer.run_loops({
                    loop_name: {
                        "max_steps": max_steps,
                        "learning_rate": learning_rate,
                        "batch_size": self.batch_size,
                    }
                })
            )

            loop_result = result.get(loop_name, {})
            inner = loop_result.get("result", {})

            final_loss = inner.get("final_loss", 0.0)
            self.set_current_loss(final_loss)

            return StepResult(
                name=f"{loop_name}_loop",
                status=StepStatus.COMPLETED,
                duration_ms=(time.time() - start) * 1000,
                details={
                    "steps_trained": inner.get("steps_trained", 0),
                    "final_loss": final_loss,
                    "checkpoint_dir": loop_result.get("checkpoint_dir"),
                },
            )

        except Exception as e:
            logger.error(f"Loop {loop_name} failed: {e}")
            return StepResult(
                name=f"{loop_name}_loop",
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    def _step_evaluate(self, context: Dict[str, Any]) -> StepResult:
        """Run evaluation benchmark."""
        start = time.time()

        # Basic evaluation using training loss
        final_loss = context.get("slow_loss", 0.0)

        return StepResult(
            name="evaluate",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start) * 1000,
            details={
                "score": 1.0 - min(final_loss, 1.0),  # Convert loss to score
                "final_loss": final_loss,
            },
        )

    def _step_export(self, context: Dict[str, Any]) -> StepResult:
        """Export checkpoint for deployment."""
        import shutil

        start = time.time()

        checkpoint_path = context.get("slow_checkpoint")
        if not checkpoint_path:
            return StepResult(
                name="export",
                status=StepStatus.SKIPPED,
                duration_ms=(time.time() - start) * 1000,
                details={"reason": "No checkpoint to export"},
            )

        try:
            # Ensure export dir exists
            self.export_dir.mkdir(parents=True, exist_ok=True)

            # Copy checkpoint to export dir
            export_path = self.export_dir / "model_checkpoint.pkl"
            src = Path(checkpoint_path)
            if src.exists():
                if src.is_file():
                    shutil.copy(src, export_path)
                else:
                    # Copy directory contents
                    shutil.copytree(src, self.export_dir, dirs_exist_ok=True)

            return StepResult(
                name="export",
                status=StepStatus.COMPLETED,
                duration_ms=(time.time() - start) * 1000,
                details={"export_path": str(export_path)},
            )

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return StepResult(
                name="export",
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
