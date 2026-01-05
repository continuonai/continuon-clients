"""
Learning Service

Domain service for learning/training functionality.
Wraps training orchestrators and provides model management.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from continuonbrain.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class LearningService:
    """
    Learning domain service implementing ILearningService.

    Handles:
    - Training orchestration
    - Model updates and hot-reload
    - Training status tracking
    - Checkpoint management
    """

    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        container: Optional["ServiceContainer"] = None,
        rlds_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize learning service.

        Args:
            config_dir: Configuration directory
            container: Service container for dependencies
            rlds_dir: Directory for RLDS episodes
            checkpoint_dir: Directory for checkpoints
        """
        self.config_dir = Path(config_dir)
        self._container = container
        self.rlds_dir = Path(rlds_dir) if rlds_dir else self.config_dir / "rlds" / "episodes"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.config_dir / "checkpoints"

        self._orchestrator = None
        self._is_training = False

    async def run_manual_training(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run manual training session."""
        try:
            from continuonbrain.trainer.wavecore_orchestrator import WaveCoreOrchestrator

            orchestrator = WaveCoreOrchestrator(
                rlds_dir=self.rlds_dir,
                checkpoint_dir=self.checkpoint_dir,
                fast_loop_steps=config.get("fast_steps", 12),
                mid_loop_steps=config.get("mid_steps", 24),
                slow_loop_steps=config.get("slow_steps", 32),
                batch_size=config.get("batch_size", 4),
            )

            self._orchestrator = orchestrator
            self._is_training = True
            orchestrator.start()

            # Wait for completion
            import asyncio
            while orchestrator.is_running():
                await asyncio.sleep(1.0)

            status = orchestrator.get_status()
            self._is_training = False

            return {
                "success": status.phase.value == "completed",
                "steps_trained": status.steps_completed,
                "final_loss": status.current_loss or 0.0,
                "checkpoint_path": str(self.checkpoint_dir),
            }

        except Exception as e:
            logger.error(f"Manual training failed: {e}")
            self._is_training = False
            return {
                "success": False,
                "steps_trained": 0,
                "final_loss": 0.0,
                "checkpoint_path": "",
                "error": str(e),
            }

    async def run_chat_learn(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run conversational learning session."""
        # Placeholder - would implement chat-based learning
        return {"success": False, "error": "Not implemented"}

    async def run_wavecore_loops(
        self,
        loop_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run WaveCore training loops."""
        try:
            from continuonbrain.services.wavecore_trainer import WavecoreTrainer

            trainer = WavecoreTrainer(
                default_rlds_dir=self.rlds_dir,
                checkpoint_dir=self.checkpoint_dir,
            )

            self._is_training = True
            result = await trainer.run_loops(loop_config)
            self._is_training = False

            return result

        except ImportError:
            return {"error": "WavecoreTrainer not available"}
        except Exception as e:
            self._is_training = False
            return {"error": str(e)}

    def hot_reload_model(
        self,
        checkpoint_path: str,
    ) -> bool:
        """Hot-reload model from checkpoint."""
        try:
            # Would implement model reloading logic
            logger.info(f"Hot reload requested for: {checkpoint_path}")
            return Path(checkpoint_path).exists()
        except Exception as e:
            logger.error(f"Hot reload failed: {e}")
            return False

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        if self._orchestrator:
            status = self._orchestrator.get_status()
            return status.to_dict()

        return {
            "is_training": self._is_training,
            "current_step": 0,
            "total_steps": 0,
            "current_loss": 0.0,
            "phase": "idle",
        }

    def start_training(self) -> bool:
        """Start background training."""
        if self._is_training:
            return False

        try:
            from continuonbrain.trainer.wavecore_orchestrator import WaveCoreOrchestrator

            self._orchestrator = WaveCoreOrchestrator(
                rlds_dir=self.rlds_dir,
                checkpoint_dir=self.checkpoint_dir,
            )
            self._is_training = self._orchestrator.start()
            return self._is_training

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False

    def stop_training(self) -> bool:
        """Stop background training."""
        if self._orchestrator:
            self._orchestrator.stop()
            self._is_training = False
            return True
        return False

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training

    def is_available(self) -> bool:
        """Check if learning service is available."""
        return True

    def shutdown(self) -> None:
        """Shutdown learning service."""
        self.stop_training()
        self._orchestrator = None
