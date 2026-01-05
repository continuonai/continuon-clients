"""
Learning Service Interface

Protocol definition for learning/training services.
"""
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ILearningService(Protocol):
    """
    Protocol for learning/training services.

    Implementations handle:
    - Training orchestration
    - Model updates and hot-reload
    - Training status tracking
    - Checkpoint management
    """

    async def run_manual_training(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run manual training session.

        Args:
            config: Training configuration dictionary

        Returns:
            Dictionary containing:
                - success: bool - Whether training completed
                - steps_trained: int - Number of steps completed
                - final_loss: float - Final training loss
                - checkpoint_path: str - Path to saved checkpoint
        """
        ...

    async def run_chat_learn(
        self,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run conversational learning session.

        Args:
            config: Learning configuration dictionary

        Returns:
            Dictionary containing training results
        """
        ...

    async def run_wavecore_loops(
        self,
        loop_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run WaveCore training loops (fast/mid/slow).

        Args:
            loop_config: Dictionary mapping loop name to config

        Returns:
            Dictionary mapping loop name to results
        """
        ...

    def hot_reload_model(
        self,
        checkpoint_path: str,
    ) -> bool:
        """
        Hot-reload model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if reload was successful
        """
        ...

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.

        Returns:
            Dictionary containing:
                - is_training: bool
                - current_step: int
                - total_steps: int
                - current_loss: float
                - phase: str
        """
        ...

    def start_training(self) -> bool:
        """Start background training."""
        ...

    def stop_training(self) -> bool:
        """Stop background training."""
        ...

    def is_training(self) -> bool:
        """Check if training is in progress."""
        ...

    def is_available(self) -> bool:
        """Check if learning service is available."""
        ...
