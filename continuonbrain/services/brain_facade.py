"""
Brain Facade

Slim orchestration layer providing unified access to domain services.
Replaces the monolithic BrainService with a clean API that delegates
to specialized domain services.

Usage:
    from continuonbrain.services.brain_facade import BrainFacade

    brain = BrainFacade()

    # Chat
    response = brain.chat.chat("Hello!", [])

    # Drive
    brain.hardware.drive(steering=0.5, throttle=0.3)

    # Detect objects
    objects = brain.perception.detect_objects()

    # Check status
    status = brain.get_status()
"""
import logging
from typing import Any, Dict, Optional

from .container import ServiceContainer, ContainerConfig, get_container
from .interfaces import (
    IChatService,
    IHardwareService,
    IPerceptionService,
    ILearningService,
    IReasoningService,
)

logger = logging.getLogger(__name__)


class BrainFacade:
    """
    Slim orchestration layer for ContinuonBrain.

    This facade provides:
    - Unified access to all domain services
    - Cross-domain coordination when needed
    - System-wide status and health
    - Graceful initialization and shutdown

    It does NOT contain business logic - that lives in domain services.
    """

    def __init__(
        self,
        container: Optional[ServiceContainer] = None,
        config: Optional[ContainerConfig] = None,
    ):
        """
        Initialize the brain facade.

        Args:
            container: Pre-configured ServiceContainer (for testing)
            config: ContainerConfig for creating new container
        """
        if container:
            self._container = container
        else:
            self._container = get_container(config)

        self._initialized = False
        logger.info("BrainFacade created")

    @property
    def chat(self) -> IChatService:
        """Get the chat service."""
        return self._container.chat

    @property
    def hardware(self) -> IHardwareService:
        """Get the hardware service."""
        return self._container.hardware

    @property
    def perception(self) -> IPerceptionService:
        """Get the perception service."""
        return self._container.perception

    @property
    def learning(self) -> ILearningService:
        """Get the learning service."""
        return self._container.learning

    @property
    def reasoning(self) -> IReasoningService:
        """Get the reasoning service."""
        return self._container.reasoning

    def initialize(self) -> None:
        """
        Initialize all services eagerly.

        Call this during startup if you want to fail fast
        on initialization errors.
        """
        if self._initialized:
            return

        logger.info("Initializing BrainFacade...")

        # Initialize in order of dependency
        try:
            # Hardware first (sensors, actuators)
            _ = self.hardware.get_capabilities()
            logger.debug("Hardware service initialized")

            # Perception (depends on hardware for camera)
            _ = self.perception.get_capabilities()
            logger.debug("Perception service initialized")

            # Chat (standalone)
            _ = self.chat.get_model_info()
            logger.debug("Chat service initialized")

            # Learning (can be lazy)
            # Reasoning (can be lazy)

            self._initialized = True
            logger.info("BrainFacade initialization complete")

        except Exception as e:
            logger.error(f"BrainFacade initialization failed: {e}")
            raise

    def shutdown(self) -> None:
        """
        Graceful shutdown of all services.

        Stops any ongoing operations and releases resources.
        """
        logger.info("Shutting down BrainFacade...")

        # Stop hardware first (safety)
        try:
            self.hardware.stop()
        except Exception as e:
            logger.warning(f"Hardware stop failed: {e}")

        # Stop training
        try:
            self.learning.stop_training()
        except Exception as e:
            logger.warning(f"Learning stop failed: {e}")

        # Shutdown container (all services)
        self._container.shutdown()

        self._initialized = False
        logger.info("BrainFacade shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dictionary with status of all services
        """
        status = {
            "initialized": self._initialized,
            "services": {},
        }

        # Hardware status
        try:
            status["services"]["hardware"] = {
                "available": self.hardware.is_available(),
                "capabilities": self.hardware.get_capabilities(),
            }
        except Exception as e:
            status["services"]["hardware"] = {"error": str(e)}

        # Perception status
        try:
            status["services"]["perception"] = {
                "available": self.perception.is_available(),
                "capabilities": self.perception.get_capabilities(),
            }
        except Exception as e:
            status["services"]["perception"] = {"error": str(e)}

        # Chat status
        try:
            status["services"]["chat"] = {
                "available": self.chat.is_available(),
                "model_info": self.chat.get_model_info(),
                "session_count": self.chat.get_session_count(),
            }
        except Exception as e:
            status["services"]["chat"] = {"error": str(e)}

        # Learning status
        try:
            status["services"]["learning"] = {
                "available": self.learning.is_available(),
                "is_training": self.learning.is_training(),
                "training_status": self.learning.get_training_status(),
            }
        except Exception as e:
            status["services"]["learning"] = {"error": str(e)}

        # Reasoning status
        try:
            status["services"]["reasoning"] = {
                "available": self.reasoning.is_available(),
            }
        except Exception as e:
            status["services"]["reasoning"] = {"error": str(e)}

        return status

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get all system capabilities.

        Returns:
            Dictionary of capability categories
        """
        return {
            "hardware": self.hardware.get_capabilities(),
            "perception": self.perception.get_capabilities(),
            "chat": {
                "available": self.chat.is_available(),
                "model": self.chat.get_model_info().get("model_type", "unknown"),
            },
            "learning": {
                "available": self.learning.is_available(),
            },
            "reasoning": {
                "available": self.reasoning.is_available(),
            },
        }

    # Convenience methods for common cross-domain operations

    def describe_scene(self) -> str:
        """
        Get a description of what the robot sees.

        Cross-domain: uses hardware (camera) + perception (detection).
        """
        return self.perception.describe_scene()

    def emergency_stop(self) -> Dict[str, Any]:
        """
        Execute emergency stop.

        Stops all motion and ongoing operations.
        """
        results = {"success": True, "stopped": []}

        # Stop hardware
        hw_result = self.hardware.stop()
        if hw_result.get("success"):
            results["stopped"].extend(hw_result.get("stopped", []))
        else:
            results["success"] = False

        # Stop training
        if self.learning.is_training():
            self.learning.stop_training()
            results["stopped"].append("training")

        return results


# Factory function for easy creation
def create_brain(
    config_dir: str = "/opt/continuonos/brain",
    prefer_real_hardware: bool = True,
    auto_detect_hardware: bool = True,
    allow_mock_fallback: bool = True,
) -> BrainFacade:
    """
    Create a BrainFacade with the specified configuration.

    Args:
        config_dir: Base configuration directory
        prefer_real_hardware: Prefer real hardware over mocks
        auto_detect_hardware: Auto-detect hardware on startup
        allow_mock_fallback: Allow mock services if real ones fail

    Returns:
        Configured BrainFacade instance
    """
    config = ContainerConfig(
        config_dir=config_dir,
        prefer_real_hardware=prefer_real_hardware,
        auto_detect_hardware=auto_detect_hardware,
        allow_mock_fallback=allow_mock_fallback,
    )

    return BrainFacade(config=config)
