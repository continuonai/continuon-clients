"""
Dependency Injection Container for ContinuonBrain services.

Provides factory methods and lazy initialization with caching.
This container is the central point for service instantiation and
dependency management.

Usage:
    from continuonbrain.services.container import get_container, ContainerConfig

    # Get the global container
    container = get_container()

    # Access services
    chat = container.chat
    hardware = container.hardware

    # For testing, create a custom container
    test_container = ServiceContainer(ContainerConfig(
        config_dir="/tmp/test",
        allow_mock_fallback=True,
    ))
    test_container.register_instance('chat', MockChatService())
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar
import logging
import threading

from .interfaces import (
    IChatService,
    IHardwareService,
    IPerceptionService,
    ILearningService,
    IReasoningService,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ContainerConfig:
    """Configuration for the DI container."""

    # Base configuration directory
    config_dir: str = "/opt/continuonos/brain"

    # Hardware options
    prefer_real_hardware: bool = True
    auto_detect_hardware: bool = True

    # Fallback options
    allow_mock_fallback: bool = True

    # Initialization options
    lazy_load: bool = True  # Defer initialization until first access

    # Service-specific configs
    chat_config: Dict[str, Any] = field(default_factory=dict)
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    perception_config: Dict[str, Any] = field(default_factory=dict)
    learning_config: Dict[str, Any] = field(default_factory=dict)
    reasoning_config: Dict[str, Any] = field(default_factory=dict)


class ServiceContainer:
    """
    Dependency Injection container for domain services.

    Provides:
    - Lazy initialization of services
    - Singleton management
    - Mock injection for testing
    - Configuration-driven service selection
    - Thread-safe access
    """

    def __init__(self, config: Optional[ContainerConfig] = None):
        self.config = config or ContainerConfig()
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()
        self._initialized = False

        self._register_default_factories()

    def _register_default_factories(self) -> None:
        """Register default service factories."""
        self._factories['chat'] = self._create_chat_service
        self._factories['hardware'] = self._create_hardware_service
        self._factories['perception'] = self._create_perception_service
        self._factories['learning'] = self._create_learning_service
        self._factories['reasoning'] = self._create_reasoning_service

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """
        Register a custom factory for testing or extension.

        Args:
            name: Service name
            factory: Callable that returns a service instance
        """
        with self._lock:
            self._factories[name] = factory
            # Clear existing instance to force recreation
            if name in self._instances:
                del self._instances[name]

    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a pre-created instance (useful for mocks).

        Args:
            name: Service name
            instance: Service instance
        """
        with self._lock:
            self._instances[name] = instance

    def get(self, name: str) -> Any:
        """
        Get or create a service instance.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If no factory is registered for the service
        """
        with self._lock:
            if name not in self._instances:
                if name not in self._factories:
                    raise KeyError(f"No factory registered for service: {name}")
                logger.debug(f"Creating service: {name}")
                self._instances[name] = self._factories[name]()
            return self._instances[name]

    def has(self, name: str) -> bool:
        """Check if a service is available."""
        return name in self._factories or name in self._instances

    def is_instantiated(self, name: str) -> bool:
        """Check if a service has been instantiated."""
        return name in self._instances

    @property
    def chat(self) -> IChatService:
        """Get the chat service."""
        return self.get('chat')

    @property
    def hardware(self) -> IHardwareService:
        """Get the hardware service."""
        return self.get('hardware')

    @property
    def perception(self) -> IPerceptionService:
        """Get the perception service."""
        return self.get('perception')

    @property
    def learning(self) -> ILearningService:
        """Get the learning service."""
        return self.get('learning')

    @property
    def reasoning(self) -> IReasoningService:
        """Get the reasoning service."""
        return self.get('reasoning')

    def initialize_all(self) -> None:
        """
        Initialize all services eagerly.

        Call this during startup if you want to fail fast
        on initialization errors.
        """
        with self._lock:
            for name in self._factories:
                if name not in self._instances:
                    try:
                        self._instances[name] = self._factories[name]()
                    except Exception as e:
                        logger.error(f"Failed to initialize service {name}: {e}")
                        if not self.config.allow_mock_fallback:
                            raise
            self._initialized = True

    def shutdown(self) -> None:
        """
        Shutdown all services.

        Call shutdown() on each service if available.
        """
        with self._lock:
            for name, instance in self._instances.items():
                if hasattr(instance, 'shutdown'):
                    try:
                        instance.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down {name}: {e}")
            self._instances.clear()
            self._initialized = False

    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            'initialized': self._initialized,
            'services': {},
        }
        for name in self._factories:
            is_instantiated = name in self._instances
            service_status = {
                'instantiated': is_instantiated,
                'available': False,
            }
            if is_instantiated:
                instance = self._instances[name]
                if hasattr(instance, 'is_available'):
                    service_status['available'] = instance.is_available()
            status['services'][name] = service_status
        return status

    # Factory methods
    def _create_chat_service(self) -> IChatService:
        """Create the chat service."""
        try:
            from .domains.chat.chat_service import ChatService
            return ChatService(
                config_dir=self.config.config_dir,
                container=self,
                **self.config.chat_config,
            )
        except ImportError as e:
            logger.warning(f"ChatService not available: {e}")
            if self.config.allow_mock_fallback:
                return _create_mock_chat_service()
            raise

    def _create_hardware_service(self) -> IHardwareService:
        """Create the hardware service."""
        try:
            from .domains.hardware.hardware_service import HardwareService
            return HardwareService(
                config_dir=self.config.config_dir,
                container=self,
                prefer_real_hardware=self.config.prefer_real_hardware,
                auto_detect=self.config.auto_detect_hardware,
                **self.config.hardware_config,
            )
        except ImportError as e:
            logger.warning(f"HardwareService not available: {e}")
            if self.config.allow_mock_fallback:
                return _create_mock_hardware_service()
            raise

    def _create_perception_service(self) -> IPerceptionService:
        """Create the perception service."""
        try:
            from .domains.perception.perception_service import PerceptionService
            return PerceptionService(
                config_dir=self.config.config_dir,
                container=self,
                **self.config.perception_config,
            )
        except ImportError as e:
            logger.warning(f"PerceptionService not available: {e}")
            if self.config.allow_mock_fallback:
                return _create_mock_perception_service()
            raise

    def _create_learning_service(self) -> ILearningService:
        """Create the learning service."""
        try:
            from .domains.learning.learning_service import LearningService
            return LearningService(
                config_dir=self.config.config_dir,
                container=self,
                **self.config.learning_config,
            )
        except ImportError as e:
            logger.warning(f"LearningService not available: {e}")
            if self.config.allow_mock_fallback:
                return _create_mock_learning_service()
            raise

    def _create_reasoning_service(self) -> IReasoningService:
        """Create the reasoning service."""
        try:
            from .domains.reasoning.reasoning_service import ReasoningService
            return ReasoningService(
                config_dir=self.config.config_dir,
                container=self,
                **self.config.reasoning_config,
            )
        except ImportError as e:
            logger.warning(f"ReasoningService not available: {e}")
            if self.config.allow_mock_fallback:
                return _create_mock_reasoning_service()
            raise


# Mock service implementations for fallback/testing
def _create_mock_chat_service() -> IChatService:
    """Create a mock chat service."""
    class MockChatService:
        def chat(self, message, history, session_id=None, fast_mode=False):
            return {
                'response': f"[Mock] Received: {message}",
                'confidence': 0.0,
                'model': 'mock',
                'session_id': session_id or 'mock-session',
                'metadata': {'mock': True},
            }
        def clear_session(self, session_id): pass
        def clear_all_sessions(self): pass
        def get_model_info(self):
            return {'model_name': 'mock', 'model_type': 'mock', 'is_loaded': False, 'capabilities': []}
        def get_session_count(self): return 0
        def is_available(self): return True
    return MockChatService()


def _create_mock_hardware_service() -> IHardwareService:
    """Create a mock hardware service."""
    class MockHardwareService:
        def drive(self, steering, throttle, duration_ms=None):
            return {'success': True, 'actual_steering': 0, 'actual_throttle': 0, 'safety_limited': True}
        def stop(self):
            return {'success': True}
        def move_arm(self, joint_positions, speed=0.5):
            return {'success': False, 'reached_target': False, 'actual_positions': []}
        def capture_frame(self):
            return None
        def get_capabilities(self):
            return {'has_arm': False, 'has_drivetrain': False, 'has_camera': False, 'has_depth': False, 'has_hailo': False}
        def get_status(self):
            return {'mock': True}
        def is_available(self): return True
    return MockHardwareService()


def _create_mock_perception_service() -> IPerceptionService:
    """Create a mock perception service."""
    class MockPerceptionService:
        def detect_objects(self, frame=None, conf_threshold=0.25):
            return []
        def describe_scene(self, frame=None):
            return "[Mock] No scene available"
        def get_scene_representation(self, frame=None):
            return {'objects': [], 'depth_map': None, 'description': '[Mock]', 'statistics': {}, 'timestamp': 0}
        def segment(self, frame, prompt=None, points=None):
            return {'masks': [], 'scores': [], 'boxes': []}
        def get_capabilities(self):
            return {'detection': False, 'segmentation': False, 'depth': False, 'scene_description': False}
        def is_available(self): return True
    return MockPerceptionService()


def _create_mock_learning_service() -> ILearningService:
    """Create a mock learning service."""
    class MockLearningService:
        async def run_manual_training(self, config):
            return {'success': False, 'steps_trained': 0, 'final_loss': 0, 'checkpoint_path': ''}
        async def run_chat_learn(self, config):
            return {'success': False}
        async def run_wavecore_loops(self, loop_config):
            return {}
        def hot_reload_model(self, checkpoint_path):
            return False
        def get_training_status(self):
            return {'is_training': False, 'current_step': 0, 'total_steps': 0, 'current_loss': 0, 'phase': 'idle'}
        def start_training(self): return False
        def stop_training(self): return False
        def is_training(self): return False
        def is_available(self): return True
    return MockLearningService()


def _create_mock_reasoning_service() -> IReasoningService:
    """Create a mock reasoning service."""
    class MockReasoningService:
        async def symbolic_search(self, config):
            return {'success': False, 'plan': [], 'confidence': 0, 'search_stats': {}}
        def get_context_subgraph(self, session_id=None, tags=None, depth=2):
            return {'nodes': [], 'edges': [], 'summary': '[Mock]'}
        def get_decision_trace(self, session_id, limit=10):
            return []
        def record_decision(self, session_id, action, outcome, reason, confidence=0.5):
            return None
        def ingest_episode(self, episode_data):
            return False
        def query_context(self, query, top_k=5):
            return []
        def is_available(self): return True
    return MockReasoningService()


# Global container instance
_default_container: Optional[ServiceContainer] = None
_container_lock = threading.Lock()


def get_container(config: Optional[ContainerConfig] = None) -> ServiceContainer:
    """
    Get the global service container, creating if needed.

    Args:
        config: Optional configuration. Only used if container doesn't exist.

    Returns:
        ServiceContainer instance
    """
    global _default_container
    with _container_lock:
        if _default_container is None:
            _default_container = ServiceContainer(config)
        return _default_container


def set_container(container: ServiceContainer) -> None:
    """
    Set the global container (for testing).

    Args:
        container: Container to use as global
    """
    global _default_container
    with _container_lock:
        _default_container = container


def reset_container() -> None:
    """Reset the global container (for testing)."""
    global _default_container
    with _container_lock:
        if _default_container is not None:
            _default_container.shutdown()
            _default_container = None
