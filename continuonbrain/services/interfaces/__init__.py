"""
ContinuonBrain Service Interfaces

Protocol definitions for domain services. These protocols define the contracts
that domain services must implement, enabling:
- Dependency injection
- Mock services for testing
- Swappable implementations
- Clear API boundaries

Usage:
    from continuonbrain.services.interfaces import (
        IChatService,
        IHardwareService,
        IPerceptionService,
        ILearningService,
        IReasoningService,
    )
"""
from .chat_service import IChatService
from .hardware_service import IHardwareService
from .perception_service import IPerceptionService
from .learning_service import ILearningService
from .reasoning_service import IReasoningService
from .audio_service import IAudioService

__all__ = [
    'IChatService',
    'IHardwareService',
    'IPerceptionService',
    'ILearningService',
    'IReasoningService',
    'IAudioService',
]
