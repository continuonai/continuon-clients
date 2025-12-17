"""Mamba Brain modules for sequence modeling inside the runtime.

This package supports an optional-dependency Mamba world model interface used
by the planner/searcher. If `mamba_ssm` is unavailable, the world model falls
back to a deterministic SSM-like stub so the runtime remains importable.
"""

from .dream_loop import MambaDreamLoop, MambaDreamConfig
from .world_model import (  # noqa: F401
    BaseWorldModel,
    MambaWorldModel,
    StubSSMWorldModel,
    WorldModelAction,
    WorldModelPredictResult,
    WorldModelState,
    build_world_model,
)

__all__ = [
    "MambaDreamLoop",
    "MambaDreamConfig",
    "BaseWorldModel",
    "MambaWorldModel",
    "StubSSMWorldModel",
    "WorldModelAction",
    "WorldModelPredictResult",
    "WorldModelState",
    "build_world_model",
]
