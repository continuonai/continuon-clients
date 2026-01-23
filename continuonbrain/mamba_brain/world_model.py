"""Backward compatibility - redirects to mambawave.world_model."""

from __future__ import annotations

from continuonbrain.mambawave.world_model import (
    BaseWorldModel,
    MambaWaveWorldModel as MambaWorldModel,
    StubWorldModel as StubSSMWorldModel,
    WorldModelAction,
    WorldModelPredictResult,
    WorldModelState,
    build_world_model,
)

__all__ = [
    "BaseWorldModel",
    "MambaWorldModel",
    "StubSSMWorldModel",
    "WorldModelState",
    "WorldModelAction",
    "WorldModelPredictResult",
    "build_world_model",
]
