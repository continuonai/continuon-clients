"""Importable alias for the `03_mamba_brain` modules.

The repo keeps scaffolding under `continuonbrain/03_mamba_brain/` for ordering,
but Python identifiers cannot start with digits, so direct imports like:
  `from continuonbrain.03_mamba_brain import ...`
are invalid syntax.

This package provides an import-safe alias:
  `from continuonbrain.mamba_brain import build_world_model, WorldModelState, ...`
"""

from __future__ import annotations

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
    "BaseWorldModel",
    "MambaWorldModel",
    "StubSSMWorldModel",
    "WorldModelAction",
    "WorldModelPredictResult",
    "WorldModelState",
    "build_world_model",
]


