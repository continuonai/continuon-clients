"""Backward compatibility alias for MambaWave.

The `mamba_brain` module is now superseded by `mambawave`, which combines
WaveCore's spectral design with Mamba SSM architecture.

This module provides import aliases to maintain backward compatibility:
    from continuonbrain.mamba_brain import build_world_model, WorldModelState

New code should use mambawave directly:
    from continuonbrain.mambawave import MambaWaveModel, MambaWaveConfig
"""

from __future__ import annotations

# Import from the evolved mambawave module
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
    "WorldModelAction",
    "WorldModelPredictResult",
    "WorldModelState",
    "build_world_model",
]

# Deprecation notice
import warnings as _warnings

_warnings.warn(
    "continuonbrain.mamba_brain is deprecated. Use continuonbrain.mambawave instead.",
    DeprecationWarning,
    stacklevel=2,
)
