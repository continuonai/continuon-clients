"""MambaWave - Unified SSM + Spectral Architecture.

MambaWave combines:
- WaveCore's spectral design (FFT-based filtering with learnable complex filters)
- Mamba SSM architecture (efficient state space model for sequences)

This is the evolved form of WaveCore, integrating selective state spaces
with spectral processing for efficient sequence modeling.

Usage:
    from continuonbrain.mambawave import MambaWaveModel, MambaWaveConfig

    config = MambaWaveConfig.default()
    model = MambaWaveModel(config)

    # Or use as a skill
    from continuonbrain.mambawave import MambaWaveSkill
    skill = MambaWaveSkill()
"""

from __future__ import annotations

from .config import MambaWaveConfig
from .models.mambawave_model import MambaWaveModel
from .layers.mamba_wave_block import MambaWaveBlock
from .layers.ssm_spectral import SSMSpectralLayer
from .skill import MambaWaveSkill
from .world_model import (
    MambaWaveWorldModel,
    WorldModelState,
    WorldModelAction,
    WorldModelPredictResult,
    build_world_model,
)

__all__ = [
    # Config
    "MambaWaveConfig",
    # Models
    "MambaWaveModel",
    # Layers
    "MambaWaveBlock",
    "SSMSpectralLayer",
    # Skill
    "MambaWaveSkill",
    # World Model
    "MambaWaveWorldModel",
    "WorldModelState",
    "WorldModelAction",
    "WorldModelPredictResult",
    "build_world_model",
]
