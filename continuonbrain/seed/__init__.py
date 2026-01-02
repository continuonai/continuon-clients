"""
ContinuonBrain Seed Model

The Seed Model is the universal initialization point for every robot
in the Continuon ecosystem. It runs on any hardware platform.

Supported Architectures:
- ARM64 (Raspberry Pi 5, Jetson)
- x86_64 (PC, Server, Cloud)
- RISC-V (Edge devices)
- Apple Silicon (M1/M2/M3)
- Quantum (Future - Pennylane/JAX)
- Neuromorphic (Future - Loihi/Lava)

Usage:
    from continuonbrain.seed import SeedModel
    
    # Initialize for current hardware (auto-detected)
    seed = SeedModel()
    
    # Or specify hardware target
    seed = SeedModel(target='pi5')  # ARM64, Hailo NPU
    seed = SeedModel(target='jetson')  # ARM64, CUDA
    seed = SeedModel(target='cloud')  # TPU
"""

from .model import SeedModel
from .config import SeedConfig
from .hardware import HardwareProfile, detect_hardware

import json
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

STABLE_SEED_PATH = Path("/opt/continuonos/brain/model/seed_stable")


def load_stable_seed(
    seed_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Load the stable seed model from disk.
    
    Args:
        seed_path: Path to seed model directory. Defaults to STABLE_SEED_PATH.
    
    Returns:
        (model, params, manifest): The CoreModel, parameters, and manifest.
    
    Example:
        from continuonbrain.seed import load_stable_seed
        
        model, params, manifest = load_stable_seed()
        
        # Run inference
        output, info = model.apply(
            {'params': params},
            x_obs=observation,
            ...
        )
    """
    from continuonbrain.jax_models.config import CoreModelConfig
    from continuonbrain.jax_models.core_model import CoreModel
    
    seed_path = seed_path or STABLE_SEED_PATH
    
    # Load manifest
    manifest_path = seed_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stable seed manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    logger.info(f"Loading stable seed v{manifest['version']} ({manifest['model']['param_count']:,} params)")
    
    # Load parameters
    checkpoint_path = seed_path / "seed_model.pkl"
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    if 'params' in params:
        params = params['params']  # Handle nested structure
    
    # Create model
    config = CoreModelConfig(**manifest['config'])
    model = CoreModel(
        config=config,
        obs_dim=manifest['input_dims']['obs_dim'],
        action_dim=manifest['input_dims']['action_dim'],
        output_dim=manifest['output_dim'],
    )
    
    logger.info(f"✅ Stable seed model loaded")
    
    return model, params, manifest


def get_seed_info() -> Dict[str, Any]:
    """
    Get information about the stable seed model without loading it.
    
    Returns:
        Manifest dict with model info, or error dict if not found.
    """
    manifest_path = STABLE_SEED_PATH / "manifest.json"
    
    if not manifest_path.exists():
        return {"error": "Stable seed model not found", "path": str(STABLE_SEED_PATH)}
    
    with open(manifest_path) as f:
        return json.load(f)


__all__ = [
    'SeedModel', 
    'SeedConfig', 
    'HardwareProfile', 
    'detect_hardware',
    'load_stable_seed',
    'get_seed_info',
    'STABLE_SEED_PATH',
]

