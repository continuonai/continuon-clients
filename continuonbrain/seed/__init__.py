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

__all__ = ['SeedModel', 'SeedConfig', 'HardwareProfile', 'detect_hardware']

