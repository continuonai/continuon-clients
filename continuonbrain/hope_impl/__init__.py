"""
HOPE Implementation Package

PyTorch-based implementation of the HOPE (Hierarchical Online Predictive Encoding)
architecture optimized for Raspberry Pi 5 deployment.

Main exports:
- HOPEBrain: Main brain interface
- HOPEConfig: Configuration dataclass
- State objects: FastState, CMSMemory, Parameters, FullState
"""

from .config import HOPEConfig
from .state import FastState, MemoryLevel, CMSMemory, Parameters, FullState
from .brain import HOPEBrain

__version__ = "0.1.0"

__all__ = [
    "HOPEBrain",
    "HOPEConfig",
    "FastState",
    "MemoryLevel",
    "CMSMemory",
    "Parameters",
    "FullState",
]
