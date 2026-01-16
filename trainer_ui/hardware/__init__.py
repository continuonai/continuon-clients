"""
Hardware management package for Trainer UI.

Provides auto-detection of peripherals and dual SO-ARM101 support.
"""

from .detector import TrainerHardwareDetector, HardwareConfig
from .arm_manager import ArmController, DualArmManager
from .audio_manager import AudioManager

__all__ = [
    "TrainerHardwareDetector",
    "HardwareConfig",
    "ArmController",
    "DualArmManager",
    "AudioManager",
]
