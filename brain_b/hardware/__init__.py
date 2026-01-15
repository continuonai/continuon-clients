"""
Brain B Hardware Abstraction

Simple motor control for RC cars.
"""

from .motors import MotorController, MockMotorController
from .safety import SafetyMonitor

__all__ = ["MotorController", "MockMotorController", "SafetyMonitor"]
