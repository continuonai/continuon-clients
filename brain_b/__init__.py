"""
Brain B - A minimal, teachable robot brain.

This package provides a simple robot brain with:
- Actor-based runtime for agent management
- Natural language conversation interface
- Hardware abstraction for motors and sensors
- Behavior teaching and playback
"""

import sys
from pathlib import Path

# Add brain_b directory to path so internal bare imports work
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from actor_runtime import ActorRuntime
from conversation import ConversationHandler
from hardware import MotorController, MockMotorController, SafetyMonitor

__all__ = [
    "ActorRuntime",
    "ConversationHandler",
    "MotorController",
    "MockMotorController",
    "SafetyMonitor",
]
