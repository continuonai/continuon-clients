"""
RobotGrid Simulator - A tile-based game for testing Brain B systems.

This module provides:
- Grid-based world with robot navigation
- Puzzle elements (keys, doors, obstacles)
- Integration with Brain B's actor runtime, sandbox, and teaching systems
- WebSocket server for real-time visualization
"""

from simulator.world import GridWorld, Robot, Tile, Direction
from simulator.game_handler import GameHandler

__all__ = ["GridWorld", "Robot", "Tile", "Direction", "GameHandler"]
