"""
RobotGrid Simulator - A tile-based game for testing Brain B systems.

This module provides:
- Grid-based world with robot navigation
- Puzzle elements (keys, doors, obstacles)
- Integration with Brain B's actor runtime, sandbox, and teaching systems
- WebSocket server for real-time visualization
- RLDS episode logging for Brain A training
- World model with surprise metrics
- Semantic search over game state history
- Procedural level generation with curriculum learning
"""

from simulator.world import GridWorld, Robot, Tile, Direction, load_level, LEVELS
from simulator.game_handler import GameHandler
from simulator.rlds_logger import RLDSLogger, WorldModelPredictor
from simulator.semantic_search import StateEmbedder, SemanticSearchIndex
from simulator.level_generator import LevelGenerator, generate_level, generate_curriculum
from simulator.training import ActionPredictor, TrainingDataset, Trainer, run_training

__all__ = [
    # World
    "GridWorld", "Robot", "Tile", "Direction", "load_level", "LEVELS",
    # Handler
    "GameHandler",
    # RLDS
    "RLDSLogger", "WorldModelPredictor",
    # Search
    "StateEmbedder", "SemanticSearchIndex",
    # Levels
    "LevelGenerator", "generate_level", "generate_curriculum",
    # Training
    "ActionPredictor", "TrainingDataset", "Trainer", "run_training",
]
