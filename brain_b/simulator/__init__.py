"""
RobotGrid Simulator - A tile-based game for testing Brain B systems.

This module provides:
- Grid-based world with robot navigation (2D)
- 3D Home exploration environment (3D)
- Puzzle elements (keys, doors, obstacles)
- Integration with Brain B's actor runtime, sandbox, and teaching systems
- WebSocket server for real-time visualization
- RLDS episode logging for Brain A training
- World model with surprise metrics
- Semantic search over game state history
- Procedural level generation with curriculum learning
"""

# 2D Grid World
from simulator.world import GridWorld, Robot, Tile, Direction, load_level, LEVELS
from simulator.game_handler import GameHandler
from simulator.rlds_logger import RLDSLogger, WorldModelPredictor
from simulator.semantic_search import StateEmbedder, SemanticSearchIndex
from simulator.level_generator import LevelGenerator, generate_level, generate_curriculum
from simulator.training import ActionPredictor, TrainingDataset, Trainer, run_training

# 3D Home World
from simulator.home_world import (
    HomeWorld,
    Robot3D,
    Position3D,
    Rotation3D,
    WorldObject,
    Room,
    RoomType,
    ObjectType,
    get_level as get_home_level,
    list_levels as list_home_levels,
    LEVELS as HOME_LEVELS,
)
from simulator.home_handler import HomeHandler, Home3DIntent, Home3DResponse
from simulator.home_rlds_logger import HomeRLDSLogger
from simulator.home_training import (
    Home3DNavigationPredictor,
    Home3DTrainingDataset,
    Home3DTrainer,
    run_home3d_training,
    HOME_ACTIONS,
)
from simulator.home_world import CURRICULUM_ORDER

__all__ = [
    # 2D World
    "GridWorld", "Robot", "Tile", "Direction", "load_level", "LEVELS",
    # 2D Handler
    "GameHandler",
    # 2D RLDS
    "RLDSLogger", "WorldModelPredictor",
    # Search
    "StateEmbedder", "SemanticSearchIndex",
    # Levels
    "LevelGenerator", "generate_level", "generate_curriculum",
    # Training
    "ActionPredictor", "TrainingDataset", "Trainer", "run_training",
    # 3D Home World
    "HomeWorld", "Robot3D", "Position3D", "Rotation3D",
    "WorldObject", "Room", "RoomType", "ObjectType",
    "get_home_level", "list_home_levels", "HOME_LEVELS",
    # 3D Handler
    "HomeHandler", "Home3DIntent", "Home3DResponse",
    # 3D RLDS
    "HomeRLDSLogger",
    # 3D Training
    "Home3DNavigationPredictor", "Home3DTrainingDataset", "Home3DTrainer",
    "run_home3d_training", "HOME_ACTIONS", "CURRICULUM_ORDER",
]
