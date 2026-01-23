#!/usr/bin/env python3
"""
Training Games Generator - Creates diverse games and levels for robot learning.

This module generates varied training scenarios across multiple game types:
1. Navigation challenges (pathfinding, obstacle avoidance)
2. Puzzle games (keys, doors, boxes, buttons)
3. Exploration missions (room discovery, item collection)
4. Interaction tasks (switches, doors, objects)
5. Multi-objective challenges (combining skills)

Usage:
    from brain_b.simulator.training_games import TrainingGamesGenerator

    generator = TrainingGamesGenerator()

    # Generate 50 varied training episodes
    episodes = generator.generate_curriculum(num_episodes=50)

    # Generate specific game type
    level = generator.generate_navigation_challenge(difficulty=5)
"""

import random
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

# Import existing level generators
try:
    from .level_generator import LevelGenerator
    from .world import GridWorld, Tile, LEVELS
    from .home_world import (
        HomeWorld, ObjectType, RoomType,
        create_simple_apartment, create_two_room_house, create_multi_floor_house,
        create_door_puzzle, create_key_hunt, create_office_layout,
        create_living_room_kitchen, get_level, list_levels,
        LEVELS as HOME_LEVELS, CURRICULUM_ORDER
    )
    # Alias for compatibility
    get_home_level = get_level
    list_home_levels = list_levels
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from level_generator import LevelGenerator
    from world import GridWorld, Tile, LEVELS
    from home_world import (
        HomeWorld, ObjectType, RoomType,
        create_simple_apartment, create_two_room_house, create_multi_floor_house,
        create_door_puzzle, create_key_hunt, create_office_layout,
        create_living_room_kitchen, get_level, list_levels,
        LEVELS as HOME_LEVELS, CURRICULUM_ORDER
    )
    get_home_level = get_level
    list_home_levels = list_levels


class GameType(str, Enum):
    """Types of training games."""
    NAVIGATION = "navigation"      # Basic pathfinding
    PUZZLE = "puzzle"              # Keys, doors, boxes
    EXPLORATION = "exploration"    # Room discovery
    INTERACTION = "interaction"    # Object manipulation
    SURVIVAL = "survival"          # Avoid hazards
    COLLECTION = "collection"      # Gather items
    MULTI_OBJECTIVE = "multi"      # Combined challenges


@dataclass
class GameConfig:
    """Configuration for a training game."""
    game_type: GameType
    difficulty: int  # 1-20
    grid_size: Tuple[int, int] = (15, 15)
    time_limit: Optional[int] = None  # Max steps
    objectives: List[str] = field(default_factory=list)
    rewards: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingGame:
    """A complete training game instance."""
    game_id: str
    config: GameConfig
    world: Any  # GridWorld or HomeWorld
    objectives: List[Dict]
    start_state: Dict
    optimal_path_length: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'game_id': self.game_id,
            'config': {
                'game_type': self.config.game_type.value,
                'difficulty': self.config.difficulty,
                'grid_size': self.config.grid_size,
                'time_limit': self.config.time_limit,
                'objectives': self.config.objectives,
            },
            'objectives': self.objectives,
            'optimal_path_length': self.optimal_path_length,
            'created_at': self.created_at,
        }


class TrainingGamesGenerator:
    """Generates diverse training games for robot learning."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
        self.level_gen = LevelGenerator(seed=seed)
        self.games_generated = 0

    def generate_curriculum(
        self,
        num_episodes: int = 50,
        game_types: Optional[List[GameType]] = None,
        difficulty_range: Tuple[int, int] = (1, 15),
    ) -> List[TrainingGame]:
        """
        Generate a diverse curriculum of training games.

        Args:
            num_episodes: Total games to generate
            game_types: Types of games to include (all if None)
            difficulty_range: (min, max) difficulty

        Returns:
            List of TrainingGame instances
        """
        if game_types is None:
            game_types = list(GameType)

        games = []
        min_diff, max_diff = difficulty_range

        for i in range(num_episodes):
            # Cycle through game types
            game_type = game_types[i % len(game_types)]

            # Progressive difficulty with some randomness
            base_diff = min_diff + (i * (max_diff - min_diff) // num_episodes)
            difficulty = min(max_diff, max(min_diff, base_diff + random.randint(-2, 2)))

            game = self._generate_game(game_type, difficulty)
            if game:
                games.append(game)
                self.games_generated += 1

        return games

    def _generate_game(self, game_type: GameType, difficulty: int) -> Optional[TrainingGame]:
        """Generate a single game of the specified type."""
        generators = {
            GameType.NAVIGATION: self.generate_navigation_challenge,
            GameType.PUZZLE: self.generate_puzzle_game,
            GameType.EXPLORATION: self.generate_exploration_mission,
            GameType.INTERACTION: self.generate_interaction_task,
            GameType.SURVIVAL: self.generate_survival_challenge,
            GameType.COLLECTION: self.generate_collection_game,
            GameType.MULTI_OBJECTIVE: self.generate_multi_objective,
        }

        generator = generators.get(game_type)
        if generator:
            return generator(difficulty)
        return None

    def generate_navigation_challenge(self, difficulty: int) -> TrainingGame:
        """
        Generate a navigation challenge.

        Focus: Pathfinding through obstacles to reach goal.
        """
        # Scale grid size with difficulty
        size = 8 + difficulty
        config = GameConfig(
            game_type=GameType.NAVIGATION,
            difficulty=difficulty,
            grid_size=(size, size),
            time_limit=size * size,
            objectives=["reach_goal"],
            rewards={"goal_reached": 10.0, "step_penalty": -0.1},
        )

        # Generate using level generator (lower difficulty = navigation focus)
        # LevelGenerator.generate() returns a GridWorld directly
        world = self.level_gen.generate(max(1, difficulty // 2))

        # Calculate optimal path
        path_length = self._calculate_path_length(world)

        game_id = f"nav_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=[{"type": "reach_goal", "reward": 10.0}],
            start_state=world.to_dict(),
            optimal_path_length=path_length,
        )

    def generate_puzzle_game(self, difficulty: int) -> TrainingGame:
        """
        Generate a puzzle game with keys, doors, and boxes.

        Focus: Solving sequential puzzles.
        """
        size = 10 + difficulty // 2
        config = GameConfig(
            game_type=GameType.PUZZLE,
            difficulty=difficulty,
            grid_size=(size, size),
            time_limit=size * size * 2,
            objectives=["collect_keys", "unlock_doors", "reach_goal"],
            rewards={
                "key_collected": 2.0,
                "door_unlocked": 3.0,
                "goal_reached": 10.0,
                "step_penalty": -0.05,
            },
        )

        # Use higher difficulty for more puzzle elements
        # LevelGenerator.generate() returns a GridWorld directly
        world = self.level_gen.generate(min(20, difficulty + 3))

        game_id = f"puzzle_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Count keys and doors in the generated world
        key_count = sum(1 for row in world.grid for tile in row if tile == Tile.KEY)
        door_count = sum(1 for row in world.grid for tile in row if tile == Tile.DOOR)

        objectives = [
            {"type": "collect_keys", "count": key_count, "reward": 2.0},
            {"type": "unlock_doors", "count": door_count, "reward": 3.0},
            {"type": "reach_goal", "reward": 10.0},
        ]

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=objectives,
            start_state=world.to_dict(),
        )

    def generate_exploration_mission(self, difficulty: int) -> TrainingGame:
        """
        Generate an exploration mission in 3D home environment.

        Focus: Discovering rooms and mapping environment.
        """
        # Select appropriate home level based on difficulty
        if difficulty <= 5:
            world = create_simple_apartment()
            level_name = "simple_apartment"
        elif difficulty <= 10:
            world = create_two_room_house()
            level_name = "two_room_house"
        else:
            world = create_multi_floor_house()
            level_name = "multi_floor_house"

        config = GameConfig(
            game_type=GameType.EXPLORATION,
            difficulty=difficulty,
            time_limit=200 + difficulty * 20,
            objectives=["visit_rooms", "find_objects"],
            rewards={
                "room_discovered": 5.0,
                "object_found": 1.0,
                "exploration_bonus": 0.1,
            },
            metadata={"level_name": level_name, "is_3d": True},
        )

        game_id = f"explore_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Count rooms and objects for objectives
        room_count = len(set(world.rooms.values())) if hasattr(world, 'rooms') else 2
        object_count = len(world.objects) if hasattr(world, 'objects') else 5

        objectives = [
            {"type": "visit_rooms", "count": room_count, "reward": 5.0},
            {"type": "find_objects", "count": min(object_count, 5), "reward": 1.0},
        ]

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=objectives,
            start_state=world.to_dict(),
        )

    def generate_interaction_task(self, difficulty: int) -> TrainingGame:
        """
        Generate an interaction-focused task.

        Focus: Manipulating objects (doors, switches, boxes).
        """
        # Use 3D world for richer interactions
        if difficulty <= 7:
            world = get_home_level("door_puzzle") or create_simple_apartment()
        elif difficulty <= 12:
            world = get_home_level("key_hunt") or create_two_room_house()
        else:
            world = get_home_level("office_layout") or create_multi_floor_house()

        config = GameConfig(
            game_type=GameType.INTERACTION,
            difficulty=difficulty,
            time_limit=150 + difficulty * 15,
            objectives=["interact_objects", "complete_task"],
            rewards={
                "door_opened": 3.0,
                "switch_toggled": 2.0,
                "item_collected": 2.0,
                "task_complete": 15.0,
            },
            metadata={"is_3d": True},
        )

        game_id = f"interact_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        objectives = [
            {"type": "interact_objects", "min_count": 1 + difficulty // 5, "reward": 2.0},
            {"type": "complete_task", "reward": 15.0},
        ]

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=objectives,
            start_state=world.to_dict(),
        )

    def generate_survival_challenge(self, difficulty: int) -> TrainingGame:
        """
        Generate a survival challenge (avoid hazards).

        Focus: Navigating dangerous terrain safely.
        """
        size = 12 + difficulty // 2
        config = GameConfig(
            game_type=GameType.SURVIVAL,
            difficulty=difficulty,
            grid_size=(size, size),
            time_limit=size * size,
            objectives=["survive", "reach_goal"],
            rewards={
                "goal_reached": 15.0,
                "survival_bonus": 0.1,
                "hazard_touched": -5.0,
            },
        )

        # Generate level with more hazards (Tier 3+ has hazards)
        world = self.level_gen.generate(max(7, difficulty))

        game_id = f"survive_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        objectives = [
            {"type": "survive", "avoid": "lava", "reward": 0.1},
            {"type": "reach_goal", "reward": 15.0},
        ]

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=objectives,
            start_state=world.to_dict(),
        )

    def generate_collection_game(self, difficulty: int) -> TrainingGame:
        """
        Generate a collection game.

        Focus: Gathering multiple items efficiently.
        """
        # Use 3D world for item variety
        if difficulty <= 6:
            world = get_home_level("key_hunt") or create_simple_apartment()
        else:
            world = get_home_level("living_kitchen") or create_two_room_house()

        # Add extra collectibles based on difficulty
        self._add_collectibles(world, 2 + difficulty // 3)

        config = GameConfig(
            game_type=GameType.COLLECTION,
            difficulty=difficulty,
            time_limit=100 + difficulty * 20,
            objectives=["collect_all", "return_home"],
            rewards={
                "item_collected": 3.0,
                "all_collected": 10.0,
                "efficiency_bonus": 0.5,
            },
            metadata={"is_3d": True},
        )

        game_id = f"collect_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        item_count = 2 + difficulty // 3
        objectives = [
            {"type": "collect_all", "count": item_count, "reward": 3.0},
            {"type": "return_home", "reward": 10.0},
        ]

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=objectives,
            start_state=world.to_dict(),
        )

    def generate_multi_objective(self, difficulty: int) -> TrainingGame:
        """
        Generate a multi-objective challenge.

        Focus: Combining navigation, puzzles, and collection.
        """
        # Use complex level
        if difficulty <= 10:
            world = create_two_room_house()
        else:
            world = create_multi_floor_house()

        # Add varied elements
        self._add_collectibles(world, 3)

        config = GameConfig(
            game_type=GameType.MULTI_OBJECTIVE,
            difficulty=difficulty,
            time_limit=300 + difficulty * 30,
            objectives=["collect_items", "solve_puzzle", "explore_area", "reach_goal"],
            rewards={
                "item_collected": 2.0,
                "puzzle_solved": 5.0,
                "area_explored": 1.0,
                "goal_reached": 20.0,
            },
            metadata={"is_3d": True, "complex": True},
        )

        game_id = f"multi_{difficulty}_{int(time.time())}_{random.randint(1000, 9999)}"

        objectives = [
            {"type": "collect_items", "count": 3, "reward": 2.0},
            {"type": "solve_puzzle", "count": 1, "reward": 5.0},
            {"type": "explore_area", "percentage": 0.7, "reward": 1.0},
            {"type": "reach_goal", "reward": 20.0},
        ]

        return TrainingGame(
            game_id=game_id,
            config=config,
            world=world,
            objectives=objectives,
            start_state=world.to_dict(),
        )

    def _calculate_path_length(self, world: GridWorld) -> Optional[int]:
        """Calculate optimal path length using BFS."""
        if not hasattr(world, 'robot') or not hasattr(world, 'find_goal'):
            return None

        from collections import deque

        start = (world.robot.x, world.robot.y)
        goal = world.find_goal()
        if not goal:
            return None

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == goal:
                return dist

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and world.is_passable(nx, ny):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

        return None

    def _add_collectibles(self, world, count: int):
        """Add collectible items to a world."""
        # Skip if not a HomeWorld with objects support
        if not hasattr(world, 'objects') or not hasattr(world, 'width'):
            return

        collectible_types = [
            ObjectType.KEY, ObjectType.REMOTE, ObjectType.PHONE,
            ObjectType.BOOK, ObjectType.CUP
        ]

        try:
            # Find empty positions
            empty_positions = []
            for x in range(1, world.width - 1):
                for y in range(1, world.depth - 1):
                    # Check no object at position
                    has_object = any(
                        getattr(o, 'position', {}).get('x') == x and
                        getattr(o, 'position', {}).get('y') == y
                        for o in world.objects
                    )
                    if not has_object:
                        empty_positions.append((x, y))

            # Place collectibles
            random.shuffle(empty_positions)
            for i, (x, y) in enumerate(empty_positions[:count]):
                obj_type = random.choice(collectible_types)
                if hasattr(world, 'add_object'):
                    world.add_object(obj_type, x, y, 0)
        except Exception:
            pass  # Skip if world doesn't support adding objects


# =============================================================================
# Batch Training Data Generation
# =============================================================================

class TrainingDataGenerator:
    """Generates training data from games."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.game_gen = TrainingGamesGenerator()

    def generate_episodes(
        self,
        num_games: int = 20,
        episodes_per_game: int = 5,
        strategy: str = "mixed",
    ) -> Dict[str, Any]:
        """
        Generate training episodes from games.

        Args:
            num_games: Number of games to create
            episodes_per_game: Episodes to generate per game
            strategy: Exploration strategy (random, goal, mixed)

        Returns:
            Statistics about generated data
        """
        games = self.game_gen.generate_curriculum(num_games)

        total_episodes = 0
        total_steps = 0
        stats = {"games": len(games), "episodes": 0, "steps": 0, "by_type": {}}

        for game in games:
            game_type = game.config.game_type.value

            for ep_num in range(episodes_per_game):
                episode = self._play_episode(game, strategy)

                if episode:
                    self._save_episode(episode, game)
                    total_episodes += 1
                    total_steps += len(episode.get('steps', []))

                    stats["by_type"][game_type] = stats["by_type"].get(game_type, 0) + 1

        stats["episodes"] = total_episodes
        stats["steps"] = total_steps

        return stats

    def _play_episode(self, game: TrainingGame, strategy: str) -> Optional[Dict]:
        """Play through a game to generate an episode."""
        world = game.world

        # Determine action vocabulary based on world type
        if isinstance(world, HomeWorld):
            actions = ["forward", "backward", "strafe_left", "strafe_right",
                      "turn_left", "turn_right", "interact"]
        else:
            actions = ["forward", "backward", "left", "right"]

        steps = []
        max_steps = game.config.time_limit or 200

        for step_idx in range(max_steps):
            # Choose action based on strategy
            if strategy == "random":
                action = random.choice(actions)
            elif strategy == "goal":
                action = self._goal_seek_action(world, actions)
            else:  # mixed
                if random.random() < 0.3:
                    action = random.choice(actions)
                else:
                    action = self._goal_seek_action(world, actions)

            # Execute action
            prev_state = world.to_dict()
            success = self._execute_action(world, action)

            # Calculate reward
            reward = self._calculate_reward(world, action, success, game)

            # Record step
            steps.append({
                "step_idx": step_idx,
                "timestamp": time.time(),
                "action": {
                    "command": action,
                    "intent": action,
                    "params": {},
                    "raw_input": action,
                },
                "observation": world.to_dict(),
                "reward": reward,
                "done": self._check_done(world, game),
            })

            if steps[-1]["done"]:
                break

        if not steps:
            return None

        return {
            "episode_id": f"{game.game_id}_{int(time.time())}",
            "game_id": game.game_id,
            "game_type": game.config.game_type.value,
            "steps": steps,
            "total_reward": sum(s["reward"] for s in steps),
            "success": steps[-1]["done"],
        }

    def _goal_seek_action(self, world, actions: List[str]) -> str:
        """Choose action that moves toward goal."""
        # Simple heuristic - prefer forward and turning toward goal
        if hasattr(world, 'goal') and hasattr(world, 'robot'):
            robot = world.robot
            goal = world.goal if hasattr(world, 'goal') else None

            if goal:
                dx = goal['x'] - robot.position['x'] if isinstance(robot.position, dict) else goal['x'] - robot.x
                dy = goal['y'] - robot.position['y'] if isinstance(robot.position, dict) else goal['y'] - robot.y

                if abs(dx) > abs(dy):
                    return "turn_right" if dx > 0 else "turn_left"
                else:
                    return "forward" if dy > 0 else "backward"

        return random.choice(actions)

    def _execute_action(self, world, action: str) -> bool:
        """Execute action on world."""
        if hasattr(world, 'step'):
            return world.step(action)
        elif hasattr(world, 'move'):
            direction_map = {
                "forward": "forward",
                "backward": "backward",
                "left": "left",
                "right": "right",
                "turn_left": "turn_left",
                "turn_right": "turn_right",
            }
            return world.move(direction_map.get(action, action))
        return False

    def _calculate_reward(self, world, action: str, success: bool, game: TrainingGame) -> float:
        """Calculate reward for action."""
        reward = -0.01  # Step penalty

        if not success:
            reward -= 0.1  # Failed action penalty

        # Check for goal reached
        if hasattr(world, 'level_complete') and world.level_complete:
            reward += game.config.rewards.get("goal_reached", 10.0)

        # Check for item collected
        if hasattr(world, 'robot') and hasattr(world.robot, 'inventory'):
            if world.robot.inventory:
                reward += game.config.rewards.get("item_collected", 1.0)

        return reward

    def _check_done(self, world, game: TrainingGame) -> bool:
        """Check if game is complete."""
        if hasattr(world, 'level_complete'):
            return world.level_complete
        if hasattr(world, 'is_won'):
            return world.is_won()
        return False

    def _save_episode(self, episode: Dict, game: TrainingGame):
        """Save episode to RLDS format."""
        episode_id = episode["episode_id"]
        episode_dir = self.output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "schema_version": "1.1",
            "episode_id": episode_id,
            "game_id": game.game_id,
            "game_type": game.config.game_type.value,
            "difficulty": game.config.difficulty,
            "num_steps": len(episode["steps"]),
            "total_reward": episode["total_reward"],
            "success": episode["success"],
            "created": datetime.now().isoformat(),
        }

        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save steps
        steps_dir = episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        with open(steps_dir / "000000.jsonl", "w") as f:
            for step in episode["steps"]:
                f.write(json.dumps(step) + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_training_games(
    num_games: int = 30,
    episodes_per_game: int = 3,
    output_dir: str = "continuonbrain/rlds/episodes",
) -> Dict[str, Any]:
    """
    Generate training games and episodes.

    Args:
        num_games: Number of different games to create
        episodes_per_game: Episodes to play per game
        output_dir: Where to save RLDS episodes

    Returns:
        Statistics about generated data
    """
    generator = TrainingDataGenerator(Path(output_dir))

    print(f"\n{'='*60}")
    print("  Training Games Generator")
    print(f"{'='*60}")
    print(f"Games to create: {num_games}")
    print(f"Episodes per game: {episodes_per_game}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    stats = generator.generate_episodes(
        num_games=num_games,
        episodes_per_game=episodes_per_game,
        strategy="mixed",
    )

    print(f"\n{'='*60}")
    print("  Generation Complete!")
    print(f"{'='*60}")
    print(f"Games created: {stats['games']}")
    print(f"Episodes generated: {stats['episodes']}")
    print(f"Total steps: {stats['steps']}")
    print(f"\nBy game type:")
    for game_type, count in stats["by_type"].items():
        print(f"  {game_type}: {count} episodes")
    print(f"{'='*60}\n")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training games")
    parser.add_argument("--games", type=int, default=30, help="Number of games")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per game")
    parser.add_argument("--output", type=str, default="continuonbrain/rlds/episodes")

    args = parser.parse_args()

    generate_training_games(
        num_games=args.games,
        episodes_per_game=args.episodes,
        output_dir=args.output,
    )
