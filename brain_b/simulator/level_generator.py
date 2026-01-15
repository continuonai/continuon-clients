"""
Procedural Level Generator for RobotGrid.

Generates levels with controlled difficulty for curriculum learning.
Ensures all generated levels have valid solutions.
"""

import random
from dataclasses import dataclass
from typing import Optional
from collections import deque

from simulator.world import GridWorld, Tile, Direction


@dataclass
class LevelConfig:
    """Configuration for level generation."""
    width: int = 10
    height: int = 10
    difficulty: int = 1  # 1-20 scale
    seed: Optional[int] = None

    # Derived parameters (set based on difficulty)
    num_keys: int = 0
    num_doors: int = 0
    num_lava_patches: int = 0
    num_boxes: int = 0
    num_buttons: int = 0
    min_path_length: int = 5
    max_dead_ends: int = 3


# Difficulty tier definitions
DIFFICULTY_TIERS = {
    # Tier 1: Tutorial (difficulty 1-3)
    1: {"keys": 0, "doors": 0, "lava": 0, "boxes": 0, "path": 5},
    2: {"keys": 0, "doors": 0, "lava": 0, "boxes": 0, "path": 8},
    3: {"keys": 0, "doors": 0, "lava": 0, "boxes": 0, "path": 12},

    # Tier 2: Keys (difficulty 4-6)
    4: {"keys": 1, "doors": 1, "lava": 0, "boxes": 0, "path": 10},
    5: {"keys": 1, "doors": 1, "lava": 0, "boxes": 0, "path": 15},
    6: {"keys": 2, "doors": 2, "lava": 0, "boxes": 0, "path": 18},

    # Tier 3: Hazards (difficulty 7-10)
    7: {"keys": 1, "doors": 1, "lava": 1, "boxes": 0, "path": 12},
    8: {"keys": 1, "doors": 1, "lava": 2, "boxes": 0, "path": 15},
    9: {"keys": 2, "doors": 1, "lava": 2, "boxes": 0, "path": 18},
    10: {"keys": 2, "doors": 2, "lava": 3, "boxes": 0, "path": 20},

    # Tier 4: Puzzles (difficulty 11-15)
    11: {"keys": 1, "doors": 1, "lava": 1, "boxes": 1, "path": 15},
    12: {"keys": 1, "doors": 1, "lava": 2, "boxes": 1, "path": 18},
    13: {"keys": 2, "doors": 2, "lava": 2, "boxes": 2, "path": 20},
    14: {"keys": 2, "doors": 2, "lava": 3, "boxes": 2, "path": 25},
    15: {"keys": 3, "doors": 2, "lava": 3, "boxes": 3, "path": 28},

    # Tier 5: Challenge (difficulty 16-20)
    16: {"keys": 2, "doors": 2, "lava": 4, "boxes": 2, "path": 25},
    17: {"keys": 3, "doors": 3, "lava": 4, "boxes": 3, "path": 30},
    18: {"keys": 3, "doors": 3, "lava": 5, "boxes": 3, "path": 35},
    19: {"keys": 4, "doors": 3, "lava": 5, "boxes": 4, "path": 40},
    20: {"keys": 4, "doors": 4, "lava": 6, "boxes": 4, "path": 50},
}


class LevelGenerator:
    """
    Generates levels with controlled difficulty.

    Algorithm:
    1. Create empty grid with walls around edges
    2. Place start and goal positions
    3. Generate path from start to goal
    4. Add keys before their doors along path
    5. Add lava patches away from solution path
    6. Add boxes and buttons for puzzles
    7. Verify solution exists
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(self, difficulty: int = 1, seed: Optional[int] = None) -> GridWorld:
        """
        Generate a level with the specified difficulty.

        Args:
            difficulty: 1-20 scale
            seed: Random seed for reproducibility

        Returns:
            Generated GridWorld with guaranteed solution
        """
        if seed is not None:
            self.rng = random.Random(seed)

        # Clamp difficulty
        difficulty = max(1, min(20, difficulty))

        # Get tier parameters
        params = DIFFICULTY_TIERS.get(difficulty, DIFFICULTY_TIERS[1])

        # Size scales with difficulty
        width = 8 + (difficulty // 4) * 2
        height = 8 + (difficulty // 4) * 2

        # Try to generate a valid level
        for attempt in range(10):
            world = self._generate_attempt(
                width=width,
                height=height,
                num_keys=params["keys"],
                num_doors=params["doors"],
                num_lava=params["lava"],
                num_boxes=params["boxes"],
                min_path=params["path"],
            )

            if world and self._verify_solution(world):
                world.level_name = f"Generated L{difficulty}"
                return world

        # Fallback: return simple level
        return self._generate_simple(width, height)

    def _generate_attempt(
        self,
        width: int,
        height: int,
        num_keys: int,
        num_doors: int,
        num_lava: int,
        num_boxes: int,
        min_path: int,
    ) -> Optional[GridWorld]:
        """Attempt to generate a level with given parameters."""
        world = GridWorld(width, height)

        # Fill with walls initially
        for y in range(height):
            for x in range(width):
                world.grid[y][x] = Tile.WALL

        # Create rooms and corridors
        self._carve_rooms(world)

        # Find start and goal positions
        floor_tiles = self._get_floor_tiles(world)
        if len(floor_tiles) < 10:
            return None

        # Place start (corner preference)
        start = self._pick_corner_position(world, floor_tiles)
        if not start:
            return None
        world.robot.x, world.robot.y = start
        world.robot.direction = self.rng.choice(list(Direction))

        # Place goal (far from start)
        goal = self._pick_far_position(world, floor_tiles, start, min_dist=min_path // 2)
        if not goal:
            return None
        world.set_tile(goal[0], goal[1], Tile.GOAL)
        floor_tiles.remove(goal)

        # Place keys and doors along path
        if num_keys > 0 and num_doors > 0:
            path = self._find_path(world, start, goal, ignore_doors=True)
            if path and len(path) >= num_keys + num_doors + 2:
                self._place_keys_and_doors(world, path, num_keys, num_doors, floor_tiles)

        # Place lava patches (away from solution path)
        if num_lava > 0:
            self._place_lava(world, num_lava, floor_tiles, start, goal)

        # Place boxes (puzzle elements)
        # For now, skip boxes - they require careful placement
        # TODO: Add box puzzle generation

        return world

    def _carve_rooms(self, world: GridWorld) -> None:
        """Carve out rooms and corridors."""
        width, height = world.width, world.height

        # Simple room carving: create a border and random internal structure
        # Leave 1-tile border as walls
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Create mostly floor with some internal walls
                if self.rng.random() < 0.7:
                    world.grid[y][x] = Tile.FLOOR

        # Add some random wall clusters
        num_clusters = (width * height) // 30
        for _ in range(num_clusters):
            cx = self.rng.randint(2, width - 3)
            cy = self.rng.randint(2, height - 3)
            size = self.rng.randint(1, 3)
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    if 1 <= cx + dx < width - 1 and 1 <= cy + dy < height - 1:
                        if self.rng.random() < 0.6:
                            world.grid[cy + dy][cx + dx] = Tile.WALL

    def _get_floor_tiles(self, world: GridWorld) -> list[tuple[int, int]]:
        """Get all floor tile positions."""
        tiles = []
        for y in range(world.height):
            for x in range(world.width):
                if world.get_tile(x, y) == Tile.FLOOR:
                    tiles.append((x, y))
        return tiles

    def _pick_corner_position(
        self,
        world: GridWorld,
        floor_tiles: list[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Pick a position near a corner."""
        # Sort by distance to corners
        corners = [
            (1, 1),
            (world.width - 2, 1),
            (1, world.height - 2),
            (world.width - 2, world.height - 2),
        ]

        for corner in self.rng.sample(corners, len(corners)):
            # Find closest floor tile to this corner
            candidates = sorted(
                floor_tiles,
                key=lambda t: abs(t[0] - corner[0]) + abs(t[1] - corner[1])
            )
            if candidates:
                return candidates[0]

        return floor_tiles[0] if floor_tiles else None

    def _pick_far_position(
        self,
        world: GridWorld,
        floor_tiles: list[tuple[int, int]],
        from_pos: tuple[int, int],
        min_dist: int = 5,
    ) -> Optional[tuple[int, int]]:
        """Pick a position far from the given position."""
        candidates = [
            t for t in floor_tiles
            if abs(t[0] - from_pos[0]) + abs(t[1] - from_pos[1]) >= min_dist
        ]

        if not candidates:
            candidates = floor_tiles

        # Sort by distance (descending) and pick from top candidates
        candidates.sort(
            key=lambda t: abs(t[0] - from_pos[0]) + abs(t[1] - from_pos[1]),
            reverse=True
        )

        return self.rng.choice(candidates[:max(1, len(candidates) // 3)])

    def _find_path(
        self,
        world: GridWorld,
        start: tuple[int, int],
        goal: tuple[int, int],
        ignore_doors: bool = False,
    ) -> Optional[list[tuple[int, int]]]:
        """Find path from start to goal using BFS."""
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()

            if pos == goal:
                return path

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if (nx, ny) in visited:
                    continue

                tile = world.get_tile(nx, ny)

                # Can pass through
                can_pass = (
                    tile.walkable or
                    tile == Tile.GOAL or
                    (ignore_doors and tile == Tile.DOOR)
                )

                if can_pass:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    def _place_keys_and_doors(
        self,
        world: GridWorld,
        path: list[tuple[int, int]],
        num_keys: int,
        num_doors: int,
        floor_tiles: list[tuple[int, int]],
    ) -> None:
        """Place keys and doors along the solution path."""
        # Ensure keys come before doors
        # Split path into segments
        segment_size = len(path) // (num_keys + num_doors + 1)

        keys_placed = 0
        doors_placed = 0

        for i, pos in enumerate(path):
            segment = i // segment_size

            # First half: place keys
            if segment < num_keys and keys_placed < num_keys:
                if world.get_tile(pos[0], pos[1]) == Tile.FLOOR:
                    world.set_tile(pos[0], pos[1], Tile.KEY)
                    if pos in floor_tiles:
                        floor_tiles.remove(pos)
                    keys_placed += 1

            # Second half: place doors
            elif segment >= num_keys and doors_placed < num_doors:
                # Place door slightly off path as chokepoint
                if world.get_tile(pos[0], pos[1]) == Tile.FLOOR:
                    world.set_tile(pos[0], pos[1], Tile.DOOR)
                    if pos in floor_tiles:
                        floor_tiles.remove(pos)
                    doors_placed += 1

    def _place_lava(
        self,
        world: GridWorld,
        num_patches: int,
        floor_tiles: list[tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        """Place lava patches away from the solution."""
        # Find solution path to avoid
        solution = self._find_path(world, start, goal, ignore_doors=True)
        solution_set = set(solution) if solution else set()

        # Add buffer around solution
        buffered = set()
        for pos in solution_set:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    buffered.add((pos[0] + dx, pos[1] + dy))

        # Place lava patches
        candidates = [t for t in floor_tiles if t not in buffered]

        for _ in range(num_patches):
            if not candidates:
                break

            pos = self.rng.choice(candidates)
            candidates.remove(pos)

            # Create small lava patch
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = pos[0] + dx, pos[1] + dy
                    tile = world.get_tile(nx, ny)
                    if tile == Tile.FLOOR and (nx, ny) not in buffered:
                        if self.rng.random() < 0.6:
                            world.set_tile(nx, ny, Tile.LAVA)
                            if (nx, ny) in floor_tiles:
                                floor_tiles.remove((nx, ny))
                            if (nx, ny) in candidates:
                                candidates.remove((nx, ny))

    def _verify_solution(self, world: GridWorld) -> bool:
        """Verify that a solution exists from start to goal."""
        # Find goal position
        goal = None
        for y in range(world.height):
            for x in range(world.width):
                if world.get_tile(x, y) == Tile.GOAL:
                    goal = (x, y)
                    break

        if not goal:
            return False

        start = (world.robot.x, world.robot.y)

        # Count keys available
        num_keys = 0
        for y in range(world.height):
            for x in range(world.width):
                if world.get_tile(x, y) == Tile.KEY:
                    num_keys += 1

        # BFS with key tracking
        queue = deque([(start, 0)])  # (position, keys_collected)
        visited = {(start, 0)}

        while queue:
            pos, keys = queue.popleft()

            if pos == goal:
                return True

            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                tile = world.get_tile(nx, ny)

                new_keys = keys

                # Check if we can enter this tile
                if tile.restricted:
                    continue
                if tile == Tile.WALL:
                    continue
                if tile == Tile.DOOR and keys == 0:
                    continue

                # Collect key
                if tile == Tile.KEY:
                    new_keys = min(keys + 1, num_keys)

                # Use key on door
                if tile == Tile.DOOR:
                    new_keys = keys - 1

                state = ((nx, ny), new_keys)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)

        return False

    def _generate_simple(self, width: int, height: int) -> GridWorld:
        """Generate a simple fallback level."""
        world = GridWorld(width, height)

        # Create border walls
        for y in range(height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    world.grid[y][x] = Tile.WALL
                else:
                    world.grid[y][x] = Tile.FLOOR

        # Place start and goal
        world.robot.x = 1
        world.robot.y = 1
        world.robot.direction = Direction.EAST
        world.set_tile(width - 2, height - 2, Tile.GOAL)

        world.level_name = "Simple Fallback"
        return world


# Convenience function
def generate_level(difficulty: int = 1, seed: Optional[int] = None) -> GridWorld:
    """Generate a level with specified difficulty."""
    generator = LevelGenerator(seed)
    return generator.generate(difficulty, seed)


def generate_curriculum(
    start_difficulty: int = 1,
    end_difficulty: int = 20,
    levels_per_tier: int = 3,
) -> list[GridWorld]:
    """
    Generate a curriculum of levels with increasing difficulty.

    Returns:
        List of levels from easy to hard
    """
    generator = LevelGenerator()
    levels = []

    for difficulty in range(start_difficulty, end_difficulty + 1):
        for i in range(levels_per_tier):
            level = generator.generate(difficulty, seed=difficulty * 1000 + i)
            levels.append(level)

    return levels
