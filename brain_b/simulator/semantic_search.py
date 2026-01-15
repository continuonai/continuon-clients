"""
Semantic Search for RobotGrid Game States.

Enables finding similar past states for:
- Strategy recall ("What worked last time?")
- Anomaly detection ("This state is unusual")
- Curriculum learning ("Start from similar solved states")
"""

import json
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from simulator.world import GridWorld, Direction, Tile


@dataclass
class StateEmbedding:
    """Embedding of a game state for similarity search."""
    episode_id: str
    frame_id: int
    vector: list[float]
    metadata: dict


@dataclass
class SearchMatch:
    """A match from semantic search."""
    episode_id: str
    frame_id: int
    similarity: float
    state: dict
    action_taken: Optional[str] = None
    outcome: Optional[str] = None


class StateEmbedder:
    """
    Embeds game states for semantic similarity search.

    Encoding dimensions:
    - Position (normalized x, y)
    - Direction (one-hot, 4 dims)
    - Inventory (binary flags)
    - Visible tiles (pattern features)
    - Level progress (goal distance, moves)
    - Game state flags (has_key, near_lava, etc.)
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        # Embedding dimension: 2 (pos) + 4 (dir) + 4 (inv) + 9 (visible) + 4 (progress) + 8 (flags) = 31
        self.embedding_dim = 31

    def embed(self, world: GridWorld) -> list[float]:
        """Create embedding vector for current game state."""
        robot = world.robot
        vector = []

        # 1. Normalized position (2 dims)
        vector.append(robot.x / max(world.width, 1))
        vector.append(robot.y / max(world.height, 1))

        # 2. Direction one-hot (4 dims)
        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_idx = {Direction.NORTH: 0, Direction.EAST: 1, Direction.SOUTH: 2, Direction.WEST: 3}
        dir_onehot[dir_idx[robot.direction]] = 1.0
        vector.extend(dir_onehot)

        # 3. Inventory flags (4 dims: key, multiple keys, has_items, item_count)
        has_key = 1.0 if "key" in robot.inventory else 0.0
        key_count = robot.inventory.count("key")
        has_items = 1.0 if robot.inventory else 0.0
        item_count = min(len(robot.inventory) / 5.0, 1.0)  # Normalized
        vector.extend([has_key, min(key_count / 3.0, 1.0), has_items, item_count])

        # 4. Visible tiles pattern (9 dims: 3x3 walkability)
        dx_map, dy_map = robot.direction.delta()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x, y = robot.x + dx, robot.y + dy
                tile = world.get_tile(x, y)
                # Encode as: 1.0 = walkable, 0.5 = restricted, 0.0 = blocked
                if tile.walkable:
                    vector.append(1.0)
                elif tile.restricted:
                    vector.append(0.5)
                else:
                    vector.append(0.0)

        # 5. Progress features (4 dims)
        # Find goal position
        goal_dist = self._find_goal_distance(world)
        vector.append(min(goal_dist / 20.0, 1.0))  # Normalized distance to goal
        vector.append(min(robot.moves / 100.0, 1.0))  # Normalized move count
        # Keys remaining (estimate from level)
        keys_on_map = self._count_tiles(world, Tile.KEY)
        doors_on_map = self._count_tiles(world, Tile.DOOR)
        vector.append(min(keys_on_map / 3.0, 1.0))
        vector.append(min(doors_on_map / 3.0, 1.0))

        # 6. Game state flags (8 dims)
        ahead_tile = self._get_ahead_tile(world)
        flags = [
            1.0 if ahead_tile == Tile.WALL else 0.0,      # Wall ahead
            1.0 if ahead_tile == Tile.LAVA else 0.0,      # Lava ahead
            1.0 if ahead_tile == Tile.KEY else 0.0,       # Key ahead
            1.0 if ahead_tile == Tile.DOOR else 0.0,      # Door ahead
            1.0 if ahead_tile == Tile.GOAL else 0.0,      # Goal ahead
            1.0 if ahead_tile == Tile.BOX else 0.0,       # Box ahead
            1.0 if self._is_near_lava(world) else 0.0,    # Near lava
            1.0 if self._is_corner(world) else 0.0,       # In corner
        ]
        vector.extend(flags)

        return vector

    def _find_goal_distance(self, world: GridWorld) -> float:
        """Find Manhattan distance to nearest goal."""
        robot = world.robot
        min_dist = float('inf')

        for y in range(world.height):
            for x in range(world.width):
                if world.get_tile(x, y) == Tile.GOAL:
                    dist = abs(x - robot.x) + abs(y - robot.y)
                    min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 20.0

    def _count_tiles(self, world: GridWorld, tile_type: Tile) -> int:
        """Count tiles of a specific type."""
        count = 0
        for y in range(world.height):
            for x in range(world.width):
                if world.get_tile(x, y) == tile_type:
                    count += 1
        return count

    def _get_ahead_tile(self, world: GridWorld) -> Tile:
        """Get tile directly ahead of robot."""
        dx, dy = world.robot.direction.delta()
        return world.get_tile(world.robot.x + dx, world.robot.y + dy)

    def _is_near_lava(self, world: GridWorld) -> bool:
        """Check if robot is adjacent to lava."""
        robot = world.robot
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if world.get_tile(robot.x + dx, robot.y + dy) == Tile.LAVA:
                    return True
        return False

    def _is_corner(self, world: GridWorld) -> bool:
        """Check if robot is in a corner (3+ adjacent walls)."""
        robot = world.robot
        wall_count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if world.get_tile(robot.x + dx, robot.y + dy) == Tile.WALL:
                    wall_count += 1
        return wall_count >= 5


class SemanticSearchIndex:
    """
    Index for semantic search over game state history.

    Stores embeddings and enables similarity queries.
    """

    def __init__(self, embedder: Optional[StateEmbedder] = None):
        self.embedder = embedder or StateEmbedder()
        self.embeddings: list[StateEmbedding] = []
        self.index_path: Optional[Path] = None

    def add(
        self,
        world: GridWorld,
        episode_id: str,
        frame_id: int,
        action_taken: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> None:
        """Add a state to the index."""
        vector = self.embedder.embed(world)

        embedding = StateEmbedding(
            episode_id=episode_id,
            frame_id=frame_id,
            vector=vector,
            metadata={
                "robot_x": world.robot.x,
                "robot_y": world.robot.y,
                "direction": world.robot.direction.name,
                "inventory": world.robot.inventory.copy(),
                "moves": world.robot.moves,
                "level": world.level_name,
                "action_taken": action_taken,
                "outcome": outcome,
            },
        )

        self.embeddings.append(embedding)

    def search(
        self,
        query_world: GridWorld,
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SearchMatch]:
        """Find k most similar past states."""
        query_vector = self.embedder.embed(query_world)

        # Compute similarities
        scored = []
        for emb in self.embeddings:
            sim = self._cosine_similarity(query_vector, emb.vector)
            if sim >= min_similarity:
                scored.append((sim, emb))

        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top k
        results = []
        for sim, emb in scored[:k]:
            results.append(SearchMatch(
                episode_id=emb.episode_id,
                frame_id=emb.frame_id,
                similarity=sim,
                state=emb.metadata,
                action_taken=emb.metadata.get("action_taken"),
                outcome=emb.metadata.get("outcome"),
            ))

        return results

    def search_by_action(
        self,
        query_world: GridWorld,
        action: str,
        k: int = 5,
    ) -> list[SearchMatch]:
        """Find similar states where a specific action was taken."""
        query_vector = self.embedder.embed(query_world)

        # Filter by action and compute similarities
        scored = []
        for emb in self.embeddings:
            if emb.metadata.get("action_taken") == action:
                sim = self._cosine_similarity(query_vector, emb.vector)
                scored.append((sim, emb))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, emb in scored[:k]:
            results.append(SearchMatch(
                episode_id=emb.episode_id,
                frame_id=emb.frame_id,
                similarity=sim,
                state=emb.metadata,
                action_taken=emb.metadata.get("action_taken"),
                outcome=emb.metadata.get("outcome"),
            ))

        return results

    def find_successful_strategies(
        self,
        query_world: GridWorld,
        k: int = 3,
    ) -> list[SearchMatch]:
        """Find similar states that led to successful outcomes."""
        query_vector = self.embedder.embed(query_world)

        # Filter by successful outcome and compute similarities
        scored = []
        for emb in self.embeddings:
            outcome = emb.metadata.get("outcome", "")
            if "success" in outcome.lower() or "complete" in outcome.lower():
                sim = self._cosine_similarity(query_vector, emb.vector)
                scored.append((sim, emb))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, emb in scored[:k]:
            results.append(SearchMatch(
                episode_id=emb.episode_id,
                frame_id=emb.frame_id,
                similarity=sim,
                state=emb.metadata,
                action_taken=emb.metadata.get("action_taken"),
                outcome=emb.metadata.get("outcome"),
            ))

        return results

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def save(self, path: str) -> None:
        """Save index to disk."""
        data = {
            "embeddings": [
                {
                    "episode_id": e.episode_id,
                    "frame_id": e.frame_id,
                    "vector": e.vector,
                    "metadata": e.metadata,
                }
                for e in self.embeddings
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f)

        self.index_path = Path(path)

    def load(self, path: str) -> None:
        """Load index from disk."""
        with open(path) as f:
            data = json.load(f)

        self.embeddings = [
            StateEmbedding(
                episode_id=e["episode_id"],
                frame_id=e["frame_id"],
                vector=e["vector"],
                metadata=e["metadata"],
            )
            for e in data["embeddings"]
        ]

        self.index_path = Path(path)

    def stats(self) -> dict:
        """Get index statistics."""
        if not self.embeddings:
            return {"total_states": 0}

        episodes = set(e.episode_id for e in self.embeddings)
        levels = set(e.metadata.get("level", "unknown") for e in self.embeddings)

        return {
            "total_states": len(self.embeddings),
            "unique_episodes": len(episodes),
            "unique_levels": len(levels),
            "embedding_dim": self.embedder.embedding_dim,
        }


def build_index_from_episodes(episodes_dir: str, index_path: str) -> SemanticSearchIndex:
    """
    Build a search index from saved RLDS episodes.

    Args:
        episodes_dir: Directory containing episode folders
        index_path: Where to save the index

    Returns:
        Populated search index
    """
    from simulator.world import GridWorld

    index = SemanticSearchIndex()
    episodes_path = Path(episodes_dir)

    for ep_dir in episodes_path.iterdir():
        if not ep_dir.is_dir():
            continue

        steps_file = ep_dir / "steps.jsonl"
        metadata_file = ep_dir / "metadata.json"

        if not steps_file.exists() or not metadata_file.exists():
            continue

        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        episode_id = metadata.get("episode_id", ep_dir.name)

        # Load steps and reconstruct states
        with open(steps_file) as f:
            for line in f:
                step = json.loads(line)
                frame_id = step["frame_id"]

                # Create minimal world from observation
                obs = step["observation"]
                robot_state = obs["robot_state"]

                # We can't fully reconstruct the grid, but we can store the embedding
                # For full reconstruction, we'd need to store the grid snapshot
                index.embeddings.append(StateEmbedding(
                    episode_id=episode_id,
                    frame_id=frame_id,
                    vector=[0.0] * index.embedder.embedding_dim,  # Placeholder
                    metadata={
                        "robot_x": robot_state["x"],
                        "robot_y": robot_state["y"],
                        "direction": robot_state["direction"],
                        "inventory": robot_state.get("inventory", []),
                        "moves": robot_state.get("moves", 0),
                        "level": metadata.get("level_name", "unknown"),
                        "action_taken": step["action"]["command"],
                        "outcome": "success" if step.get("done") else "in_progress",
                    },
                ))

    index.save(index_path)
    return index
