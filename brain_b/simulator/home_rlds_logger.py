"""
RLDS Episode Logger for 3D Home Exploration.

Captures game sessions as RLDS episodes for Brain A training.
Follows the schema defined in docs/rlds-schema.md v1.1.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from pathlib import Path

from simulator.home_world import HomeWorld, ObjectType, RoomType


@dataclass
class Home3DObservation:
    """Observation at a single step in 3D home."""
    robot_state: dict
    current_room: str
    visible_objects: List[dict]
    goal_distance: float
    position_3d: dict
    rotation_3d: dict
    inventory: List[str]


@dataclass
class Home3DAction:
    """Action taken at a single step."""
    command: str
    intent: str
    params: dict = field(default_factory=dict)
    raw_input: str = ""


@dataclass
class Home3DStep:
    """A single step in an RLDS episode."""
    frame_id: int
    timestamp_us: int
    observation: Home3DObservation
    action: Home3DAction
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Home3DEpisodeMetadata:
    """Metadata for a 3D home RLDS episode."""
    schema_version: str = "1.1"
    episode_id: str = ""
    robot_id: str = "home3d_simulator_v1"
    robot_model: str = "Home3D/ExplorerBot"
    capabilities: List[str] = field(default_factory=lambda: [
        "navigation_3d",
        "object_interaction",
        "room_exploration",
    ])

    # Game-specific metadata
    level_id: str = ""
    level_description: str = ""
    world_size: dict = field(default_factory=dict)  # width, depth, height
    initial_state: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)
    success: bool = False
    total_moves: int = 0
    rooms_visited: List[str] = field(default_factory=list)
    objects_interacted: List[str] = field(default_factory=list)
    items_collected: List[str] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0

    # Provenance
    source: str = "home3d_simulator"
    session_id: str = ""


class HomeRLDSLogger:
    """
    Logs 3D home game sessions as RLDS episodes.

    Usage:
        logger = HomeRLDSLogger(output_dir="./home_rlds_episodes")
        logger.start_episode(world, level_id="simple_apartment")

        # For each game action:
        logger.log_step(world, action, result)

        # When done:
        logger.end_episode(success=True)
    """

    def __init__(self, output_dir: str = "./brain_b_data/home_rlds_episodes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_episode: Optional[Home3DEpisodeMetadata] = None
        self.steps: List[Home3DStep] = []
        self.frame_counter: int = 0

        # Track progress
        self._rooms_visited: set = set()
        self._objects_interacted: set = set()
        self._items_collected: List[str] = []

    def start_episode(
        self,
        world: HomeWorld,
        level_id: str = "",
        session_id: str = "",
    ) -> str:
        """
        Start recording a new episode.

        Returns:
            episode_id: Unique ID for this episode
        """
        episode_id = f"home3d_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        self.current_episode = Home3DEpisodeMetadata(
            episode_id=episode_id,
            level_id=level_id,
            level_description=world.goal_description,
            world_size={
                "width": world.width,
                "depth": world.depth,
                "height": world.height,
            },
            initial_state=world.to_dict(),
            start_time=time.time(),
            session_id=session_id or str(uuid.uuid4()),
        )

        self.steps = []
        self.frame_counter = 0
        self._rooms_visited = set()
        self._objects_interacted = set()
        self._items_collected = []

        # Record initial room
        room = world.get_room_at(world.robot.position)
        if room:
            self._rooms_visited.add(room.room_type.value)

        return episode_id

    def log_step(
        self,
        world: HomeWorld,
        action_command: str,
        action_intent: str,
        action_params: dict,
        raw_input: str,
        success: bool,
        sandbox_denied: bool = False,
        item_collected: Optional[str] = None,
        interacted_object: Optional[str] = None,
        level_complete: bool = False,
    ) -> None:
        """Log a single step in the episode."""
        if self.current_episode is None:
            return

        # Track room visits
        room = world.get_room_at(world.robot.position)
        if room:
            self._rooms_visited.add(room.room_type.value)

        # Track interactions
        if interacted_object:
            self._objects_interacted.add(interacted_object)

        # Track collections
        if item_collected:
            self._items_collected.append(item_collected)

        # Build observation
        observation = self._build_observation(world)

        # Build action
        action = Home3DAction(
            command=action_command,
            intent=action_intent,
            params=action_params,
            raw_input=raw_input,
        )

        # Compute reward
        reward = self._compute_reward(
            success=success,
            sandbox_denied=sandbox_denied,
            item_collected=item_collected,
            interacted_object=interacted_object,
            level_complete=level_complete,
        )

        step = Home3DStep(
            frame_id=self.frame_counter,
            timestamp_us=int(time.time() * 1_000_000),
            observation=observation,
            action=action,
            reward=reward,
            done=level_complete,
            info={
                "success": success,
                "sandbox_denied": sandbox_denied,
                "item_collected": item_collected,
                "interacted_object": interacted_object,
            },
        )

        self.steps.append(step)
        self.frame_counter += 1

    def _build_observation(self, world: HomeWorld) -> Home3DObservation:
        """Build observation from current world state."""
        robot = world.robot
        room = world.get_room_at(robot.position)

        return Home3DObservation(
            robot_state=robot.to_dict(),
            current_room=room.room_type.value if room else "unknown",
            visible_objects=world.get_visible_objects(max_distance=5.0),
            goal_distance=robot.position.distance_to(world.goal_position) if world.goal_position else -1,
            position_3d=robot.position.to_dict(),
            rotation_3d=robot.rotation.to_dict(),
            inventory=robot.inventory.copy(),
        )

    def _compute_reward(
        self,
        success: bool,
        sandbox_denied: bool,
        item_collected: Optional[str],
        interacted_object: Optional[str],
        level_complete: bool,
    ) -> float:
        """Compute reward for this step."""
        reward = 0.0

        if level_complete:
            reward += 10.0
        elif success:
            reward += 0.1
        elif sandbox_denied:
            reward -= 1.0
        else:
            reward -= 0.05  # Small penalty for failed actions

        if item_collected:
            # Bonus for collecting important items
            if item_collected in ("key", "remote", "phone"):
                reward += 2.0
            else:
                reward += 0.5

        if interacted_object:
            reward += 0.3  # Bonus for interactions

        return reward

    def end_episode(self, world: HomeWorld, success: bool) -> str:
        """
        End the current episode and save to disk.

        Returns:
            Path to saved episode directory
        """
        if self.current_episode is None:
            raise ValueError("No episode in progress")

        # Update metadata
        self.current_episode.final_state = world.to_dict()
        self.current_episode.success = success
        self.current_episode.total_moves = world.robot.moves
        self.current_episode.rooms_visited = list(self._rooms_visited)
        self.current_episode.objects_interacted = list(self._objects_interacted)
        self.current_episode.items_collected = self._items_collected.copy()
        self.current_episode.end_time = time.time()
        self.current_episode.duration_s = (
            self.current_episode.end_time - self.current_episode.start_time
        )

        # Build episode data
        episode_data = {
            "metadata": asdict(self.current_episode),
            "steps": [self._step_to_dict(s) for s in self.steps],
        }

        # Save to file
        episode_dir = self.output_dir / self.current_episode.episode_id
        episode_dir.mkdir(exist_ok=True)

        # Save metadata.json
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(episode_data["metadata"], f, indent=2)

        # Save steps directory with 000000.jsonl (RLDS format)
        steps_dir = episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)
        steps_path = steps_dir / "000000.jsonl"
        with open(steps_path, "w") as f:
            for step in episode_data["steps"]:
                f.write(json.dumps(step) + "\n")

        # Also save flat steps.jsonl for compatibility
        flat_steps_path = episode_dir / "steps.jsonl"
        with open(flat_steps_path, "w") as f:
            for step in episode_data["steps"]:
                f.write(json.dumps(step) + "\n")

        # Reset state
        result_path = str(episode_dir)
        self.current_episode = None
        self.steps = []
        self.frame_counter = 0

        return result_path

    def _step_to_dict(self, step: Home3DStep) -> dict:
        """Convert step to dictionary for serialization."""
        return {
            "frame_id": step.frame_id,
            "timestamp_us": step.timestamp_us,
            "observation": asdict(step.observation),
            "action": asdict(step.action),
            "reward": step.reward,
            "done": step.done,
            "info": step.info,
        }

    def list_episodes(self) -> List[dict]:
        """List all recorded episodes."""
        episodes = []
        for ep_dir in self.output_dir.iterdir():
            if ep_dir.is_dir():
                metadata_path = ep_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        episodes.append(json.load(f))
        return sorted(episodes, key=lambda e: e.get("start_time", 0), reverse=True)

    def load_episode(self, episode_id: str) -> dict:
        """Load a specific episode."""
        ep_dir = self.output_dir / episode_id
        if not ep_dir.exists():
            raise ValueError(f"Episode not found: {episode_id}")

        metadata_path = ep_dir / "metadata.json"

        # Try both step file locations
        steps_path = ep_dir / "steps" / "000000.jsonl"
        if not steps_path.exists():
            steps_path = ep_dir / "steps.jsonl"

        with open(metadata_path) as f:
            metadata = json.load(f)

        steps = []
        with open(steps_path) as f:
            for line in f:
                steps.append(json.loads(line))

        return {"metadata": metadata, "steps": steps}

    def get_statistics(self) -> dict:
        """Get statistics about recorded episodes."""
        episodes = self.list_episodes()

        if not episodes:
            return {
                "total_episodes": 0,
                "successful_episodes": 0,
                "total_moves": 0,
                "avg_duration": 0,
            }

        successful = sum(1 for e in episodes if e.get("success", False))
        total_moves = sum(e.get("total_moves", 0) for e in episodes)
        total_duration = sum(e.get("duration_s", 0) for e in episodes)

        return {
            "total_episodes": len(episodes),
            "successful_episodes": successful,
            "success_rate": successful / len(episodes) if episodes else 0,
            "total_moves": total_moves,
            "avg_moves": total_moves / len(episodes) if episodes else 0,
            "total_duration": total_duration,
            "avg_duration": total_duration / len(episodes) if episodes else 0,
            "unique_levels": len(set(e.get("level_id", "") for e in episodes)),
        }
