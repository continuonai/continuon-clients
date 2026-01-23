"""
RLDS Episode Logger for RobotGrid.

Captures game sessions as RLDS episodes for Brain A training.
Follows the schema defined in docs/rlds-schema.md v1.1.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from simulator.world import GridWorld, Direction, Tile


@dataclass
class WorldModelData:
    """World model prediction data for a step."""
    predicted_state: Optional[dict] = None
    actual_state: Optional[dict] = None
    surprise: float = 0.0
    belief_confidence: float = 1.0
    latent_tokens: Optional[list] = None


@dataclass
class StepObservation:
    """Observation at a single step."""
    robot_state: dict
    visible_tiles: list[dict]
    look_ahead: dict
    grid_snapshot: Optional[list[list[str]]] = None


@dataclass
class StepAction:
    """Action taken at a single step."""
    command: str
    intent: str
    params: dict = field(default_factory=dict)
    raw_input: str = ""


@dataclass
class Step:
    """A single step in an RLDS episode."""
    frame_id: int
    timestamp_us: int
    observation: StepObservation
    action: StepAction
    world_model: WorldModelData
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class EpisodeMetadata:
    """Metadata for an RLDS episode."""
    schema_version: str = "1.1"
    episode_id: str = ""
    robot_id: str = "simulator_v1"
    robot_model: str = "RobotGrid/GridBot"
    capabilities: list[str] = field(default_factory=lambda: ["navigation", "manipulation", "planning"])

    # Game-specific metadata
    level_id: str = ""
    level_name: str = ""
    level_difficulty: int = 1
    initial_state: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)
    success: bool = False
    total_moves: int = 0
    sandbox_denials: int = 0
    behaviors_used: list[str] = field(default_factory=list)
    keys_collected: int = 0
    doors_opened: int = 0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0

    # Provenance
    source: str = "robotgrid_simulator"
    session_id: str = ""


class RLDSLogger:
    """
    Logs game sessions as RLDS episodes.

    Usage:
        logger = RLDSLogger(output_dir="./rlds_episodes")
        logger.start_episode(world, level_id="tutorial")

        # For each game action:
        logger.log_step(world, action, result)

        # When done:
        logger.end_episode(success=True)
    """

    def __init__(self, output_dir: str = "./brain_b_data/rlds_episodes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_episode: Optional[EpisodeMetadata] = None
        self.steps: list[Step] = []
        self.frame_counter: int = 0
        self.world_model: Optional["WorldModelPredictor"] = None

        # Track behaviors used
        self._behaviors_used: set[str] = set()
        self._sandbox_denials: int = 0
        self._keys_collected: int = 0
        self._doors_opened: int = 0

    def set_world_model(self, model: "WorldModelPredictor") -> None:
        """Set world model for prediction and surprise computation."""
        self.world_model = model

    def start_episode(
        self,
        world: GridWorld,
        level_id: str = "",
        session_id: str = "",
        difficulty: int = 1,
    ) -> str:
        """
        Start recording a new episode.

        Returns:
            episode_id: Unique ID for this episode
        """
        episode_id = f"robotgrid_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        self.current_episode = EpisodeMetadata(
            episode_id=episode_id,
            level_id=level_id,
            level_name=world.level_name,
            level_difficulty=difficulty,
            initial_state=world.to_dict(),
            start_time=time.time(),
            session_id=session_id or str(uuid.uuid4()),
        )

        self.steps = []
        self.frame_counter = 0
        self._behaviors_used = set()
        self._sandbox_denials = 0
        self._keys_collected = 0
        self._doors_opened = 0

        return episode_id

    def log_step(
        self,
        world: GridWorld,
        action_command: str,
        action_intent: str,
        action_params: dict,
        raw_input: str,
        success: bool,
        sandbox_denied: bool = False,
        item_collected: Optional[str] = None,
        door_opened: bool = False,
        behavior_invoked: Optional[str] = None,
        level_complete: bool = False,
    ) -> None:
        """Log a single step in the episode."""
        if self.current_episode is None:
            return

        # Track counters
        if sandbox_denied:
            self._sandbox_denials += 1
        if item_collected == "key":
            self._keys_collected += 1
        if door_opened:
            self._doors_opened += 1
        if behavior_invoked:
            self._behaviors_used.add(behavior_invoked)

        # Build observation
        observation = self._build_observation(world)

        # Build action
        action = StepAction(
            command=action_command,
            intent=action_intent,
            params=action_params,
            raw_input=raw_input,
        )

        # World model prediction and surprise
        world_model_data = self._compute_world_model(world, action)

        # Compute reward
        reward = self._compute_reward(
            success=success,
            sandbox_denied=sandbox_denied,
            item_collected=item_collected,
            level_complete=level_complete,
        )

        step = Step(
            frame_id=self.frame_counter,
            timestamp_us=int(time.time() * 1_000_000),
            observation=observation,
            action=action,
            world_model=world_model_data,
            reward=reward,
            done=level_complete,
            info={
                "success": success,
                "sandbox_denied": sandbox_denied,
                "item_collected": item_collected,
                "door_opened": door_opened,
            },
        )

        self.steps.append(step)
        self.frame_counter += 1

    def _build_observation(self, world: GridWorld) -> StepObservation:
        """Build observation from current world state."""
        robot = world.robot

        # Get visible tiles (3x3 around robot)
        visible = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x, y = robot.x + dx, robot.y + dy
                tile = world.get_tile(x, y)
                visible.append({
                    "x": x,
                    "y": y,
                    "dx": dx,
                    "dy": dy,
                    "tile": tile.name,
                    "walkable": tile.walkable,
                    "restricted": tile.restricted,
                })

        # Look ahead in facing direction
        dx, dy = robot.direction.delta()
        look_x, look_y = robot.x + dx, robot.y + dy
        look_tile = world.get_tile(look_x, look_y)

        # Count distance to obstacle
        distance = 0
        check_x, check_y = robot.x, robot.y
        while distance < 10:
            check_x += dx
            check_y += dy
            t = world.get_tile(check_x, check_y)
            if not t.walkable or t.restricted:
                break
            distance += 1

        return StepObservation(
            robot_state={
                "x": robot.x,
                "y": robot.y,
                "direction": robot.direction.name,
                "inventory": robot.inventory.copy(),
                "moves": robot.moves,
            },
            visible_tiles=visible,
            look_ahead={
                "tile": look_tile.name,
                "distance": distance,
                "walkable": look_tile.walkable,
                "restricted": look_tile.restricted,
            },
        )

    def _compute_world_model(self, world: GridWorld, action: StepAction) -> WorldModelData:
        """Compute world model prediction and surprise."""
        if self.world_model is None:
            return WorldModelData()

        # Get prediction before action
        predicted = self.world_model.predict(world, action.command)

        # Actual state after action (already applied)
        actual = {
            "x": world.robot.x,
            "y": world.robot.y,
            "direction": world.robot.direction.name,
            "inventory": world.robot.inventory.copy(),
        }

        # Compute surprise (simple state divergence)
        surprise = self.world_model.compute_surprise(predicted, actual)

        return WorldModelData(
            predicted_state=predicted,
            actual_state=actual,
            surprise=surprise,
            belief_confidence=1.0 - surprise,
        )

    def _compute_reward(
        self,
        success: bool,
        sandbox_denied: bool,
        item_collected: Optional[str],
        level_complete: bool,
    ) -> float:
        """Compute reward for this step."""
        reward = 0.0

        if level_complete:
            reward += 10.0
        elif success:
            reward += 0.1
        elif sandbox_denied:
            reward -= 1.0  # Penalty for hitting restricted zones
        else:
            reward -= 0.1  # Small penalty for failed actions

        if item_collected:
            reward += 1.0

        return reward

    def end_episode(self, world: GridWorld, success: bool) -> str:
        """
        End the current episode and save to disk.

        Returns:
            Path to saved episode file
        """
        if self.current_episode is None:
            raise ValueError("No episode in progress")

        # Update metadata
        self.current_episode.final_state = world.to_dict()
        self.current_episode.success = success
        self.current_episode.total_moves = world.robot.moves
        self.current_episode.sandbox_denials = self._sandbox_denials
        self.current_episode.behaviors_used = list(self._behaviors_used)
        self.current_episode.keys_collected = self._keys_collected
        self.current_episode.doors_opened = self._doors_opened
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

        # Save steps/000000.jsonl (standard RLDS v2.0 format)
        steps_dir = episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)
        steps_path = steps_dir / "000000.jsonl"
        with open(steps_path, "w") as f:
            for step in episode_data["steps"]:
                f.write(json.dumps(step) + "\n")

        # Reset state
        result_path = str(episode_dir)
        self.current_episode = None
        self.steps = []
        self.frame_counter = 0

        return result_path

    def _step_to_dict(self, step: Step) -> dict:
        """Convert step to dictionary for serialization."""
        return {
            "frame_id": step.frame_id,
            "timestamp_us": step.timestamp_us,
            "observation": asdict(step.observation),
            "action": asdict(step.action),
            "world_model": asdict(step.world_model),
            "reward": step.reward,
            "done": step.done,
            "info": step.info,
        }

    def list_episodes(self) -> list[dict]:
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
        steps_path = ep_dir / "steps.jsonl"

        with open(metadata_path) as f:
            metadata = json.load(f)

        steps = []
        with open(steps_path) as f:
            for line in f:
                steps.append(json.loads(line))

        return {"metadata": metadata, "steps": steps}


class WorldModelPredictor:
    """
    Simple world model for predicting next game state.

    Initially rule-based, can be replaced with learned model.
    """

    def __init__(self):
        pass

    def predict(self, world: GridWorld, action: str) -> dict:
        """Predict next state given current state and action."""
        robot = world.robot
        x, y = robot.x, robot.y
        direction = robot.direction

        if action == "forward":
            dx, dy = direction.delta()
            target = world.get_tile(x + dx, y + dy)
            if target.walkable and not target.restricted:
                x, y = x + dx, y + dy
        elif action == "backward":
            opposite = direction.turn_left().turn_left()
            dx, dy = opposite.delta()
            target = world.get_tile(x + dx, y + dy)
            if target.walkable and not target.restricted:
                x, y = x + dx, y + dy
        elif action == "left":
            direction = direction.turn_left()
        elif action == "right":
            direction = direction.turn_right()

        return {
            "x": x,
            "y": y,
            "direction": direction.name,
            "inventory": robot.inventory.copy(),
        }

    def compute_surprise(self, predicted: dict, actual: dict) -> float:
        """Compute surprise as state divergence."""
        if predicted is None:
            return 0.0

        # Position mismatch
        pos_match = (predicted["x"] == actual["x"] and predicted["y"] == actual["y"])

        # Direction mismatch
        dir_match = (predicted["direction"] == actual["direction"])

        # Inventory mismatch
        inv_match = (set(predicted.get("inventory", [])) == set(actual.get("inventory", [])))

        # Weighted surprise
        surprise = 0.0
        if not pos_match:
            surprise += 0.5
        if not dir_match:
            surprise += 0.3
        if not inv_match:
            surprise += 0.2

        return surprise
