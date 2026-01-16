#!/usr/bin/env python3
"""
HomeScan - 3D Home Exploration Simulator Integration for Trainer UI

Integrates the brain_b 3D home simulator with trainer_ui for:
- Real-time 3D navigation training
- RLDS episode recording for Brain A training
- Visual rendering of the 3D world
- Action prediction from trained models

Usage:
    # From trainer_ui server, after importing:
    home_scan = HomeScanIntegration()
    await home_scan.initialize()
    response = await home_scan.execute_command("forward")
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any
import sys

# Add parent directory and brain_b for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "brain_b"))

# Import 3D home simulator components
try:
    from brain_b.simulator.home_world import (
        HomeWorld,
        Robot3D,
        Position3D,
        Rotation3D,
        ObjectType,
        RoomType,
        LEVELS as HOME_LEVELS,
        CURRICULUM_ORDER,
        get_level as get_home_level,
        list_levels as list_home_levels,
    )
    from brain_b.simulator.home_handler import HomeHandler, Home3DIntent, Home3DResponse
    from brain_b.simulator.home_rlds_logger import HomeRLDSLogger
    from brain_b.simulator.home_training import (
        Home3DNavigationPredictor,
        HOME_ACTIONS,
    )
    HAS_HOME_SIMULATOR = True
except ImportError as e:
    print(f"HomeScan: 3D home simulator not available: {e}")
    HAS_HOME_SIMULATOR = False

# Export for external use
HAS_HOME_SCAN = HAS_HOME_SIMULATOR


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HomeScanConfig:
    """Configuration for HomeScan integration."""
    rlds_output_path: Path = field(default_factory=lambda: Path("../brain_b/brain_b_data/home_rlds"))
    model_path: Optional[Path] = field(default_factory=lambda: Path("../brain_b/brain_b_data/home_models/home3d_nav_model.json"))
    default_level: str = "empty_room"
    auto_record: bool = True  # Auto-start RLDS recording
    render_mode: str = "ascii"  # ascii, json, or visual
    auto_load_model: bool = True  # Auto-load model if available


# ============================================================================
# HomeScan State
# ============================================================================

@dataclass
class HomeScanState:
    """Current state of the HomeScan simulator."""
    active: bool = False
    level_name: str = ""
    robot_position: Dict[str, float] = field(default_factory=dict)
    robot_rotation: Dict[str, float] = field(default_factory=dict)
    current_room: str = ""
    visible_objects: List[Dict[str, Any]] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    battery: float = 1.0
    moves: int = 0
    goal_reached: bool = False
    goal_distance: float = 0.0
    recording: bool = False
    episode_id: Optional[str] = None
    last_action: str = ""
    last_reward: float = 0.0
    predictor_ready: bool = False


# ============================================================================
# HomeScan Integration
# ============================================================================

class HomeScanIntegration:
    """
    Integrates 3D home simulator with trainer_ui.

    Provides:
    - Level management (load, list, curriculum)
    - Command execution (forward, turn, interact, etc.)
    - RLDS episode recording
    - State serialization for WebSocket broadcast
    - Action prediction from trained models
    """

    def __init__(self, config: Optional[HomeScanConfig] = None):
        self.config = config or HomeScanConfig()
        self.state = HomeScanState()

        # Simulator components
        self.world: Optional[HomeWorld] = None
        self.handler: Optional[HomeHandler] = None
        self.logger: Optional[HomeRLDSLogger] = None
        self.predictor: Optional[Home3DNavigationPredictor] = None

        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if home simulator is available."""
        return HAS_HOME_SIMULATOR

    async def initialize(self, level_name: Optional[str] = None) -> bool:
        """
        Initialize the HomeScan simulator.

        Args:
            level_name: Level to load (default: config.default_level)

        Returns:
            True if initialized successfully
        """
        if not HAS_HOME_SIMULATOR:
            print("HomeScan: Simulator not available")
            return False

        try:
            level = level_name or self.config.default_level

            # Load the world
            self.world = get_home_level(level)
            if not self.world:
                print(f"HomeScan: Level not found: {level}")
                return False

            # Note: We don't use HomeHandler directly as it requires ActorRuntime
            # Instead, we call world methods directly for navigation
            self.handler = None  # Not using handler

            # Create RLDS logger
            self.config.rlds_output_path.mkdir(parents=True, exist_ok=True)
            self.logger = HomeRLDSLogger(self.config.rlds_output_path)

            # Try to load predictor (auto-load if model exists)
            if self.config.auto_load_model and self.config.model_path:
                model_path = Path(self.config.model_path)
                if model_path.exists():
                    try:
                        self.predictor = Home3DNavigationPredictor()
                        self.predictor.load(model_path)
                        self.state.predictor_ready = self.predictor.is_ready
                        if self.state.predictor_ready:
                            print(f"HomeScan: Loaded trained model from {model_path}")
                    except Exception as e:
                        print(f"HomeScan: Failed to load model: {e}")
                        self.predictor = None
                else:
                    print(f"HomeScan: No trained model found at {model_path}")

            # Update state
            self.state.active = True
            self.state.level_name = level
            self._update_state_from_world()

            # Auto-start recording
            if self.config.auto_record:
                self.start_recording()

            self._initialized = True
            print(f"HomeScan: Initialized with level '{level}'")
            return True

        except Exception as e:
            print(f"HomeScan: Initialization error: {e}")
            return False

    def _update_state_from_world(self):
        """Update state from world model."""
        if not self.world:
            return

        robot = self.world.robot
        self.state.robot_position = {
            "x": robot.position.x,
            "y": robot.position.y,
            "z": robot.position.z,
        }
        self.state.robot_rotation = {
            "pitch": robot.rotation.pitch,
            "yaw": robot.rotation.yaw,
            "roll": robot.rotation.roll,
        }
        current_room = self.world.get_room_at(robot.position)
        self.state.current_room = current_room.room_type.value if current_room else "unknown"
        # get_visible_objects returns list of dicts directly
        self.state.visible_objects = self.world.get_visible_objects()[:10]  # Limit to 10
        self.state.inventory = list(robot.inventory)
        self.state.battery = robot.battery
        self.state.moves = robot.moves
        self.state.goal_reached = self.world.level_complete
        self.state.goal_distance = (
            robot.position.distance_to(self.world.goal_position)
            if self.world.goal_position else 0.0
        )

    async def execute_command(self, command: str, raw_input: str = "") -> Dict[str, Any]:
        """
        Execute a navigation command.

        Args:
            command: Command to execute (forward, backward, turn_left, etc.)
            raw_input: Original user input

        Returns:
            Response dict with result, message, and updated state
        """
        if not self._initialized or not self.world:
            return {
                "success": False,
                "message": "HomeScan not initialized",
                "state": asdict(self.state),
            }

        try:
            # Execute command directly on world
            command_map = {
                "forward": self.world.move_forward,
                "backward": self.world.move_backward,
                "strafe_left": self.world.strafe_left,
                "strafe_right": self.world.strafe_right,
                "turn_left": self.world.turn_left,
                "turn_right": self.world.turn_right,
                "look_up": self.world.look_up,
                "look_down": self.world.look_down,
                "interact": self.world.interact,
            }

            handler_func = command_map.get(command.lower())
            if not handler_func:
                return {
                    "success": False,
                    "message": f"Unknown command: {command}",
                    "reward": -0.1,
                    "done": False,
                    "state": asdict(self.state),
                }

            # Execute the command
            result = handler_func()
            success = result.success
            message = result.message

            # Calculate reward
            reward = 0.1 if success else -0.05
            done = self.world.level_complete
            if done:
                reward = 1.0

            # Record step if logging
            if self.logger and self.state.recording:
                self.logger.log_step(
                    world=self.world,
                    action_command=command,
                    action_intent=command,
                    action_params={},
                    raw_input=raw_input or command,
                    success=success,
                    level_complete=done,
                )

            # Update state
            self._update_state_from_world()
            self.state.last_action = command
            self.state.last_reward = reward

            return {
                "success": success,
                "message": message,
                "reward": reward,
                "done": done,
                "state": asdict(self.state),
            }

        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "state": asdict(self.state),
            }

    async def load_level(self, level_name: str) -> Dict[str, Any]:
        """
        Load a new level.

        Args:
            level_name: Name of the level to load

        Returns:
            Response with success status and new state
        """
        # End current recording if active
        if self.state.recording:
            self.stop_recording()

        # Re-initialize with new level
        success = await self.initialize(level_name)

        return {
            "success": success,
            "level": level_name,
            "state": asdict(self.state),
        }

    def start_recording(self) -> str:
        """
        Start RLDS episode recording.

        Returns:
            Episode ID
        """
        if not self.logger:
            return ""

        if self.state.recording:
            self.stop_recording()

        episode_id = self.logger.start_episode(
            world=self.world,
            level_id=self.state.level_name,
            session_id=f"home_scan_{int(time.time())}",
        )
        self.state.recording = True
        self.state.episode_id = episode_id

        # Log initial state
        if self.world:
            self.logger.log_step(
                world=self.world,
                action_command="init",
                action_intent="init",
                action_params={},
                raw_input="",
                success=True,
            )

        return episode_id

    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and save episode.

        Returns:
            Path to saved episode, or None if not recording
        """
        if not self.logger or not self.state.recording or not self.world:
            return None

        path = self.logger.end_episode(
            world=self.world,
            success=self.state.goal_reached,
        )
        self.state.recording = False
        self.state.episode_id = None

        return path

    def predict_next_action(self) -> Optional[Dict[str, Any]]:
        """
        Use trained model to predict next action.

        Returns:
            Prediction dict with action and confidence, or None
        """
        if not self.predictor or not self.predictor.is_ready:
            return None

        if not self.world:
            return None

        # Build observation
        obs = self._build_observation_vector()

        # Get prediction - predict() returns list of probabilities
        probs = self.predictor.predict(obs)
        action_idx = probs.index(max(probs))

        return {
            "action": HOME_ACTIONS[action_idx],
            "action_idx": action_idx,
            "confidence": float(probs[action_idx]),
            "all_probs": {
                HOME_ACTIONS[i]: float(p)
                for i, p in enumerate(probs)
            },
        }

    def _build_observation_vector(self) -> List[float]:
        """Build 48-dim observation vector for predictor."""
        if not self.world:
            return [0.0] * 48

        robot = self.world.robot

        # Position (3)
        obs = [robot.position.x / 20.0, robot.position.y / 20.0, robot.position.z / 5.0]

        # Rotation (3, normalized)
        obs.extend([
            robot.rotation.pitch / 90.0,
            robot.rotation.yaw / 360.0,
            robot.rotation.roll / 90.0,
        ])

        # Room encoding (one-hot, 8 room types)
        current_room = self.world.get_room_at(robot.position)
        room_name = current_room.room_type.value if current_room else "unknown"
        room_vec = [0.0] * 8
        room_types = list(RoomType)
        for i, rt in enumerate(room_types[:8]):
            if rt.value in room_name.lower():
                room_vec[i] = 1.0
                break
        obs.extend(room_vec)

        # Visible objects summary (16: 2 per object type x 8 types)
        # get_visible_objects returns list of dicts with 'distance' key
        objects = self.world.get_visible_objects()[:8]
        for i in range(8):
            if i < len(objects):
                obj = objects[i]
                dist = obj.get("distance", 10.0)
                obs.extend([1.0, min(1.0, 1.0 / (dist + 0.1))])
            else:
                obs.extend([0.0, 0.0])

        # Inventory (4, one-hot for common items)
        inv_vec = [0.0] * 4
        for item in robot.inventory[:4]:
            if "key" in item.lower():
                inv_vec[0] = 1.0
            elif "food" in item.lower():
                inv_vec[1] = 1.0
            elif "tool" in item.lower():
                inv_vec[2] = 1.0
            else:
                inv_vec[3] = 1.0
        obs.extend(inv_vec)

        # State (6)
        goal_distance = (
            robot.position.distance_to(self.world.goal_position)
            if self.world.goal_position else 20.0
        )
        obs.extend([
            robot.battery,
            min(1.0, robot.moves / 100.0),
            1.0 if self.world.level_complete else 0.0,
            min(1.0, goal_distance / 20.0),
            0.0,  # Reserved
            0.0,  # Reserved
        ])

        # Ensure 48 dimensions
        while len(obs) < 48:
            obs.append(0.0)

        return obs[:48]

    def get_render(self) -> str:
        """
        Get ASCII render of the current world state.

        Returns:
            ASCII art representation
        """
        if not self.world:
            return "No world loaded"

        return self.world.render_top_down()

    def get_available_levels(self) -> List[str]:
        """Get list of available levels."""
        if not HAS_HOME_SIMULATOR:
            return []
        return list_home_levels()

    def get_curriculum(self) -> List[str]:
        """Get curriculum order for training."""
        if not HAS_HOME_SIMULATOR:
            return []
        return CURRICULUM_ORDER

    def get_state_dict(self) -> Dict[str, Any]:
        """Get serializable state dictionary."""
        return asdict(self.state)

    def reset(self) -> Dict[str, Any]:
        """
        Reset the current level.

        Returns:
            Response with new state
        """
        if not self.world:
            return {
                "success": False,
                "message": "No world loaded",
            }

        # Reset world
        self.world.reset()

        # Update state
        self._update_state_from_world()
        self.state.last_action = "reset"
        self.state.last_reward = 0.0

        # Start new recording if auto-record enabled
        if self.config.auto_record:
            self.start_recording()

        return {
            "success": True,
            "message": "Level reset",
            "state": asdict(self.state),
        }

    # =========================================================================
    # Training Methods
    # =========================================================================

    async def generate_training_data(
        self,
        num_episodes: int = 10,
        strategy: str = "mixed",
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate training data using exploration strategies.

        Args:
            num_episodes: Number of episodes to generate
            strategy: Exploration strategy (random, goal, wall, or mixed)
            max_steps: Max steps per episode

        Returns:
            Results dict with stats
        """
        if not HAS_HOME_SIMULATOR:
            return {"success": False, "error": "Simulator not available"}

        try:
            from brain_b.simulator.home_rlds_logger import HomeRLDSLogger

            # Setup logger
            output_path = self.config.rlds_output_path
            output_path.mkdir(parents=True, exist_ok=True)
            logger = HomeRLDSLogger(output_path)

            # Get levels
            levels = list_home_levels()
            strategies = ["random", "goal", "wall"] if strategy == "mixed" else [strategy]

            results = []
            total = num_episodes
            count = 0

            for ep_num in range(num_episodes):
                count += 1
                level = levels[ep_num % len(levels)]
                strat = strategies[ep_num % len(strategies)]

                # Generate episode
                result = await self._generate_single_episode(
                    level, strat, logger, max_steps
                )
                results.append(result)

            # Stats
            successful = [r for r in results if r.get("success")]
            goals_reached = [r for r in successful if r.get("goal_reached")]
            total_steps = sum(r.get("steps", 0) for r in successful)

            return {
                "success": True,
                "episodes": len(successful),
                "goals_reached": len(goals_reached),
                "total_steps": total_steps,
                "success_rate": len(goals_reached) / len(successful) if successful else 0,
                "output_path": str(output_path),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_single_episode(
        self,
        level_name: str,
        strategy: str,
        logger: "HomeRLDSLogger",
        max_steps: int,
    ) -> Dict[str, Any]:
        """Generate a single training episode."""
        import time
        import random

        # Load level
        world = get_home_level(level_name)
        if not world:
            return {"success": False, "error": f"Level not found: {level_name}"}

        # Start episode
        episode_id = logger.start_episode(
            world=world,
            level_id=level_name,
            session_id=f"gen_{strategy}_{int(time.time())}",
        )

        # Run exploration strategy
        actions = []
        action_vocab = ["forward", "backward", "strafe_left", "strafe_right",
                        "turn_left", "turn_right", "interact"]

        for step in range(max_steps):
            # Choose action based on strategy
            if strategy == "random":
                action = random.choice(action_vocab)
            elif strategy == "goal":
                action = self._goal_seek_action(world)
            elif strategy == "wall":
                action = self._wall_follow_action(world, actions)
            else:
                action = random.choice(action_vocab)

            actions.append(action)

            # Execute action
            self._execute_action_on_world(world, action)

            # Log step
            logger.log_step(
                world=world,
                action_command=action,
                action_intent=action,
                action_params={},
                raw_input=action,
                success=True,
                level_complete=world.level_complete,
            )

            if world.level_complete:
                break

        # End episode
        path = logger.end_episode(world=world, success=world.level_complete)

        return {
            "success": True,
            "episode_id": episode_id,
            "level": level_name,
            "strategy": strategy,
            "steps": len(actions),
            "goal_reached": world.level_complete,
            "path": str(path),
        }

    def _goal_seek_action(self, world: "HomeWorld") -> str:
        """Goal-seeking action selection."""
        import random

        if not world.goal_position:
            return random.choice(["forward", "turn_left", "turn_right"])

        robot = world.robot
        dx = world.goal_position.x - robot.position.x
        dy = world.goal_position.y - robot.position.y

        # Decide target direction
        if abs(dx) > abs(dy):
            target_yaw = 90 if dx > 0.5 else 270
        else:
            target_yaw = 0 if dy < -0.5 else 180

        yaw_diff = (target_yaw - robot.rotation.yaw + 180) % 360 - 180

        if abs(yaw_diff) > 30:
            return "turn_right" if yaw_diff > 0 else "turn_left"
        else:
            return "forward"

    def _wall_follow_action(self, world: "HomeWorld", history: List[str]) -> str:
        """Wall-following action selection."""
        import random

        # Check recent actions
        recent_turns = sum(1 for a in history[-3:] if "turn" in a)

        if recent_turns > 2:
            return "forward"
        elif len(history) > 0 and history[-1] == "forward":
            return "forward" if random.random() > 0.3 else "turn_right"
        else:
            return "turn_right" if random.random() > 0.5 else "forward"

    def _execute_action_on_world(self, world: "HomeWorld", action: str):
        """Execute an action directly on a world instance."""
        action_map = {
            "forward": world.move_forward,
            "backward": world.move_backward,
            "strafe_left": world.strafe_left,
            "strafe_right": world.strafe_right,
            "turn_left": world.turn_left,
            "turn_right": world.turn_right,
            "look_up": world.look_up,
            "look_down": world.look_down,
            "interact": world.interact,
        }

        method = action_map.get(action)
        if method:
            method()

    async def train_model(
        self,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Train the navigation model on RLDS episodes.

        Args:
            epochs: Training epochs
            batch_size: Training batch size

        Returns:
            Training results
        """
        if not HAS_HOME_SIMULATOR:
            return {"success": False, "error": "Simulator not available"}

        try:
            from brain_b.simulator.home_training import run_home3d_training

            # Paths - run_home3d_training saves to {output_dir}/home3d_nav_model.json
            episodes_dir = str(self.config.rlds_output_path)
            if self.config.model_path:
                output_dir = str(self.config.model_path.parent)
            else:
                output_dir = "../brain_b/brain_b_data/home_models"

            # The training function saves with fixed name
            actual_model_path = Path(output_dir) / "home3d_nav_model.json"

            # Run training (returns Home3DTrainingMetrics dataclass)
            metrics = run_home3d_training(
                episodes_dir=episodes_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
            )

            # Reload model from actual saved path
            if metrics.samples_seen > 0:
                # Create predictor if needed
                if not self.predictor:
                    self.predictor = Home3DNavigationPredictor()
                self.predictor.load(actual_model_path)
                self.state.predictor_ready = self.predictor.is_ready

            return {
                "success": metrics.samples_seen > 0,
                "accuracy": metrics.accuracy,
                "loss": metrics.loss,
                "samples_seen": metrics.samples_seen,
                "episodes_processed": metrics.episodes_processed,
                "epochs": epochs,
                "model_path": str(actual_model_path),
                "predictor_ready": self.state.predictor_ready,
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and stats."""
        # Count RLDS episodes (stored as subdirectories with metadata.json)
        episodes_path = self.config.rlds_output_path
        episode_count = 0
        total_steps = 0

        if episodes_path.exists():
            import json
            for ep_dir in episodes_path.iterdir():
                if not ep_dir.is_dir():
                    continue
                # Check if it's an episode directory
                metadata_file = ep_dir / "metadata.json"
                steps_file = ep_dir / "steps.jsonl"
                if metadata_file.exists() or steps_file.exists():
                    episode_count += 1
                    # Count steps
                    if steps_file.exists():
                        try:
                            with open(steps_file) as f:
                                total_steps += sum(1 for _ in f)
                        except:
                            pass

        # Model info
        model_exists = self.config.model_path and self.config.model_path.exists()

        return {
            "episodes": episode_count,
            "total_steps": total_steps,
            "model_exists": model_exists,
            "model_path": str(self.config.model_path) if self.config.model_path else None,
            "predictor_ready": self.state.predictor_ready,
        }

    def shutdown(self):
        """Clean shutdown of HomeScan."""
        if self.state.recording:
            self.stop_recording()

        self.state.active = False
        self._initialized = False


# ============================================================================
# WebSocket Message Handlers for Integration with trainer_ui
# ============================================================================

def create_home_scan_handlers(home_scan: HomeScanIntegration) -> Dict[str, callable]:
    """
    Create WebSocket message handlers for HomeScan integration.

    Args:
        home_scan: HomeScanIntegration instance

    Returns:
        Dict mapping message types to handler functions
    """

    async def handle_home_init(data: dict) -> dict:
        """Initialize HomeScan with optional level."""
        level = data.get("level")
        success = await home_scan.initialize(level)
        return {
            "type": "home_scan_state",
            "success": success,
            "state": home_scan.get_state_dict(),
            "levels": home_scan.get_available_levels(),
            "curriculum": home_scan.get_curriculum(),
        }

    async def handle_home_command(data: dict) -> dict:
        """Execute a navigation command."""
        command = data.get("command", "")
        raw_input = data.get("raw_input", command)
        result = await home_scan.execute_command(command, raw_input)
        return {
            "type": "home_scan_result",
            **result,
        }

    async def handle_home_load_level(data: dict) -> dict:
        """Load a new level."""
        level = data.get("level", "")
        result = await home_scan.load_level(level)
        return {
            "type": "home_scan_state",
            **result,
        }

    async def handle_home_reset(data: dict) -> dict:
        """Reset current level."""
        result = home_scan.reset()
        return {
            "type": "home_scan_state",
            **result,
        }

    async def handle_home_predict(data: dict) -> dict:
        """Get action prediction from trained model."""
        prediction = home_scan.predict_next_action()
        return {
            "type": "home_scan_prediction",
            "prediction": prediction,
        }

    async def handle_home_render(data: dict) -> dict:
        """Get ASCII render of world."""
        render = home_scan.get_render()
        return {
            "type": "home_scan_render",
            "render": render,
        }

    async def handle_home_recording(data: dict) -> dict:
        """Toggle RLDS recording."""
        action = data.get("action", "toggle")

        if action == "start" or (action == "toggle" and not home_scan.state.recording):
            episode_id = home_scan.start_recording()
            return {
                "type": "home_scan_recording",
                "recording": True,
                "episode_id": episode_id,
            }
        elif action == "stop" or (action == "toggle" and home_scan.state.recording):
            path = home_scan.stop_recording()
            return {
                "type": "home_scan_recording",
                "recording": False,
                "path": str(path) if path else None,
            }

        return {
            "type": "home_scan_recording",
            "recording": home_scan.state.recording,
        }

    async def handle_home_state(data: dict) -> dict:
        """Get current state."""
        return {
            "type": "home_scan_state",
            "state": home_scan.get_state_dict(),
        }

    async def handle_home_generate_data(data: dict) -> dict:
        """Generate training data."""
        num_episodes = data.get("num_episodes", 10)
        strategy = data.get("strategy", "mixed")
        max_steps = data.get("max_steps", 100)

        result = await home_scan.generate_training_data(
            num_episodes=num_episodes,
            strategy=strategy,
            max_steps=max_steps,
        )
        return {
            "type": "home_training_result",
            "action": "generate_data",
            **result,
        }

    async def handle_home_train(data: dict) -> dict:
        """Train the navigation model."""
        epochs = data.get("epochs", 20)
        learning_rate = data.get("learning_rate", 0.01)

        result = await home_scan.train_model(
            epochs=epochs,
            learning_rate=learning_rate,
        )
        return {
            "type": "home_training_result",
            "action": "train",
            **result,
        }

    async def handle_home_training_status(data: dict) -> dict:
        """Get training status."""
        status = home_scan.get_training_status()
        return {
            "type": "home_training_status",
            **status,
        }

    return {
        "home_init": handle_home_init,
        "home_command": handle_home_command,
        "home_load_level": handle_home_load_level,
        "home_reset": handle_home_reset,
        "home_predict": handle_home_predict,
        "home_render": handle_home_render,
        "home_recording": handle_home_recording,
        "home_state": handle_home_state,
        "home_generate_data": handle_home_generate_data,
        "home_train": handle_home_train,
        "home_training_status": handle_home_training_status,
    }


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_home_scan():
        """Test HomeScan integration."""
        print("Testing HomeScan Integration...")

        config = HomeScanConfig(
            rlds_output_path=Path("../brain_b/brain_b_data/test_home_scan"),
            auto_record=True,
        )

        home_scan = HomeScanIntegration(config)

        if not home_scan.is_available:
            print("HomeScan not available - simulator modules not found")
            return

        # Initialize
        success = await home_scan.initialize("simple_home")
        print(f"Initialized: {success}")
        print(f"State: {home_scan.get_state_dict()}")

        # Test commands
        commands = ["forward", "turn_right", "forward", "forward", "interact"]
        for cmd in commands:
            result = await home_scan.execute_command(cmd)
            print(f"Command '{cmd}': success={result['success']}, reward={result.get('reward', 0)}")

        # Render
        print("\nWorld render:")
        print(home_scan.get_render())

        # Stop recording
        path = home_scan.stop_recording()
        print(f"\nRecording saved to: {path}")

        # Show available levels
        print(f"\nAvailable levels: {home_scan.get_available_levels()}")
        print(f"Curriculum: {home_scan.get_curriculum()}")

        home_scan.shutdown()
        print("\nHomeScan test complete!")

    asyncio.run(test_home_scan())
