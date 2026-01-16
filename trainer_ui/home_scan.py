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
    model_path: Optional[Path] = None
    default_level: str = "empty_room"
    auto_record: bool = True  # Auto-start RLDS recording
    render_mode: str = "ascii"  # ascii, json, or visual


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

            # Try to load predictor
            if self.config.model_path and self.config.model_path.exists():
                self.predictor = Home3DNavigationPredictor()
                self.predictor.load(self.config.model_path)
                self.state.predictor_ready = self.predictor.is_ready

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

        # Get prediction
        action_idx, probs = self.predictor.predict(obs)

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
        room_name = self.world.get_current_room_name()
        room_vec = [0.0] * 8
        room_types = list(RoomType)
        for i, rt in enumerate(room_types[:8]):
            if rt.value in room_name.lower():
                room_vec[i] = 1.0
                break
        obs.extend(room_vec)

        # Visible objects summary (16: 2 per object type x 8 types)
        objects = self.world.get_visible_objects()[:8]
        for i in range(8):
            if i < len(objects):
                obj, dist = objects[i]
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
        obs.extend([
            robot.battery,
            min(1.0, robot.moves / 100.0),
            1.0 if self.world.is_goal_reached() else 0.0,
            min(1.0, self.world.get_goal_distance() / 20.0),
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

    return {
        "home_init": handle_home_init,
        "home_command": handle_home_command,
        "home_load_level": handle_home_load_level,
        "home_reset": handle_home_reset,
        "home_predict": handle_home_predict,
        "home_render": handle_home_render,
        "home_recording": handle_home_recording,
        "home_state": handle_home_state,
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
