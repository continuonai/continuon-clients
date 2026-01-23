#!/usr/bin/env python3
"""
HomeScan Simulator Utilities

Provides RLDS episode recording and environment management for the
HomeScan 3D robot training simulator.

Features:
- Record simulator sessions as RLDS episodes
- Environment state management
- Collision detection data
- Training data export

Usage:
    from simulator_utils import SimulatorRecorder, SimulatorEnvironment

    recorder = SimulatorRecorder(output_path)
    recorder.start_episode("sim_001")
    recorder.record_step(action, observation, reward)
    recorder.end_episode()
"""

import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any


@dataclass
class SimulatorObservation:
    """Observation from the simulator environment."""

    robot_position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    robot_rotation: float = 0.0
    obstacle_count: int = 0
    obstacle_positions: List[List[float]] = field(default_factory=list)
    collision_detected: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SimulatorAction:
    """Action taken in the simulator."""

    type: str = "noop"  # move, rotate, spawn_asset, spawn_obstacle, reset
    direction: Optional[str] = None  # forward, backward, left, right
    speed: float = 0.0
    angle: float = 0.0
    asset_type: Optional[int] = None
    position: Optional[List[float]] = None


@dataclass
class SimulatorStep:
    """Single step in a simulator episode."""

    step_idx: int
    timestamp: float
    action: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorEpisode:
    """Complete simulator episode for RLDS export."""

    episode_id: str
    session_id: str
    environment_id: str = "homescan_simulator"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    steps: List[SimulatorStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)


class SimulatorRecorder:
    """Records simulator sessions as RLDS episodes."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.current_episode: Optional[SimulatorEpisode] = None
        self.episodes: List[str] = []

    def start_episode(self, session_id: str, metadata: Optional[Dict] = None) -> str:
        """Start recording a new episode."""
        episode_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}"

        self.current_episode = SimulatorEpisode(
            episode_id=episode_id,
            session_id=session_id,
            metadata=metadata or {},
        )

        print(f"[SimRecorder] Started episode: {episode_id}")
        return episode_id

    def record_step(
        self,
        action: Dict[str, Any],
        observation: Dict[str, Any],
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict] = None,
    ):
        """Record a single step in the current episode."""
        if not self.current_episode:
            return

        step = SimulatorStep(
            step_idx=len(self.current_episode.steps),
            timestamp=time.time(),
            action=action,
            observation=observation,
            reward=reward,
            done=done,
            info=info or {},
        )

        self.current_episode.steps.append(step)

    def end_episode(self) -> Optional[Path]:
        """End the current episode and save to disk."""
        if not self.current_episode:
            return None

        self.current_episode.end_time = time.time()

        # Create episode directory
        episode_dir = self.output_path / self.current_episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "episode_id": self.current_episode.episode_id,
            "session_id": self.current_episode.session_id,
            "environment_id": self.current_episode.environment_id,
            "num_steps": len(self.current_episode.steps),
            "duration_s": self.current_episode.duration_s,
            "total_reward": self.current_episode.total_reward,
            "start_time": datetime.fromtimestamp(self.current_episode.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.current_episode.end_time).isoformat(),
            "created": datetime.now().isoformat(),
            **self.current_episode.metadata,
        }

        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save steps in RLDS format
        steps_dir = episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        with open(steps_dir / "000000.jsonl", "w") as f:
            for step in self.current_episode.steps:
                f.write(json.dumps(asdict(step)) + "\n")

        print(f"[SimRecorder] Saved episode: {self.current_episode.episode_id}")
        print(f"  - Steps: {len(self.current_episode.steps)}")
        print(f"  - Duration: {self.current_episode.duration_s:.1f}s")
        print(f"  - Total Reward: {self.current_episode.total_reward:.2f}")
        print(f"  - Path: {episode_dir}")

        self.episodes.append(self.current_episode.episode_id)
        episode_path = episode_dir
        self.current_episode = None

        return episode_path

    def is_recording(self) -> bool:
        """Check if currently recording an episode."""
        return self.current_episode is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        current_steps = len(self.current_episode.steps) if self.current_episode else 0
        current_reward = self.current_episode.total_reward if self.current_episode else 0

        return {
            "recording": self.is_recording(),
            "current_episode": self.current_episode.episode_id if self.current_episode else None,
            "current_steps": current_steps,
            "current_reward": current_reward,
            "total_episodes": len(self.episodes),
            "episodes": self.episodes[-10:],  # Last 10 episodes
        }


class SimulatorEnvironment:
    """Manages simulator environment state."""

    def __init__(self):
        self.obstacles: List[Dict[str, Any]] = []
        self.robot_position: Dict[str, float] = {"x": 0, "y": 0, "z": 0}
        self.robot_rotation: float = 0.0
        self.step_count: int = 0

    def reset(self):
        """Reset environment to initial state."""
        self.obstacles = []
        self.robot_position = {"x": 0, "y": 0, "z": 0}
        self.robot_rotation = 0.0
        self.step_count = 0

    def add_obstacle(self, position: List[float], obstacle_type: str = "cylinder"):
        """Add an obstacle to the environment."""
        self.obstacles.append({
            "id": f"obs_{len(self.obstacles)}",
            "type": obstacle_type,
            "position": position,
            "created_at": time.time(),
        })

    def update_robot(self, position: Dict[str, float], rotation: float):
        """Update robot state."""
        self.robot_position = position
        self.robot_rotation = rotation
        self.step_count += 1

    def check_collision(self, collision_radius: float = 1.0) -> bool:
        """Check if robot collides with any obstacle."""
        rx, rz = self.robot_position["x"], self.robot_position["z"]

        for obs in self.obstacles:
            ox, oz = obs["position"][0], obs["position"][2]
            distance = ((rx - ox) ** 2 + (rz - oz) ** 2) ** 0.5
            if distance < collision_radius:
                return True

        return False

    def get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        return {
            "robot_position": self.robot_position,
            "robot_rotation": self.robot_rotation,
            "obstacle_count": len(self.obstacles),
            "obstacle_positions": [obs["position"] for obs in self.obstacles],
            "collision_detected": self.check_collision(),
            "step_count": self.step_count,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get full environment state."""
        return {
            "robot": {
                "position": self.robot_position,
                "rotation": self.robot_rotation,
            },
            "obstacles": self.obstacles,
            "step_count": self.step_count,
        }


class SimulatorSession:
    """Manages a complete simulator session with recording."""

    def __init__(self, rlds_path: Path):
        self.recorder = SimulatorRecorder(rlds_path)
        self.environment = SimulatorEnvironment()
        self.session_id: Optional[str] = None
        self.active = False

    def start(self, session_id: str) -> str:
        """Start a new simulator session."""
        self.session_id = session_id
        self.environment.reset()
        self.active = True
        return session_id

    def start_recording(self) -> str:
        """Start recording the current session."""
        if not self.session_id:
            self.session_id = f"session_{int(time.time())}"
        return self.recorder.start_episode(self.session_id)

    def stop_recording(self) -> Optional[Path]:
        """Stop recording and save episode."""
        return self.recorder.end_episode()

    def process_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a simulator step from the client."""
        action = data.get("action", {})
        observation = data.get("observation", {})
        reward = data.get("reward", 0.0)
        done = data.get("done", False)

        # Update environment state
        if "robot_position" in observation:
            self.environment.update_robot(
                observation["robot_position"],
                observation.get("robot_rotation", 0.0)
            )

        # Check for obstacle spawns
        if action.get("type") == "spawn_obstacle":
            pos = action.get("position", [0, 0, 0])
            self.environment.add_obstacle(pos)

        # Record step
        self.recorder.record_step(action, observation, reward, done)

        # Return updated state
        return {
            "step_recorded": True,
            "step_idx": data.get("step_idx", 0),
            "collision": self.environment.check_collision(),
            "stats": self.recorder.get_stats(),
        }

    def reset(self):
        """Reset the session."""
        self.environment.reset()
        if self.recorder.is_recording():
            self.recorder.record_step(
                {"type": "reset"},
                self.environment.get_observation(),
                0.0,
                True,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        return {
            "session_id": self.session_id,
            "active": self.active,
            "recording": self.recorder.is_recording(),
            "environment": self.environment.get_state(),
            "stats": self.recorder.get_stats(),
        }


# Singleton session manager for the trainer server
_simulator_session: Optional[SimulatorSession] = None


def get_simulator_session(rlds_path: Optional[Path] = None) -> SimulatorSession:
    """Get or create the simulator session singleton."""
    global _simulator_session

    if _simulator_session is None:
        if rlds_path is None:
            rlds_path = Path(__file__).parent.parent / "continuonbrain" / "rlds" / "episodes"
        _simulator_session = SimulatorSession(rlds_path)

    return _simulator_session


if __name__ == "__main__":
    # Test the simulator utilities
    print("Testing SimulatorRecorder...")

    recorder = SimulatorRecorder(Path("/tmp/test_sim_episodes"))

    # Start episode
    episode_id = recorder.start_episode("test_session")
    print(f"Started episode: {episode_id}")

    # Record some steps
    for i in range(5):
        recorder.record_step(
            action={"type": "move", "direction": "forward", "speed": 0.5},
            observation={"robot_position": {"x": i * 0.5, "y": 0, "z": 0}, "robot_rotation": 0},
            reward=0.1,
        )

    # End episode
    path = recorder.end_episode()
    print(f"Episode saved to: {path}")

    # Test environment
    print("\nTesting SimulatorEnvironment...")
    env = SimulatorEnvironment()
    env.add_obstacle([5, 0, 5])
    env.update_robot({"x": 1, "y": 0, "z": 1}, 0.5)
    print(f"Observation: {env.get_observation()}")

    print("\nAll tests passed!")
