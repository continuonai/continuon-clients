"""
Training Integration for House 3D Renderer

Connects the house_3d renderer to existing training pipelines:
- high_fidelity_trainer.py
- training_with_inference.py
- perception_system.py

Generates photorealistic training episodes with 3D rendered frames.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from .scene import HouseScene, SceneObject
from .renderer import HouseRenderer, RenderMode
from .camera import RobotCamera, OrthographicCamera
from .assets import HOUSE_TEMPLATES


@dataclass
class Visual3DFrame:
    """
    Single frame from 3D renderer for training.

    Contains RGB, depth, segmentation, and metadata.
    """
    timestamp: float
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32
    segmentation: np.ndarray  # (H, W, 3) uint8
    overhead: np.ndarray     # (S, S, 3) uint8

    robot_position: Tuple[float, float, float]
    robot_yaw: float
    robot_pitch: float

    objects_in_view: List[Dict] = field(default_factory=list)
    room_name: str = ""

    def to_dict(self) -> Dict:
        """Serialize for RLDS storage (without raw arrays)."""
        return {
            'timestamp': self.timestamp,
            'robot_position': list(self.robot_position),
            'robot_yaw': self.robot_yaw,
            'robot_pitch': self.robot_pitch,
            'objects_in_view': self.objects_in_view,
            'room_name': self.room_name,
            'frame_shape': {
                'rgb': list(self.rgb.shape),
                'depth': list(self.depth.shape),
                'segmentation': list(self.segmentation.shape),
                'overhead': list(self.overhead.shape),
            }
        }


@dataclass
class Visual3DEpisodeStep:
    """Single step in a visual training episode."""
    step_id: int
    frame: Visual3DFrame
    action: str
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


class Visual3DTrainingEnvironment:
    """
    3D training environment for generating photorealistic episodes.

    Integrates with existing game systems while providing
    high-fidelity visual output for training.
    """

    def __init__(
        self,
        scene: Optional[HouseScene] = None,
        template: str = 'studio_apartment',
        render_width: int = 640,
        render_height: int = 480,
        overhead_size: int = 256,
    ):
        """
        Initialize 3D training environment.

        Args:
            scene: Pre-built HouseScene, or None to use template
            template: Template name if scene is None
            render_width, render_height: POV render dimensions
            overhead_size: Overhead map size
        """
        if scene is not None:
            self.scene = scene
        else:
            self.scene = HouseScene.from_template(template)

        self.renderer = HouseRenderer(render_width, render_height)
        self.overhead_size = overhead_size

        # Robot state
        self.robot_position = [3.0, 0.0, 3.0]  # x, y (height), z
        self.robot_yaw = 0.0
        self.robot_pitch = 0.0

        # Episode tracking
        self.steps: List[Visual3DEpisodeStep] = []
        self.total_reward = 0.0
        self.episode_id = ""

        # Movement parameters
        self.move_speed = 0.5  # meters per step
        self.turn_speed = 15.0  # degrees per step

        # Floor bounds for collision
        bounds = self.scene.get_floor_bounds()
        self.floor_min = bounds[0]
        self.floor_max = bounds[1]

    def reset(self, start_position: Optional[Tuple[float, float, float]] = None) -> Visual3DFrame:
        """
        Reset environment for new episode.

        Args:
            start_position: Optional starting position

        Returns:
            Initial frame
        """
        self.steps = []
        self.total_reward = 0.0
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if start_position:
            self.robot_position = list(start_position)
        else:
            # Random position within bounds
            self.robot_position = [
                np.random.uniform(self.floor_min[0] + 1, self.floor_max[0] - 1),
                0.0,
                np.random.uniform(self.floor_min[1] + 1, self.floor_max[1] - 1),
            ]
        self.robot_yaw = np.random.uniform(0, 360)
        self.robot_pitch = 0.0

        return self._render_frame()

    def step(self, action: str) -> Tuple[Visual3DFrame, float, bool, Dict]:
        """
        Execute action and return new frame.

        Args:
            action: One of 'forward', 'backward', 'left', 'right',
                   'turn_left', 'turn_right', 'look_up', 'look_down'

        Returns:
            (frame, reward, done, info)
        """
        reward = -0.01  # Small step penalty

        # Execute action
        if action == 'forward':
            reward += self._move(1)
        elif action == 'backward':
            reward += self._move(-1)
        elif action == 'left':
            reward += self._strafe(-1)
        elif action == 'right':
            reward += self._strafe(1)
        elif action == 'turn_left':
            self.robot_yaw = (self.robot_yaw - self.turn_speed) % 360
        elif action == 'turn_right':
            self.robot_yaw = (self.robot_yaw + self.turn_speed) % 360
        elif action == 'look_up':
            self.robot_pitch = min(45, self.robot_pitch + self.turn_speed)
        elif action == 'look_down':
            self.robot_pitch = max(-45, self.robot_pitch - self.turn_speed)

        # Render new frame
        frame = self._render_frame()

        # Check done condition
        done = len(self.steps) >= 1000  # Max steps

        info = {
            'step': len(self.steps),
            'action': action,
            'position': tuple(self.robot_position),
            'yaw': self.robot_yaw,
        }

        # Record step
        step = Visual3DEpisodeStep(
            step_id=len(self.steps),
            frame=frame,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )
        self.steps.append(step)
        self.total_reward += reward

        return frame, reward, done, info

    def _move(self, direction: float) -> float:
        """Move forward/backward."""
        import math
        yaw_rad = math.radians(self.robot_yaw)

        dx = math.sin(yaw_rad) * self.move_speed * direction
        dz = math.cos(yaw_rad) * self.move_speed * direction

        new_x = self.robot_position[0] + dx
        new_z = self.robot_position[2] + dz

        # Check bounds
        if self._is_valid_position(new_x, new_z):
            self.robot_position[0] = new_x
            self.robot_position[2] = new_z
            return 0.0
        else:
            return -0.5  # Collision penalty

    def _strafe(self, direction: float) -> float:
        """Move left/right."""
        import math
        yaw_rad = math.radians(self.robot_yaw + 90)

        dx = math.sin(yaw_rad) * self.move_speed * direction
        dz = math.cos(yaw_rad) * self.move_speed * direction

        new_x = self.robot_position[0] + dx
        new_z = self.robot_position[2] + dz

        if self._is_valid_position(new_x, new_z):
            self.robot_position[0] = new_x
            self.robot_position[2] = new_z
            return 0.0
        else:
            return -0.5

    def _is_valid_position(self, x: float, z: float) -> bool:
        """Check if position is valid (within bounds, no collision)."""
        # Floor bounds
        margin = 0.5
        if x < self.floor_min[0] + margin or x > self.floor_max[0] - margin:
            return False
        if z < self.floor_min[1] + margin or z > self.floor_max[1] - margin:
            return False

        # Check obstacle collisions
        for obj in self.scene.get_obstacles():
            bbox = obj.get_bounding_box()
            if (bbox.min_point.x - 0.3 <= x <= bbox.max_point.x + 0.3 and
                bbox.min_point.z - 0.3 <= z <= bbox.max_point.z + 0.3):
                return False

        return True

    def _render_frame(self) -> Visual3DFrame:
        """Render current view."""
        pos = tuple(self.robot_position)

        # Render all views
        result = self.renderer.render_split_view(
            self.scene,
            pos,
            self.robot_yaw,
            self.robot_pitch,
            overhead_size=self.overhead_size,
        )

        # Get segmentation
        camera = RobotCamera(eye_height=0.8, aspect=self.renderer.width / self.renderer.height)
        camera.update_from_robot(pos, self.robot_yaw, self.robot_pitch)
        segmentation = self.renderer.render(self.scene, camera, RenderMode.SEGMENTATION)

        # Find objects in view
        objects_in_view = self._get_objects_in_view()

        return Visual3DFrame(
            timestamp=datetime.now().timestamp(),
            rgb=result['pov'],
            depth=result['depth'],
            segmentation=segmentation,
            overhead=result['overhead'],
            robot_position=pos,
            robot_yaw=self.robot_yaw,
            robot_pitch=self.robot_pitch,
            objects_in_view=objects_in_view,
            room_name=self._get_current_room(),
        )

    def _get_objects_in_view(self) -> List[Dict]:
        """Get list of objects visible from current position."""
        import math
        objects = []

        for obj in self.scene.objects:
            # Calculate relative position
            dx = obj.transform.position[0] - self.robot_position[0]
            dz = obj.transform.position[2] - self.robot_position[2]
            distance = math.sqrt(dx * dx + dz * dz)

            if distance > 10:
                continue

            # Calculate angle to object
            angle_to_obj = math.degrees(math.atan2(dx, dz))
            relative_angle = (angle_to_obj - self.robot_yaw + 180) % 360 - 180

            # Check if in field of view
            if abs(relative_angle) < 45:
                objects.append({
                    'name': obj.name,
                    'distance': round(distance, 2),
                    'angle': round(relative_angle, 1),
                    'tags': obj.tags,
                })

        return objects

    def _get_current_room(self) -> str:
        """Get name of room robot is currently in."""
        for room in self.scene.rooms:
            bounds = room.get_navigable_bounds()
            if (bounds.min_point.x <= self.robot_position[0] <= bounds.max_point.x and
                bounds.min_point.z <= self.robot_position[2] <= bounds.max_point.z):
                return room.name
        return "unknown"

    def get_episode_data(self) -> Dict:
        """Get complete episode data for saving."""
        return {
            'episode_id': self.episode_id,
            'template': self.scene.name,
            'total_steps': len(self.steps),
            'total_reward': self.total_reward,
            'steps': [
                {
                    'step_id': s.step_id,
                    'action': s.action,
                    'reward': s.reward,
                    'done': s.done,
                    'frame': s.frame.to_dict(),
                    'info': s.info,
                }
                for s in self.steps
            ]
        }

    def save_episode(self, output_dir: str):
        """Save episode with frames."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        episode_dir = output_path / f"visual_episode_{self.episode_id}"
        episode_dir.mkdir(exist_ok=True)

        # Save metadata
        with open(episode_dir / "episode.json", 'w') as f:
            json.dump(self.get_episode_data(), f, indent=2)

        # Save frames as numpy arrays
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for step in self.steps:
            np.save(frames_dir / f"rgb_{step.step_id:04d}.npy", step.frame.rgb)
            np.save(frames_dir / f"depth_{step.step_id:04d}.npy", step.frame.depth)

        print(f"Episode saved to {episode_dir}")
        return str(episode_dir)


def create_visual_training_env(
    template: str = 'studio_apartment',
    width: int = 640,
    height: int = 480,
) -> Visual3DTrainingEnvironment:
    """Factory function for creating training environments."""
    return Visual3DTrainingEnvironment(
        template=template,
        render_width=width,
        render_height=height,
    )


def run_visual_training_episode(
    env: Visual3DTrainingEnvironment,
    policy: str = 'random',
    max_steps: int = 100,
) -> Dict:
    """
    Run a single training episode.

    Args:
        env: Training environment
        policy: 'random' or 'explore'
        max_steps: Maximum steps per episode

    Returns:
        Episode statistics
    """
    actions = ['forward', 'turn_left', 'turn_right', 'left', 'right']

    frame = env.reset()
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        # Select action
        if policy == 'random':
            action = np.random.choice(actions)
        elif policy == 'explore':
            # Prefer forward, occasionally turn
            weights = [0.5, 0.15, 0.15, 0.1, 0.1]
            action = np.random.choice(actions, p=weights)
        else:
            action = np.random.choice(actions)

        frame, reward, done, info = env.step(action)
        step_count += 1

    return {
        'episode_id': env.episode_id,
        'steps': step_count,
        'total_reward': env.total_reward,
        'final_position': tuple(env.robot_position),
    }


def generate_visual_training_batch(
    num_episodes: int = 10,
    template: str = 'studio_apartment',
    output_dir: str = 'brain_b_data/visual_episodes',
    save_frames: bool = True,
) -> List[Dict]:
    """
    Generate a batch of visual training episodes.

    Args:
        num_episodes: Number of episodes to generate
        template: House template to use
        output_dir: Where to save episodes
        save_frames: Whether to save frame data

    Returns:
        List of episode statistics
    """
    env = create_visual_training_env(template)
    results = []

    for i in range(num_episodes):
        print(f"Generating episode {i+1}/{num_episodes}...")
        stats = run_visual_training_episode(env, policy='explore', max_steps=100)

        if save_frames:
            env.save_episode(output_dir)

        results.append(stats)
        print(f"  Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

    return results


# Integration with perception_system.py
class House3DPerceptionAdapter:
    """
    Adapter to convert house_3d frames to perception_system format.

    Makes house_3d compatible with existing high_fidelity_trainer.py.
    """

    def __init__(self, env: Visual3DTrainingEnvironment):
        self.env = env

    def generate_perception_frame(self) -> Dict:
        """Generate frame in perception_system.py format."""
        frame = self.env._render_frame()

        return {
            'timestamp': frame.timestamp,
            'lighting': 'daylight',  # Could be varied
            'rgb': {
                'data': frame.rgb,
                'width': frame.rgb.shape[1],
                'height': frame.rgb.shape[0],
            },
            'depth': {
                'data': frame.depth,
                'min_depth': 0.1,
                'max_depth': 10.0,
            },
            'segmentation': {
                'data': frame.segmentation,
                'classes': ['floor', 'wall', 'furniture', 'ceiling'],
            },
            'object_detections': [
                {
                    'class': obj.get('tags', ['object'])[0] if obj.get('tags') else 'object',
                    'distance': obj['distance'],
                    'angle': obj['angle'],
                    'name': obj['name'],
                }
                for obj in frame.objects_in_view
            ],
            'robot_state': {
                'position': frame.robot_position,
                'yaw': frame.robot_yaw,
                'pitch': frame.robot_pitch,
                'room': frame.room_name,
            }
        }


if __name__ == '__main__':
    # Demo: generate training episodes
    print("Generating visual training batch...")
    results = generate_visual_training_batch(
        num_episodes=3,
        template='studio_apartment',
        save_frames=False,  # Set True to save frame data
    )

    print(f"\nGenerated {len(results)} episodes")
    total_steps = sum(r['steps'] for r in results)
    total_reward = sum(r['total_reward'] for r in results)
    print(f"Total steps: {total_steps}")
    print(f"Average reward: {total_reward / len(results):.2f}")
