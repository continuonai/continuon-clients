#!/usr/bin/env python3
"""
High-Fidelity Training Episode Generator

Generates training episodes with realistic perception data that approaches
human-level perceptual complexity:

- Full RGB images with lighting variations
- Depth maps for 3D understanding
- Semantic segmentation for scene parsing
- Object detection with 3D bounding boxes
- LiDAR scans for navigation
- Audio events for environmental awareness
- Proprioception for body awareness

This creates a bridge between simulation and real-world deployment.

Usage:
    python brain_b/simulator/high_fidelity_trainer.py --episodes 50
    python brain_b/simulator/high_fidelity_trainer.py --curriculum 100 --save
"""

import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from .home_robot_games import (
        HomeRobotGameGenerator, GameState, ActionType,
        ObjectType, RoomType, TaskType
    )
    from .perception_system import (
        PerceptionEngine, PerceptionFrame, LightingCondition,
        RGBImage, DepthMap, SemanticSegmentation
    )
except ImportError:
    from home_robot_games import (
        HomeRobotGameGenerator, GameState, ActionType,
        ObjectType, RoomType, TaskType
    )
    from perception_system import (
        PerceptionEngine, PerceptionFrame, LightingCondition,
        RGBImage, DepthMap, SemanticSegmentation
    )


class EpisodeStep:
    """Single step in a training episode."""

    def __init__(
        self,
        step_id: int,
        perception: PerceptionFrame,
        action: ActionType,
        reward: float,
        done: bool,
        info: Dict,
    ):
        self.step_id = step_id
        self.perception = perception
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info

    def to_dict(self) -> Dict:
        """Serialize step for saving."""
        return {
            "step_id": self.step_id,
            "timestamp": self.perception.timestamp,
            "action": self.action.value if self.action else "none",
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "perception": {
                "lighting": self.perception.lighting.value,
                "num_detections": len(self.perception.object_detections),
                "has_audio": len(self.perception.audio_events) > 0,
            }
        }


class TrainingEpisode:
    """Complete training episode with high-fidelity perception."""

    def __init__(
        self,
        episode_id: str,
        game_state: GameState,
        perception_engine: PerceptionEngine,
    ):
        self.episode_id = episode_id
        self.initial_state = game_state
        self.perception_engine = perception_engine
        self.steps: List[EpisodeStep] = []
        self.total_reward = 0.0
        self.start_time = datetime.now()
        self.end_time = None

        # Current state
        self.robot_x = game_state.robot_x
        self.robot_y = game_state.robot_y
        self.robot_dir = game_state.robot_dir
        self.inventory = list(game_state.inventory)
        self.completed_objectives = []

    def get_state_dict(self) -> Dict:
        """Get current state as dictionary."""
        return {
            "robot": {
                "x": self.robot_x,
                "y": self.robot_y,
                "dir": self.robot_dir,
                "room": self.initial_state.robot_room,
                "inventory": [o.type.value for o in self.inventory],
            },
            "rooms": [
                {
                    "name": r.name,
                    "type": r.type.value,
                    "bounds": [r.x, r.y, r.width, r.height],
                    "objects": [
                        {"type": o.type.value, "x": o.x, "y": o.y}
                        for o in r.objects
                    ],
                }
                for r in self.initial_state.rooms
            ],
            "task": self.initial_state.task.to_dict(),
        }

    def step(self, action: ActionType) -> Tuple[PerceptionFrame, float, bool, Dict]:
        """Execute action and get perception."""
        # Update robot state based on action
        reward = self._execute_action(action)

        # Generate high-fidelity perception
        state_dict = self.get_state_dict()
        perception = self.perception_engine.generate_perception(state_dict)

        # Check if task is complete
        done = self._check_completion()

        # Episode info
        info = {
            "step": len(self.steps),
            "action": action.value,
            "robot_pos": (self.robot_x, self.robot_y),
            "objectives_done": len(self.completed_objectives),
            "objectives_total": len(self.initial_state.task.objectives),
        }

        # Record step
        step = EpisodeStep(
            step_id=len(self.steps),
            perception=perception,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )
        self.steps.append(step)
        self.total_reward += reward

        if done:
            self.end_time = datetime.now()

        return perception, reward, done, info

    def _execute_action(self, action: ActionType) -> float:
        """Execute action and return reward."""
        reward = -0.1  # Small step penalty

        # Direction vectors
        dir_vectors = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
        }

        # Direction rotations
        left_turn = {"north": "west", "west": "south", "south": "east", "east": "north"}
        right_turn = {"north": "east", "east": "south", "south": "west", "west": "north"}

        if action == ActionType.MOVE_FORWARD:
            dx, dy = dir_vectors[self.robot_dir]
            new_x = self.robot_x + dx
            new_y = self.robot_y + dy

            # Check bounds and collisions
            if self._is_valid_position(new_x, new_y):
                self.robot_x = new_x
                self.robot_y = new_y
                reward = 0.0  # Neutral for valid move
            else:
                reward = -0.5  # Penalty for collision

        elif action == ActionType.MOVE_BACKWARD:
            dx, dy = dir_vectors[self.robot_dir]
            new_x = self.robot_x - dx
            new_y = self.robot_y - dy

            if self._is_valid_position(new_x, new_y):
                self.robot_x = new_x
                self.robot_y = new_y

        elif action == ActionType.TURN_LEFT:
            self.robot_dir = left_turn[self.robot_dir]

        elif action == ActionType.TURN_RIGHT:
            self.robot_dir = right_turn[self.robot_dir]

        elif action == ActionType.PICK_UP:
            # Check for nearby objects
            reward += self._try_pickup()

        elif action == ActionType.PUT_DOWN:
            if self.inventory:
                self.inventory.pop()
                reward += 0.5

        elif action == ActionType.TOGGLE:
            reward += self._try_toggle()

        # Check objective progress
        reward += self._check_objective_progress()

        return reward

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid."""
        # Check room bounds
        for room in self.initial_state.rooms:
            if room.name == self.initial_state.robot_room:
                if x < room.x + 1 or x >= room.x + room.width - 1:
                    return False
                if y < room.y + 1 or y >= room.y + room.height - 1:
                    return False

                # Check object collisions
                for obj in room.objects:
                    if not obj.pickable and obj.x == x and obj.y == y:
                        return False

                return True
        return False

    def _try_pickup(self) -> float:
        """Try to pick up nearby object."""
        for room in self.initial_state.rooms:
            if room.name == self.initial_state.robot_room:
                for obj in room.objects:
                    if obj.pickable:
                        dist = abs(obj.x - self.robot_x) + abs(obj.y - self.robot_y)
                        if dist <= 1:
                            self.inventory.append(obj)
                            room.objects.remove(obj)
                            return 1.0  # Reward for pickup
        return -0.2  # Penalty for failed pickup

    def _try_toggle(self) -> float:
        """Try to toggle nearby switch/door."""
        for room in self.initial_state.rooms:
            if room.name == self.initial_state.robot_room:
                for obj in room.objects:
                    if obj.type in [ObjectType.LIGHT_SWITCH, ObjectType.DOOR]:
                        dist = abs(obj.x - self.robot_x) + abs(obj.y - self.robot_y)
                        if dist <= 1:
                            obj.state["on"] = not obj.state.get("on", False)
                            return 0.5
        return -0.1

    def _check_objective_progress(self) -> float:
        """Check if any objectives were completed."""
        reward = 0.0
        task = self.initial_state.task

        for obj in task.objectives:
            obj_key = json.dumps(obj, sort_keys=True)
            if obj_key in self.completed_objectives:
                continue

            completed = False

            if obj.get("type") == "reach":
                target_x, target_y = obj.get("x"), obj.get("y")
                if self.robot_x == target_x and self.robot_y == target_y:
                    completed = True

            elif obj.get("type") == "pick_up":
                target_type = obj.get("object")
                for item in self.inventory:
                    if item.type.value == target_type:
                        completed = True
                        break

            elif obj.get("type") == "toggle":
                target_type = obj.get("object")
                target_state = obj.get("target_state")
                for room in self.initial_state.rooms:
                    for o in room.objects:
                        if o.type.value == target_type:
                            if o.state.get("on") == target_state:
                                completed = True

            if completed:
                self.completed_objectives.append(obj_key)
                reward += task.reward / len(task.objectives)

        return reward

    def _check_completion(self) -> bool:
        """Check if episode is complete."""
        return len(self.completed_objectives) >= len(self.initial_state.task.objectives)

    def to_dict(self) -> Dict:
        """Serialize episode for saving."""
        return {
            "episode_id": self.episode_id,
            "task": self.initial_state.task.to_dict(),
            "tier": self.initial_state.task.tier,
            "total_reward": self.total_reward,
            "num_steps": len(self.steps),
            "success": self._check_completion(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "steps": [s.to_dict() for s in self.steps],
        }


class HighFidelityTrainer:
    """Trains robot with high-fidelity perception data."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        output_dir: str = "continuonbrain/rlds/episodes",
    ):
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.game_generator = HomeRobotGameGenerator()
        self.episodes_generated = 0
        self.total_steps = 0

        # Lighting variations for training diversity
        self.lighting_conditions = list(LightingCondition)

    def generate_episode(
        self,
        tier: int = 1,
        difficulty: int = 1,
        max_steps: int = 100,
        policy: str = "random",
    ) -> TrainingEpisode:
        """Generate a single training episode."""
        # Generate game
        game = self.game_generator.generate_game(tier, difficulty)

        # Random lighting condition
        lighting = random.choice(self.lighting_conditions)

        # Create perception engine
        engine = PerceptionEngine(
            resolution=self.resolution,
            enable_noise=True,
            lighting=lighting,
        )

        # Create episode
        episode_id = f"hf_episode_{self.episodes_generated:06d}"
        episode = TrainingEpisode(episode_id, game, engine)

        # Run episode with policy
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Select action
            if policy == "random":
                action = random.choice([
                    ActionType.MOVE_FORWARD,
                    ActionType.TURN_LEFT,
                    ActionType.TURN_RIGHT,
                    ActionType.PICK_UP,
                    ActionType.TOGGLE,
                ])
            elif policy == "smart_random":
                action = self._smart_random_policy(episode)
            else:
                action = ActionType.MOVE_FORWARD

            # Execute step
            perception, reward, done, info = episode.step(action)
            steps += 1

        self.episodes_generated += 1
        self.total_steps += steps

        return episode

    def _smart_random_policy(self, episode: TrainingEpisode) -> ActionType:
        """Slightly smarter random policy."""
        # Get current perception
        state = episode.get_state_dict()
        detections = episode.perception_engine.generate_perception(state).object_detections

        # If we see a graspable object nearby, try to pick it up
        for det in detections:
            if det.graspable and det.position_3d[0] < 2:
                return ActionType.PICK_UP

        # Otherwise random navigation with weighted probabilities
        nav_actions = [
            ActionType.MOVE_FORWARD,
            ActionType.MOVE_BACKWARD,
            ActionType.TURN_LEFT,
            ActionType.TURN_RIGHT,
            ActionType.PICK_UP,
            ActionType.TOGGLE,
        ]
        weights = [0.40, 0.05, 0.25, 0.25, 0.03, 0.02]
        return random.choices(nav_actions, weights=weights)[0]

    def generate_curriculum(
        self,
        total_episodes: int = 100,
        max_steps: int = 100,
        save: bool = True,
    ) -> List[TrainingEpisode]:
        """Generate progressive curriculum of episodes."""
        episodes = []

        # Progressive tier distribution
        tier_distribution = [
            (1, 0.30),  # 30% basic
            (2, 0.25),  # 25% object interaction
            (3, 0.20),  # 20% multi-room
            (4, 0.15),  # 15% task completion
            (5, 0.10),  # 10% complex
        ]

        print(f"\nGenerating {total_episodes} high-fidelity episodes...")
        print(f"Resolution: {self.resolution}")
        print(f"Output: {self.output_dir}\n")

        for tier, ratio in tier_distribution:
            num_episodes = int(total_episodes * ratio)

            for i in range(num_episodes):
                # Increasing difficulty within tier
                difficulty = 1 + (i * 2 // num_episodes)

                episode = self.generate_episode(
                    tier=tier,
                    difficulty=difficulty,
                    max_steps=max_steps,
                    policy="smart_random",
                )
                episodes.append(episode)

                # Progress indicator
                if len(episodes) % 10 == 0:
                    print(f"  Generated {len(episodes)}/{total_episodes} episodes "
                          f"(Tier {tier}, {episode.total_reward:.1f} reward)")

        # Save if requested
        if save:
            self.save_episodes(episodes)

        return episodes

    def save_episodes(self, episodes: List[TrainingEpisode]):
        """Save episodes as RLDS format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.output_dir / f"high_fidelity_batch_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "source": "high_fidelity_trainer",
            "timestamp": timestamp,
            "num_episodes": len(episodes),
            "total_steps": sum(len(e.steps) for e in episodes),
            "resolution": self.resolution,
            "tier_distribution": {},
        }

        for ep in episodes:
            tier = ep.initial_state.task.tier
            metadata["tier_distribution"][tier] = metadata["tier_distribution"].get(tier, 0) + 1

        with open(batch_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save each episode
        for episode in episodes:
            ep_dir = batch_dir / episode.episode_id
            ep_dir.mkdir(exist_ok=True)

            # Save episode data
            with open(ep_dir / "episode.json", 'w') as f:
                json.dump(episode.to_dict(), f, indent=2)

            # Save initial state
            with open(ep_dir / "initial_state.json", 'w') as f:
                json.dump(episode.get_state_dict(), f, indent=2)

            # Save perception data (first and last frames as samples)
            if episode.steps:
                # Save RGB images as raw numpy arrays
                first_rgb = episode.steps[0].perception.rgb_image
                if first_rgb:
                    np.save(ep_dir / "first_rgb.npy", first_rgb.data)

                # Save depth map
                first_depth = episode.steps[0].perception.depth_map
                if first_depth:
                    np.save(ep_dir / "first_depth.npy", first_depth.data)

                # Save semantic segmentation
                first_seg = episode.steps[0].perception.semantic_seg
                if first_seg:
                    np.save(ep_dir / "first_semantic.npy", first_seg.labels)

                # Save LiDAR
                first_lidar = episode.steps[0].perception.lidar_scan
                if first_lidar:
                    np.save(ep_dir / "first_lidar_ranges.npy", first_lidar.ranges)
                    np.save(ep_dir / "first_lidar_angles.npy", first_lidar.angles)

        print(f"\nSaved {len(episodes)} episodes to {batch_dir}")
        print(f"Total steps: {sum(len(e.steps) for e in episodes)}")

    def print_summary(self, episodes: List[TrainingEpisode]):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)

        total_reward = sum(e.total_reward for e in episodes)
        successes = sum(1 for e in episodes if e._check_completion())
        total_steps = sum(len(e.steps) for e in episodes)

        print(f"Episodes: {len(episodes)}")
        print(f"Total steps: {total_steps}")
        print(f"Success rate: {successes/len(episodes)*100:.1f}%")
        print(f"Average reward: {total_reward/len(episodes):.2f}")

        # Per-tier stats
        print("\nPer-Tier Statistics:")
        tier_stats = {}
        for ep in episodes:
            tier = ep.initial_state.task.tier
            if tier not in tier_stats:
                tier_stats[tier] = {"count": 0, "success": 0, "reward": 0}
            tier_stats[tier]["count"] += 1
            tier_stats[tier]["success"] += 1 if ep._check_completion() else 0
            tier_stats[tier]["reward"] += ep.total_reward

        for tier in sorted(tier_stats.keys()):
            stats = tier_stats[tier]
            print(f"  Tier {tier}: {stats['count']} episodes, "
                  f"{stats['success']/stats['count']*100:.0f}% success, "
                  f"{stats['reward']/stats['count']:.1f} avg reward")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="High-Fidelity Training Episode Generator")
    parser.add_argument("--episodes", type=int, default=10, help="Number of single episodes")
    parser.add_argument("--curriculum", type=int, default=0, help="Generate progressive curriculum")
    parser.add_argument("--tier", type=int, default=1, help="Tier for single episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--resolution", type=str, default="640x480", help="Image resolution WxH")
    parser.add_argument("--save", action="store_true", help="Save episodes to RLDS")
    parser.add_argument("--output", type=str, default="continuonbrain/rlds/episodes", help="Output directory")

    args = parser.parse_args()

    # Parse resolution
    w, h = map(int, args.resolution.split("x"))
    resolution = (w, h)

    # Create trainer
    trainer = HighFidelityTrainer(
        resolution=resolution,
        output_dir=args.output,
    )

    if args.curriculum > 0:
        # Generate progressive curriculum
        episodes = trainer.generate_curriculum(
            total_episodes=args.curriculum,
            max_steps=args.max_steps,
            save=args.save,
        )
        trainer.print_summary(episodes)

    else:
        # Generate single-tier episodes
        print(f"Generating {args.episodes} tier-{args.tier} episodes...")
        episodes = []
        for i in range(args.episodes):
            episode = trainer.generate_episode(
                tier=args.tier,
                difficulty=1 + i % 3,
                max_steps=args.max_steps,
            )
            episodes.append(episode)
            print(f"  Episode {i+1}: {len(episode.steps)} steps, "
                  f"{episode.total_reward:.1f} reward, "
                  f"{'SUCCESS' if episode._check_completion() else 'INCOMPLETE'}")

        if args.save:
            trainer.save_episodes(episodes)

        trainer.print_summary(episodes)


if __name__ == "__main__":
    main()
