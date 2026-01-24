#!/usr/bin/env python3
"""
Expert Perception Training System

Generates high-quality perception training data using expert policies
that make correct decisions based on visual/sensor input.

Key insight: Random policies produce biased, low-quality training data.
Expert policies demonstrate optimal behavior for each perceptual situation.

Components:
1. ExpertPerceptionPolicy - Makes optimal decisions based on perception
2. PerceptionDataGenerator - Generates diverse training scenarios
3. PerceptionTrainer - Trains perception model on expert data

Usage:
    python brain_b/trainer/expert_perception.py --generate 5000
    python brain_b/trainer/expert_perception.py --train --epochs 100
    python brain_b/trainer/expert_perception.py --evaluate
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.perception_system import (
    PerceptionEngine, PerceptionFrame, LightingCondition,
    ObjectDetection, MaterialType
)
from simulator.expert_navigation import NavigationState, ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION


@dataclass
class PerceptionScenario:
    """A perception scenario with expected expert action."""
    name: str
    description: str

    # Perception features
    rgb_brightness: float  # 0-1
    depth_mean: float  # meters
    depth_min: float  # closest obstacle
    obstacle_ahead: bool
    floor_visible: float  # ratio 0-1
    wall_visible: float  # ratio 0-1

    # Object detection
    num_objects: int
    nearest_object_dist: float
    graspable_nearby: bool
    target_visible: bool

    # LiDAR summary (8 sectors)
    lidar_front: float
    lidar_left: float
    lidar_right: float
    lidar_back: float

    # Navigation context
    goal_direction: float  # -1 to 1
    goal_distance: float  # 0-1 normalized

    # Expert action
    expert_action: str
    confidence: float


class ExpertPerceptionPolicy:
    """
    Expert policy that makes optimal decisions based on perception.

    Decision hierarchy:
    1. Safety: Avoid collisions
    2. Task: Complete objectives (reach goal, pick up objects)
    3. Exploration: Efficiently explore unknown areas
    """

    # Thresholds
    COLLISION_DIST = 0.5  # meters
    CAUTION_DIST = 1.5    # meters
    GRASP_DIST = 1.0      # meters

    def __init__(self, task_type: str = "navigate"):
        """
        Args:
            task_type: "navigate", "fetch", "explore", "patrol"
        """
        self.task_type = task_type

    def decide(self, scenario: PerceptionScenario) -> Tuple[str, float]:
        """
        Make expert decision based on perception.

        Returns:
            (action, confidence)
        """
        # Priority 1: Collision avoidance
        if scenario.obstacle_ahead and scenario.depth_min < self.COLLISION_DIST:
            # Immediate danger - must turn or back up
            if scenario.lidar_left > scenario.lidar_right:
                return "rotate_left", 0.95
            elif scenario.lidar_right > scenario.lidar_left:
                return "rotate_right", 0.95
            else:
                return "move_backward", 0.90

        # Priority 2: Cautious navigation near obstacles
        if scenario.depth_min < self.CAUTION_DIST:
            # Getting close to something
            if scenario.lidar_front < self.CAUTION_DIST:
                # Front blocked, find open direction
                if scenario.lidar_left > scenario.lidar_right + 0.5:
                    return "rotate_left", 0.85
                elif scenario.lidar_right > scenario.lidar_left + 0.5:
                    return "rotate_right", 0.85
                elif scenario.lidar_back > self.COLLISION_DIST:
                    return "move_backward", 0.80

        # Priority 3: Task-specific behavior
        if self.task_type == "fetch" and scenario.graspable_nearby:
            if scenario.nearest_object_dist < self.GRASP_DIST:
                return "pick_up", 0.90
            else:
                # Move towards graspable object
                return "move_forward", 0.85

        if self.task_type == "navigate" and scenario.target_visible:
            # Move towards target
            if abs(scenario.goal_direction) > 0.3:
                if scenario.goal_direction < 0:
                    return "rotate_left", 0.85
                else:
                    return "rotate_right", 0.85
            else:
                return "move_forward", 0.90

        # Priority 4: Goal-directed movement
        if scenario.goal_distance > 0.1:  # Not at goal
            if abs(scenario.goal_direction) > 0.4:
                # Need to turn towards goal
                if scenario.goal_direction < 0 and scenario.lidar_left > self.COLLISION_DIST:
                    return "rotate_left", 0.80
                elif scenario.goal_direction > 0 and scenario.lidar_right > self.COLLISION_DIST:
                    return "rotate_right", 0.80

            # Path is clear, move forward
            if scenario.lidar_front > self.CAUTION_DIST:
                return "move_forward", 0.85

        # Priority 5: Exploration (when no specific goal)
        if self.task_type == "explore":
            # Prefer unexplored directions (simulate with randomness)
            if scenario.lidar_front > self.CAUTION_DIST:
                return "move_forward", 0.70
            elif scenario.lidar_left > scenario.lidar_right:
                return "rotate_left", 0.65
            else:
                return "rotate_right", 0.65

        # Default: Move forward if safe
        if scenario.lidar_front > self.CAUTION_DIST:
            return "move_forward", 0.75
        elif scenario.lidar_left > scenario.lidar_right:
            return "rotate_left", 0.70
        else:
            return "rotate_right", 0.70


class PerceptionDataGenerator:
    """Generates diverse perception training scenarios with expert labels."""

    # Scenario templates
    SCENARIO_TYPES = [
        "clear_path",
        "obstacle_ahead",
        "obstacle_left",
        "obstacle_right",
        "narrow_passage",
        "corner_left",
        "corner_right",
        "dead_end",
        "open_space",
        "cluttered_room",
        "doorway",
        "near_wall",
        "object_in_view",
        "graspable_nearby",
        "target_visible",
        "dark_environment",
        "bright_environment",
        "goal_left",
        "goal_right",
        "goal_ahead",
    ]

    def __init__(self):
        self.expert = ExpertPerceptionPolicy()
        self.samples: List[Dict] = []

    def generate_scenario(self, scenario_type: str) -> PerceptionScenario:
        """Generate a specific scenario type."""

        # Base values
        brightness = random.uniform(0.3, 0.9)
        depth_mean = random.uniform(2.0, 8.0)
        floor_visible = random.uniform(0.2, 0.5)
        wall_visible = random.uniform(0.1, 0.4)
        num_objects = random.randint(0, 5)
        goal_direction = random.uniform(-0.3, 0.3)
        goal_distance = random.uniform(0.3, 0.8)

        # LiDAR defaults (clear)
        lidar_front = random.uniform(3.0, 10.0)
        lidar_left = random.uniform(3.0, 10.0)
        lidar_right = random.uniform(3.0, 10.0)
        lidar_back = random.uniform(3.0, 10.0)

        obstacle_ahead = False
        graspable_nearby = False
        target_visible = False
        nearest_object_dist = random.uniform(2.0, 8.0)
        depth_min = random.uniform(1.5, 5.0)

        # Customize based on scenario type
        if scenario_type == "clear_path":
            lidar_front = random.uniform(5.0, 10.0)
            depth_min = random.uniform(3.0, 8.0)

        elif scenario_type == "obstacle_ahead":
            lidar_front = random.uniform(0.3, 1.0)
            depth_min = random.uniform(0.3, 1.0)
            obstacle_ahead = True

        elif scenario_type == "obstacle_left":
            lidar_left = random.uniform(0.3, 1.0)
            lidar_front = random.uniform(2.0, 5.0)

        elif scenario_type == "obstacle_right":
            lidar_right = random.uniform(0.3, 1.0)
            lidar_front = random.uniform(2.0, 5.0)

        elif scenario_type == "narrow_passage":
            lidar_left = random.uniform(0.5, 1.5)
            lidar_right = random.uniform(0.5, 1.5)
            lidar_front = random.uniform(3.0, 8.0)

        elif scenario_type == "corner_left":
            lidar_front = random.uniform(0.3, 1.2)
            lidar_left = random.uniform(0.3, 1.0)
            lidar_right = random.uniform(2.0, 5.0)
            obstacle_ahead = True
            depth_min = random.uniform(0.3, 1.2)

        elif scenario_type == "corner_right":
            lidar_front = random.uniform(0.3, 1.2)
            lidar_right = random.uniform(0.3, 1.0)
            lidar_left = random.uniform(2.0, 5.0)
            obstacle_ahead = True
            depth_min = random.uniform(0.3, 1.2)

        elif scenario_type == "dead_end":
            lidar_front = random.uniform(0.3, 0.8)
            lidar_left = random.uniform(0.3, 1.0)
            lidar_right = random.uniform(0.3, 1.0)
            lidar_back = random.uniform(2.0, 5.0)
            obstacle_ahead = True
            depth_min = random.uniform(0.3, 0.8)

        elif scenario_type == "open_space":
            lidar_front = random.uniform(6.0, 10.0)
            lidar_left = random.uniform(5.0, 10.0)
            lidar_right = random.uniform(5.0, 10.0)
            depth_min = random.uniform(4.0, 8.0)
            floor_visible = random.uniform(0.4, 0.6)

        elif scenario_type == "cluttered_room":
            num_objects = random.randint(5, 10)
            nearest_object_dist = random.uniform(0.5, 2.0)
            lidar_front = random.uniform(1.5, 4.0)
            depth_min = random.uniform(1.0, 3.0)

        elif scenario_type == "doorway":
            lidar_left = random.uniform(0.4, 0.8)
            lidar_right = random.uniform(0.4, 0.8)
            lidar_front = random.uniform(3.0, 8.0)
            wall_visible = random.uniform(0.3, 0.5)

        elif scenario_type == "near_wall":
            wall_visible = random.uniform(0.4, 0.7)
            side = random.choice(["left", "right", "front"])
            if side == "left":
                lidar_left = random.uniform(0.3, 1.0)
            elif side == "right":
                lidar_right = random.uniform(0.3, 1.0)
            else:
                lidar_front = random.uniform(0.5, 1.5)
                obstacle_ahead = True
                depth_min = random.uniform(0.5, 1.5)

        elif scenario_type == "object_in_view":
            num_objects = random.randint(1, 3)
            nearest_object_dist = random.uniform(1.0, 4.0)

        elif scenario_type == "graspable_nearby":
            graspable_nearby = True
            nearest_object_dist = random.uniform(0.3, 1.5)
            num_objects = random.randint(1, 3)

        elif scenario_type == "target_visible":
            target_visible = True
            goal_distance = random.uniform(0.2, 0.6)

        elif scenario_type == "dark_environment":
            brightness = random.uniform(0.05, 0.2)

        elif scenario_type == "bright_environment":
            brightness = random.uniform(0.8, 1.0)

        elif scenario_type == "goal_left":
            goal_direction = random.uniform(-0.9, -0.5)
            lidar_left = random.uniform(2.0, 6.0)  # Ensure left is open

        elif scenario_type == "goal_right":
            goal_direction = random.uniform(0.5, 0.9)
            lidar_right = random.uniform(2.0, 6.0)  # Ensure right is open

        elif scenario_type == "goal_ahead":
            goal_direction = random.uniform(-0.2, 0.2)
            lidar_front = random.uniform(3.0, 8.0)  # Ensure front is open
            target_visible = True

        # Create scenario
        scenario = PerceptionScenario(
            name=scenario_type,
            description=f"Generated {scenario_type} scenario",
            rgb_brightness=brightness,
            depth_mean=depth_mean,
            depth_min=depth_min,
            obstacle_ahead=obstacle_ahead,
            floor_visible=floor_visible,
            wall_visible=wall_visible,
            num_objects=num_objects,
            nearest_object_dist=nearest_object_dist,
            graspable_nearby=graspable_nearby,
            target_visible=target_visible,
            lidar_front=lidar_front,
            lidar_left=lidar_left,
            lidar_right=lidar_right,
            lidar_back=lidar_back,
            goal_direction=goal_direction,
            goal_distance=goal_distance,
            expert_action="",
            confidence=0.0,
        )

        # Get expert action
        action, confidence = self.expert.decide(scenario)
        scenario.expert_action = action
        scenario.confidence = confidence

        return scenario

    def scenario_to_features(self, scenario: PerceptionScenario) -> List[float]:
        """Convert scenario to feature vector for training."""
        features = [
            scenario.rgb_brightness,
            scenario.depth_mean / 10.0,  # Normalize
            scenario.depth_min / 5.0,
            float(scenario.obstacle_ahead),
            scenario.floor_visible,
            scenario.wall_visible,
            scenario.num_objects / 10.0,
            scenario.nearest_object_dist / 10.0,
            float(scenario.graspable_nearby),
            float(scenario.target_visible),
            scenario.lidar_front / 10.0,
            scenario.lidar_left / 10.0,
            scenario.lidar_right / 10.0,
            scenario.lidar_back / 10.0,
            scenario.goal_direction,
            scenario.goal_distance,
        ]

        # Pad to 32 dimensions for compatibility
        while len(features) < 32:
            features.append(0.0)

        return features[:32]

    def generate_balanced_dataset(self, total_samples: int = 5000) -> List[Dict]:
        """Generate balanced dataset across all scenario types."""
        samples = []

        # Distribution across scenarios (emphasize challenging ones)
        scenario_weights = {
            "clear_path": 0.08,
            "obstacle_ahead": 0.12,  # Critical
            "obstacle_left": 0.06,
            "obstacle_right": 0.06,
            "narrow_passage": 0.06,
            "corner_left": 0.08,
            "corner_right": 0.08,
            "dead_end": 0.10,  # Critical
            "open_space": 0.05,
            "cluttered_room": 0.05,
            "doorway": 0.04,
            "near_wall": 0.04,
            "object_in_view": 0.03,
            "graspable_nearby": 0.05,
            "target_visible": 0.04,
            "dark_environment": 0.02,
            "bright_environment": 0.02,
            "goal_left": 0.04,
            "goal_right": 0.04,
            "goal_ahead": 0.04,
        }

        for scenario_type, weight in scenario_weights.items():
            n_samples = int(total_samples * weight)
            for _ in range(n_samples):
                scenario = self.generate_scenario(scenario_type)
                sample = {
                    "scenario_type": scenario_type,
                    "features": self.scenario_to_features(scenario),
                    "action": scenario.expert_action,
                    "action_idx": ACTION_TO_IDX.get(scenario.expert_action, 0),
                    "confidence": scenario.confidence,
                    "raw": {
                        "brightness": scenario.rgb_brightness,
                        "depth_min": scenario.depth_min,
                        "obstacle_ahead": scenario.obstacle_ahead,
                        "lidar": [scenario.lidar_front, scenario.lidar_left,
                                  scenario.lidar_right, scenario.lidar_back],
                    }
                }
                samples.append(sample)

        # Fill remaining with random
        while len(samples) < total_samples:
            scenario_type = random.choice(self.SCENARIO_TYPES)
            scenario = self.generate_scenario(scenario_type)
            sample = {
                "scenario_type": scenario_type,
                "features": self.scenario_to_features(scenario),
                "action": scenario.expert_action,
                "action_idx": ACTION_TO_IDX.get(scenario.expert_action, 0),
                "confidence": scenario.confidence,
            }
            samples.append(sample)

        random.shuffle(samples)
        self.samples = samples

        return samples

    def save(self, filepath: str):
        """Save generated samples."""
        with open(filepath, 'w') as f:
            json.dump(self.samples, f, indent=2)
        print(f"Saved {len(self.samples)} expert perception samples to {filepath}")

    def load(self, filepath: str) -> List[Dict]:
        """Load samples from file."""
        with open(filepath, 'r') as f:
            self.samples = json.load(f)
        return self.samples


class ExpertPerceptionModel:
    """
    Neural network for perception-based action prediction.

    Architecture: 3-layer MLP
    Input: 32-dim perception features
    Hidden: 64 -> 32 neurons
    Output: 5 actions (softmax)
    """

    INPUT_DIM = 32
    HIDDEN1_DIM = 64
    HIDDEN2_DIM = 32
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self):
        np.random.seed(42)

        # Xavier initialization
        self.W1 = np.random.randn(self.INPUT_DIM, self.HIDDEN1_DIM) * np.sqrt(2.0 / self.INPUT_DIM)
        self.b1 = np.zeros(self.HIDDEN1_DIM)

        self.W2 = np.random.randn(self.HIDDEN1_DIM, self.HIDDEN2_DIM) * np.sqrt(2.0 / self.HIDDEN1_DIM)
        self.b2 = np.zeros(self.HIDDEN2_DIM)

        self.W3 = np.random.randn(self.HIDDEN2_DIM, self.NUM_ACTIONS) * np.sqrt(2.0 / self.HIDDEN2_DIM)
        self.b3 = np.zeros(self.NUM_ACTIONS)

        self.is_ready = False

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass with intermediate activations for backprop."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        logits = h2 @ self.W3 + self.b3

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return probs, h1, h2, logits

    def predict(self, features: List[float]) -> Tuple[str, float, List[float]]:
        """Predict action from perception features."""
        x = np.array(features)
        probs, _, _, _ = self.forward(x)
        action_idx = int(np.argmax(probs))
        return IDX_TO_ACTION[action_idx], float(probs[action_idx]), probs.tolist()

    def train_step(self, features: List[float], target_idx: int, lr: float = 0.001) -> float:
        """Single training step with backpropagation."""
        x = np.array(features)

        # Forward
        probs, h1, h2, logits = self.forward(x)

        # Loss
        loss = -np.log(probs[target_idx] + 1e-10)

        # Backward
        d_logits = probs.copy()
        d_logits[target_idx] -= 1

        # Layer 3
        d_W3 = np.outer(h2, d_logits)
        d_b3 = d_logits

        # Layer 2
        d_h2 = d_logits @ self.W3.T
        d_h2 = d_h2 * (h2 > 0)
        d_W2 = np.outer(h1, d_h2)
        d_b2 = d_h2

        # Layer 1
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        d_W1 = np.outer(x, d_h1)
        d_b1 = d_h1

        # Update
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3

        self.is_ready = True
        return loss

    def save(self, filepath: str):
        """Save model."""
        data = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load model."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.W1 = np.array(data["W1"])
        self.b1 = np.array(data["b1"])
        self.W2 = np.array(data["W2"])
        self.b2 = np.array(data["b2"])
        self.W3 = np.array(data["W3"])
        self.b3 = np.array(data["b3"])
        self.is_ready = True


def train_expert_perception(data_dir: str = "brain_b_data", epochs: int = 100, samples: int = 5000):
    """Train expert perception model."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Generate data
    print(f"Generating {samples} expert perception samples...")
    generator = PerceptionDataGenerator()
    data = generator.generate_balanced_dataset(samples)

    # Print distribution
    action_counts = {}
    scenario_counts = {}
    for s in data:
        action_counts[s["action"]] = action_counts.get(s["action"], 0) + 1
        scenario_counts[s["scenario_type"]] = scenario_counts.get(s["scenario_type"], 0) + 1

    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count} ({count/len(data)*100:.1f}%)")

    # Save data
    data_file = data_path / "expert_perception_data.json"
    generator.save(str(data_file))

    # Train
    print(f"\nTraining expert perception model for {epochs} epochs...")
    model = ExpertPerceptionModel()

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        correct = 0

        for sample in data:
            loss = model.train_step(sample["features"], sample["action_idx"])
            total_loss += loss

            pred, _, _ = model.predict(sample["features"])
            if pred == sample["action"]:
                correct += 1

        if (epoch + 1) % 10 == 0:
            acc = correct / len(data)
            print(f"  Epoch {epoch + 1}: loss={total_loss/len(data):.4f}, acc={acc*100:.1f}%")

    # Final accuracy
    correct = 0
    for sample in data:
        pred, _, _ = model.predict(sample["features"])
        if pred == sample["action"]:
            correct += 1
    final_acc = correct / len(data)
    print(f"\nFinal accuracy: {final_acc*100:.1f}%")

    # Save model
    model_file = data_path / "expert_perception_model.json"
    model.save(str(model_file))
    print(f"Model saved to {model_file}")

    # Test on scenarios
    print("\nTesting on key scenarios:")
    test_scenarios = [
        ("clear_path", {"obstacle_ahead": False, "depth_min": 5.0, "lidar_front": 8.0}),
        ("obstacle_ahead", {"obstacle_ahead": True, "depth_min": 0.5, "lidar_front": 0.5}),
        ("dead_end", {"obstacle_ahead": True, "depth_min": 0.5, "lidar_front": 0.5,
                      "lidar_left": 0.5, "lidar_right": 0.5, "lidar_back": 5.0}),
        ("goal_left", {"goal_direction": -0.7, "lidar_left": 5.0}),
    ]

    for name, overrides in test_scenarios:
        scenario = generator.generate_scenario(name)
        features = generator.scenario_to_features(scenario)
        pred, conf, _ = model.predict(features)
        print(f"  {name}: {pred} (conf={conf:.2f})")

    return model, final_acc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Expert Perception Training")
    parser.add_argument("--generate", type=int, default=0, help="Generate N samples")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--data-dir", type=str, default="brain_b_data", help="Data directory")

    args = parser.parse_args()

    if args.generate > 0:
        generator = PerceptionDataGenerator()
        generator.generate_balanced_dataset(args.generate)
        generator.save(f"{args.data_dir}/expert_perception_data.json")

    elif args.train:
        train_expert_perception(args.data_dir, args.epochs, args.samples)

    elif args.evaluate:
        model = ExpertPerceptionModel()
        model.load(f"{args.data_dir}/expert_perception_model.json")

        generator = PerceptionDataGenerator()
        data = generator.load(f"{args.data_dir}/expert_perception_data.json")

        correct = 0
        for sample in data:
            pred, _, _ = model.predict(sample["features"])
            if pred == sample["action"]:
                correct += 1

        print(f"Evaluation accuracy: {correct/len(data)*100:.1f}%")

    else:
        print("Run with --generate N, --train, or --evaluate")


class ExpertPerceptionTrainer:
    """High-level trainer class for use by training daemon."""

    def __init__(self, data_dir: str = "brain_b_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.generator = PerceptionDataGenerator()
        self.model = ExpertPerceptionModel()
        self.samples = []

    def generate_expert_samples(self, num_samples: int = 2000):
        """Generate expert perception samples."""
        self.samples = self.generator.generate_balanced_dataset(num_samples)
        return self.samples

    def train(self, epochs: int = 50):
        """Train the expert perception model."""
        if not self.samples:
            self.generate_expert_samples(2000)

        for epoch in range(epochs):
            random.shuffle(self.samples)
            total_loss = 0.0

            for sample in self.samples:
                loss = self.model.train_step(sample["features"], sample["action_idx"])
                total_loss += loss

        # Calculate final accuracy
        correct = 0
        for sample in self.samples:
            pred, _, _ = self.model.predict(sample["features"])
            if pred == sample["action"]:
                correct += 1

        accuracy = correct / len(self.samples)
        return {"accuracy": accuracy, "samples": len(self.samples), "epochs": epochs}

    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save(filepath)


if __name__ == "__main__":
    main()
