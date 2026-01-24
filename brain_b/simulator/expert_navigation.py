#!/usr/bin/env python3
"""
Expert Navigation Policy and Training Data Generator

Creates high-quality training data for navigation using expert policies
that correctly handle obstacles, walls, and navigation scenarios.

The key insight: random policies produce biased data (mostly forward).
Expert policies demonstrate correct behavior for all scenarios.

Usage:
    python brain_b/simulator/expert_navigation.py --generate 1000
    python brain_b/simulator/expert_navigation.py --train
    python brain_b/simulator/expert_navigation.py --test
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


# Navigation actions
ACTIONS = [
    "move_forward",
    "move_backward",
    "rotate_left",
    "rotate_right",
    "noop",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


@dataclass
class NavigationState:
    """State representation for navigation."""
    # Distances in 8 directions (normalized 0-1, 0=close obstacle, 1=far/clear)
    front: float = 1.0
    front_left: float = 1.0
    front_right: float = 1.0
    left: float = 1.0
    right: float = 1.0
    back: float = 1.0
    back_left: float = 1.0
    back_right: float = 1.0

    # Goal direction (relative angle, -1 to 1 where 0=straight ahead)
    goal_direction: float = 0.0

    # Goal distance (normalized, 0=at goal, 1=far)
    goal_distance: float = 0.5

    # Additional context
    in_narrow_passage: bool = False
    near_corner: bool = False

    def to_vector(self) -> List[float]:
        """Convert to feature vector."""
        return [
            self.front,
            self.front_left,
            self.front_right,
            self.left,
            self.right,
            self.back,
            self.back_left,
            self.back_right,
            self.goal_direction,
            self.goal_distance,
            float(self.in_narrow_passage),
            float(self.near_corner),
        ]

    @classmethod
    def from_lidar(cls, ranges: List[float], angles: List[float],
                   goal_dir: float = 0.0, goal_dist: float = 0.5) -> 'NavigationState':
        """Create state from LiDAR data."""
        # Bin ranges into 8 directions
        sectors = {
            'front': [], 'front_left': [], 'front_right': [],
            'left': [], 'right': [],
            'back': [], 'back_left': [], 'back_right': [],
        }

        for angle, dist in zip(angles, ranges):
            deg = math.degrees(angle)
            # Normalize to 0-1 (assuming max range 10m)
            norm_dist = min(dist / 10.0, 1.0)

            if -22.5 <= deg < 22.5:
                sectors['front'].append(norm_dist)
            elif 22.5 <= deg < 67.5:
                sectors['front_left'].append(norm_dist)
            elif 67.5 <= deg < 112.5:
                sectors['left'].append(norm_dist)
            elif 112.5 <= deg < 157.5:
                sectors['back_left'].append(norm_dist)
            elif deg >= 157.5 or deg < -157.5:
                sectors['back'].append(norm_dist)
            elif -157.5 <= deg < -112.5:
                sectors['back_right'].append(norm_dist)
            elif -112.5 <= deg < -67.5:
                sectors['right'].append(norm_dist)
            elif -67.5 <= deg < -22.5:
                sectors['front_right'].append(norm_dist)

        # Get minimum distance per sector (closest obstacle)
        state = cls()
        state.front = min(sectors['front']) if sectors['front'] else 1.0
        state.front_left = min(sectors['front_left']) if sectors['front_left'] else 1.0
        state.front_right = min(sectors['front_right']) if sectors['front_right'] else 1.0
        state.left = min(sectors['left']) if sectors['left'] else 1.0
        state.right = min(sectors['right']) if sectors['right'] else 1.0
        state.back = min(sectors['back']) if sectors['back'] else 1.0
        state.back_left = min(sectors['back_left']) if sectors['back_left'] else 1.0
        state.back_right = min(sectors['back_right']) if sectors['back_right'] else 1.0

        state.goal_direction = goal_dir
        state.goal_distance = goal_dist

        # Detect narrow passage
        state.in_narrow_passage = state.left < 0.3 and state.right < 0.3

        # Detect corner
        state.near_corner = (state.front < 0.3 and
                            (state.left < 0.3 or state.right < 0.3))

        return state


class ExpertNavigationPolicy:
    """
    Rule-based expert policy for navigation.

    This policy makes correct decisions for all scenarios:
    - Clear path: move towards goal
    - Wall ahead: turn away from wall
    - Narrow passage: proceed carefully
    - Corner: turn towards open space
    """

    OBSTACLE_THRESHOLD = 0.2  # Normalized distance below which is "too close"
    CAUTION_THRESHOLD = 0.4   # Distance at which to start being careful

    def __init__(self, randomness: float = 0.1):
        """
        Args:
            randomness: Probability of adding random exploration (0-1)
        """
        self.randomness = randomness

    def get_action(self, state: NavigationState) -> Tuple[str, float]:
        """
        Get the expert action for a given state.

        Returns:
            (action_name, confidence)
        """
        # Add some randomness for exploration diversity
        if random.random() < self.randomness:
            action = random.choice(ACTIONS)
            return action, 0.5

        # Priority 0: Dead end detection - all forward directions blocked
        is_dead_end = (state.front < self.OBSTACLE_THRESHOLD and
                       state.front_left < self.CAUTION_THRESHOLD and
                       state.front_right < self.CAUTION_THRESHOLD and
                       state.left < self.CAUTION_THRESHOLD and
                       state.right < self.CAUTION_THRESHOLD)
        if is_dead_end and state.back > self.OBSTACLE_THRESHOLD:
            return "move_backward", 0.95

        # Priority 1: Avoid immediate collision
        if state.front < self.OBSTACLE_THRESHOLD:
            # Wall directly ahead - must turn or back up
            # Check if both sides are also blocked
            both_sides_blocked = (state.left < self.CAUTION_THRESHOLD and
                                  state.right < self.CAUTION_THRESHOLD)
            if both_sides_blocked and state.back > self.OBSTACLE_THRESHOLD:
                return "move_backward", 0.92

            if state.left > state.right:
                return "rotate_left", 0.95
            elif state.right > state.left:
                return "rotate_right", 0.95
            else:
                # Both sides equally blocked, back up
                return "move_backward", 0.9

        # Priority 2: Handle near-obstacles cautiously
        if state.front < self.CAUTION_THRESHOLD:
            # Getting close to obstacle
            if state.goal_direction < -0.3 and state.left > self.OBSTACLE_THRESHOLD:
                return "rotate_left", 0.85
            elif state.goal_direction > 0.3 and state.right > self.OBSTACLE_THRESHOLD:
                return "rotate_right", 0.85
            elif state.left > state.right:
                return "rotate_left", 0.8
            else:
                return "rotate_right", 0.8

        # Priority 3: Turn towards goal if not facing it
        if abs(state.goal_direction) > 0.3:
            if state.goal_direction < 0 and state.left > self.OBSTACLE_THRESHOLD:
                return "rotate_left", 0.75
            elif state.goal_direction > 0 and state.right > self.OBSTACLE_THRESHOLD:
                return "rotate_right", 0.75

        # Priority 4: Move forward if clear
        if state.front > self.CAUTION_THRESHOLD:
            return "move_forward", 0.9

        # Default: small rotation to find better path
        if state.left > state.right:
            return "rotate_left", 0.6
        else:
            return "rotate_right", 0.6

    def get_action_probs(self, state: NavigationState) -> List[float]:
        """Get probability distribution over actions."""
        action, confidence = self.get_action(state)
        action_idx = ACTION_TO_IDX[action]

        # Create soft distribution
        probs = [0.05] * len(ACTIONS)  # Small baseline for all
        probs[action_idx] = confidence

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        return probs


@dataclass
class ExpertTrainingSample:
    """Training sample from expert policy."""
    state: NavigationState
    action: str
    action_idx: int
    confidence: float
    scenario: str  # Description of scenario


class ExpertDataGenerator:
    """Generates expert training data for navigation."""

    def __init__(self):
        self.expert = ExpertNavigationPolicy(randomness=0.05)
        self.samples: List[ExpertTrainingSample] = []

    def generate_scenario(self, scenario_type: str) -> ExpertTrainingSample:
        """Generate a single training sample for a specific scenario."""

        if scenario_type == "clear_ahead":
            # Open space ahead, goal roughly forward
            state = NavigationState(
                front=random.uniform(0.6, 1.0),
                front_left=random.uniform(0.5, 1.0),
                front_right=random.uniform(0.5, 1.0),
                left=random.uniform(0.4, 1.0),
                right=random.uniform(0.4, 1.0),
                back=random.uniform(0.3, 1.0),
                back_left=random.uniform(0.3, 1.0),
                back_right=random.uniform(0.3, 1.0),
                goal_direction=random.uniform(-0.2, 0.2),
                goal_distance=random.uniform(0.3, 0.8),
            )

        elif scenario_type == "wall_ahead":
            # Wall directly in front
            state = NavigationState(
                front=random.uniform(0.05, 0.2),  # Very close!
                front_left=random.uniform(0.2, 0.6),
                front_right=random.uniform(0.2, 0.6),
                left=random.uniform(0.3, 1.0),
                right=random.uniform(0.3, 1.0),
                back=random.uniform(0.4, 1.0),
                back_left=random.uniform(0.4, 1.0),
                back_right=random.uniform(0.4, 1.0),
                goal_direction=random.uniform(-0.5, 0.5),
                goal_distance=random.uniform(0.3, 0.8),
            )

        elif scenario_type == "wall_left":
            # Wall on the left
            state = NavigationState(
                front=random.uniform(0.4, 1.0),
                front_left=random.uniform(0.05, 0.2),
                front_right=random.uniform(0.4, 1.0),
                left=random.uniform(0.05, 0.2),
                right=random.uniform(0.4, 1.0),
                back=random.uniform(0.3, 1.0),
                back_left=random.uniform(0.1, 0.3),
                back_right=random.uniform(0.4, 1.0),
                goal_direction=random.uniform(-0.3, 0.3),
                goal_distance=random.uniform(0.3, 0.8),
            )

        elif scenario_type == "wall_right":
            # Wall on the right
            state = NavigationState(
                front=random.uniform(0.4, 1.0),
                front_left=random.uniform(0.4, 1.0),
                front_right=random.uniform(0.05, 0.2),
                left=random.uniform(0.4, 1.0),
                right=random.uniform(0.05, 0.2),
                back=random.uniform(0.3, 1.0),
                back_left=random.uniform(0.4, 1.0),
                back_right=random.uniform(0.1, 0.3),
                goal_direction=random.uniform(-0.3, 0.3),
                goal_distance=random.uniform(0.3, 0.8),
            )

        elif scenario_type == "corner_left":
            # Corner with wall ahead and left
            state = NavigationState(
                front=random.uniform(0.05, 0.25),
                front_left=random.uniform(0.05, 0.2),
                front_right=random.uniform(0.3, 0.7),
                left=random.uniform(0.05, 0.25),
                right=random.uniform(0.4, 1.0),
                back=random.uniform(0.3, 0.8),
                back_left=random.uniform(0.1, 0.4),
                back_right=random.uniform(0.4, 1.0),
                goal_direction=random.uniform(-0.5, 0.5),
                goal_distance=random.uniform(0.3, 0.8),
                near_corner=True,
            )

        elif scenario_type == "corner_right":
            # Corner with wall ahead and right
            state = NavigationState(
                front=random.uniform(0.05, 0.25),
                front_left=random.uniform(0.3, 0.7),
                front_right=random.uniform(0.05, 0.2),
                left=random.uniform(0.4, 1.0),
                right=random.uniform(0.05, 0.25),
                back=random.uniform(0.3, 0.8),
                back_left=random.uniform(0.4, 1.0),
                back_right=random.uniform(0.1, 0.4),
                goal_direction=random.uniform(-0.5, 0.5),
                goal_distance=random.uniform(0.3, 0.8),
                near_corner=True,
            )

        elif scenario_type == "narrow_passage":
            # Narrow corridor - walls on both sides
            state = NavigationState(
                front=random.uniform(0.4, 1.0),
                front_left=random.uniform(0.1, 0.3),
                front_right=random.uniform(0.1, 0.3),
                left=random.uniform(0.1, 0.25),
                right=random.uniform(0.1, 0.25),
                back=random.uniform(0.3, 1.0),
                back_left=random.uniform(0.1, 0.3),
                back_right=random.uniform(0.1, 0.3),
                goal_direction=random.uniform(-0.15, 0.15),
                goal_distance=random.uniform(0.3, 0.8),
                in_narrow_passage=True,
            )

        elif scenario_type == "dead_end":
            # Dead end - walls on three sides
            state = NavigationState(
                front=random.uniform(0.05, 0.2),
                front_left=random.uniform(0.05, 0.2),
                front_right=random.uniform(0.05, 0.2),
                left=random.uniform(0.05, 0.25),
                right=random.uniform(0.05, 0.25),
                back=random.uniform(0.4, 1.0),
                back_left=random.uniform(0.2, 0.5),
                back_right=random.uniform(0.2, 0.5),
                goal_direction=random.uniform(-0.5, 0.5),
                goal_distance=random.uniform(0.3, 0.8),
            )

        elif scenario_type == "goal_left":
            # Clear path but goal is to the left
            state = NavigationState(
                front=random.uniform(0.5, 1.0),
                front_left=random.uniform(0.4, 1.0),
                front_right=random.uniform(0.4, 1.0),
                left=random.uniform(0.4, 1.0),
                right=random.uniform(0.4, 1.0),
                back=random.uniform(0.3, 1.0),
                back_left=random.uniform(0.3, 1.0),
                back_right=random.uniform(0.3, 1.0),
                goal_direction=random.uniform(-0.9, -0.4),  # Goal to the left
                goal_distance=random.uniform(0.3, 0.8),
            )

        elif scenario_type == "goal_right":
            # Clear path but goal is to the right
            state = NavigationState(
                front=random.uniform(0.5, 1.0),
                front_left=random.uniform(0.4, 1.0),
                front_right=random.uniform(0.4, 1.0),
                left=random.uniform(0.4, 1.0),
                right=random.uniform(0.4, 1.0),
                back=random.uniform(0.3, 1.0),
                back_left=random.uniform(0.3, 1.0),
                back_right=random.uniform(0.3, 1.0),
                goal_direction=random.uniform(0.4, 0.9),  # Goal to the right
                goal_distance=random.uniform(0.3, 0.8),
            )

        else:
            # Random state
            state = NavigationState(
                front=random.uniform(0.1, 1.0),
                front_left=random.uniform(0.1, 1.0),
                front_right=random.uniform(0.1, 1.0),
                left=random.uniform(0.1, 1.0),
                right=random.uniform(0.1, 1.0),
                back=random.uniform(0.1, 1.0),
                back_left=random.uniform(0.1, 1.0),
                back_right=random.uniform(0.1, 1.0),
                goal_direction=random.uniform(-1.0, 1.0),
                goal_distance=random.uniform(0.1, 1.0),
            )

        # Get expert action
        action, confidence = self.expert.get_action(state)

        return ExpertTrainingSample(
            state=state,
            action=action,
            action_idx=ACTION_TO_IDX[action],
            confidence=confidence,
            scenario=scenario_type,
        )

    def generate_balanced_dataset(self, total_samples: int = 1000) -> List[ExpertTrainingSample]:
        """Generate a balanced dataset covering all scenarios."""
        scenarios = [
            ("clear_ahead", 0.12),       # 12%
            ("wall_ahead", 0.18),        # 18% - key scenario!
            ("wall_left", 0.08),
            ("wall_right", 0.08),
            ("corner_left", 0.10),
            ("corner_right", 0.10),
            ("narrow_passage", 0.06),
            ("dead_end", 0.15),          # 15% - increased for move_backward learning!
            ("goal_left", 0.06),
            ("goal_right", 0.06),
        ]

        samples = []

        for scenario, ratio in scenarios:
            n = int(total_samples * ratio)
            for _ in range(n):
                sample = self.generate_scenario(scenario)
                samples.append(sample)

        # Fill remaining with random
        while len(samples) < total_samples:
            sample = self.generate_scenario("random")
            samples.append(sample)

        random.shuffle(samples)
        self.samples = samples

        return samples

    def save_samples(self, filepath: str):
        """Save samples to file."""
        data = []
        for sample in self.samples:
            data.append({
                "state": sample.state.to_vector(),
                "action": sample.action,
                "action_idx": sample.action_idx,
                "confidence": sample.confidence,
                "scenario": sample.scenario,
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(data)} expert samples to {filepath}")

    def load_samples(self, filepath: str) -> List[ExpertTrainingSample]:
        """Load samples from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        samples = []
        for item in data:
            state = NavigationState()
            vec = item["state"]
            state.front = vec[0]
            state.front_left = vec[1]
            state.front_right = vec[2]
            state.left = vec[3]
            state.right = vec[4]
            state.back = vec[5]
            state.back_left = vec[6]
            state.back_right = vec[7]
            state.goal_direction = vec[8]
            state.goal_distance = vec[9]
            state.in_narrow_passage = bool(vec[10]) if len(vec) > 10 else False
            state.near_corner = bool(vec[11]) if len(vec) > 11 else False

            samples.append(ExpertTrainingSample(
                state=state,
                action=item["action"],
                action_idx=item["action_idx"],
                confidence=item["confidence"],
                scenario=item["scenario"],
            ))

        self.samples = samples
        return samples


class ImprovedNavigationModel:
    """
    Improved navigation model with better architecture.

    - Two hidden layers instead of one
    - Larger capacity
    - Trained on expert data
    """

    INPUT_DIM = 12  # From NavigationState.to_vector()
    HIDDEN_DIM = 32
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self):
        np.random.seed(42)

        # Two-layer MLP
        self.W1 = np.random.randn(self.INPUT_DIM, self.HIDDEN_DIM) * 0.1
        self.b1 = np.zeros(self.HIDDEN_DIM)
        self.W2 = np.random.randn(self.HIDDEN_DIM, self.HIDDEN_DIM) * 0.1
        self.b2 = np.zeros(self.HIDDEN_DIM)
        self.W3 = np.random.randn(self.HIDDEN_DIM, self.NUM_ACTIONS) * 0.1
        self.b3 = np.zeros(self.NUM_ACTIONS)

        self.is_ready = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Layer 1
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        # Layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        # Output
        logits = h2 @ self.W3 + self.b3
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def predict(self, state: NavigationState) -> Tuple[str, float, List[float]]:
        """Predict action from state."""
        x = np.array(state.to_vector())
        probs = self.forward(x)
        action_idx = int(np.argmax(probs))
        return IDX_TO_ACTION[action_idx], float(probs[action_idx]), probs.tolist()

    def train_step(self, state: NavigationState, target_idx: int, lr: float = 0.01) -> float:
        """Single training step with backpropagation."""
        x = np.array(state.to_vector())

        # Forward pass
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Loss
        loss = -np.log(probs[target_idx] + 1e-10)

        # Backward pass
        d_logits = probs.copy()
        d_logits[target_idx] -= 1

        # Layer 3 gradients
        d_W3 = np.outer(h2, d_logits)
        d_b3 = d_logits

        # Layer 2 gradients
        d_h2 = d_logits @ self.W3.T
        d_h2 = d_h2 * (h2 > 0)
        d_W2 = np.outer(h1, d_h2)
        d_b2 = d_h2

        # Layer 1 gradients
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        d_W1 = np.outer(x, d_h1)
        d_b1 = d_h1

        # Update weights
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3

        self.is_ready = True
        return loss

    def save(self, filepath: str):
        """Save model weights."""
        data = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load model weights."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.W1 = np.array(data["W1"])
        self.b1 = np.array(data["b1"])
        self.W2 = np.array(data["W2"])
        self.b2 = np.array(data["b2"])
        self.W3 = np.array(data["W3"])
        self.b3 = np.array(data["b3"])
        self.is_ready = True


def train_improved_navigation(data_dir: str = "brain_b_data", epochs: int = 50):
    """Train the improved navigation model on expert data."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Generate expert data
    print("Generating expert navigation data...")
    generator = ExpertDataGenerator()
    samples = generator.generate_balanced_dataset(2000)

    # Print distribution
    scenario_counts = {}
    action_counts = {}
    for s in samples:
        scenario_counts[s.scenario] = scenario_counts.get(s.scenario, 0) + 1
        action_counts[s.action] = action_counts.get(s.action, 0) + 1

    print(f"\nScenario distribution:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")

    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")

    # Save expert data
    expert_file = data_path / "expert_navigation_data.json"
    generator.save_samples(str(expert_file))

    # Train model
    print(f"\nTraining improved navigation model for {epochs} epochs...")
    model = ImprovedNavigationModel()

    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        correct = 0

        for sample in samples:
            loss = model.train_step(sample.state, sample.action_idx, lr=0.01)
            total_loss += loss

            # Check accuracy
            pred_action, _, _ = model.predict(sample.state)
            if pred_action == sample.action:
                correct += 1

        if (epoch + 1) % 10 == 0:
            acc = correct / len(samples)
            print(f"  Epoch {epoch + 1}: loss={total_loss/len(samples):.4f}, acc={acc*100:.1f}%")

    # Save model
    model_file = data_path / "improved_navigation_model.json"
    model.save(str(model_file))
    print(f"\nModel saved to {model_file}")

    # Test on scenarios
    print("\nTesting on key scenarios:")
    test_scenarios = [
        ("clear_ahead", NavigationState(front=0.8, goal_direction=0.0)),
        ("wall_ahead", NavigationState(front=0.1, left=0.6, right=0.6)),
        ("corner_left", NavigationState(front=0.1, left=0.1, right=0.7)),
        ("corner_right", NavigationState(front=0.1, left=0.7, right=0.1)),
        ("goal_left", NavigationState(front=0.8, goal_direction=-0.7)),
        ("goal_right", NavigationState(front=0.8, goal_direction=0.7)),
    ]

    for name, state in test_scenarios:
        action, conf, _ = model.predict(state)
        print(f"  {name}: {action} (conf={conf:.2f})")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Expert Navigation Training")
    parser.add_argument("--generate", type=int, default=0, help="Generate N expert samples")
    parser.add_argument("--train", action="store_true", help="Train improved model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--test", action="store_true", help="Test model on scenarios")
    parser.add_argument("--data-dir", type=str, default="brain_b_data", help="Data directory")

    args = parser.parse_args()

    if args.generate > 0:
        generator = ExpertDataGenerator()
        samples = generator.generate_balanced_dataset(args.generate)
        generator.save_samples(f"{args.data_dir}/expert_navigation_data.json")

    elif args.train:
        train_improved_navigation(args.data_dir, args.epochs)

    elif args.test:
        model = ImprovedNavigationModel()
        model.load(f"{args.data_dir}/improved_navigation_model.json")

        # Run test scenarios
        print("Testing navigation model:")
        test_cases = [
            ("Clear ahead", NavigationState(front=0.9)),
            ("Wall ahead", NavigationState(front=0.1)),
            ("Wall left", NavigationState(front=0.5, left=0.1)),
            ("Wall right", NavigationState(front=0.5, right=0.1)),
            ("Dead end", NavigationState(front=0.1, left=0.1, right=0.1)),
        ]

        for name, state in test_cases:
            action, conf, probs = model.predict(state)
            print(f"  {name}: {action} (conf={conf:.2f})")

    else:
        print("Run with --generate N, --train, or --test")


if __name__ == "__main__":
    main()
