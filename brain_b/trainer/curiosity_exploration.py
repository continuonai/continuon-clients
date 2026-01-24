"""
Curiosity-Driven Exploration for Brain B

Implements intrinsic motivation rewards to encourage exploration of novel states.
Uses prediction error as a curiosity signal - the robot is rewarded for encountering
situations where its world model predictions are inaccurate.

Key Components:
1. Forward Model: Predicts next state given current state and action
2. Inverse Model: Predicts action given current and next state
3. Intrinsic Reward: Based on prediction error (novelty)
4. Exploration Policy: Balances curiosity with task rewards

Based on the ICM (Intrinsic Curiosity Module) architecture.
"""

import math
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime


@dataclass
class StateFeatures:
    """Feature representation of robot state for curiosity computation."""
    # Position features
    position: Tuple[float, float] = (0.0, 0.0)
    orientation: float = 0.0

    # Sensor features
    distances: List[float] = field(default_factory=lambda: [1.0] * 8)  # 8 directions

    # Object features
    detected_objects: List[str] = field(default_factory=list)
    object_distances: List[float] = field(default_factory=list)

    # Visual features (simplified)
    visual_hash: int = 0  # Hash of visual scene
    brightness: float = 0.5
    color_dominant: str = "gray"

    def to_vector(self) -> List[float]:
        """Convert to feature vector for neural network."""
        vec = list(self.position) + [self.orientation / (2 * math.pi)]
        vec.extend(self.distances[:8])  # Ensure exactly 8
        while len(vec) < 11:
            vec.append(1.0)

        # Object encoding (up to 5 objects)
        obj_types = {"obstacle": 1, "goal": 2, "person": 3, "unknown": 4}
        for i in range(5):
            if i < len(self.detected_objects):
                obj_type = obj_types.get(self.detected_objects[i], 4)
                obj_dist = self.object_distances[i] if i < len(self.object_distances) else 1.0
                vec.extend([obj_type / 4.0, obj_dist])
            else:
                vec.extend([0.0, 1.0])

        # Visual features
        vec.extend([self.visual_hash / 1000000.0, self.brightness])
        color_encoding = {"red": 0.2, "green": 0.4, "blue": 0.6, "gray": 0.8, "black": 0.0, "white": 1.0}
        vec.append(color_encoding.get(self.color_dominant, 0.5))

        return vec  # 24 features total


class ForwardModel:
    """
    Predicts next state features given current state and action.
    High prediction error = novel/surprising situation.
    """

    ACTIONS = ["move_forward", "move_backward", "rotate_left", "rotate_right", "stop"]

    def __init__(self, input_dim: int = 29, hidden_dim: int = 64, output_dim: int = 24):
        self.input_dim = input_dim  # state (24) + action (5)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W1 = [[random.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * hidden_dim
        self.W3 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b3 = [0.0] * output_dim

        self.learning_rate = 0.001

    def _relu(self, x: float) -> float:
        return max(0, x)

    def _action_encoding(self, action: str) -> List[float]:
        """One-hot encode action."""
        encoding = [0.0] * len(self.ACTIONS)
        if action in self.ACTIONS:
            encoding[self.ACTIONS.index(action)] = 1.0
        return encoding

    def predict(self, state: StateFeatures, action: str) -> List[float]:
        """Predict next state features."""
        # Combine state and action
        input_vec = state.to_vector() + self._action_encoding(action)

        # Ensure input dimension matches
        while len(input_vec) < self.input_dim:
            input_vec.append(0.0)
        input_vec = input_vec[:self.input_dim]

        # Forward pass
        # Layer 1
        h1 = []
        for j in range(self.hidden_dim):
            total = self.b1[j]
            for i in range(self.input_dim):
                total += self.W1[j][i] * input_vec[i]
            h1.append(self._relu(total))

        # Layer 2
        h2 = []
        for j in range(self.hidden_dim):
            total = self.b2[j]
            for i in range(self.hidden_dim):
                total += self.W2[j][i] * h1[i]
            h2.append(self._relu(total))

        # Output layer
        output = []
        for j in range(self.output_dim):
            total = self.b3[j]
            for i in range(self.hidden_dim):
                total += self.W3[j][i] * h2[i]
            output.append(total)

        return output

    def compute_prediction_error(self, state: StateFeatures, action: str,
                                   next_state: StateFeatures) -> float:
        """Compute prediction error (curiosity signal)."""
        predicted = self.predict(state, action)
        actual = next_state.to_vector()

        # Mean squared error
        error = 0.0
        for p, a in zip(predicted, actual):
            error += (p - a) ** 2
        return error / len(predicted)

    def train_step(self, state: StateFeatures, action: str, next_state: StateFeatures) -> float:
        """Train on a single transition, return loss."""
        input_vec = state.to_vector() + self._action_encoding(action)
        while len(input_vec) < self.input_dim:
            input_vec.append(0.0)
        input_vec = input_vec[:self.input_dim]

        target = next_state.to_vector()

        # Forward pass with stored activations
        h1 = []
        for j in range(self.hidden_dim):
            total = self.b1[j]
            for i in range(self.input_dim):
                total += self.W1[j][i] * input_vec[i]
            h1.append(self._relu(total))

        h2 = []
        for j in range(self.hidden_dim):
            total = self.b2[j]
            for i in range(self.hidden_dim):
                total += self.W2[j][i] * h1[i]
            h2.append(self._relu(total))

        output = []
        for j in range(self.output_dim):
            total = self.b3[j]
            for i in range(self.hidden_dim):
                total += self.W3[j][i] * h2[i]
            output.append(total)

        # Compute loss
        loss = 0.0
        d_output = []
        for j in range(self.output_dim):
            diff = output[j] - target[j]
            loss += diff ** 2
            d_output.append(2 * diff / self.output_dim)
        loss /= self.output_dim

        # Backprop through output layer
        d_h2 = [0.0] * self.hidden_dim
        for j in range(self.output_dim):
            for i in range(self.hidden_dim):
                d_h2[i] += self.W3[j][i] * d_output[j]
                self.W3[j][i] -= self.learning_rate * d_output[j] * h2[i]
            self.b3[j] -= self.learning_rate * d_output[j]

        # ReLU derivative
        d_h2 = [d * (1 if h2[i] > 0 else 0) for i, d in enumerate(d_h2)]

        # Backprop through hidden layer 2
        d_h1 = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            for i in range(self.hidden_dim):
                d_h1[i] += self.W2[j][i] * d_h2[j]
                self.W2[j][i] -= self.learning_rate * d_h2[j] * h1[i]
            self.b2[j] -= self.learning_rate * d_h2[j]

        # ReLU derivative
        d_h1 = [d * (1 if h1[i] > 0 else 0) for i, d in enumerate(d_h1)]

        # Backprop through input layer
        for j in range(self.hidden_dim):
            for i in range(self.input_dim):
                self.W1[j][i] -= self.learning_rate * d_h1[j] * input_vec[i]
            self.b1[j] -= self.learning_rate * d_h1[j]

        return loss


class InverseModel:
    """
    Predicts action given current and next state.
    Used to learn action-relevant features.
    """

    ACTIONS = ["move_forward", "move_backward", "rotate_left", "rotate_right", "stop"]

    def __init__(self, input_dim: int = 48, hidden_dim: int = 64):
        self.input_dim = input_dim  # state (24) + next_state (24)
        self.hidden_dim = hidden_dim
        self.num_actions = len(self.ACTIONS)

        # Initialize weights
        self.W1 = [[random.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(self.num_actions)]
        self.b2 = [0.0] * self.num_actions

        self.learning_rate = 0.001

    def _relu(self, x: float) -> float:
        return max(0, x)

    def _softmax(self, logits: List[float]) -> List[float]:
        max_val = max(logits)
        exp_vals = [math.exp(x - max_val) for x in logits]
        total = sum(exp_vals)
        return [e / total for e in exp_vals]

    def predict(self, state: StateFeatures, next_state: StateFeatures) -> Tuple[str, float]:
        """Predict action that caused transition."""
        input_vec = state.to_vector() + next_state.to_vector()
        while len(input_vec) < self.input_dim:
            input_vec.append(0.0)
        input_vec = input_vec[:self.input_dim]

        # Forward pass
        h1 = []
        for j in range(self.hidden_dim):
            total = self.b1[j]
            for i in range(self.input_dim):
                total += self.W1[j][i] * input_vec[i]
            h1.append(self._relu(total))

        logits = []
        for j in range(self.num_actions):
            total = self.b2[j]
            for i in range(self.hidden_dim):
                total += self.W2[j][i] * h1[i]
            logits.append(total)

        probs = self._softmax(logits)
        best_idx = probs.index(max(probs))
        return self.ACTIONS[best_idx], probs[best_idx]

    def train_step(self, state: StateFeatures, action: str, next_state: StateFeatures) -> float:
        """Train on a single transition, return loss."""
        input_vec = state.to_vector() + next_state.to_vector()
        while len(input_vec) < self.input_dim:
            input_vec.append(0.0)
        input_vec = input_vec[:self.input_dim]

        # Forward pass
        h1 = []
        for j in range(self.hidden_dim):
            total = self.b1[j]
            for i in range(self.input_dim):
                total += self.W1[j][i] * input_vec[i]
            h1.append(self._relu(total))

        logits = []
        for j in range(self.num_actions):
            total = self.b2[j]
            for i in range(self.hidden_dim):
                total += self.W2[j][i] * h1[i]
            logits.append(total)

        probs = self._softmax(logits)

        # Cross-entropy loss
        target_idx = self.ACTIONS.index(action) if action in self.ACTIONS else 0
        loss = -math.log(max(probs[target_idx], 1e-10))

        # Backprop
        d_logits = probs.copy()
        d_logits[target_idx] -= 1.0

        d_h1 = [0.0] * self.hidden_dim
        for j in range(self.num_actions):
            for i in range(self.hidden_dim):
                d_h1[i] += self.W2[j][i] * d_logits[j]
                self.W2[j][i] -= self.learning_rate * d_logits[j] * h1[i]
            self.b2[j] -= self.learning_rate * d_logits[j]

        d_h1 = [d * (1 if h1[i] > 0 else 0) for i, d in enumerate(d_h1)]

        for j in range(self.hidden_dim):
            for i in range(self.input_dim):
                self.W1[j][i] -= self.learning_rate * d_h1[j] * input_vec[i]
            self.b1[j] -= self.learning_rate * d_h1[j]

        return loss


class NoveltyBuffer:
    """Maintains a buffer of visited states for novelty computation."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states: List[List[float]] = []
        self.visit_counts: Dict[int, int] = {}  # hash -> count

    def _hash_state(self, state_vec: List[float], resolution: float = 0.1) -> int:
        """Discretize and hash state for counting."""
        discretized = tuple(int(v / resolution) for v in state_vec[:5])  # Use position features
        return hash(discretized)

    def add_state(self, state: StateFeatures) -> int:
        """Add state and return visit count."""
        vec = state.to_vector()
        state_hash = self._hash_state(vec)

        self.visit_counts[state_hash] = self.visit_counts.get(state_hash, 0) + 1

        if len(self.states) < self.capacity:
            self.states.append(vec)
        else:
            # Replace oldest
            self.states.pop(0)
            self.states.append(vec)

        return self.visit_counts[state_hash]

    def compute_novelty(self, state: StateFeatures) -> float:
        """Compute novelty score (0-1) based on visit count and distance."""
        if not self.states:
            return 1.0  # Maximum novelty for first state

        vec = state.to_vector()
        state_hash = self._hash_state(vec)

        # Count-based novelty
        count = self.visit_counts.get(state_hash, 0)
        count_novelty = 1.0 / (1.0 + count)

        # Distance-based novelty (average distance to k nearest)
        k = min(5, len(self.states))
        distances = []
        for stored in self.states:
            dist = sum((a - b) ** 2 for a, b in zip(vec, stored)) ** 0.5
            distances.append(dist)
        distances.sort()
        avg_dist = sum(distances[:k]) / k
        dist_novelty = 1.0 - math.exp(-avg_dist)

        # Combine
        return 0.5 * count_novelty + 0.5 * dist_novelty


class CuriosityModule:
    """
    Main curiosity-driven exploration module.
    Combines forward model, inverse model, and novelty detection.
    """

    def __init__(self, intrinsic_weight: float = 0.5, novelty_weight: float = 0.3):
        self.forward_model = ForwardModel()
        self.inverse_model = InverseModel()
        self.novelty_buffer = NoveltyBuffer()

        self.intrinsic_weight = intrinsic_weight  # Weight for prediction error
        self.novelty_weight = novelty_weight  # Weight for state novelty

        self.transition_buffer: List[Dict] = []
        self.total_intrinsic_reward = 0.0
        self.episode_count = 0

    def compute_intrinsic_reward(self, state: StateFeatures, action: str,
                                   next_state: StateFeatures) -> float:
        """
        Compute intrinsic reward for a transition.
        Higher reward for novel/surprising outcomes.
        """
        # Prediction error (surprise)
        pred_error = self.forward_model.compute_prediction_error(state, action, next_state)

        # Novelty of next state
        novelty = self.novelty_buffer.compute_novelty(next_state)

        # Add state to buffer
        self.novelty_buffer.add_state(next_state)

        # Combine rewards (normalized)
        intrinsic = self.intrinsic_weight * min(pred_error, 1.0) + \
                    self.novelty_weight * novelty

        return intrinsic

    def observe_transition(self, state: StateFeatures, action: str,
                           next_state: StateFeatures, extrinsic_reward: float = 0.0) -> float:
        """
        Observe a transition and compute total reward.
        Returns combined intrinsic + extrinsic reward.
        """
        intrinsic = self.compute_intrinsic_reward(state, action, next_state)
        self.total_intrinsic_reward += intrinsic

        # Store transition for batch training
        self.transition_buffer.append({
            "state": state,
            "action": action,
            "next_state": next_state,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic_reward
        })

        return intrinsic + extrinsic_reward

    def train_batch(self, batch_size: int = 32) -> Dict[str, float]:
        """Train models on buffered transitions."""
        if len(self.transition_buffer) < batch_size:
            return {"forward_loss": 0.0, "inverse_loss": 0.0}

        # Sample batch
        batch = random.sample(self.transition_buffer, batch_size)

        forward_loss = 0.0
        inverse_loss = 0.0

        for t in batch:
            forward_loss += self.forward_model.train_step(
                t["state"], t["action"], t["next_state"]
            )
            inverse_loss += self.inverse_model.train_step(
                t["state"], t["action"], t["next_state"]
            )

        return {
            "forward_loss": forward_loss / batch_size,
            "inverse_loss": inverse_loss / batch_size
        }

    def select_action(self, state: StateFeatures, available_actions: List[str],
                      exploration_rate: float = 0.3) -> Tuple[str, float]:
        """
        Select action based on curiosity.
        Prioritizes actions that would lead to novel states.
        """
        if random.random() < exploration_rate:
            # Random exploration
            return random.choice(available_actions), 0.0

        # Evaluate each action's potential for curiosity
        action_values = []
        for action in available_actions:
            # Predict next state
            predicted_features = self.forward_model.predict(state, action)

            # Estimate novelty of predicted state (simplified)
            predicted_state = StateFeatures()
            # Use predicted position features
            if len(predicted_features) >= 2:
                predicted_state.position = (predicted_features[0], predicted_features[1])

            novelty = self.novelty_buffer.compute_novelty(predicted_state)
            action_values.append((action, novelty))

        # Select action with highest expected novelty
        best_action, best_value = max(action_values, key=lambda x: x[1])
        return best_action, best_value

    def end_episode(self) -> Dict[str, Any]:
        """End current episode and return statistics."""
        self.episode_count += 1

        stats = {
            "episode": self.episode_count,
            "transitions": len(self.transition_buffer),
            "total_intrinsic_reward": self.total_intrinsic_reward,
            "avg_intrinsic_reward": self.total_intrinsic_reward / max(len(self.transition_buffer), 1),
            "unique_states": len(self.novelty_buffer.visit_counts),
            "buffer_size": len(self.novelty_buffer.states)
        }

        # Train on episode data
        if self.transition_buffer:
            train_stats = self.train_batch(min(64, len(self.transition_buffer)))
            stats.update(train_stats)

        # Reset for next episode
        self.total_intrinsic_reward = 0.0

        return stats

    def save(self, filepath: str):
        """Save curiosity module state."""
        data = {
            "episode_count": self.episode_count,
            "visit_counts": self.novelty_buffer.visit_counts,
            "buffer_size": len(self.novelty_buffer.states),
            "intrinsic_weight": self.intrinsic_weight,
            "novelty_weight": self.novelty_weight
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load curiosity module state."""
        with open(filepath) as f:
            data = json.load(f)
        self.episode_count = data.get("episode_count", 0)
        self.novelty_buffer.visit_counts = {
            int(k): v for k, v in data.get("visit_counts", {}).items()
        }


class CuriousExplorer:
    """
    High-level exploration policy that uses curiosity for autonomous learning.
    """

    ACTIONS = ["move_forward", "move_backward", "rotate_left", "rotate_right", "stop"]

    def __init__(self):
        self.curiosity = CuriosityModule()
        self.current_state: Optional[StateFeatures] = None
        self.exploration_history: List[Dict] = []

    def start_exploration(self, initial_state: StateFeatures):
        """Begin exploration from initial state."""
        self.current_state = initial_state
        self.curiosity.novelty_buffer.add_state(initial_state)

    def step(self, sensor_data: Dict) -> Tuple[str, float]:
        """
        Take one exploration step.
        Returns (action, expected_reward).
        """
        # Update current state from sensors
        new_state = StateFeatures(
            position=sensor_data.get("position", (0.0, 0.0)),
            orientation=sensor_data.get("orientation", 0.0),
            distances=sensor_data.get("distances", [1.0] * 8),
            detected_objects=sensor_data.get("objects", []),
            object_distances=sensor_data.get("object_distances", []),
            brightness=sensor_data.get("brightness", 0.5),
            color_dominant=sensor_data.get("dominant_color", "gray")
        )

        # If we have previous state, observe the transition
        if self.current_state is not None:
            # Determine what action caused this transition (from history)
            last_action = self.exploration_history[-1]["action"] if self.exploration_history else "stop"
            intrinsic_reward = self.curiosity.observe_transition(
                self.current_state, last_action, new_state
            )

        # Select next action
        action, expected_novelty = self.curiosity.select_action(new_state, self.ACTIONS)

        # Record
        self.exploration_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": new_state.to_vector()[:5],  # Just position for logging
            "action": action,
            "expected_novelty": expected_novelty
        })

        self.current_state = new_state
        return action, expected_novelty

    def end_exploration(self) -> Dict[str, Any]:
        """End exploration session and return statistics."""
        stats = self.curiosity.end_episode()
        stats["exploration_steps"] = len(self.exploration_history)

        self.exploration_history = []
        self.current_state = None

        return stats


def demo_curiosity_exploration():
    """Demonstrate curiosity-driven exploration."""
    print("Curiosity-Driven Exploration Demo")
    print("=" * 50)

    explorer = CuriousExplorer()

    # Simulate exploration in a simple environment
    print("\nSimulating 50 exploration steps...")

    # Start at origin
    position = [0.0, 0.0]
    orientation = 0.0

    initial_state = StateFeatures(position=tuple(position), orientation=orientation)
    explorer.start_exploration(initial_state)

    total_reward = 0.0
    action_counts = {}

    for step in range(50):
        # Generate sensor data based on position
        distances = [
            max(0.1, 1.0 - abs(math.sin(position[0] + position[1] + i)))
            for i in range(8)
        ]

        sensor_data = {
            "position": tuple(position),
            "orientation": orientation,
            "distances": distances,
            "objects": ["obstacle"] if distances[0] < 0.3 else [],
            "object_distances": [distances[0]] if distances[0] < 0.3 else [],
            "brightness": 0.5 + 0.2 * math.sin(step / 10),
            "dominant_color": "gray"
        }

        action, expected_novelty = explorer.step(sensor_data)
        action_counts[action] = action_counts.get(action, 0) + 1

        # Simulate action effects
        if action == "move_forward":
            position[0] += 0.1 * math.cos(orientation)
            position[1] += 0.1 * math.sin(orientation)
        elif action == "move_backward":
            position[0] -= 0.1 * math.cos(orientation)
            position[1] -= 0.1 * math.sin(orientation)
        elif action == "rotate_left":
            orientation += 0.3
        elif action == "rotate_right":
            orientation -= 0.3

        if step % 10 == 0:
            print(f"  Step {step}: at ({position[0]:.2f}, {position[1]:.2f}), action={action}")

    # End exploration
    stats = explorer.end_exploration()

    print("\nExploration Statistics:")
    print(f"  Total steps: {stats['exploration_steps']}")
    print(f"  Unique states visited: {stats['unique_states']}")
    print(f"  Average intrinsic reward: {stats['avg_intrinsic_reward']:.4f}")
    print(f"  Forward model loss: {stats.get('forward_loss', 0):.4f}")
    print(f"  Inverse model loss: {stats.get('inverse_loss', 0):.4f}")

    print("\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count} ({100*count/50:.1f}%)")

    return stats


if __name__ == "__main__":
    stats = demo_curiosity_exploration()
    print("\nDemo completed successfully!")
