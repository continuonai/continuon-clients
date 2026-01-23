"""
Simulator Training for Brain B.

Converts HomeScan simulator RLDS episodes to Brain B training data
and trains action prediction models for robot navigation.

Usage:
    from brain_b.simulator.simulator_training import (
        SimulatorTrainer,
        SimulatorDataset,
        run_simulator_training,
    )

    # Train from RLDS episodes
    metrics = run_simulator_training(
        episodes_dir="continuonbrain/rlds/episodes",
        output_dir="brain_b_data/simulator_models"
    )
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Any


# Simulator action vocabulary
SIMULATOR_ACTIONS = [
    "move_forward",
    "move_backward",
    "rotate_left",
    "rotate_right",
    "spawn_asset",
    "spawn_obstacle",
    "reset",
    "noop",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(SIMULATOR_ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(SIMULATOR_ACTIONS)}

# Asset types
ASSET_TYPES = ["crate", "pillar", "blob"]
ASSET_TO_IDX = {a: i for i, a in enumerate(ASSET_TYPES)}


@dataclass
class SimulatorTrainingSample:
    """A single training sample from simulator episode."""
    state_vector: List[float]
    action_index: int
    action_name: str
    reward: float
    episode_id: str
    step_idx: int
    robot_position: Dict[str, float] = field(default_factory=dict)
    robot_rotation: float = 0.0
    obstacle_count: int = 0


@dataclass
class SimulatorTrainingMetrics:
    """Metrics from simulator training."""
    loss: float
    accuracy: float
    samples_seen: int
    episodes_processed: int
    action_distribution: Dict[str, int] = field(default_factory=dict)


class SimulatorActionPredictor:
    """
    Action prediction model for simulator navigation.

    Architecture: Linear layer with softmax
    Input: State embedding (32 dims)
    Output: Action probabilities (8 dims for SIMULATOR_ACTIONS)

    State vector components:
    - Robot position: 3 values (x, y, z normalized)
    - Robot rotation: 1 value (normalized to 0-1)
    - Obstacle count: 1 value (normalized)
    - Recent actions: 8 values (one-hot of last action)
    - Step progress: 1 value (normalized)
    - Asset spawned recently: 1 value
    - Collision flag: 1 value
    - Padding: 16 values
    Total: 32 dims
    """

    INPUT_DIM = 32
    NUM_ACTIONS = len(SIMULATOR_ACTIONS)

    def __init__(self, input_dim: int = 32, num_actions: int = 8):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self._loaded = False

        # Initialize weights with small random values
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(input_dim)]
            for _ in range(num_actions)
        ]

        self.biases = [0.0 for _ in range(num_actions)]
        self.learning_rate = 0.01

    @property
    def is_ready(self) -> bool:
        """Check if model has been loaded/trained."""
        return self._loaded and len(self.weights) == self.num_actions

    def predict(self, state_vector: List[float]) -> List[float]:
        """Predict action probabilities given state."""
        # Ensure correct dimension
        if len(state_vector) < self.input_dim:
            state_vector = state_vector + [0.0] * (self.input_dim - len(state_vector))

        # Linear transformation
        logits = []
        for i, action_weights in enumerate(self.weights):
            logit = sum(w * x for w, x in zip(action_weights, state_vector))
            logit += self.biases[i]
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        return probs

    def predict_action(self, state_vector: List[float]) -> tuple:
        """Predict most likely action and confidence."""
        probs = self.predict(state_vector)
        best_idx = probs.index(max(probs))
        return IDX_TO_ACTION[best_idx], probs[best_idx]

    def predict_top_k(self, state_vector: List[float], k: int = 3) -> List[tuple]:
        """Predict top-k actions with probabilities."""
        probs = self.predict(state_vector)
        indexed = [(IDX_TO_ACTION[i], p) for i, p in enumerate(probs)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]

    def train_step(self, state_vector: List[float], target_action: int) -> float:
        """
        Single training step with gradient descent.
        Returns cross-entropy loss.
        """
        # Ensure correct dimension
        if len(state_vector) < self.input_dim:
            state_vector = state_vector + [0.0] * (self.input_dim - len(state_vector))

        probs = self.predict(state_vector)

        # Cross-entropy loss
        loss = -math.log(max(probs[target_action], 1e-10))

        # Gradient: softmax derivative
        gradients = probs.copy()
        gradients[target_action] -= 1.0

        # Update weights and biases
        for action_idx in range(self.num_actions):
            for dim in range(self.input_dim):
                self.weights[action_idx][dim] -= (
                    self.learning_rate * gradients[action_idx] * state_vector[dim]
                )
            self.biases[action_idx] -= self.learning_rate * gradients[action_idx]

        return loss

    def save(self, path: str) -> None:
        """Save model weights."""
        with open(path, "w") as f:
            json.dump({
                "model_type": "simulator_action_predictor",
                "version": "1.0",
                "input_dim": self.input_dim,
                "num_actions": self.num_actions,
                "action_vocab": SIMULATOR_ACTIONS,
                "weights": self.weights,
                "biases": self.biases,
                "learning_rate": self.learning_rate,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)

    def load(self, path: str) -> None:
        """Load model weights."""
        with open(path) as f:
            data = json.load(f)

        self.input_dim = data["input_dim"]
        self.num_actions = data["num_actions"]
        self.weights = data["weights"]
        self.biases = data.get("biases", [0.0] * self.num_actions)
        self.learning_rate = data.get("learning_rate", 0.01)
        self._loaded = True


class SimulatorDataset:
    """
    Dataset of training samples from HomeScan simulator RLDS episodes.
    """

    def __init__(self, episodes_dir: str):
        self.episodes_dir = Path(episodes_dir)
        self.samples: List[SimulatorTrainingSample] = []
        self.episodes_loaded: int = 0
        self.action_counts: Dict[str, int] = {a: 0 for a in SIMULATOR_ACTIONS}
        self._last_action: Optional[str] = None

    def load_episodes(self, max_episodes: Optional[int] = None, filter_prefix: str = "") -> int:
        """
        Load episodes and extract training samples.

        Args:
            max_episodes: Maximum episodes to load
            filter_prefix: Only load episodes with this prefix (e.g., "trainer_", "sim_")
                          If empty, loads trainer_, sim_, and home3d_ prefixed episodes
        """
        count = 0

        if not self.episodes_dir.exists():
            print(f"[SimulatorDataset] Episodes directory not found: {self.episodes_dir}")
            return 0

        # Default prefixes for simulator-compatible episodes
        valid_prefixes = [
            "trainer_", "sim_", "home3d_",
            # Training games prefixes
            "nav_", "puzzle_", "explore_", "interact_",
            "survive_", "collect_", "multi_",
        ]

        for ep_dir in sorted(self.episodes_dir.iterdir()):
            if not ep_dir.is_dir():
                continue

            # Apply prefix filter
            if filter_prefix:
                if not ep_dir.name.startswith(filter_prefix):
                    continue
            else:
                # If no filter, only load simulator-compatible episodes
                if not any(ep_dir.name.startswith(p) for p in valid_prefixes):
                    continue

            if max_episodes and count >= max_episodes:
                break

            # Try both file locations
            steps_file = ep_dir / "steps" / "000000.jsonl"
            if not steps_file.exists():
                steps_file = ep_dir / "steps.jsonl"
            if not steps_file.exists():
                continue

            loaded = self._load_episode(ep_dir.name, steps_file)
            if loaded > 0:
                count += 1

        self.episodes_loaded = count
        return count

    def _load_episode(self, episode_id: str, steps_file: Path) -> int:
        """Load a single episode into samples."""
        loaded = 0
        self._last_action = None

        with open(steps_file) as f:
            for line in f:
                try:
                    step = json.loads(line)
                    sample = self._step_to_sample(episode_id, step)
                    if sample:
                        self.samples.append(sample)
                        self.action_counts[sample.action_name] += 1
                        self._last_action = sample.action_name
                        loaded += 1
                except json.JSONDecodeError:
                    continue

        return loaded

    def _step_to_sample(self, episode_id: str, step: dict) -> Optional[SimulatorTrainingSample]:
        """Convert a step to a training sample."""
        action_data = step.get("action", {})
        action_type = action_data.get("type", "")

        # Map simulator actions to normalized names
        action_map = {
            "move": self._parse_move_action,
            "rotate": self._parse_rotate_action,
            "spawn_asset": lambda d: "spawn_asset",
            "spawn_obstacle": lambda d: "spawn_obstacle",
            "reset": lambda d: "reset",
            "sim_action": self._parse_sim_action,
        }

        normalized_action = None
        if action_type in action_map:
            normalized_action = action_map[action_type](action_data)

        # Also handle home3d and training_games format (uses "command" instead of "type")
        if normalized_action is None and "command" in action_data:
            command = action_data.get("command", "")
            # Map commands to normalized action names
            command_map = {
                # Home3D commands
                "forward": "move_forward",
                "backward": "move_backward",
                "strafe_left": "move_forward",
                "strafe_right": "move_forward",
                "turn_left": "rotate_left",
                "turn_right": "rotate_right",
                "interact": "noop",
                # GridWorld commands
                "left": "rotate_left",
                "right": "rotate_right",
                # Additional mappings
                "move_forward": "move_forward",
                "move_backward": "move_backward",
                "rotate_left": "rotate_left",
                "rotate_right": "rotate_right",
            }
            normalized_action = command_map.get(command)

        if normalized_action is None or normalized_action not in ACTION_TO_IDX:
            return None

        # Build state vector
        obs = step.get("observation", step.get("state", {}))
        state_vector = self._obs_to_vector(obs, action_data)

        # Extract position info
        robot_pos = obs.get("robot_position", {})
        if not robot_pos and "state" in step:
            state = step["state"]
            robot_pos = {"x": 0, "y": 0, "z": 0}

        return SimulatorTrainingSample(
            state_vector=state_vector,
            action_index=ACTION_TO_IDX[normalized_action],
            action_name=normalized_action,
            reward=step.get("reward", 0.0),
            episode_id=episode_id,
            step_idx=step.get("step_idx", 0),
            robot_position=robot_pos,
            robot_rotation=obs.get("robot_rotation", 0.0),
            obstacle_count=obs.get("obstacle_count", 0),
        )

    def _parse_move_action(self, action_data: dict) -> Optional[str]:
        """Parse move action to normalized name."""
        direction = action_data.get("direction", "")
        if direction == "forward":
            return "move_forward"
        elif direction == "backward":
            return "move_backward"
        return None

    def _parse_rotate_action(self, action_data: dict) -> Optional[str]:
        """Parse rotate action to normalized name."""
        direction = action_data.get("direction", "")
        if direction == "left":
            return "rotate_left"
        elif direction == "right":
            return "rotate_right"
        return None

    def _parse_sim_action(self, action_data: dict) -> Optional[str]:
        """Parse generic sim_action to normalized name."""
        # Check nested action type
        inner_type = action_data.get("type", "")
        if inner_type == "spawn_asset":
            return "spawn_asset"
        elif inner_type == "spawn_obstacle":
            return "spawn_obstacle"
        return None

    def _obs_to_vector(self, obs: dict, action_data: dict) -> List[float]:
        """Convert observation to state vector."""
        vector = []

        # Robot position (3 values, normalized to 0-1 assuming ±50 range)
        pos = obs.get("robot_position", {})
        vector.append((pos.get("x", 0) + 50) / 100.0)
        vector.append((pos.get("y", 0) + 50) / 100.0)
        vector.append((pos.get("z", 0) + 50) / 100.0)

        # Robot rotation (1 value, normalized to 0-1)
        rotation = obs.get("robot_rotation", 0.0)
        vector.append((rotation + math.pi) / (2 * math.pi))

        # Obstacle count (1 value, normalized)
        obstacle_count = obs.get("obstacle_count", 0)
        vector.append(min(obstacle_count / 10.0, 1.0))

        # Last action one-hot (8 values)
        last_action_onehot = [0.0] * len(SIMULATOR_ACTIONS)
        if self._last_action and self._last_action in ACTION_TO_IDX:
            last_action_onehot[ACTION_TO_IDX[self._last_action]] = 1.0
        vector.extend(last_action_onehot)

        # Step progress (1 value)
        step_idx = obs.get("step_count", 0)
        vector.append(min(step_idx / 100.0, 1.0))

        # Asset spawned flag (1 value)
        asset_type = action_data.get("asset_type")
        vector.append(1.0 if asset_type is not None else 0.0)

        # Collision flag (1 value)
        collision = obs.get("collision_detected", False)
        vector.append(1.0 if collision else 0.0)

        # Padding to 32 dimensions
        while len(vector) < 32:
            vector.append(0.0)

        return vector[:32]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[SimulatorTrainingSample]:
        return iter(self.samples)

    def shuffle(self) -> None:
        """Shuffle samples."""
        random.shuffle(self.samples)

    def batches(self, batch_size: int) -> Iterator[List[SimulatorTrainingSample]]:
        """Yield batches of samples."""
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i:i + batch_size]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "episodes": self.episodes_loaded,
            "samples": len(self.samples),
            "action_distribution": self.action_counts,
        }


class SimulatorTrainer:
    """
    Trainer for simulator action prediction model.
    """

    def __init__(
        self,
        model: Optional[SimulatorActionPredictor] = None,
        checkpoint_dir: str = "./brain_b_data/simulator_checkpoints",
    ):
        self.model = model or SimulatorActionPredictor()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.total_samples = 0
        self.total_episodes = 0
        self.history: List[SimulatorTrainingMetrics] = []

    def train(
        self,
        dataset: SimulatorDataset,
        epochs: int = 10,
        batch_size: int = 16,
        checkpoint_every: int = 50,
        verbose: bool = True,
    ) -> SimulatorTrainingMetrics:
        """
        Train the model on the dataset.
        Returns final metrics.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("  HomeScan Simulator Training - Brain B")
            print(f"{'=' * 60}")
            print(f"Episodes: {dataset.episodes_loaded}")
            print(f"Samples: {len(dataset)}")
            print(f"Actions: {dataset.action_counts}")
            print(f"{'=' * 60}\n")

        for epoch in range(epochs):
            dataset.shuffle()

            epoch_loss = 0.0
            epoch_correct = 0
            samples_this_epoch = 0

            for batch in dataset.batches(batch_size):
                for sample in batch:
                    # Train step
                    loss = self.model.train_step(
                        sample.state_vector,
                        sample.action_index
                    )
                    epoch_loss += loss

                    # Check accuracy
                    pred_action, _ = self.model.predict_action(sample.state_vector)
                    if pred_action == sample.action_name:
                        epoch_correct += 1

                    samples_this_epoch += 1
                    self.total_samples += 1

                    # Checkpoint
                    if checkpoint_every > 0 and self.total_samples % checkpoint_every == 0:
                        self._save_checkpoint()

            # Epoch metrics
            metrics = SimulatorTrainingMetrics(
                loss=epoch_loss / max(samples_this_epoch, 1),
                accuracy=epoch_correct / max(samples_this_epoch, 1),
                samples_seen=self.total_samples,
                episodes_processed=dataset.episodes_loaded,
                action_distribution=dataset.action_counts.copy(),
            )
            self.history.append(metrics)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"loss={metrics.loss:.4f}, "
                      f"acc={metrics.accuracy:.2%}")

        self.total_episodes = dataset.episodes_loaded
        self.model._loaded = True
        return self.history[-1] if self.history else SimulatorTrainingMetrics(0, 0, 0, 0)

    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"sim_model_{self.total_samples}.json"
        self.model.save(str(path))
        return str(path)

    def evaluate(self, dataset: SimulatorDataset) -> SimulatorTrainingMetrics:
        """Evaluate model on dataset without training."""
        total_loss = 0.0
        correct = 0

        for sample in dataset:
            probs = self.model.predict(sample.state_vector)
            loss = -math.log(max(probs[sample.action_index], 1e-10))
            total_loss += loss

            pred_action, _ = self.model.predict_action(sample.state_vector)
            if pred_action == sample.action_name:
                correct += 1

        n = len(dataset)
        return SimulatorTrainingMetrics(
            loss=total_loss / max(n, 1),
            accuracy=correct / max(n, 1),
            samples_seen=n,
            episodes_processed=dataset.episodes_loaded,
        )

    def save(self, path: str) -> None:
        """Save trainer state and model."""
        self.model.save(path)

        # Save metadata
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "total_samples": self.total_samples,
                "total_episodes": self.total_episodes,
                "saved_at": datetime.now().isoformat(),
                "history": [
                    {
                        "loss": m.loss,
                        "accuracy": m.accuracy,
                        "samples_seen": m.samples_seen,
                        "episodes_processed": m.episodes_processed,
                    }
                    for m in self.history
                ],
            }, f, indent=2)

    def load(self, path: str) -> None:
        """Load trainer state and model."""
        self.model.load(path)

        meta_path = path.replace(".json", "_meta.json")
        if Path(meta_path).exists():
            with open(meta_path) as f:
                data = json.load(f)
            self.total_samples = data.get("total_samples", 0)
            self.total_episodes = data.get("total_episodes", 0)


def run_simulator_training(
    episodes_dir: str = "../continuonbrain/rlds/episodes",
    output_dir: str = "./brain_b_data/simulator_models",
    epochs: int = 10,
    batch_size: int = 16,
    filter_prefix: str = "trainer_",
) -> SimulatorTrainingMetrics:
    """
    Run the full simulator training pipeline.

    Args:
        episodes_dir: Directory containing RLDS episodes
        output_dir: Where to save model and metrics
        epochs: Number of training epochs
        batch_size: Training batch size
        filter_prefix: Only train on episodes with this prefix

    Returns:
        Final training metrics
    """
    print(f"[SimulatorTraining] Loading episodes from {episodes_dir}")

    dataset = SimulatorDataset(episodes_dir)
    num_eps = dataset.load_episodes(filter_prefix=filter_prefix)

    print(f"[SimulatorTraining] Loaded {num_eps} episodes, {len(dataset)} samples")

    if len(dataset) == 0:
        print("[SimulatorTraining] No training data found!")
        return SimulatorTrainingMetrics(0, 0, 0, 0)

    trainer = SimulatorTrainer(checkpoint_dir=output_dir)
    metrics = trainer.train(dataset, epochs=epochs, batch_size=batch_size)

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "simulator_action_model.json"
    trainer.save(str(model_path))

    print(f"\n{'=' * 60}")
    print("  Simulator Training Complete!")
    print(f"{'=' * 60}")
    print(f"Final accuracy: {metrics.accuracy:.2%}")
    print(f"Model saved to: {model_path}")
    print(f"{'=' * 60}")

    return metrics


# Singleton predictor for inference
_simulator_predictor: Optional[SimulatorActionPredictor] = None


def get_simulator_predictor(model_path: Optional[str] = None) -> SimulatorActionPredictor:
    """Get or create the simulator predictor singleton."""
    global _simulator_predictor

    if _simulator_predictor is None:
        _simulator_predictor = SimulatorActionPredictor()

        if model_path and Path(model_path).exists():
            _simulator_predictor.load(model_path)

    return _simulator_predictor


if __name__ == "__main__":
    import sys

    episodes_dir = sys.argv[1] if len(sys.argv) > 1 else "../continuonbrain/rlds/episodes"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./brain_b_data/simulator_models"

    run_simulator_training(episodes_dir, output_dir)
