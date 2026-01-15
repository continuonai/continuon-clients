"""
Training Loop for RobotGrid Action Prediction.

Uses RLDS episodes to train next-action prediction models,
aligned with Brain A's slow loop training methodology.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Optional, Iterator
from pathlib import Path

from simulator.semantic_search import StateEmbedder


@dataclass
class TrainingSample:
    """A single training sample: state -> action."""
    state_vector: list[float]
    action_index: int
    action_name: str
    reward: float
    surprise: float
    episode_id: str
    frame_id: int


@dataclass
class TrainingMetrics:
    """Metrics from training."""
    loss: float
    accuracy: float
    mean_surprise: float
    samples_seen: int
    episodes_processed: int


# Action vocabulary
ACTIONS = ["forward", "backward", "left", "right", "look", "wait"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


class ActionPredictor:
    """
    Simple action prediction model.

    Architecture: Linear layer with softmax
    Input: State embedding (31 dims)
    Output: Action probabilities (6 dims)

    This is intentionally simple - can be replaced with
    neural network from Brain A training.
    """

    def __init__(self, input_dim: int = 31, num_actions: int = 6):
        self.input_dim = input_dim
        self.num_actions = num_actions

        # Simple linear weights (no bias for simplicity)
        # Initialize with small random values
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(input_dim)]
            for _ in range(num_actions)
        ]

        self.learning_rate = 0.01

    def predict(self, state_vector: list[float]) -> list[float]:
        """Predict action probabilities given state."""
        # Linear transformation
        logits = []
        for action_weights in self.weights:
            logit = sum(w * x for w, x in zip(action_weights, state_vector))
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        return probs

    def predict_action(self, state_vector: list[float]) -> tuple[str, float]:
        """Predict most likely action."""
        probs = self.predict(state_vector)
        best_idx = probs.index(max(probs))
        return IDX_TO_ACTION[best_idx], probs[best_idx]

    def train_step(self, state_vector: list[float], target_action: int) -> float:
        """
        Single training step with gradient descent.

        Returns cross-entropy loss.
        """
        probs = self.predict(state_vector)

        # Cross-entropy loss
        loss = -math.log(max(probs[target_action], 1e-10))

        # Gradient: softmax derivative
        # d_loss/d_logit = prob - (1 if target else 0)
        gradients = probs.copy()
        gradients[target_action] -= 1.0

        # Update weights
        for action_idx in range(self.num_actions):
            for dim in range(self.input_dim):
                self.weights[action_idx][dim] -= (
                    self.learning_rate * gradients[action_idx] * state_vector[dim]
                )

        return loss

    def save(self, path: str) -> None:
        """Save model weights."""
        with open(path, "w") as f:
            json.dump({
                "input_dim": self.input_dim,
                "num_actions": self.num_actions,
                "weights": self.weights,
                "learning_rate": self.learning_rate,
            }, f)

    def load(self, path: str) -> None:
        """Load model weights."""
        with open(path) as f:
            data = json.load(f)

        self.input_dim = data["input_dim"]
        self.num_actions = data["num_actions"]
        self.weights = data["weights"]
        self.learning_rate = data.get("learning_rate", 0.01)


class TrainingDataset:
    """
    Dataset of training samples from RLDS episodes.

    Loads episodes and converts to training samples.
    """

    def __init__(self, episodes_dir: str):
        self.episodes_dir = Path(episodes_dir)
        self.embedder = StateEmbedder()
        self.samples: list[TrainingSample] = []
        self.episodes_loaded: int = 0

    def load_episodes(self, max_episodes: Optional[int] = None) -> int:
        """Load episodes and extract training samples."""
        count = 0

        for ep_dir in self.episodes_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            if max_episodes and count >= max_episodes:
                break

            steps_file = ep_dir / "steps.jsonl"
            if not steps_file.exists():
                continue

            self._load_episode(ep_dir.name, steps_file)
            count += 1

        self.episodes_loaded = count
        return count

    def _load_episode(self, episode_id: str, steps_file: Path) -> None:
        """Load a single episode into samples."""
        with open(steps_file) as f:
            for line in f:
                step = json.loads(line)
                sample = self._step_to_sample(episode_id, step)
                if sample:
                    self.samples.append(sample)

    def _step_to_sample(self, episode_id: str, step: dict) -> Optional[TrainingSample]:
        """Convert a step to a training sample."""
        action = step["action"]["command"]
        if action not in ACTION_TO_IDX:
            return None

        # Build state vector from observation
        obs = step["observation"]
        robot = obs["robot_state"]

        # Reconstruct partial state vector
        # Note: Full reconstruction requires grid state, which we don't store
        # This is a simplified version using available data
        state_vector = self._obs_to_vector(obs)

        return TrainingSample(
            state_vector=state_vector,
            action_index=ACTION_TO_IDX[action],
            action_name=action,
            reward=step.get("reward", 0.0),
            surprise=step.get("world_model", {}).get("surprise", 0.0),
            episode_id=episode_id,
            frame_id=step["frame_id"],
        )

    def _obs_to_vector(self, obs: dict) -> list[float]:
        """Convert observation to state vector."""
        robot = obs["robot_state"]
        visible = obs.get("visible_tiles", [])
        look = obs.get("look_ahead", {})

        vector = []

        # Position (normalized, assume 20x20 grid)
        vector.append(robot["x"] / 20.0)
        vector.append(robot["y"] / 20.0)

        # Direction one-hot
        dir_map = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
        dir_onehot = [0.0, 0.0, 0.0, 0.0]
        dir_idx = dir_map.get(robot["direction"], 0)
        dir_onehot[dir_idx] = 1.0
        vector.extend(dir_onehot)

        # Inventory
        inv = robot.get("inventory", [])
        has_key = 1.0 if "key" in inv else 0.0
        vector.extend([has_key, min(len(inv) / 3.0, 1.0), 1.0 if inv else 0.0, min(len(inv) / 5.0, 1.0)])

        # Visible tiles (9 values)
        if len(visible) >= 9:
            for tile in visible[:9]:
                if tile.get("walkable"):
                    vector.append(1.0)
                elif tile.get("restricted"):
                    vector.append(0.5)
                else:
                    vector.append(0.0)
        else:
            vector.extend([0.5] * 9)  # Unknown = middle value

        # Progress (simplified)
        vector.extend([0.5, robot.get("moves", 0) / 100.0, 0.5, 0.5])

        # Flags (from look_ahead)
        tile_ahead = look.get("tile", "FLOOR")
        flags = [
            1.0 if tile_ahead == "WALL" else 0.0,
            1.0 if tile_ahead == "LAVA" else 0.0,
            1.0 if tile_ahead == "KEY" else 0.0,
            1.0 if tile_ahead == "DOOR" else 0.0,
            1.0 if tile_ahead == "GOAL" else 0.0,
            1.0 if tile_ahead == "BOX" else 0.0,
            0.0,  # Near lava (unknown)
            0.0,  # In corner (unknown)
        ]
        vector.extend(flags)

        return vector

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self.samples)

    def shuffle(self) -> None:
        """Shuffle samples."""
        random.shuffle(self.samples)

    def batches(self, batch_size: int) -> Iterator[list[TrainingSample]]:
        """Yield batches of samples."""
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i:i + batch_size]


class Trainer:
    """
    Trainer for action prediction model.

    Implements slow loop training methodology:
    - Loads curated RLDS episodes
    - Trains prediction model
    - Tracks surprise metrics
    - Supports checkpointing
    """

    def __init__(
        self,
        model: Optional[ActionPredictor] = None,
        checkpoint_dir: str = "./brain_b_data/checkpoints",
    ):
        self.model = model or ActionPredictor()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.total_samples = 0
        self.total_episodes = 0
        self.history: list[TrainingMetrics] = []

    def train(
        self,
        dataset: TrainingDataset,
        epochs: int = 10,
        batch_size: int = 32,
        checkpoint_every: int = 100,
    ) -> TrainingMetrics:
        """
        Train the model on the dataset.

        Returns final metrics.
        """
        for epoch in range(epochs):
            dataset.shuffle()

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_surprise = 0.0
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

                    epoch_surprise += sample.surprise
                    samples_this_epoch += 1
                    self.total_samples += 1

                    # Checkpoint
                    if self.total_samples % checkpoint_every == 0:
                        self._save_checkpoint()

            # Epoch metrics
            metrics = TrainingMetrics(
                loss=epoch_loss / max(samples_this_epoch, 1),
                accuracy=epoch_correct / max(samples_this_epoch, 1),
                mean_surprise=epoch_surprise / max(samples_this_epoch, 1),
                samples_seen=self.total_samples,
                episodes_processed=dataset.episodes_loaded,
            )
            self.history.append(metrics)

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"loss={metrics.loss:.4f}, "
                  f"acc={metrics.accuracy:.4f}, "
                  f"surprise={metrics.mean_surprise:.4f}")

        self.total_episodes = dataset.episodes_loaded
        return self.history[-1] if self.history else TrainingMetrics(0, 0, 0, 0, 0)

    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"model_{self.total_samples}.json"
        self.model.save(str(path))
        return str(path)

    def evaluate(self, dataset: TrainingDataset) -> TrainingMetrics:
        """Evaluate model on dataset without training."""
        total_loss = 0.0
        correct = 0
        total_surprise = 0.0

        for sample in dataset:
            probs = self.model.predict(sample.state_vector)
            loss = -math.log(max(probs[sample.action_index], 1e-10))
            total_loss += loss

            pred_action, _ = self.model.predict_action(sample.state_vector)
            if pred_action == sample.action_name:
                correct += 1

            total_surprise += sample.surprise

        n = len(dataset)
        return TrainingMetrics(
            loss=total_loss / max(n, 1),
            accuracy=correct / max(n, 1),
            mean_surprise=total_surprise / max(n, 1),
            samples_seen=n,
            episodes_processed=dataset.episodes_loaded,
        )

    def save(self, path: str) -> None:
        """Save trainer state."""
        self.model.save(path)
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "total_samples": self.total_samples,
                "total_episodes": self.total_episodes,
                "history": [
                    {
                        "loss": m.loss,
                        "accuracy": m.accuracy,
                        "mean_surprise": m.mean_surprise,
                        "samples_seen": m.samples_seen,
                    }
                    for m in self.history
                ],
            }, f)

    def load(self, path: str) -> None:
        """Load trainer state."""
        self.model.load(path)
        meta_path = path.replace(".json", "_meta.json")
        if Path(meta_path).exists():
            with open(meta_path) as f:
                data = json.load(f)
            self.total_samples = data.get("total_samples", 0)
            self.total_episodes = data.get("total_episodes", 0)


def run_training(
    episodes_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
) -> TrainingMetrics:
    """
    Run the full training pipeline.

    Args:
        episodes_dir: Directory containing RLDS episodes
        output_dir: Where to save model and metrics
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Final training metrics
    """
    print(f"Loading episodes from {episodes_dir}")
    dataset = TrainingDataset(episodes_dir)
    num_eps = dataset.load_episodes()
    print(f"Loaded {num_eps} episodes, {len(dataset)} samples")

    if len(dataset) == 0:
        print("No training data found!")
        return TrainingMetrics(0, 0, 0, 0, 0)

    trainer = Trainer(checkpoint_dir=output_dir)
    metrics = trainer.train(dataset, epochs=epochs, batch_size=batch_size)

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path / "final_model.json"))

    print(f"\nTraining complete!")
    print(f"Final accuracy: {metrics.accuracy:.2%}")
    print(f"Model saved to {output_path / 'final_model.json'}")

    return metrics


if __name__ == "__main__":
    import sys
    episodes_dir = sys.argv[1] if len(sys.argv) > 1 else "./brain_b_data/rlds_episodes"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./brain_b_data/models"
    run_training(episodes_dir, output_dir)
