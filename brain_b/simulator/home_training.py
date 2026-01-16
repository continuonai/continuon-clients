"""
Training Loop for 3D Home Navigation Action Prediction.

Uses RLDS episodes from home exploration to train next-action prediction models,
aligned with Brain A's slow loop training methodology.
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict
from pathlib import Path


@dataclass
class Home3DTrainingSample:
    """A single training sample: state -> action."""
    state_vector: List[float]
    action_index: int
    action_name: str
    reward: float
    episode_id: str
    frame_id: int
    room_type: str = ""
    goal_distance: float = -1.0


@dataclass
class Home3DTrainingMetrics:
    """Metrics from training."""
    loss: float
    accuracy: float
    samples_seen: int
    episodes_processed: int
    action_distribution: Dict[str, int] = field(default_factory=dict)


# 3D Home action vocabulary
HOME_ACTIONS = [
    "forward",
    "backward",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "interact",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(HOME_ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(HOME_ACTIONS)}

# Room type encoding
ROOM_TYPES = [
    "unknown",
    "living_room",
    "kitchen",
    "bedroom",
    "bathroom",
    "hallway",
    "garage",
    "office",
    "dining_room",
]
ROOM_TO_IDX = {r: i for i, r in enumerate(ROOM_TYPES)}

# Object type encoding (top items)
OBJECT_TYPES = [
    "wall", "door", "couch", "table", "chair", "bed", "desk", "shelf",
    "fridge", "stove", "sink", "tv", "lamp", "key", "remote", "phone", "book",
]
OBJ_TO_IDX = {o: i for i, o in enumerate(OBJECT_TYPES)}


class Home3DNavigationPredictor:
    """
    Action prediction model for 3D home navigation.

    Architecture: Linear layer with softmax
    Input: State embedding (48 dims)
    Output: Action probabilities (9 dims for HOME_ACTIONS)

    State vector components:
    - Position: 3 values (x, y, z normalized)
    - Rotation: 2 values (pitch, yaw normalized)
    - Room type: 9 values (one-hot)
    - Goal distance: 1 value (normalized)
    - Visible objects: 17 values (count of each type, normalized)
    - Inventory: 5 values (has key, has remote, etc.)
    - Battery: 1 value
    - Progress: 2 values (moves, level_complete flag)
    Total: 3 + 2 + 9 + 1 + 17 + 5 + 1 + 2 = 40 dims (padded to 48)
    """

    INPUT_DIM = 48
    NUM_ACTIONS = len(HOME_ACTIONS)

    def __init__(self, input_dim: int = 48, num_actions: int = 9):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self._loaded = False

        # Initialize weights with small random values
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(input_dim)]
            for _ in range(num_actions)
        ]

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
        for action_weights in self.weights:
            logit = sum(w * x for w, x in zip(action_weights, state_vector))
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        return probs

    def predict_action(self, state_vector: List[float]) -> tuple:
        """Predict most likely action."""
        probs = self.predict(state_vector)
        best_idx = probs.index(max(probs))
        return IDX_TO_ACTION[best_idx], probs[best_idx]

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
                "model_type": "home3d_navigation",
                "input_dim": self.input_dim,
                "num_actions": self.num_actions,
                "action_vocab": HOME_ACTIONS,
                "weights": self.weights,
                "learning_rate": self.learning_rate,
            }, f)

    def load(self, path) -> None:
        """Load model weights."""
        with open(str(path)) as f:
            data = json.load(f)

        self.input_dim = data["input_dim"]
        self.num_actions = data["num_actions"]
        self.weights = data["weights"]
        self.learning_rate = data.get("learning_rate", 0.01)
        self._loaded = True


class Home3DTrainingDataset:
    """
    Dataset of training samples from 3D home RLDS episodes.
    """

    def __init__(self, episodes_dir: str):
        self.episodes_dir = Path(episodes_dir)
        self.samples: List[Home3DTrainingSample] = []
        self.episodes_loaded: int = 0
        self.action_counts: Dict[str, int] = {a: 0 for a in HOME_ACTIONS}

    def load_episodes(self, max_episodes: Optional[int] = None) -> int:
        """Load episodes and extract training samples."""
        count = 0

        if not self.episodes_dir.exists():
            print(f"Episodes directory not found: {self.episodes_dir}")
            return 0

        for ep_dir in self.episodes_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            if max_episodes and count >= max_episodes:
                break

            # Try both file locations
            steps_file = ep_dir / "steps" / "000000.jsonl"
            if not steps_file.exists():
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
                    self.action_counts[sample.action_name] += 1

    def _step_to_sample(self, episode_id: str, step: dict) -> Optional[Home3DTrainingSample]:
        """Convert a step to a training sample."""
        action_data = step.get("action", {})
        action = action_data.get("command", "")

        # Normalize action names
        action_map = {
            "forward": "forward",
            "backward": "backward",
            "strafe_left": "strafe_left",
            "strafe_right": "strafe_right",
            "turn_left": "turn_left",
            "turn_right": "turn_right",
            "look_up": "look_up",
            "look_down": "look_down",
            "interact": "interact",
        }

        normalized_action = action_map.get(action)
        if normalized_action is None or normalized_action not in ACTION_TO_IDX:
            return None

        # Build state vector from observation
        obs = step.get("observation", {})
        state_vector = self._obs_to_vector(obs)

        room_type = obs.get("current_room", "unknown")
        goal_dist = obs.get("goal_distance", -1.0)

        return Home3DTrainingSample(
            state_vector=state_vector,
            action_index=ACTION_TO_IDX[normalized_action],
            action_name=normalized_action,
            reward=step.get("reward", 0.0),
            episode_id=episode_id,
            frame_id=step.get("frame_id", 0),
            room_type=room_type,
            goal_distance=goal_dist,
        )

    def _obs_to_vector(self, obs: dict) -> List[float]:
        """Convert observation to state vector."""
        vector = []

        # Position (3 values, normalized to 0-1 assuming 20x20x6 world)
        pos = obs.get("position_3d", {})
        vector.append(pos.get("x", 0) / 20.0)
        vector.append(pos.get("y", 0) / 20.0)
        vector.append(pos.get("z", 0) / 6.0)

        # Rotation (2 values: pitch -90 to 90, yaw 0 to 360)
        rot = obs.get("rotation_3d", {})
        vector.append((rot.get("pitch", 0) + 90) / 180.0)  # 0 to 1
        vector.append(rot.get("yaw", 0) / 360.0)  # 0 to 1

        # Room type one-hot (9 values)
        room = obs.get("current_room", "unknown")
        room_idx = ROOM_TO_IDX.get(room, 0)
        room_onehot = [0.0] * len(ROOM_TYPES)
        room_onehot[room_idx] = 1.0
        vector.extend(room_onehot)

        # Goal distance (1 value, normalized)
        goal_dist = obs.get("goal_distance", -1)
        if goal_dist < 0:
            vector.append(1.0)  # No goal = far
        else:
            vector.append(min(goal_dist / 20.0, 1.0))

        # Visible objects (17 values - count of each type, normalized)
        visible = obs.get("visible_objects", [])
        obj_counts = [0.0] * len(OBJECT_TYPES)
        for obj in visible:
            obj_type = obj.get("type", "")
            if obj_type in OBJ_TO_IDX:
                idx = OBJ_TO_IDX[obj_type]
                obj_counts[idx] += 1.0 / 5.0  # Normalize by max expected
        vector.extend(obj_counts)

        # Inventory (5 values)
        inventory = obs.get("inventory", [])
        has_key = 1.0 if "key" in inventory else 0.0
        has_remote = 1.0 if "remote" in inventory else 0.0
        has_phone = 1.0 if "phone" in inventory else 0.0
        has_book = 1.0 if "book" in inventory else 0.0
        inv_count = min(len(inventory) / 5.0, 1.0)
        vector.extend([has_key, has_remote, has_phone, has_book, inv_count])

        # Robot state
        robot = obs.get("robot_state", {})
        battery = robot.get("battery", 1.0)
        vector.append(battery)

        # Progress (2 values)
        moves = robot.get("moves", 0)
        vector.append(min(moves / 100.0, 1.0))
        vector.append(1.0 if obs.get("level_complete", False) else 0.0)

        # Pad to 48 dimensions
        while len(vector) < 48:
            vector.append(0.0)

        return vector[:48]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Home3DTrainingSample]:
        return iter(self.samples)

    def shuffle(self) -> None:
        """Shuffle samples."""
        random.shuffle(self.samples)

    def batches(self, batch_size: int) -> Iterator[List[Home3DTrainingSample]]:
        """Yield batches of samples."""
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i:i + batch_size]


class Home3DTrainer:
    """
    Trainer for 3D home navigation prediction model.
    """

    def __init__(
        self,
        model: Optional[Home3DNavigationPredictor] = None,
        checkpoint_dir: str = "./brain_b_data/home_checkpoints",
    ):
        self.model = model or Home3DNavigationPredictor()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.total_samples = 0
        self.total_episodes = 0
        self.history: List[Home3DTrainingMetrics] = []

    def train(
        self,
        dataset: Home3DTrainingDataset,
        epochs: int = 10,
        batch_size: int = 32,
        checkpoint_every: int = 100,
    ) -> Home3DTrainingMetrics:
        """
        Train the model on the dataset.
        Returns final metrics.
        """
        print(f"\n{'=' * 60}")
        print("  3D Home Navigation Training")
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
                    if self.total_samples % checkpoint_every == 0:
                        self._save_checkpoint()

            # Epoch metrics
            metrics = Home3DTrainingMetrics(
                loss=epoch_loss / max(samples_this_epoch, 1),
                accuracy=epoch_correct / max(samples_this_epoch, 1),
                samples_seen=self.total_samples,
                episodes_processed=dataset.episodes_loaded,
                action_distribution=dataset.action_counts.copy(),
            )
            self.history.append(metrics)

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"loss={metrics.loss:.4f}, "
                  f"acc={metrics.accuracy:.2%}")

        self.total_episodes = dataset.episodes_loaded
        return self.history[-1] if self.history else Home3DTrainingMetrics(0, 0, 0, 0)

    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"home3d_model_{self.total_samples}.json"
        self.model.save(str(path))
        return str(path)

    def evaluate(self, dataset: Home3DTrainingDataset) -> Home3DTrainingMetrics:
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
        return Home3DTrainingMetrics(
            loss=total_loss / max(n, 1),
            accuracy=correct / max(n, 1),
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


def run_home3d_training(
    episodes_dir: str = "./brain_b_data/home_rlds_episodes",
    output_dir: str = "./brain_b_data/home_models",
    epochs: int = 10,
    batch_size: int = 32,
) -> Home3DTrainingMetrics:
    """
    Run the full 3D home training pipeline.

    Args:
        episodes_dir: Directory containing RLDS episodes
        output_dir: Where to save model and metrics
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Final training metrics
    """
    print(f"Loading 3D home episodes from {episodes_dir}")
    dataset = Home3DTrainingDataset(episodes_dir)
    num_eps = dataset.load_episodes()
    print(f"Loaded {num_eps} episodes, {len(dataset)} samples")

    if len(dataset) == 0:
        print("No training data found!")
        return Home3DTrainingMetrics(0, 0, 0, 0)

    trainer = Home3DTrainer(checkpoint_dir=output_dir)
    metrics = trainer.train(dataset, epochs=epochs, batch_size=batch_size)

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output_path / "home3d_nav_model.json"))

    print(f"\n{'=' * 60}")
    print("  Training Complete!")
    print(f"{'=' * 60}")
    print(f"Final accuracy: {metrics.accuracy:.2%}")
    print(f"Model saved to: {output_path / 'home3d_nav_model.json'}")
    print(f"{'=' * 60}")

    return metrics


if __name__ == "__main__":
    import sys
    episodes_dir = sys.argv[1] if len(sys.argv) > 1 else "./brain_b_data/home_rlds_episodes"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./brain_b_data/home_models"
    run_home3d_training(episodes_dir, output_dir)
