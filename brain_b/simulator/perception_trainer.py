#!/usr/bin/env python3
"""
Perception-Aware Navigation Trainer

Trains navigation models using high-fidelity perception data:
- RGB image features
- Depth map features
- Semantic segmentation features
- LiDAR range features
- Object detection features

This creates models that understand real-world visual complexity.

Usage:
    python brain_b/simulator/perception_trainer.py --train
    python brain_b/simulator/perception_trainer.py --evaluate
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


# Action vocabulary (same as simulator_training)
ACTIONS = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "pick_up",
    "put_down",
    "toggle",
    "scan",
    "noop",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


@dataclass
class PerceptionFeatures:
    """Extracted features from perception data."""
    # RGB features (from spatial pooling)
    rgb_color_hist: List[float] = field(default_factory=list)  # 16-bin histogram
    rgb_edge_density: float = 0.0  # Edge detection score
    rgb_brightness: float = 0.0  # Average brightness

    # Depth features
    depth_mean: float = 0.0
    depth_min: float = 0.0
    depth_max: float = 0.0
    depth_variance: float = 0.0
    obstacle_ahead: float = 0.0  # Proximity to obstacle

    # Semantic features
    floor_ratio: float = 0.0  # How much floor visible
    wall_ratio: float = 0.0  # How much wall visible
    object_ratio: float = 0.0  # How much objects visible
    graspable_visible: int = 0  # Number of graspable objects

    # LiDAR features (8-sector)
    lidar_sectors: List[float] = field(default_factory=list)

    # Detection features
    num_objects: int = 0
    nearest_object_dist: float = 10.0
    nearest_graspable_dist: float = 10.0

    def to_vector(self) -> List[float]:
        """Convert to flat feature vector."""
        vec = []
        vec.extend(self.rgb_color_hist[:16] if self.rgb_color_hist else [0.0]*16)
        vec.append(self.rgb_edge_density)
        vec.append(self.rgb_brightness)
        vec.append(self.depth_mean / 10.0)  # Normalize
        vec.append(self.depth_min / 10.0)
        vec.append(self.depth_max / 10.0)
        vec.append(self.depth_variance)
        vec.append(self.obstacle_ahead)
        vec.append(self.floor_ratio)
        vec.append(self.wall_ratio)
        vec.append(self.object_ratio)
        vec.append(float(self.graspable_visible) / 10.0)
        vec.extend(self.lidar_sectors[:8] if self.lidar_sectors else [1.0]*8)
        vec.append(float(self.num_objects) / 10.0)
        vec.append(self.nearest_object_dist / 10.0)
        vec.append(self.nearest_graspable_dist / 10.0)
        # Pad to 40 dimensions
        while len(vec) < 40:
            vec.append(0.0)
        return vec[:40]  # 40 dimensions


class PerceptionFeatureExtractor:
    """Extracts features from perception data for training."""

    def __init__(self):
        self.feature_dim = 40

    def extract_from_rgb(self, rgb_data: np.ndarray) -> Dict[str, float]:
        """Extract features from RGB image."""
        if rgb_data is None or rgb_data.size == 0:
            return {
                "color_hist": [0.0] * 16,
                "edge_density": 0.0,
                "brightness": 0.0,
            }

        # Color histogram (simplified)
        gray = np.mean(rgb_data, axis=2)
        hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 256))
        color_hist = (hist / hist.sum()).tolist()

        # Brightness
        brightness = np.mean(gray) / 255.0

        # Edge density (simple gradient)
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edge_density = (np.mean(gx) + np.mean(gy)) / 255.0

        return {
            "color_hist": color_hist,
            "edge_density": edge_density,
            "brightness": brightness,
        }

    def extract_from_depth(self, depth_data: np.ndarray) -> Dict[str, float]:
        """Extract features from depth map."""
        if depth_data is None or depth_data.size == 0:
            return {
                "mean": 5.0,
                "min": 0.1,
                "max": 10.0,
                "variance": 0.0,
                "obstacle_ahead": 0.0,
            }

        # Central region for obstacle detection
        h, w = depth_data.shape
        center = depth_data[h//3:2*h//3, w//4:3*w//4]

        return {
            "mean": float(np.mean(depth_data)),
            "min": float(np.min(depth_data)),
            "max": float(np.max(depth_data)),
            "variance": float(np.var(depth_data)),
            "obstacle_ahead": float(np.min(center) < 1.0),  # 1 if close obstacle
        }

    def extract_from_semantic(self, labels: np.ndarray) -> Dict[str, float]:
        """Extract features from semantic segmentation."""
        if labels is None or labels.size == 0:
            return {
                "floor_ratio": 0.3,
                "wall_ratio": 0.3,
                "object_ratio": 0.1,
            }

        total = labels.size

        # Count class pixels (using standard class IDs)
        floor_ratio = np.sum(labels == 1) / total  # floor class
        wall_ratio = np.sum(labels == 2) / total  # wall class
        object_ratio = np.sum((labels >= 6) & (labels <= 20)) / total  # objects

        return {
            "floor_ratio": float(floor_ratio),
            "wall_ratio": float(wall_ratio),
            "object_ratio": float(object_ratio),
        }

    def extract_from_lidar(self, ranges: np.ndarray, angles: np.ndarray) -> List[float]:
        """Extract 8-sector LiDAR features."""
        if ranges is None or len(ranges) == 0:
            return [10.0] * 8

        # Divide into 8 sectors (45 degrees each)
        sector_ranges = [[] for _ in range(8)]

        for i, (angle, dist) in enumerate(zip(angles, ranges)):
            # Map angle to sector
            sector = int((angle + math.pi) / (math.pi / 4)) % 8
            sector_ranges[sector].append(dist)

        # Get minimum range per sector
        sector_mins = []
        for sr in sector_ranges:
            if sr:
                sector_mins.append(min(sr))
            else:
                sector_mins.append(10.0)  # Max range

        return sector_mins

    def extract_from_detections(self, detections: List[Dict]) -> Dict[str, float]:
        """Extract features from object detections."""
        if not detections:
            return {
                "num_objects": 0,
                "nearest_object_dist": 10.0,
                "nearest_graspable_dist": 10.0,
                "graspable_count": 0,
            }

        nearest_dist = 10.0
        nearest_graspable = 10.0
        graspable_count = 0

        for det in detections:
            pos = det.get("position_3d", [10, 0, 0])
            dist = math.sqrt(pos[0]**2 + pos[1]**2) if pos else 10.0

            nearest_dist = min(nearest_dist, dist)

            if det.get("graspable", False):
                nearest_graspable = min(nearest_graspable, dist)
                graspable_count += 1

        return {
            "num_objects": len(detections),
            "nearest_object_dist": nearest_dist,
            "nearest_graspable_dist": nearest_graspable,
            "graspable_count": graspable_count,
        }

    def extract(
        self,
        rgb: np.ndarray = None,
        depth: np.ndarray = None,
        semantic: np.ndarray = None,
        lidar_ranges: np.ndarray = None,
        lidar_angles: np.ndarray = None,
        detections: List[Dict] = None,
    ) -> PerceptionFeatures:
        """Extract all features from perception data."""
        rgb_feat = self.extract_from_rgb(rgb)
        depth_feat = self.extract_from_depth(depth)
        sem_feat = self.extract_from_semantic(semantic)
        lidar_feat = self.extract_from_lidar(lidar_ranges, lidar_angles)
        det_feat = self.extract_from_detections(detections or [])

        return PerceptionFeatures(
            rgb_color_hist=rgb_feat["color_hist"],
            rgb_edge_density=rgb_feat["edge_density"],
            rgb_brightness=rgb_feat["brightness"],
            depth_mean=depth_feat["mean"],
            depth_min=depth_feat["min"],
            depth_max=depth_feat["max"],
            depth_variance=depth_feat["variance"],
            obstacle_ahead=depth_feat["obstacle_ahead"],
            floor_ratio=sem_feat["floor_ratio"],
            wall_ratio=sem_feat["wall_ratio"],
            object_ratio=sem_feat["object_ratio"],
            graspable_visible=det_feat["graspable_count"],
            lidar_sectors=lidar_feat,
            num_objects=det_feat["num_objects"],
            nearest_object_dist=det_feat["nearest_object_dist"],
            nearest_graspable_dist=det_feat["nearest_graspable_dist"],
        )


@dataclass
class PerceptionTrainingSample:
    """Training sample with perception features."""
    features: PerceptionFeatures
    action_idx: int
    action_name: str
    reward: float
    episode_id: str
    step_idx: int


class PerceptionActionPredictor:
    """
    Neural-network-like action predictor using perception features.

    Architecture: Two-layer MLP with softmax output
    Input: Perception features (40 dims)
    Hidden: 64 neurons
    Output: Action probabilities (9 actions)
    """

    INPUT_DIM = 40
    HIDDEN_DIM = 64
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self):
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(self.INPUT_DIM, self.HIDDEN_DIM) * 0.1
        self.b1 = np.zeros(self.HIDDEN_DIM)
        self.W2 = np.random.randn(self.HIDDEN_DIM, self.NUM_ACTIONS) * 0.1
        self.b2 = np.zeros(self.NUM_ACTIONS)

        self._loaded = False
        self.is_ready = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Hidden layer with ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)
        # Output with softmax
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def predict(self, features: PerceptionFeatures) -> Tuple[int, float, List[float]]:
        """Predict action from perception features."""
        x = np.array(features.to_vector())
        probs = self.forward(x)
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])
        return action_idx, confidence, probs.tolist()

    def train_step(
        self,
        features: PerceptionFeatures,
        target_action: int,
        learning_rate: float = 0.01,
    ) -> float:
        """Single training step with backpropagation."""
        x = np.array(features.to_vector())

        # Forward pass
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Cross-entropy loss
        loss = -np.log(probs[target_action] + 1e-10)

        # Backward pass
        d_logits = probs.copy()
        d_logits[target_action] -= 1

        # Gradients for output layer
        d_W2 = np.outer(h, d_logits)
        d_b2 = d_logits

        # Backprop through hidden layer
        d_h = d_logits @ self.W2.T
        d_h = d_h * (h > 0)  # ReLU gradient

        # Gradients for hidden layer
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h

        # Update weights
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2

        self.is_ready = True
        return loss

    def save(self, filepath: str):
        """Save model weights."""
        data = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "input_dim": self.INPUT_DIM,
            "hidden_dim": self.HIDDEN_DIM,
            "num_actions": self.NUM_ACTIONS,
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
        self._loaded = True
        self.is_ready = True


class PerceptionTrainer:
    """Trains navigation model from high-fidelity perception data."""

    def __init__(self, data_dir: str = "brain_b_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = PerceptionFeatureExtractor()
        self.predictor = PerceptionActionPredictor()

        self.checkpoint_dir = self.data_dir / "perception_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.samples: List[PerceptionTrainingSample] = []

    def load_episodes(self, episodes_dir: str) -> List[Dict]:
        """Load high-fidelity episodes from directory."""
        episodes_path = Path(episodes_dir)
        episodes = []

        if not episodes_path.exists():
            print(f"Episodes directory not found: {episodes_dir}")
            return episodes

        # Find batch directories
        for batch_dir in sorted(episodes_path.glob("high_fidelity_batch_*")):
            for ep_dir in sorted(batch_dir.glob("hf_episode_*")):
                episode = self._load_episode(ep_dir)
                if episode:
                    episodes.append(episode)

        print(f"Loaded {len(episodes)} high-fidelity episodes")
        return episodes

    def _load_episode(self, ep_dir: Path) -> Optional[Dict]:
        """Load single episode with perception data."""
        episode_file = ep_dir / "episode.json"
        if not episode_file.exists():
            return None

        with open(episode_file, 'r') as f:
            episode = json.load(f)

        # Load perception data if available
        rgb_file = ep_dir / "first_rgb.npy"
        depth_file = ep_dir / "first_depth.npy"
        semantic_file = ep_dir / "first_semantic.npy"
        lidar_ranges_file = ep_dir / "first_lidar_ranges.npy"
        lidar_angles_file = ep_dir / "first_lidar_angles.npy"

        perception_data = {}

        if rgb_file.exists():
            perception_data["rgb"] = np.load(rgb_file)
        if depth_file.exists():
            perception_data["depth"] = np.load(depth_file)
        if semantic_file.exists():
            perception_data["semantic"] = np.load(semantic_file)
        if lidar_ranges_file.exists():
            perception_data["lidar_ranges"] = np.load(lidar_ranges_file)
        if lidar_angles_file.exists():
            perception_data["lidar_angles"] = np.load(lidar_angles_file)

        episode["perception_data"] = perception_data
        return episode

    def prepare_samples(self, episodes: List[Dict]) -> int:
        """Convert episodes to training samples."""
        self.samples = []

        for episode in episodes:
            perception_data = episode.get("perception_data", {})

            # Extract features from perception
            features = self.feature_extractor.extract(
                rgb=perception_data.get("rgb"),
                depth=perception_data.get("depth"),
                semantic=perception_data.get("semantic"),
                lidar_ranges=perception_data.get("lidar_ranges"),
                lidar_angles=perception_data.get("lidar_angles"),
            )

            # Create samples from steps
            for step in episode.get("steps", []):
                action_name = step.get("action", "noop")
                if action_name in ACTION_TO_IDX:
                    sample = PerceptionTrainingSample(
                        features=features,
                        action_idx=ACTION_TO_IDX[action_name],
                        action_name=action_name,
                        reward=step.get("reward", 0.0),
                        episode_id=episode.get("episode_id", "unknown"),
                        step_idx=step.get("step_id", 0),
                    )
                    self.samples.append(sample)

        print(f"Prepared {len(self.samples)} training samples")
        return len(self.samples)

    def train(
        self,
        epochs: int = 20,
        learning_rate: float = 0.01,
        batch_size: int = 32,
    ) -> Dict:
        """Train the perception-based action predictor."""
        if not self.samples:
            print("No samples to train on")
            return {"loss": 0, "accuracy": 0}

        print(f"\nTraining perception model on {len(self.samples)} samples...")

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            random.shuffle(self.samples)
            epoch_loss = 0.0
            epoch_correct = 0

            for i, sample in enumerate(self.samples):
                # Train step
                loss = self.predictor.train_step(
                    sample.features,
                    sample.action_idx,
                    learning_rate,
                )
                epoch_loss += loss

                # Check accuracy
                pred_idx, _, _ = self.predictor.predict(sample.features)
                if pred_idx == sample.action_idx:
                    epoch_correct += 1

            epoch_acc = epoch_correct / len(self.samples)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: "
                      f"loss={epoch_loss/len(self.samples):.4f}, "
                      f"acc={epoch_acc*100:.1f}%")

        # Final evaluation
        for sample in self.samples:
            pred_idx, _, _ = self.predictor.predict(sample.features)
            if pred_idx == sample.action_idx:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.checkpoint_dir / f"perception_model_{timestamp}.json"
        self.predictor.save(str(model_path))
        print(f"\nModel saved to {model_path}")

        return {
            "loss": epoch_loss / len(self.samples) if self.samples else 0,
            "accuracy": accuracy,
            "samples": len(self.samples),
            "epochs": epochs,
        }

    def evaluate(self, test_episodes: List[Dict]) -> Dict:
        """Evaluate model on test episodes."""
        # Prepare test samples
        old_samples = self.samples
        self.prepare_samples(test_episodes)
        test_samples = self.samples
        self.samples = old_samples

        if not test_samples:
            return {"accuracy": 0, "samples": 0}

        correct = 0
        action_accuracy = {a: {"correct": 0, "total": 0} for a in ACTIONS}

        for sample in test_samples:
            pred_idx, confidence, _ = self.predictor.predict(sample.features)

            action_accuracy[sample.action_name]["total"] += 1
            if pred_idx == sample.action_idx:
                correct += 1
                action_accuracy[sample.action_name]["correct"] += 1

        return {
            "accuracy": correct / len(test_samples),
            "samples": len(test_samples),
            "per_action": {
                a: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                for a, stats in action_accuracy.items()
            },
        }

    def load_model(self, filepath: str = None):
        """Load latest model checkpoint."""
        if filepath:
            self.predictor.load(filepath)
            return

        # Find latest checkpoint
        checkpoints = list(self.checkpoint_dir.glob("perception_model_*.json"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            self.predictor.load(str(latest))
            print(f"Loaded model: {latest.name}")


def run_perception_training(
    episodes_dir: str = "continuonbrain/rlds/episodes",
    data_dir: str = "brain_b_data",
    epochs: int = 20,
) -> Dict:
    """Run full perception-based training pipeline."""
    trainer = PerceptionTrainer(data_dir)

    # Load episodes
    episodes = trainer.load_episodes(episodes_dir)
    if not episodes:
        print("No episodes found")
        return {"error": "no_episodes"}

    # Prepare samples
    trainer.prepare_samples(episodes)

    # Train
    metrics = trainer.train(epochs=epochs)

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Perception-Aware Navigation Trainer")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--episodes", type=str, default="continuonbrain/rlds/episodes",
                       help="Episodes directory")
    parser.add_argument("--data-dir", type=str, default="brain_b_data",
                       help="Data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")

    args = parser.parse_args()

    if args.train:
        metrics = run_perception_training(
            episodes_dir=args.episodes,
            data_dir=args.data_dir,
            epochs=args.epochs,
        )
        print(f"\nTraining complete:")
        print(f"  Loss: {metrics.get('loss', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0)*100:.1f}%")
        print(f"  Samples: {metrics.get('samples', 0)}")

    elif args.evaluate:
        trainer = PerceptionTrainer(args.data_dir)
        trainer.load_model()
        episodes = trainer.load_episodes(args.episodes)
        metrics = trainer.evaluate(episodes)
        print(f"\nEvaluation:")
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  Samples: {metrics['samples']}")

    else:
        print("Run with --train or --evaluate")


if __name__ == "__main__":
    main()
