"""
Training Loop for Claude Code Tool Prediction.

Uses RLDS episodes recorded from Claude Code sessions to train
next-tool prediction models for Brain B slow loop.
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict, Any
from pathlib import Path
from datetime import datetime


@dataclass
class TrainingSample:
    """A single training sample: context -> tool."""
    context_vector: list[float]
    tool_index: int
    tool_name: str
    success: bool
    episode_id: str
    step_idx: int


@dataclass
class TrainingMetrics:
    """Metrics from training."""
    loss: float
    accuracy: float
    samples_seen: int
    episodes_processed: int
    tool_distribution: Dict[str, int] = field(default_factory=dict)


# Tool vocabulary from Claude Code sessions
TOOLS = [
    "Bash", "Read", "Write", "Edit", "Grep", "Glob",
    "Task", "WebFetch", "WebSearch", "TodoWrite",
    "AskUserQuestion", "Other"
]
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOLS)}
IDX_TO_TOOL = {i: t for i, t in enumerate(TOOLS)}


class ToolPredictor:
    """
    Tool prediction model for Claude Code sessions.

    Architecture: Linear layer with softmax
    Input: Context embedding
    Output: Tool probabilities
    """

    def __init__(self, input_dim: int = 32, num_tools: int = None):
        self.input_dim = input_dim
        self.num_tools = num_tools or len(TOOLS)

        # Simple linear weights
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(input_dim)]
            for _ in range(self.num_tools)
        ]

        self.learning_rate = 0.01

    def predict(self, context_vector: list[float]) -> list[float]:
        """Predict tool probabilities given context."""
        # Pad or truncate to match input_dim
        vec = context_vector[:self.input_dim]
        if len(vec) < self.input_dim:
            vec = vec + [0.0] * (self.input_dim - len(vec))

        # Linear transformation
        logits = []
        for tool_weights in self.weights:
            logit = sum(w * x for w, x in zip(tool_weights, vec))
            logits.append(logit)

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        return probs

    def predict_tool(self, context_vector: list[float]) -> tuple[str, float]:
        """Predict most likely tool."""
        probs = self.predict(context_vector)
        best_idx = probs.index(max(probs))
        return IDX_TO_TOOL[best_idx], probs[best_idx]

    def train_step(self, context_vector: list[float], target_tool: int) -> float:
        """Single training step with gradient descent."""
        vec = context_vector[:self.input_dim]
        if len(vec) < self.input_dim:
            vec = vec + [0.0] * (self.input_dim - len(vec))

        probs = self.predict(vec)

        # Cross-entropy loss
        loss = -math.log(max(probs[target_tool], 1e-10))

        # Gradient
        gradients = probs.copy()
        gradients[target_tool] -= 1.0

        # Update weights
        for tool_idx in range(self.num_tools):
            for dim in range(self.input_dim):
                self.weights[tool_idx][dim] -= (
                    self.learning_rate * gradients[tool_idx] * vec[dim]
                )

        return loss

    def save(self, path: str) -> None:
        """Save model weights."""
        with open(path, "w") as f:
            json.dump({
                "input_dim": self.input_dim,
                "num_tools": self.num_tools,
                "weights": self.weights,
                "learning_rate": self.learning_rate,
                "tool_vocab": TOOLS,
            }, f, indent=2)

    def load(self, path: str) -> None:
        """Load model weights."""
        with open(path) as f:
            data = json.load(f)

        self.input_dim = data["input_dim"]
        self.num_tools = data["num_tools"]
        self.weights = data["weights"]
        self.learning_rate = data.get("learning_rate", 0.01)


class ClaudeCodeDataset:
    """
    Dataset of training samples from Claude Code RLDS episodes.
    """

    def __init__(self, episodes_dir: str):
        self.episodes_dir = Path(episodes_dir)
        self.samples: List[TrainingSample] = []
        self.episodes_loaded: int = 0
        self.tool_counts: Dict[str, int] = {}

    def load_episodes(self, max_episodes: Optional[int] = None) -> int:
        """Load episodes and extract training samples."""
        count = 0

        for ep_dir in sorted(self.episodes_dir.iterdir()):
            if not ep_dir.is_dir():
                continue

            if max_episodes and count >= max_episodes:
                break

            # Try different step file locations
            steps_file = None
            for candidate in [
                ep_dir / "steps" / "000000.jsonl",
                ep_dir / "steps.jsonl",
            ]:
                if candidate.exists():
                    steps_file = candidate
                    break

            if not steps_file:
                continue

            loaded = self._load_episode(ep_dir.name, steps_file)
            if loaded > 0:
                count += 1

        self.episodes_loaded = count
        return count

    def _load_episode(self, episode_id: str, steps_file: Path) -> int:
        """Load a single episode into samples."""
        samples_added = 0

        with open(steps_file) as f:
            prev_context = None
            for line in f:
                try:
                    step = json.loads(line)
                    sample = self._step_to_sample(episode_id, step, prev_context)
                    if sample:
                        self.samples.append(sample)
                        samples_added += 1
                        self.tool_counts[sample.tool_name] = self.tool_counts.get(sample.tool_name, 0) + 1
                    prev_context = step
                except json.JSONDecodeError:
                    continue

        return samples_added

    def _step_to_sample(
        self,
        episode_id: str,
        step: dict,
        prev_context: Optional[dict]
    ) -> Optional[TrainingSample]:
        """Convert a step to a training sample."""
        action = step.get("action", {})
        tool_name = action.get("tool_name", "Other")

        # Map to known tools or Other
        if tool_name not in TOOL_TO_IDX:
            tool_name = "Other"

        # Build context vector from observation
        context_vector = self._build_context_vector(step, prev_context)

        return TrainingSample(
            context_vector=context_vector,
            tool_index=TOOL_TO_IDX[tool_name],
            tool_name=tool_name,
            success=action.get("success", True),
            episode_id=episode_id,
            step_idx=step.get("step_idx", 0),
        )

    def _build_context_vector(self, step: dict, prev_context: Optional[dict]) -> list[float]:
        """Convert observation to context vector."""
        vector = []
        obs = step.get("observation", {})
        action = step.get("action", {})
        context = obs.get("context", {})

        # Tool availability one-hot (12 features)
        available_tool = context.get("tool_available", "")
        for tool in TOOLS:
            vector.append(1.0 if tool == available_tool else 0.0)

        # Step position (normalized)
        step_idx = step.get("step_idx", 0)
        vector.append(min(step_idx / 100.0, 1.0))

        # Previous tool (if any)
        if prev_context:
            prev_tool = prev_context.get("action", {}).get("tool_name", "")
            for tool in TOOLS:
                vector.append(1.0 if tool == prev_tool else 0.0)
        else:
            vector.extend([0.0] * len(TOOLS))

        # Success of previous action
        if prev_context:
            vector.append(1.0 if prev_context.get("action", {}).get("success", True) else 0.0)
        else:
            vector.append(1.0)

        # Reward signal
        vector.append(step.get("reward", 0.0))

        # Terminal flag
        vector.append(1.0 if step.get("is_terminal", False) else 0.0)

        # Padding to fixed size
        while len(vector) < 32:
            vector.append(0.0)

        return vector[:32]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self.samples)

    def shuffle(self) -> None:
        """Shuffle samples."""
        random.shuffle(self.samples)

    def batches(self, batch_size: int) -> Iterator[List[TrainingSample]]:
        """Yield batches of samples."""
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i:i + batch_size]


class ClaudeCodeTrainer:
    """
    Trainer for Claude Code tool prediction model.
    """

    def __init__(
        self,
        model: Optional[ToolPredictor] = None,
        checkpoint_dir: str = "./brain_b_data/checkpoints",
    ):
        self.model = model or ToolPredictor()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.total_samples = 0
        self.total_episodes = 0
        self.history: List[TrainingMetrics] = []

    def train(
        self,
        dataset: ClaudeCodeDataset,
        epochs: int = 10,
        batch_size: int = 32,
        checkpoint_every: int = 100,
    ) -> TrainingMetrics:
        """Train the model on the dataset."""
        print(f"\n{'='*50}")
        print("Starting Training Cycle")
        print(f"{'='*50}")
        print(f"Samples: {len(dataset)}")
        print(f"Episodes: {dataset.episodes_loaded}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"\nTool distribution:")
        for tool, count in sorted(dataset.tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")
        print(f"{'='*50}\n")

        for epoch in range(epochs):
            dataset.shuffle()

            epoch_loss = 0.0
            epoch_correct = 0
            samples_this_epoch = 0

            for batch in dataset.batches(batch_size):
                for sample in batch:
                    # Train step
                    loss = self.model.train_step(
                        sample.context_vector,
                        sample.tool_index
                    )
                    epoch_loss += loss

                    # Check accuracy
                    pred_tool, _ = self.model.predict_tool(sample.context_vector)
                    if pred_tool == sample.tool_name:
                        epoch_correct += 1

                    samples_this_epoch += 1
                    self.total_samples += 1

                    # Checkpoint
                    if self.total_samples % checkpoint_every == 0:
                        self._save_checkpoint()

            # Epoch metrics
            metrics = TrainingMetrics(
                loss=epoch_loss / max(samples_this_epoch, 1),
                accuracy=epoch_correct / max(samples_this_epoch, 1),
                samples_seen=self.total_samples,
                episodes_processed=dataset.episodes_loaded,
                tool_distribution=dataset.tool_counts.copy(),
            )
            self.history.append(metrics)

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"loss={metrics.loss:.4f}, "
                  f"acc={metrics.accuracy:.2%}")

        self.total_episodes = dataset.episodes_loaded
        return self.history[-1] if self.history else TrainingMetrics(0, 0, 0, 0, {})

    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"tool_model_{self.total_samples}.json"
        self.model.save(str(path))
        return str(path)

    def evaluate(self, dataset: ClaudeCodeDataset) -> TrainingMetrics:
        """Evaluate model on dataset without training."""
        total_loss = 0.0
        correct = 0

        for sample in dataset:
            probs = self.model.predict(sample.context_vector)
            loss = -math.log(max(probs[sample.tool_index], 1e-10))
            total_loss += loss

            pred_tool, _ = self.model.predict_tool(sample.context_vector)
            if pred_tool == sample.tool_name:
                correct += 1

        n = len(dataset)
        return TrainingMetrics(
            loss=total_loss / max(n, 1),
            accuracy=correct / max(n, 1),
            samples_seen=n,
            episodes_processed=dataset.episodes_loaded,
            tool_distribution=dataset.tool_counts.copy(),
        )

    def save(self, path: str) -> None:
        """Save trainer state."""
        self.model.save(path)
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "total_samples": self.total_samples,
                "total_episodes": self.total_episodes,
                "trained_at": datetime.now().isoformat(),
                "history": [
                    {
                        "loss": m.loss,
                        "accuracy": m.accuracy,
                        "samples_seen": m.samples_seen,
                    }
                    for m in self.history
                ],
            }, f, indent=2)

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
    batch_size: int = 16,
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
    print(f"\n{'='*60}")
    print("  Brain B Training Cycle - Claude Code Tool Prediction")
    print(f"{'='*60}")
    print(f"\nLoading episodes from: {episodes_dir}")

    dataset = ClaudeCodeDataset(episodes_dir)
    num_eps = dataset.load_episodes()
    print(f"Loaded {num_eps} episodes, {len(dataset)} samples")

    if len(dataset) == 0:
        print("No training data found!")
        return TrainingMetrics(0, 0, 0, 0, {})

    trainer = ClaudeCodeTrainer(checkpoint_dir=output_dir)
    metrics = trainer.train(dataset, epochs=epochs, batch_size=batch_size)

    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "tool_predictor_model.json"
    trainer.save(str(model_path))

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"{'='*60}")
    print(f"  Final accuracy: {metrics.accuracy:.2%}")
    print(f"  Total samples: {metrics.samples_seen}")
    print(f"  Episodes processed: {metrics.episodes_processed}")
    print(f"  Model saved to: {model_path}")
    print(f"{'='*60}\n")

    return metrics


if __name__ == "__main__":
    import sys

    # Default paths for ContinuonXR
    default_episodes = "../continuonbrain/rlds/episodes"
    default_output = "./brain_b_data/models"

    episodes_dir = sys.argv[1] if len(sys.argv) > 1 else default_episodes
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    run_training(episodes_dir, output_dir, epochs=epochs)
