"""
Auto-Training Service for Brain B.

Monitors RLDS episodes and triggers retraining when enough new data is available.
Provides training status and metrics for the dashboard.
"""

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable

try:
    from trainer.claude_code_trainer import (
        ClaudeCodeDataset,
        ClaudeCodeTrainer,
        ToolPredictor,
        TrainingMetrics,
    )
except ImportError:
    from .claude_code_trainer import (
        ClaudeCodeDataset,
        ClaudeCodeTrainer,
        ToolPredictor,
        TrainingMetrics,
    )


@dataclass
class TrainingStatus:
    """Current training status."""
    is_training: bool = False
    last_trained: Optional[str] = None
    last_accuracy: float = 0.0
    last_loss: float = 0.0
    episodes_available: int = 0
    episodes_trained: int = 0
    samples_trained: int = 0
    model_version: int = 0
    next_retrain_at: int = 0  # Episode count threshold
    error: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for auto-training."""
    episodes_dir: str = "../continuonbrain/rlds/episodes"
    models_dir: str = "./brain_b_data/models"
    min_episodes_for_retrain: int = 5  # Minimum new episodes before retraining
    epochs: int = 10
    batch_size: int = 16
    auto_retrain: bool = True


class AutoTrainer:
    """
    Auto-training service that monitors episodes and retrains when needed.

    Features:
    - Monitors episode directory for new data
    - Triggers retraining when threshold is reached
    - Provides training status and metrics
    - Supports manual retrain trigger
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.status = TrainingStatus()
        self._lock = threading.Lock()
        self._training_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[TrainingStatus], None]] = []

        # Initialize paths
        self.episodes_path = Path(self.config.episodes_dir)
        self.models_path = Path(self.config.models_dir)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Load existing status
        self._load_status()

    def _load_status(self):
        """Load training status from disk."""
        status_file = self.models_path / "training_status.json"
        if status_file.exists():
            try:
                with open(status_file) as f:
                    data = json.load(f)
                self.status.last_trained = data.get("last_trained")
                self.status.last_accuracy = data.get("last_accuracy", 0.0)
                self.status.last_loss = data.get("last_loss", 0.0)
                self.status.episodes_trained = data.get("episodes_trained", 0)
                self.status.samples_trained = data.get("samples_trained", 0)
                self.status.model_version = data.get("model_version", 0)
            except Exception as e:
                print(f"[AutoTrainer] Failed to load status: {e}")

        # Count available episodes
        self._update_episode_count()

    def _save_status(self):
        """Save training status to disk."""
        status_file = self.models_path / "training_status.json"
        with open(status_file, "w") as f:
            json.dump({
                "last_trained": self.status.last_trained,
                "last_accuracy": self.status.last_accuracy,
                "last_loss": self.status.last_loss,
                "episodes_trained": self.status.episodes_trained,
                "samples_trained": self.status.samples_trained,
                "model_version": self.status.model_version,
            }, f, indent=2)

    def _update_episode_count(self):
        """Update the count of available episodes."""
        count = 0
        if self.episodes_path.exists():
            for ep_dir in self.episodes_path.iterdir():
                if ep_dir.is_dir():
                    # Check for steps file
                    for candidate in [
                        ep_dir / "steps" / "000000.jsonl",
                        ep_dir / "steps.jsonl",
                    ]:
                        if candidate.exists():
                            count += 1
                            break
        self.status.episodes_available = count
        self.status.next_retrain_at = self.status.episodes_trained + self.config.min_episodes_for_retrain

    def get_status(self) -> Dict:
        """Get current training status as dictionary."""
        with self._lock:
            self._update_episode_count()
            return asdict(self.status)

    def should_retrain(self) -> bool:
        """Check if retraining should be triggered."""
        self._update_episode_count()
        new_episodes = self.status.episodes_available - self.status.episodes_trained
        return new_episodes >= self.config.min_episodes_for_retrain

    def trigger_retrain(self, force: bool = False) -> Dict:
        """
        Trigger a training run.

        Args:
            force: If True, train even if threshold not reached

        Returns:
            Status dict with result
        """
        with self._lock:
            if self.status.is_training:
                return {"status": "already_training", "message": "Training already in progress"}

            if not force and not self.should_retrain():
                return {
                    "status": "skipped",
                    "message": f"Not enough new episodes. Have {self.status.episodes_available}, trained {self.status.episodes_trained}, need {self.config.min_episodes_for_retrain} new.",
                }

            self.status.is_training = True
            self.status.error = None

        # Start training in background thread
        self._training_thread = threading.Thread(target=self._run_training, daemon=True)
        self._training_thread.start()

        return {"status": "started", "message": "Training started in background"}

    def _run_training(self):
        """Run the training loop (called in background thread)."""
        try:
            print(f"\n[AutoTrainer] Starting training...")
            print(f"[AutoTrainer] Episodes: {self.status.episodes_available}")

            # Load dataset
            dataset = ClaudeCodeDataset(str(self.episodes_path))
            num_eps = dataset.load_episodes()

            if len(dataset) == 0:
                raise ValueError("No training samples found")

            print(f"[AutoTrainer] Loaded {num_eps} episodes, {len(dataset)} samples")

            # Train
            trainer = ClaudeCodeTrainer(checkpoint_dir=str(self.models_path))
            metrics = trainer.train(
                dataset,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
            )

            # Save model
            model_path = self.models_path / "tool_predictor_model.json"
            trainer.save(str(model_path))

            # Update status
            with self._lock:
                self.status.last_trained = datetime.now().isoformat()
                self.status.last_accuracy = metrics.accuracy
                self.status.last_loss = metrics.loss
                self.status.episodes_trained = num_eps
                self.status.samples_trained = metrics.samples_seen
                self.status.model_version += 1
                self.status.is_training = False
                self._save_status()

            print(f"[AutoTrainer] Training complete!")
            print(f"[AutoTrainer] Accuracy: {metrics.accuracy:.2%}")

            # Notify callbacks
            self._notify_callbacks()

        except Exception as e:
            print(f"[AutoTrainer] Training failed: {e}")
            with self._lock:
                self.status.is_training = False
                self.status.error = str(e)

    def add_callback(self, callback: Callable[[TrainingStatus], None]):
        """Add a callback to be notified when training completes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all callbacks of status change."""
        for callback in self._callbacks:
            try:
                callback(self.status)
            except Exception as e:
                print(f"[AutoTrainer] Callback error: {e}")

    def get_training_history(self) -> List[Dict]:
        """Get training history from saved metadata files."""
        history = []
        meta_file = self.models_path / "tool_predictor_model_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                history = data.get("history", [])
            except Exception:
                pass
        return history

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        model_file = self.models_path / "tool_predictor_model.json"
        if not model_file.exists():
            return {"exists": False}

        try:
            with open(model_file) as f:
                data = json.load(f)
            return {
                "exists": True,
                "input_dim": data.get("input_dim", 0),
                "num_tools": data.get("num_tools", 0),
                "tool_vocab": data.get("tool_vocab", []),
                "file_size_kb": model_file.stat().st_size / 1024,
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}


# Singleton instance
_auto_trainer: Optional[AutoTrainer] = None


def get_auto_trainer(config: Optional[TrainingConfig] = None) -> AutoTrainer:
    """Get or create the auto-trainer singleton."""
    global _auto_trainer
    if _auto_trainer is None:
        _auto_trainer = AutoTrainer(config)
    return _auto_trainer


if __name__ == "__main__":
    # Test the auto-trainer
    trainer = AutoTrainer()

    print("=== Auto-Trainer Status ===")
    status = trainer.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n=== Model Info ===")
    model_info = trainer.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("\n=== Should Retrain? ===")
    print(f"  {trainer.should_retrain()}")

    if trainer.should_retrain():
        print("\n=== Triggering Retrain ===")
        result = trainer.trigger_retrain()
        print(f"  {result}")

        # Wait for completion
        if result.get("status") == "started":
            print("  Waiting for training to complete...")
            while trainer.status.is_training:
                time.sleep(1)
            print(f"  Done! Accuracy: {trainer.status.last_accuracy:.2%}")
