"""
Simulator Training Integration for Brain B.

Connects HomeScan 3D simulator RLDS episodes to Brain B training,
allowing the robot to learn from simulated navigation experiences.

This module provides:
- Unified training interface for navigation and tool prediction
- Episode conversion from Home3D format to Brain B format
- Training status and metrics aggregation
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable


@dataclass
class SimulatorTrainingConfig:
    """Configuration for simulator-based training."""
    # Episode sources
    home_episodes_dir: str = "./brain_b_data/home_rlds_episodes"
    claude_episodes_dir: str = "../continuonbrain/rlds/episodes"

    # Model outputs
    navigation_model_dir: str = "./brain_b_data/home_models"
    tool_model_dir: str = "./brain_b_data/models"

    # Training parameters
    navigation_epochs: int = 10
    tool_epochs: int = 10
    batch_size: int = 32

    # Auto-training thresholds
    min_episodes_for_nav_train: int = 3
    min_episodes_for_tool_train: int = 5


@dataclass
class TrainingResult:
    """Result of a training run."""
    success: bool
    model_type: str  # "navigation" or "tool"
    accuracy: float = 0.0
    loss: float = 0.0
    episodes_processed: int = 0
    samples_seen: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None
    model_path: Optional[str] = None


@dataclass
class UnifiedTrainingStatus:
    """Combined status for navigation and tool training."""
    # Navigation model status
    nav_model_exists: bool = False
    nav_model_accuracy: float = 0.0
    nav_episodes_available: int = 0
    nav_episodes_trained: int = 0
    nav_last_trained: Optional[str] = None

    # Tool model status
    tool_model_exists: bool = False
    tool_model_accuracy: float = 0.0
    tool_episodes_available: int = 0
    tool_episodes_trained: int = 0
    tool_last_trained: Optional[str] = None

    # Overall status
    is_training: bool = False
    current_training_type: Optional[str] = None
    error: Optional[str] = None


class SimulatorTrainingIntegration:
    """
    Integrates HomeScan simulator training with Brain B.

    Provides:
    - Training on navigation episodes (forward, turn, etc.)
    - Training on tool episodes (Bash, Read, etc.)
    - Unified status and metrics
    - Auto-training when new episodes are available
    """

    def __init__(self, config: Optional[SimulatorTrainingConfig] = None):
        self.config = config or SimulatorTrainingConfig()
        self.status = UnifiedTrainingStatus()
        self._callbacks: List[Callable[[UnifiedTrainingStatus], None]] = []

        # Ensure directories exist
        Path(self.config.navigation_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.tool_model_dir).mkdir(parents=True, exist_ok=True)

        # Load persisted status
        self._load_status()

    def _load_status(self):
        """Load training status from disk."""
        status_file = Path(self.config.navigation_model_dir) / "integration_status.json"
        if status_file.exists():
            try:
                with open(status_file) as f:
                    data = json.load(f)
                self.status.nav_model_accuracy = data.get("nav_model_accuracy", 0.0)
                self.status.nav_episodes_trained = data.get("nav_episodes_trained", 0)
                self.status.nav_last_trained = data.get("nav_last_trained")
                self.status.tool_model_accuracy = data.get("tool_model_accuracy", 0.0)
                self.status.tool_episodes_trained = data.get("tool_episodes_trained", 0)
                self.status.tool_last_trained = data.get("tool_last_trained")
            except Exception as e:
                print(f"[SimTrainer] Failed to load status: {e}")

        # Update current counts
        self._update_counts()

    def _save_status(self):
        """Save training status to disk."""
        status_file = Path(self.config.navigation_model_dir) / "integration_status.json"
        with open(status_file, "w") as f:
            json.dump({
                "nav_model_accuracy": self.status.nav_model_accuracy,
                "nav_episodes_trained": self.status.nav_episodes_trained,
                "nav_last_trained": self.status.nav_last_trained,
                "tool_model_accuracy": self.status.tool_model_accuracy,
                "tool_episodes_trained": self.status.tool_episodes_trained,
                "tool_last_trained": self.status.tool_last_trained,
            }, f, indent=2)

    def _update_counts(self):
        """Update episode counts."""
        # Navigation episodes
        nav_path = Path(self.config.home_episodes_dir)
        self.status.nav_episodes_available = self._count_episodes(nav_path)

        # Tool episodes
        tool_path = Path(self.config.claude_episodes_dir)
        self.status.tool_episodes_available = self._count_episodes(tool_path)

        # Check model existence
        nav_model = Path(self.config.navigation_model_dir) / "home3d_nav_model.json"
        self.status.nav_model_exists = nav_model.exists()

        tool_model = Path(self.config.tool_model_dir) / "tool_predictor_model.json"
        self.status.tool_model_exists = tool_model.exists()

    def _count_episodes(self, path: Path) -> int:
        """Count valid episodes in a directory."""
        if not path.exists():
            return 0

        count = 0
        for ep_dir in path.iterdir():
            if not ep_dir.is_dir():
                continue
            # Check for steps file
            for candidate in [
                ep_dir / "steps" / "000000.jsonl",
                ep_dir / "steps.jsonl",
            ]:
                if candidate.exists():
                    count += 1
                    break
        return count

    def get_status(self) -> Dict[str, Any]:
        """Get unified training status."""
        self._update_counts()
        return asdict(self.status)

    def should_train_navigation(self) -> bool:
        """Check if navigation training should be triggered."""
        self._update_counts()
        new_eps = self.status.nav_episodes_available - self.status.nav_episodes_trained
        return new_eps >= self.config.min_episodes_for_nav_train

    def should_train_tools(self) -> bool:
        """Check if tool training should be triggered."""
        self._update_counts()
        new_eps = self.status.tool_episodes_available - self.status.tool_episodes_trained
        return new_eps >= self.config.min_episodes_for_tool_train

    def train_navigation(self, force: bool = False) -> TrainingResult:
        """
        Train the navigation model on Home3D episodes.

        Args:
            force: If True, train even if threshold not reached

        Returns:
            TrainingResult with metrics
        """
        if self.status.is_training:
            return TrainingResult(
                success=False,
                model_type="navigation",
                error="Training already in progress"
            )

        if not force and not self.should_train_navigation():
            return TrainingResult(
                success=False,
                model_type="navigation",
                error=f"Not enough episodes. Have {self.status.nav_episodes_available}, need {self.config.min_episodes_for_nav_train} new."
            )

        self.status.is_training = True
        self.status.current_training_type = "navigation"
        self.status.error = None

        start_time = time.time()

        try:
            # Import here to avoid circular imports
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
            from home_training import (
                Home3DTrainingDataset,
                Home3DTrainer,
            )

            # Load dataset
            dataset = Home3DTrainingDataset(self.config.home_episodes_dir)
            num_eps = dataset.load_episodes()

            if len(dataset) == 0:
                raise ValueError("No navigation training samples found")

            print(f"[SimTrainer] Loaded {num_eps} navigation episodes, {len(dataset)} samples")

            # Train
            trainer = Home3DTrainer(checkpoint_dir=self.config.navigation_model_dir)
            metrics = trainer.train(
                dataset,
                epochs=self.config.navigation_epochs,
                batch_size=self.config.batch_size,
            )

            # Save model
            model_path = Path(self.config.navigation_model_dir) / "home3d_nav_model.json"
            trainer.save(str(model_path))

            # Update status
            self.status.nav_model_accuracy = metrics.accuracy
            self.status.nav_episodes_trained = num_eps
            self.status.nav_last_trained = datetime.now().isoformat()
            self.status.nav_model_exists = True
            self._save_status()

            duration = time.time() - start_time

            print(f"[SimTrainer] Navigation training complete! Accuracy: {metrics.accuracy:.2%}")

            return TrainingResult(
                success=True,
                model_type="navigation",
                accuracy=metrics.accuracy,
                loss=metrics.loss,
                episodes_processed=num_eps,
                samples_seen=metrics.samples_seen,
                duration_s=duration,
                model_path=str(model_path),
            )

        except Exception as e:
            self.status.error = str(e)
            print(f"[SimTrainer] Navigation training failed: {e}")
            return TrainingResult(
                success=False,
                model_type="navigation",
                error=str(e)
            )
        finally:
            self.status.is_training = False
            self.status.current_training_type = None
            self._notify_callbacks()

    def train_tools(self, force: bool = False) -> TrainingResult:
        """
        Train the tool predictor on Claude Code episodes.

        Args:
            force: If True, train even if threshold not reached

        Returns:
            TrainingResult with metrics
        """
        if self.status.is_training:
            return TrainingResult(
                success=False,
                model_type="tool",
                error="Training already in progress"
            )

        if not force and not self.should_train_tools():
            return TrainingResult(
                success=False,
                model_type="tool",
                error=f"Not enough episodes. Have {self.status.tool_episodes_available}, need {self.config.min_episodes_for_tool_train} new."
            )

        self.status.is_training = True
        self.status.current_training_type = "tool"
        self.status.error = None

        start_time = time.time()

        try:
            from trainer.claude_code_trainer import (
                ClaudeCodeDataset,
                ClaudeCodeTrainer,
            )

            # Load dataset
            dataset = ClaudeCodeDataset(self.config.claude_episodes_dir)
            num_eps = dataset.load_episodes()

            if len(dataset) == 0:
                raise ValueError("No tool training samples found")

            print(f"[SimTrainer] Loaded {num_eps} tool episodes, {len(dataset)} samples")

            # Train
            trainer = ClaudeCodeTrainer(checkpoint_dir=self.config.tool_model_dir)
            metrics = trainer.train(
                dataset,
                epochs=self.config.tool_epochs,
                batch_size=self.config.batch_size,
            )

            # Save model
            model_path = Path(self.config.tool_model_dir) / "tool_predictor_model.json"
            trainer.save(str(model_path))

            # Update status
            self.status.tool_model_accuracy = metrics.accuracy
            self.status.tool_episodes_trained = num_eps
            self.status.tool_last_trained = datetime.now().isoformat()
            self.status.tool_model_exists = True
            self._save_status()

            duration = time.time() - start_time

            print(f"[SimTrainer] Tool training complete! Accuracy: {metrics.accuracy:.2%}")

            return TrainingResult(
                success=True,
                model_type="tool",
                accuracy=metrics.accuracy,
                loss=metrics.loss,
                episodes_processed=num_eps,
                samples_seen=metrics.samples_seen,
                duration_s=duration,
                model_path=str(model_path),
            )

        except Exception as e:
            self.status.error = str(e)
            print(f"[SimTrainer] Tool training failed: {e}")
            return TrainingResult(
                success=False,
                model_type="tool",
                error=str(e)
            )
        finally:
            self.status.is_training = False
            self.status.current_training_type = None
            self._notify_callbacks()

    def train_all(self, force: bool = False) -> Dict[str, TrainingResult]:
        """
        Train both navigation and tool models if needed.

        Returns:
            Dictionary mapping model type to result
        """
        results = {}

        # Train navigation first
        if force or self.should_train_navigation():
            results["navigation"] = self.train_navigation(force=force)
        else:
            results["navigation"] = TrainingResult(
                success=False,
                model_type="navigation",
                error="Not enough new episodes"
            )

        # Then train tools
        if force or self.should_train_tools():
            results["tool"] = self.train_tools(force=force)
        else:
            results["tool"] = TrainingResult(
                success=False,
                model_type="tool",
                error="Not enough new episodes"
            )

        return results

    def convert_navigation_to_tool_episodes(
        self,
        output_dir: Optional[str] = None,
        action_to_tool_map: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Convert Home3D navigation episodes to tool prediction format.

        This allows the tool predictor to learn from navigation patterns
        by treating navigation actions as "tools".

        Args:
            output_dir: Where to save converted episodes
            action_to_tool_map: Map navigation actions to tool names

        Returns:
            Number of episodes converted
        """
        if output_dir is None:
            output_dir = self.config.claude_episodes_dir

        if action_to_tool_map is None:
            # Default mapping: navigation actions to pseudo-tools
            action_to_tool_map = {
                "forward": "Navigate",
                "backward": "Navigate",
                "strafe_left": "Navigate",
                "strafe_right": "Navigate",
                "turn_left": "Navigate",
                "turn_right": "Navigate",
                "look_up": "Observe",
                "look_down": "Observe",
                "interact": "Interact",
            }

        nav_path = Path(self.config.home_episodes_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        converted = 0

        for ep_dir in nav_path.iterdir():
            if not ep_dir.is_dir():
                continue

            # Find steps file
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

            # Load and convert
            metadata_file = ep_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            else:
                metadata = {"episode_id": ep_dir.name, "source": "home3d"}

            # Read steps
            steps = []
            with open(steps_file) as f:
                for line in f:
                    step = json.loads(line)
                    steps.append(step)

            # Convert to tool format
            tool_steps = []
            prev_tool = ""

            for step in steps:
                action = step.get("action", {})
                command = action.get("command", "")
                tool = action_to_tool_map.get(command, "Other")

                tool_step = {
                    "frame_id": step.get("frame_id", 0),
                    "timestamp_us": step.get("timestamp_us", 0),
                    "observation": {
                        "current_tool": "",
                        "prev_tool": prev_tool,
                        "prev_success": step.get("info", {}).get("success", True),
                        "step_idx": step.get("frame_id", 0),
                    },
                    "action": {
                        "tool": tool,
                        "source_command": command,
                    },
                    "reward": step.get("reward", 0.0),
                    "done": step.get("done", False),
                }
                tool_steps.append(tool_step)
                prev_tool = tool

            # Save converted episode
            out_ep_dir = out_path / f"nav_{ep_dir.name}"
            out_ep_dir.mkdir(exist_ok=True)

            # Save metadata
            metadata["converted_from"] = "home3d"
            metadata["original_episode"] = ep_dir.name
            with open(out_ep_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save steps
            with open(out_ep_dir / "steps.jsonl", "w") as f:
                for step in tool_steps:
                    f.write(json.dumps(step) + "\n")

            converted += 1

        print(f"[SimTrainer] Converted {converted} navigation episodes to tool format")
        return converted

    def add_callback(self, callback: Callable[[UnifiedTrainingStatus], None]):
        """Add a callback for training status updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all callbacks."""
        for callback in self._callbacks:
            try:
                callback(self.status)
            except Exception as e:
                print(f"[SimTrainer] Callback error: {e}")

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of all available episodes."""
        nav_summary = self._get_episode_dir_summary(Path(self.config.home_episodes_dir))
        tool_summary = self._get_episode_dir_summary(Path(self.config.claude_episodes_dir))

        return {
            "navigation": nav_summary,
            "tool": tool_summary,
            "total_episodes": nav_summary["count"] + tool_summary["count"],
            "total_samples": nav_summary["total_steps"] + tool_summary["total_steps"],
        }

    def _get_episode_dir_summary(self, path: Path) -> Dict[str, Any]:
        """Get summary for an episode directory."""
        if not path.exists():
            return {"count": 0, "total_steps": 0, "levels": [], "success_rate": 0.0}

        count = 0
        total_steps = 0
        successes = 0
        levels = set()

        for ep_dir in path.iterdir():
            if not ep_dir.is_dir():
                continue

            metadata_file = ep_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        meta = json.load(f)
                    count += 1
                    total_steps += meta.get("total_moves", 0)
                    if meta.get("success"):
                        successes += 1
                    if meta.get("level_id"):
                        levels.add(meta["level_id"])
                except Exception:
                    pass

        return {
            "count": count,
            "total_steps": total_steps,
            "levels": list(levels),
            "success_rate": successes / count if count > 0 else 0.0,
        }


# Singleton instance
_integration: Optional[SimulatorTrainingIntegration] = None


def get_training_integration(
    config: Optional[SimulatorTrainingConfig] = None
) -> SimulatorTrainingIntegration:
    """Get or create the training integration singleton."""
    global _integration
    if _integration is None:
        _integration = SimulatorTrainingIntegration(config)
    return _integration


if __name__ == "__main__":
    # Test the integration
    import sys

    integration = SimulatorTrainingIntegration()

    print("=" * 60)
    print("  Simulator Training Integration Status")
    print("=" * 60)

    status = integration.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("  Episode Summary")
    print("=" * 60)

    summary = integration.get_episode_summary()
    print(f"  Navigation episodes: {summary['navigation']['count']}")
    print(f"  Tool episodes: {summary['tool']['count']}")
    print(f"  Total samples: {summary['total_samples']}")

    if "--train" in sys.argv:
        print("\n" + "=" * 60)
        print("  Training All Models")
        print("=" * 60)

        results = integration.train_all(force="--force" in sys.argv)

        for model_type, result in results.items():
            print(f"\n{model_type.upper()}:")
            if result.success:
                print(f"  Accuracy: {result.accuracy:.2%}")
                print(f"  Episodes: {result.episodes_processed}")
                print(f"  Duration: {result.duration_s:.1f}s")
            else:
                print(f"  Error: {result.error}")
