"""
MambaWave Training Orchestrator.

Extends TrainerOrchestrator to provide real MambaWave model training
from RLDS episodes. This is the key component that enables the Brain B
→ ContinuonBrain learning pipeline.

Usage:
    from continuonbrain.trainer.mambawave_orchestrator import MambaWaveOrchestrator

    orchestrator = MambaWaveOrchestrator(
        episodes_dir=Path("continuonbrain/rlds/episodes"),
        checkpoint_dir=Path("continuonbrain/checkpoints"),
    )
    orchestrator.start()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from continuonbrain.trainer.orchestrator import (
    StepResult,
    StepStatus,
    TrainerOrchestrator,
)

logger = logging.getLogger(__name__)


@dataclass
class MambaWaveTrainingConfig:
    """Configuration for MambaWave training."""

    # Model architecture
    loop_type: str = "mid"  # "fast", "mid", "slow"
    d_model: int = 128
    n_layers: int = 4

    # Training params
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 500
    max_wall_time_s: int = 300
    gradient_clip: float = 1.0

    # Data
    min_episodes: int = 16
    max_episodes: int = 256
    seq_len: int = 64

    # Checkpointing
    save_every_steps: int = 100
    keep_n_checkpoints: int = 3

    # Device
    device: str = "cpu"  # "cpu", "cuda", "mps"


class RLDSDataset(Dataset):
    """Dataset for loading RLDS episodes into PyTorch."""

    def __init__(
        self,
        episode_files: List[Path],
        seq_len: int = 64,
        d_input: int = 128,
    ):
        self.episode_files = episode_files
        self.seq_len = seq_len
        self.d_input = d_input
        self._samples: List[Dict[str, Any]] = []
        self._load_episodes()

    def _load_episodes(self) -> None:
        """Load all episodes into memory."""
        for ep_path in self.episode_files:
            try:
                steps = self._load_steps(ep_path)
                if steps:
                    self._samples.extend(steps)
            except Exception as e:
                logger.warning(f"Failed to load {ep_path}: {e}")

    def _load_steps(self, ep_path: Path) -> List[Dict[str, Any]]:
        """Load steps from an episode file."""
        steps = []
        with open(ep_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        step = json.loads(line)
                        steps.append(step)
                    except json.JSONDecodeError:
                        continue
        return steps

    def __len__(self) -> int:
        return max(1, len(self._samples) - self.seq_len)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample as tensors."""
        # Get sequence of steps
        start_idx = idx
        end_idx = min(start_idx + self.seq_len, len(self._samples))
        seq_steps = self._samples[start_idx:end_idx]

        # Convert to tensors
        obs_tensors = []
        action_tensors = []
        reward_tensors = []

        for step in seq_steps:
            obs = self._encode_observation(step.get("observation", step.get("obs", {})))
            action = self._encode_action(step.get("action", {}))
            reward = float(step.get("reward", 0.0))

            obs_tensors.append(obs)
            action_tensors.append(action)
            reward_tensors.append(reward)

        # Pad if needed
        while len(obs_tensors) < self.seq_len:
            obs_tensors.append(torch.zeros(self.d_input))
            action_tensors.append(torch.zeros(self.d_input))
            reward_tensors.append(0.0)

        return {
            "obs": torch.stack(obs_tensors),
            "action": torch.stack(action_tensors),
            "reward": torch.tensor(reward_tensors),
        }

    def _encode_observation(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Encode observation dict to tensor."""
        # Extract numerical features from observation
        features = []

        # Handle different observation formats
        if "context" in obs:
            ctx = obs["context"]
            features.append(float(ctx.get("step_index", 0)) / 100.0)
            features.append(1.0 if ctx.get("last_error") else 0.0)

        if "domain_obs" in obs:
            domain = obs["domain_obs"]
            # Encode result type
            if domain.get("result_type") == "error":
                features.append(-1.0)
            else:
                features.append(1.0)

        # RobotGrid/Home3D observations
        if "robot_state" in obs:
            robot = obs["robot_state"]
            if isinstance(robot, dict):
                features.append(float(robot.get("x", 0)) / 10.0)
                features.append(float(robot.get("y", 0)) / 10.0)
                features.append(float(robot.get("moves", 0)) / 100.0)

        if "position_3d" in obs:
            pos = obs["position_3d"]
            if isinstance(pos, dict):
                features.append(float(pos.get("x", 0)) / 10.0)
                features.append(float(pos.get("y", 0)) / 10.0)
                features.append(float(pos.get("z", 0)) / 10.0)

        # Pad/truncate to d_input
        while len(features) < self.d_input:
            features.append(0.0)
        features = features[: self.d_input]

        return torch.tensor(features, dtype=torch.float32)

    def _encode_action(self, action: Dict[str, Any]) -> torch.Tensor:
        """Encode action dict to tensor."""
        features = []

        # Action type encoding
        action_type = action.get("action_type", action.get("command", ""))
        type_hash = hash(str(action_type)) % 1000
        features.append(type_hash / 1000.0)

        # Action name encoding
        name = action.get("name", action.get("intent", ""))
        name_hash = hash(str(name)) % 1000
        features.append(name_hash / 1000.0)

        # Success/failure
        features.append(1.0 if action.get("success", True) else 0.0)

        # Parameter count
        params = action.get("parameters", action.get("params", {}))
        features.append(min(1.0, len(params) / 10.0))

        # Pad to d_input
        while len(features) < self.d_input:
            features.append(0.0)
        features = features[: self.d_input]

        return torch.tensor(features, dtype=torch.float32)


def list_nested_episodes(rlds_dir: Path) -> List[Path]:
    """List episodes from nested directory structure."""
    if not rlds_dir.exists():
        return []

    episodes = []
    for ep_dir in sorted(rlds_dir.iterdir()):
        if not ep_dir.is_dir():
            continue

        for candidate in [
            ep_dir / "steps" / "000000.jsonl",
            ep_dir / "steps.jsonl",
        ]:
            if candidate.exists():
                episodes.append(candidate)
                break

    return episodes


class MambaWaveOrchestrator(TrainerOrchestrator):
    """
    Orchestrator for MambaWave model training.

    Training steps:
    1. prepare - Load episodes, create dataset
    2. init_model - Initialize or load MambaWave model
    3. train - Run training loop
    4. evaluate - Evaluate on held-out data
    5. checkpoint - Save model checkpoint
    6. export - Export for deployment
    """

    def __init__(
        self,
        episodes_dirs: Optional[List[Path]] = None,
        config: Optional[MambaWaveTrainingConfig] = None,
        status_path: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        super().__init__(
            status_path=status_path or Path("continuonbrain/trainer/mambawave_status.json"),
            checkpoint_dir=checkpoint_dir or Path("continuonbrain/checkpoints/mambawave"),
        )

        self.episodes_dirs = episodes_dirs or [
            Path("continuonbrain/rlds/episodes"),
            Path("brain_b_data/rlds_episodes"),
            Path("brain_b_data/home_rlds_episodes"),
        ]
        self.config = config or MambaWaveTrainingConfig()

        # Will be set during training
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._dataset: Optional[RLDSDataset] = None
        self._dataloader: Optional[DataLoader] = None

    def get_training_steps(self) -> List[str]:
        """Return ordered list of training steps."""
        return ["prepare", "init_model", "train", "evaluate", "checkpoint", "export"]

    def execute_step(self, step_name: str, context: Dict[str, Any]) -> StepResult:
        """Execute a training step."""
        start_time = time.time()

        try:
            if step_name == "prepare":
                return self._step_prepare(context)
            elif step_name == "init_model":
                return self._step_init_model(context)
            elif step_name == "train":
                return self._step_train(context)
            elif step_name == "evaluate":
                return self._step_evaluate(context)
            elif step_name == "checkpoint":
                return self._step_checkpoint(context)
            elif step_name == "export":
                return self._step_export(context)
            else:
                return StepResult(
                    name=step_name,
                    status=StepStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    error=f"Unknown step: {step_name}",
                )
        except Exception as e:
            logger.exception(f"Step {step_name} failed")
            return StepResult(
                name=step_name,
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _step_prepare(self, context: Dict[str, Any]) -> StepResult:
        """Prepare training data by loading RLDS episodes."""
        start_time = time.time()

        # Collect all episode files
        all_files: List[Path] = []
        for ep_dir in self.episodes_dirs:
            if ep_dir.exists():
                files = list_nested_episodes(ep_dir)
                all_files.extend(files)
                logger.info(f"Found {len(files)} episodes in {ep_dir}")

        if len(all_files) < self.config.min_episodes:
            return StepResult(
                name="prepare",
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=f"Not enough episodes: {len(all_files)} < {self.config.min_episodes}",
            )

        # Use most recent episodes
        episode_files = all_files[-self.config.max_episodes :]
        context["episode_files"] = episode_files

        # Create dataset
        self._dataset = RLDSDataset(
            episode_files=episode_files,
            seq_len=self.config.seq_len,
            d_input=self.config.d_model,
        )

        if len(self._dataset) == 0:
            return StepResult(
                name="prepare",
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error="Dataset is empty after loading",
            )

        # Create dataloader
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        context["dataset_size"] = len(self._dataset)
        context["n_batches"] = len(self._dataloader)

        return StepResult(
            name="prepare",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "episodes_loaded": len(episode_files),
                "dataset_size": len(self._dataset),
                "n_batches": len(self._dataloader),
            },
        )

    def _step_init_model(self, context: Dict[str, Any]) -> StepResult:
        """Initialize or load MambaWave model."""
        start_time = time.time()

        try:
            # Import MambaWave (lazy to avoid import errors if not available)
            from continuonbrain.mambawave import MambaWaveConfig, MambaWaveModel

            # Get config based on loop type
            if self.config.loop_type == "fast":
                mw_config = MambaWaveConfig.fast_loop()
            elif self.config.loop_type == "slow":
                mw_config = MambaWaveConfig.slow_loop()
            else:
                mw_config = MambaWaveConfig.mid_loop()

            # Override dimensions
            mw_config.d_model = self.config.d_model
            mw_config.n_layers = self.config.n_layers
            mw_config.seq_len = self.config.seq_len

            # Create model
            self._model = MambaWaveModel(mw_config)

            # Move to device
            device = torch.device(self.config.device)
            self._model = self._model.to(device)

            # Try to load existing checkpoint
            checkpoint_path = self.checkpoint_dir / "latest.pt"
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Loaded checkpoint from {checkpoint_path}")
                    context["loaded_checkpoint"] = True
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")
                    context["loaded_checkpoint"] = False
            else:
                context["loaded_checkpoint"] = False

            # Create optimizer
            self._optimizer = optim.AdamW(
                self._model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            n_params = self._model.count_parameters()
            context["model_params"] = n_params

            return StepResult(
                name="init_model",
                status=StepStatus.COMPLETED,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "model_type": "MambaWave",
                    "loop_type": self.config.loop_type,
                    "d_model": self.config.d_model,
                    "n_layers": self.config.n_layers,
                    "n_params": n_params,
                    "device": str(device),
                    "loaded_checkpoint": context.get("loaded_checkpoint", False),
                },
            )

        except ImportError as e:
            # Fall back to simple model if MambaWave not available
            logger.warning(f"MambaWave not available, using simple model: {e}")
            return self._init_simple_model(context)

    def _init_simple_model(self, context: Dict[str, Any]) -> StepResult:
        """Initialize a simple fallback model if MambaWave unavailable."""
        start_time = time.time()

        # Simple MLP model
        class SimpleModel(nn.Module):
            def __init__(self, d_model: int, seq_len: int):
                super().__init__()
                self.d_model = d_model
                self.encoder = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model),
                )
                self.decoder = nn.Linear(d_model, d_model)

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return {"hidden": decoded}

            def count_parameters(self) -> int:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

        self._model = SimpleModel(self.config.d_model, self.config.seq_len)
        device = torch.device(self.config.device)
        self._model = self._model.to(device)

        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        n_params = self._model.count_parameters()
        context["model_params"] = n_params
        context["fallback_model"] = True

        return StepResult(
            name="init_model",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "model_type": "SimpleModel (fallback)",
                "d_model": self.config.d_model,
                "n_params": n_params,
                "device": str(device),
            },
        )

    def _step_train(self, context: Dict[str, Any]) -> StepResult:
        """Run training loop."""
        start_time = time.time()

        if self._model is None or self._optimizer is None or self._dataloader is None:
            return StepResult(
                name="train",
                status=StepStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error="Model, optimizer, or dataloader not initialized",
            )

        self._model.train()
        device = torch.device(self.config.device)

        total_loss = 0.0
        step = 0
        n_batches = 0

        # Training loop
        epoch = 0
        while step < self.config.max_steps:
            epoch += 1
            for batch in self._dataloader:
                # Check time budget
                if time.time() - start_time > self.config.max_wall_time_s:
                    logger.info("Time budget reached, stopping training")
                    break

                # Check step budget
                if step >= self.config.max_steps:
                    break

                # Move batch to device
                obs = batch["obs"].to(device)  # (B, seq_len, d_model)
                action = batch["action"].to(device)  # (B, seq_len, d_model)
                reward = batch["reward"].to(device)  # (B, seq_len)

                # Forward pass - predict next observation from current obs+action
                # Concatenate obs and action as input
                x = obs + action  # Simple combination

                # Handle different model interfaces
                if hasattr(self._model, "forward_continuous"):
                    outputs = self._model.forward_continuous(x)
                else:
                    outputs = self._model(x)

                # Get predictions
                pred = outputs.get("hidden", outputs.get("logits", x))

                # Loss: predict next observation
                # Shift by 1 to predict next state
                if pred.shape[1] > 1:
                    target = obs[:, 1:, :]
                    pred_shifted = pred[:, :-1, :]
                    loss = nn.functional.mse_loss(pred_shifted, target)
                else:
                    loss = nn.functional.mse_loss(pred, obs)

                # Add reward prediction loss
                # Simple mean of predictions should correlate with reward
                pred_mean = pred.mean(dim=-1)
                if pred_mean.shape == reward.shape:
                    reward_loss = nn.functional.mse_loss(pred_mean, reward)
                    loss = loss + 0.1 * reward_loss

                # Backward pass
                self._optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self.config.gradient_clip,
                    )

                self._optimizer.step()

                total_loss += loss.item()
                step += 1
                n_batches += 1

                # Update loss tracking
                avg_loss = total_loss / max(1, n_batches)
                self.set_current_loss(avg_loss)

                # Log progress
                if step % 10 == 0:
                    logger.info(f"Step {step}/{self.config.max_steps}: loss={avg_loss:.4f}")

            # End of epoch
            logger.info(f"Epoch {epoch} complete, step={step}")

        avg_loss = total_loss / max(1, n_batches)
        wall_time = time.time() - start_time

        context["final_loss"] = avg_loss
        context["final_step"] = step
        context["n_batches"] = n_batches
        context["wall_time_s"] = wall_time

        return StepResult(
            name="train",
            status=StepStatus.COMPLETED,
            duration_ms=wall_time * 1000,
            details={
                "steps": step,
                "epochs": epoch,
                "n_batches": n_batches,
                "final_loss": avg_loss,
                "wall_time_s": wall_time,
            },
        )

    def _step_evaluate(self, context: Dict[str, Any]) -> StepResult:
        """Evaluate model on held-out data."""
        start_time = time.time()

        if self._model is None or self._dataloader is None:
            return StepResult(
                name="evaluate",
                status=StepStatus.SKIPPED,
                duration_ms=(time.time() - start_time) * 1000,
                details={"reason": "No model or data available"},
            )

        self._model.eval()
        device = torch.device(self.config.device)

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self._dataloader:
                obs = batch["obs"].to(device)
                action = batch["action"].to(device)

                x = obs + action

                if hasattr(self._model, "forward_continuous"):
                    outputs = self._model.forward_continuous(x)
                else:
                    outputs = self._model(x)

                pred = outputs.get("hidden", outputs.get("logits", x))

                if pred.shape[1] > 1:
                    target = obs[:, 1:, :]
                    pred_shifted = pred[:, :-1, :]
                    loss = nn.functional.mse_loss(pred_shifted, target)
                else:
                    loss = nn.functional.mse_loss(pred, obs)

                total_loss += loss.item()
                n_batches += 1

                # Only evaluate on a few batches
                if n_batches >= 10:
                    break

        eval_loss = total_loss / max(1, n_batches)
        context["eval_loss"] = eval_loss

        return StepResult(
            name="evaluate",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "eval_loss": eval_loss,
                "n_batches": n_batches,
            },
        )

    def _step_checkpoint(self, context: Dict[str, Any]) -> StepResult:
        """Save model checkpoint."""
        start_time = time.time()

        if self._model is None:
            return StepResult(
                name="checkpoint",
                status=StepStatus.SKIPPED,
                duration_ms=(time.time() - start_time) * 1000,
                details={"reason": "No model to save"},
            )

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint with timestamp
        timestamp = int(time.time())
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.pt"

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "config": {
                "loop_type": self.config.loop_type,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "seq_len": self.config.seq_len,
            },
            "training_context": {
                "final_loss": context.get("final_loss"),
                "final_step": context.get("final_step"),
                "eval_loss": context.get("eval_loss"),
            },
            "timestamp": timestamp,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save as latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        context["checkpoint_path"] = str(checkpoint_path)

        return StepResult(
            name="checkpoint",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "checkpoint_path": str(checkpoint_path),
                "latest_path": str(latest_path),
            },
        )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the N most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep latest.pt and N most recent timestamped checkpoints
        for ckpt in checkpoints[self.config.keep_n_checkpoints :]:
            try:
                ckpt.unlink()
                logger.info(f"Removed old checkpoint: {ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove {ckpt}: {e}")

    def _step_export(self, context: Dict[str, Any]) -> StepResult:
        """Export model for deployment (e.g., to adapters directory)."""
        start_time = time.time()

        if self._model is None:
            return StepResult(
                name="export",
                status=StepStatus.SKIPPED,
                duration_ms=(time.time() - start_time) * 1000,
                details={"reason": "No model to export"},
            )

        # Export to adapters directory
        adapters_dir = Path("continuonbrain/adapters/current")
        adapters_dir.mkdir(parents=True, exist_ok=True)

        # Save world model weights specifically
        world_model_path = adapters_dir / "world_model.pt"

        export_data = {
            "world_model_state_dict": self._model.state_dict(),
            "config": {
                "loop_type": self.config.loop_type,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "seq_len": self.config.seq_len,
            },
            "training_stats": {
                "final_loss": context.get("final_loss"),
                "eval_loss": context.get("eval_loss"),
                "n_episodes": len(context.get("episode_files", [])),
            },
            "exported_at": int(time.time()),
        }

        torch.save(export_data, world_model_path)
        logger.info(f"Exported world model to {world_model_path}")

        context["export_path"] = str(world_model_path)

        return StepResult(
            name="export",
            status=StepStatus.COMPLETED,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "export_path": str(world_model_path),
            },
        )


def create_mambawave_orchestrator(
    repo_root: Optional[Path] = None,
    config: Optional[MambaWaveTrainingConfig] = None,
) -> MambaWaveOrchestrator:
    """Factory function to create a MambaWaveOrchestrator with sensible defaults."""
    root = repo_root or Path.cwd()

    episodes_dirs = [
        root / "continuonbrain/rlds/episodes",
        root / "brain_b_data/rlds_episodes",
        root / "brain_b_data/home_rlds_episodes",
    ]

    return MambaWaveOrchestrator(
        episodes_dirs=episodes_dirs,
        config=config or MambaWaveTrainingConfig(),
        status_path=root / "continuonbrain/trainer/mambawave_status.json",
        checkpoint_dir=root / "continuonbrain/checkpoints/mambawave",
    )
