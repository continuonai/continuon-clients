"""
MambaWave Model Hooks for Training.

Provides ModelHooks implementation that uses real MambaWave models
instead of stubs. This enables actual neural network training in
the Brain B → ContinuonBrain pipeline.

Usage:
    from continuonbrain.trainer.mambawave_hooks import build_mambawave_hooks

    hooks = build_mambawave_hooks(loop_type="mid")
    model = hooks.build_model()
    trainable = hooks.attach_lora_adapters(model, ())
    optimizer = hooks.make_optimizer(trainable, lr=3e-4, wd=0.01)

    for batch in batches:
        loss = hooks.train_step(model, optimizer, batch)

    hooks.save_adapters(model, Path("adapters.pt"))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class MambaWaveModelHooks:
    """
    ModelHooks implementation using MambaWave models.

    Provides the same interface as the stub hooks but with real
    neural network training.
    """

    def __init__(
        self,
        loop_type: str = "mid",
        d_model: int = 128,
        n_layers: int = 4,
        seq_len: int = 64,
        device: str = "cpu",
    ):
        self.loop_type = loop_type
        self.d_model = d_model
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.device = torch.device(device)
        self._model_config = None

    def build_model(self) -> nn.Module:
        """Build a MambaWave model, or fallback to simple model."""
        try:
            from continuonbrain.mambawave import MambaWaveConfig, MambaWaveModel

            # Get config based on loop type
            if self.loop_type == "fast":
                config = MambaWaveConfig.fast_loop()
            elif self.loop_type == "slow":
                config = MambaWaveConfig.slow_loop()
            else:
                config = MambaWaveConfig.mid_loop()

            # Override with our settings
            config.d_model = self.d_model
            config.n_layers = self.n_layers
            config.seq_len = self.seq_len

            self._model_config = config
            model = MambaWaveModel(config)
            logger.info(
                f"Built MambaWave model: d_model={self.d_model}, "
                f"n_layers={self.n_layers}, params={model.count_parameters():,}"
            )
            return model.to(self.device)

        except ImportError as e:
            logger.warning(f"MambaWave not available, using fallback: {e}")
            return self._build_fallback_model()

    def _build_fallback_model(self) -> nn.Module:
        """Build a simple fallback model."""

        class FallbackModel(nn.Module):
            def __init__(self, d_model: int, seq_len: int):
                super().__init__()
                self.d_model = d_model
                self.seq_len = seq_len
                self.encoder = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 2, d_model),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return {"hidden": decoded}

            def forward_continuous(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return self.forward(x)

            def count_parameters(self) -> int:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

        model = FallbackModel(self.d_model, self.seq_len)
        logger.info(f"Built fallback model: params={model.count_parameters():,}")
        return model.to(self.device)

    def attach_lora_adapters(
        self,
        model: nn.Module,
        adapter_state: Tuple[Any, ...],
    ) -> Iterable[nn.Parameter]:
        """
        Attach LoRA adapters to the model.

        For MambaWave, we train all parameters directly instead of LoRA
        for simplicity. In production, you'd want actual LoRA layers.
        """
        # For now, return all trainable parameters
        # TODO: Implement actual LoRA adapters
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Attached {len(trainable)} trainable parameter groups")
        return trainable

    def make_optimizer(
        self,
        trainable_params: Iterable[nn.Parameter],
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        """Create optimizer for the model."""
        params_list = list(trainable_params)
        optimizer = optim.AdamW(
            params_list,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        logger.info(f"Created AdamW optimizer: lr={learning_rate}, wd={weight_decay}")
        return optimizer

    def train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch: Dict[str, Any],
    ) -> float:
        """
        Execute one training step.

        Args:
            model: The model to train
            optimizer: The optimizer
            batch: Collated batch with 'obs' and 'action' as lists

        Returns:
            Loss value for this step
        """
        model.train()

        # Handle collated batch format: {"obs": [...], "action": [...]}
        obs_data = batch.get("obs", [])
        action_data = batch.get("action", [])
        reward_data = batch.get("reward", [0.0] * len(obs_data))

        # Convert batch to tensors
        obs_list = []
        action_list = []
        reward_list = []

        for i, (obs, action) in enumerate(zip(obs_data, action_data)):
            obs_tensor = self._encode_observation(obs if isinstance(obs, dict) else {})
            action_tensor = self._encode_action(action if isinstance(action, dict) else {})
            reward = float(reward_data[i]) if i < len(reward_data) else 0.0

            obs_list.append(obs_tensor)
            action_list.append(action_tensor)
            reward_list.append(reward)

        # Stack into tensors (B, d_model)
        obs_tensor = torch.stack(obs_list).to(self.device)
        action_tensor = torch.stack(action_list).to(self.device)
        reward_tensor = torch.tensor(reward_list, device=self.device)

        # Add sequence dimension if needed (B, 1, d_model)
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
            action_tensor = action_tensor.unsqueeze(1)

        # Combine observation and action as input
        x = obs_tensor + action_tensor

        # Forward pass
        if hasattr(model, "forward_continuous"):
            outputs = model.forward_continuous(x)
        else:
            outputs = model(x)

        pred = outputs.get("hidden", outputs.get("logits", x))

        # Loss: predict observation (reconstruction-style)
        loss = nn.functional.mse_loss(pred, obs_tensor)

        # Add reward correlation loss
        pred_mean = pred.mean(dim=(-1, -2))  # (B,)
        if pred_mean.shape == reward_tensor.shape:
            reward_loss = nn.functional.mse_loss(pred_mean, reward_tensor)
            loss = loss + 0.1 * reward_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        return loss.item()

    def _encode_observation(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Encode observation dict to tensor."""
        features = []

        # Context features
        if "context" in obs:
            ctx = obs["context"]
            features.append(float(ctx.get("step_index", 0)) / 100.0)
            features.append(1.0 if ctx.get("last_error") else 0.0)

        if "domain_obs" in obs:
            domain = obs["domain_obs"]
            features.append(-1.0 if domain.get("result_type") == "error" else 1.0)

        # Robot state
        if "robot_state" in obs:
            robot = obs["robot_state"]
            if isinstance(robot, dict):
                features.append(float(robot.get("x", 0)) / 10.0)
                features.append(float(robot.get("y", 0)) / 10.0)
                features.append(float(robot.get("moves", 0)) / 100.0)

        # 3D position
        if "position_3d" in obs:
            pos = obs["position_3d"]
            if isinstance(pos, dict):
                features.append(float(pos.get("x", 0)) / 10.0)
                features.append(float(pos.get("y", 0)) / 10.0)
                features.append(float(pos.get("z", 0)) / 10.0)

        # Pad to d_model
        while len(features) < self.d_model:
            features.append(0.0)
        features = features[: self.d_model]

        return torch.tensor(features, dtype=torch.float32)

    def _encode_action(self, action: Dict[str, Any]) -> torch.Tensor:
        """Encode action dict to tensor."""
        features = []

        # Action type
        action_type = action.get("action_type", action.get("command", ""))
        features.append((hash(str(action_type)) % 1000) / 1000.0)

        # Action name
        name = action.get("name", action.get("intent", ""))
        features.append((hash(str(name)) % 1000) / 1000.0)

        # Success
        features.append(1.0 if action.get("success", True) else 0.0)

        # Pad to d_model
        while len(features) < self.d_model:
            features.append(0.0)
        features = features[: self.d_model]

        return torch.tensor(features, dtype=torch.float32)

    def save_adapters(self, model: nn.Module, path: Path) -> None:
        """Save model weights (adapters)."""
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model_state_dict": model.state_dict(),
            "config": {
                "loop_type": self.loop_type,
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "seq_len": self.seq_len,
            },
        }

        torch.save(save_data, path)
        logger.info(f"Saved model to {path}")

    def load_adapters(self, model: nn.Module, path: Path) -> None:
        """Load model weights (adapters)."""
        if not path.exists():
            logger.warning(f"Adapter path does not exist: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {path}")

    def eval_batch(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate model on a batch (for safety gate)."""
        model.eval()

        # Handle collated batch format
        obs_data = batch.get("obs", [])
        action_data = batch.get("action", [])

        obs_list = []
        action_list = []

        for obs, action in zip(obs_data, action_data):
            obs_tensor = self._encode_observation(obs if isinstance(obs, dict) else {})
            action_tensor = self._encode_action(action if isinstance(action, dict) else {})
            obs_list.append(obs_tensor)
            action_list.append(action_tensor)

        obs_tensor = torch.stack(obs_list).to(self.device)
        action_tensor = torch.stack(action_list).to(self.device)

        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
            action_tensor = action_tensor.unsqueeze(1)

        x = obs_tensor + action_tensor

        with torch.no_grad():
            if hasattr(model, "forward_continuous"):
                outputs = model.forward_continuous(x)
            else:
                outputs = model(x)

            pred = outputs.get("hidden", outputs.get("logits", x))
            loss = nn.functional.mse_loss(pred, obs_tensor)

        return {
            "eval_loss": loss.item(),
            "n_samples": len(batch),
        }


def build_mambawave_hooks(
    loop_type: str = "mid",
    d_model: int = 128,
    n_layers: int = 4,
    seq_len: int = 64,
    device: str = "cpu",
) -> MambaWaveModelHooks:
    """Factory function to create MambaWave hooks."""
    return MambaWaveModelHooks(
        loop_type=loop_type,
        d_model=d_model,
        n_layers=n_layers,
        seq_len=seq_len,
        device=device,
    )
