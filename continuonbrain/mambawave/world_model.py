"""MambaWave World Model - Evolved from mamba_brain.

Provides state prediction for planning and control, combining:
- WaveCore spectral processing for temporal patterns
- Mamba SSM for efficient state tracking
- World modeling API for robot planning

This module maintains backward compatibility with the mamba_brain API
while providing enhanced capabilities through MambaWave architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class WorldModelState:
    """State representation for the world model.

    Initially focused on robot arm joint positions, but extensible
    to include other state variables (gripper, sensors, etc.).
    """

    joint_pos: List[float]

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert to tensor."""
        return torch.tensor(self.joint_pos, dtype=torch.float32, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "WorldModelState":
        """Create from tensor."""
        return cls(joint_pos=tensor.tolist())


@dataclass
class WorldModelAction:
    """Action representation for the world model.

    Represents changes to apply to the state (joint deltas).
    """

    joint_delta: List[float]

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert to tensor."""
        return torch.tensor(self.joint_delta, dtype=torch.float32, device=device)


@dataclass
class WorldModelPredictResult:
    """Result of a world model prediction."""

    next_state: WorldModelState
    uncertainty: float
    debug: Dict[str, Any]


class BaseWorldModel:
    """Base class for world models."""

    def predict(
        self, state: WorldModelState, action: WorldModelAction
    ) -> WorldModelPredictResult:
        """Predict next state from current state + action."""
        raise NotImplementedError

    def rollout(
        self,
        state: WorldModelState,
        actions: Sequence[WorldModelAction],
    ) -> Tuple[WorldModelState, List[WorldModelPredictResult]]:
        """Rollout a sequence of actions."""
        current = state
        results: List[WorldModelPredictResult] = []
        for action in actions:
            out = self.predict(current, action)
            results.append(out)
            current = out.next_state
        return current, results


class StubWorldModel(BaseWorldModel):
    """Deterministic fallback when neural model is unavailable."""

    def __init__(self, joint_limit: float = 1.0):
        self.joint_limit = float(joint_limit)

    def predict(
        self, state: WorldModelState, action: WorldModelAction
    ) -> WorldModelPredictResult:
        joint_pos = list(state.joint_pos)
        delta = list(action.joint_delta)

        # Pad/truncate to match dimensions
        if len(delta) != len(joint_pos):
            if len(delta) < len(joint_pos):
                delta = delta + [0.0] * (len(joint_pos) - len(delta))
            else:
                delta = delta[: len(joint_pos)]

        # Apply action with limits
        next_pos = []
        for j, dj in zip(joint_pos, delta):
            val = float(j) + float(dj)
            val = max(-self.joint_limit, min(self.joint_limit, val))
            next_pos.append(val)

        # Uncertainty heuristic: larger deltas = less certain
        mag = math.sqrt(sum(float(d) ** 2 for d in delta) / max(1, len(delta)))
        uncertainty = min(1.0, 0.05 + mag)

        return WorldModelPredictResult(
            next_state=WorldModelState(joint_pos=next_pos),
            uncertainty=float(uncertainty),
            debug={"backend": "stub", "delta_rms": mag},
        )


class MambaWaveWorldModel(BaseWorldModel):
    """MambaWave-powered world model for state prediction.

    Uses the MambaWave architecture to predict next states from
    current state + action pairs. Falls back to stub if the
    neural model is not available or not trained.
    """

    def __init__(
        self,
        *,
        joint_dim: int = 6,
        joint_limit: float = 1.0,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.joint_dim = joint_dim
        self.joint_limit = joint_limit
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        # Fallback stub
        self._stub = StubWorldModel(joint_limit=joint_limit)

        # Neural model (lazy loaded)
        self._model: Optional[nn.Module] = None
        self._model_loaded = False

    def _load_model(self) -> bool:
        """Attempt to load the neural model."""
        if self._model_loaded:
            return self._model is not None

        self._model_loaded = True

        if self.checkpoint_path is None:
            return False

        try:
            # Import here to avoid circular deps
            from .models.mambawave_model import MambaWaveModel
            from .config import MambaWaveConfig

            # Load checkpoint first to get config
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Extract config from checkpoint if available
            if isinstance(checkpoint, dict) and "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                loop_type = ckpt_config.get("loop_type", "fast")

                if loop_type == "slow":
                    config = MambaWaveConfig.slow_loop()
                elif loop_type == "mid":
                    config = MambaWaveConfig.mid_loop()
                else:
                    config = MambaWaveConfig.fast_loop()

                # Apply saved config values
                config.d_model = ckpt_config.get("d_model", config.d_model)
                config.n_layers = ckpt_config.get("n_layers", config.n_layers)
                config.seq_len = ckpt_config.get("seq_len", config.seq_len)
            else:
                # Fallback to small model for world prediction
                config = MambaWaveConfig.fast_loop()
                config.vocab_size = 1
                config.d_model = self.joint_dim * 2
                config.seq_len = 1

            self._model = MambaWaveModel(config).to(self.device)

            # Load state dict from checkpoint (try multiple keys)
            if isinstance(checkpoint, dict):
                state_dict = None
                for key in ["world_model_state_dict", "model_state_dict", "state_dict"]:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break

                if state_dict is not None:
                    self._model.load_state_dict(state_dict)

            self._model.eval()
            return True

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load world model: {e}")
            self._model = None
            return False

    def predict(
        self, state: WorldModelState, action: WorldModelAction
    ) -> WorldModelPredictResult:
        """Predict next state using MambaWave or fallback to stub."""
        # Try neural model
        if self._load_model() and self._model is not None:
            try:
                return self._predict_neural(state, action)
            except Exception:
                pass

        # Fallback to stub
        result = self._stub.predict(state, action)
        result.debug["backend"] = "mambawave_stub_fallback"
        result.debug["checkpoint_path"] = self.checkpoint_path
        return result

    def _predict_neural(
        self, state: WorldModelState, action: WorldModelAction
    ) -> WorldModelPredictResult:
        """Neural prediction using MambaWave."""
        assert self._model is not None

        # Prepare input: concatenate state and action
        state_tensor = state.to_tensor(self.device)
        action_tensor = action.to_tensor(self.device)

        # Ensure dimensions match joint_dim
        if len(state_tensor) != self.joint_dim:
            state_tensor = torch.nn.functional.pad(
                state_tensor, (0, self.joint_dim - len(state_tensor))
            )
        if len(action_tensor) != self.joint_dim:
            action_tensor = torch.nn.functional.pad(
                action_tensor, (0, self.joint_dim - len(action_tensor))
            )

        # Get model's expected input dimension
        d_model = self._model.config.d_model if hasattr(self._model, 'config') else 64

        # Concatenate state and action, then pad to model's d_model
        combined = torch.cat([state_tensor, action_tensor])  # (joint_dim * 2,)

        # Pad to d_model if needed
        if len(combined) < d_model:
            combined = torch.nn.functional.pad(
                combined, (0, d_model - len(combined))
            )
        elif len(combined) > d_model:
            combined = combined[:d_model]

        # Reshape for model: (batch=1, seq_len=1, d_model)
        x = combined.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = self._model.forward_continuous(x)
            hidden = outputs["hidden"]

            # Decode prediction (first joint_dim elements as next state delta)
            pred = hidden[0, 0, : self.joint_dim]

            # Apply residual: next_state = state + predicted_delta
            next_pos = (state_tensor + pred).clamp(-self.joint_limit, self.joint_limit)

            # Estimate uncertainty from hidden state variance
            uncertainty = float(hidden.std().item())

        return WorldModelPredictResult(
            next_state=WorldModelState(joint_pos=next_pos.tolist()),
            uncertainty=min(1.0, uncertainty),
            debug={"backend": "mambawave_neural", "checkpoint_path": self.checkpoint_path},
        )


def build_world_model(
    prefer_neural: bool = True,
    **kwargs: Any,
) -> BaseWorldModel:
    """Factory function to create a world model.

    Args:
        prefer_neural: If True, try to use MambaWave neural model
        **kwargs: Arguments passed to the world model

    Returns:
        A world model instance
    """
    if prefer_neural:
        return MambaWaveWorldModel(**kwargs)
    return StubWorldModel(joint_limit=float(kwargs.get("joint_limit", 1.0)))


# Backward compatibility aliases
MambaWorldModel = MambaWaveWorldModel
StubSSMWorldModel = StubWorldModel
