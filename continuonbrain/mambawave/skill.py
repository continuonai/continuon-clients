"""MambaWave Skill - Integration with ContinuonBrain.

Provides a skill interface for ContinuonBrain to use MambaWave
for sequence modeling, world prediction, and planning tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import logging

import torch

from .config import MambaWaveConfig
from .models.mambawave_model import MambaWaveModel
from .world_model import MambaWaveWorldModel, WorldModelState, WorldModelAction

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result from a skill execution."""

    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MambaWaveSkill:
    """MambaWave as a skill for ContinuonBrain.

    This skill provides:
    1. Sequence modeling (token prediction, generation)
    2. World modeling (state prediction for planning)
    3. Embedding extraction (for other skills to use)

    Integrates with Ralph learning loops:
    - Fast loop: Real-time inference
    - Mid loop: Online learning
    - Slow loop: Batch training
    """

    name = "mambawave"
    description = "Unified SSM + Spectral architecture for sequence and world modeling"
    version = "1.0.0"

    def __init__(
        self,
        config: Optional[MambaWaveConfig] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.config = config or MambaWaveConfig.default()
        self.config.device = device
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Models (lazy loaded)
        self._sequence_model: Optional[MambaWaveModel] = None
        self._world_model: Optional[MambaWaveWorldModel] = None

        # Skill registry for sub-capabilities
        self._capabilities: Dict[str, Callable] = {
            "predict_next": self.predict_next,
            "generate": self.generate,
            "embed": self.embed,
            "predict_world": self.predict_world,
            "rollout": self.rollout,
        }

    @property
    def sequence_model(self) -> MambaWaveModel:
        """Lazy load sequence model."""
        if self._sequence_model is None:
            self._sequence_model = MambaWaveModel(self.config).to(self.device)
            if self.checkpoint_path:
                self._load_checkpoint(self._sequence_model, self.checkpoint_path)
            self._sequence_model.eval()
        return self._sequence_model

    @property
    def world_model(self) -> MambaWaveWorldModel:
        """Lazy load world model."""
        if self._world_model is None:
            self._world_model = MambaWaveWorldModel(
                joint_dim=self.config.joint_dim,
                joint_limit=self.config.joint_limit,
                checkpoint_path=self.checkpoint_path,
            )
        return self._world_model

    def _load_checkpoint(self, model: torch.nn.Module, path: str) -> None:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def execute(self, capability: str, **kwargs) -> SkillResult:
        """Execute a skill capability.

        Args:
            capability: Name of the capability to execute
            **kwargs: Arguments for the capability

        Returns:
            SkillResult with output or error
        """
        if capability not in self._capabilities:
            return SkillResult(
                success=False,
                output=None,
                error=f"Unknown capability: {capability}. Available: {list(self._capabilities.keys())}",
            )

        try:
            output = self._capabilities[capability](**kwargs)
            return SkillResult(success=True, output=output)
        except Exception as e:
            logger.exception(f"Skill execution failed: {capability}")
            return SkillResult(success=False, output=None, error=str(e))

    def predict_next(
        self,
        input_ids: List[int],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Predict next token probabilities.

        Args:
            input_ids: List of token IDs
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and probabilities
        """
        model = self.sequence_model
        ids_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            outputs = model(ids_tensor)
            logits = outputs["logits"][:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.size(-1)))

        return {
            "top_tokens": top_ids[0].tolist(),
            "top_probs": top_probs[0].tolist(),
        }

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """Generate a sequence.

        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated token IDs
        """
        model = self.sequence_model
        ids_tensor = torch.tensor([input_ids], device=self.device)

        generated = model.generate(
            ids_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return generated[0].tolist()

    def embed(self, input_ids: List[int]) -> List[float]:
        """Get embeddings for a sequence.

        Args:
            input_ids: Token IDs to embed

        Returns:
            Mean-pooled embedding vector
        """
        model = self.sequence_model
        ids_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            outputs = model(ids_tensor, return_hidden=True)
            # Use last hidden state, mean pooled
            hidden = outputs["hidden_states"][-1]
            embedding = hidden.mean(dim=1)

        return embedding[0].tolist()

    def predict_world(
        self,
        joint_pos: List[float],
        joint_delta: List[float],
    ) -> Dict[str, Any]:
        """Predict next world state.

        Args:
            joint_pos: Current joint positions
            joint_delta: Action (joint deltas)

        Returns:
            Predicted next state with uncertainty
        """
        state = WorldModelState(joint_pos=joint_pos)
        action = WorldModelAction(joint_delta=joint_delta)

        result = self.world_model.predict(state, action)

        return {
            "next_joint_pos": result.next_state.joint_pos,
            "uncertainty": result.uncertainty,
            "debug": result.debug,
        }

    def rollout(
        self,
        joint_pos: List[float],
        actions: List[List[float]],
    ) -> Dict[str, Any]:
        """Rollout a sequence of actions.

        Args:
            joint_pos: Starting joint positions
            actions: List of joint deltas

        Returns:
            Trajectory of states
        """
        state = WorldModelState(joint_pos=joint_pos)
        action_objs = [WorldModelAction(joint_delta=a) for a in actions]

        final_state, results = self.world_model.rollout(state, action_objs)

        return {
            "final_joint_pos": final_state.joint_pos,
            "trajectory": [
                {
                    "joint_pos": r.next_state.joint_pos,
                    "uncertainty": r.uncertainty,
                }
                for r in results
            ],
        }

    def get_info(self) -> Dict[str, Any]:
        """Get skill information."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "config": {
                "loop_type": self.config.loop_type,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "use_ssm": self.config.use_ssm,
                "use_spectral": self.config.use_spectral,
                "use_attention": self.config.use_attention,
            },
            "capabilities": list(self._capabilities.keys()),
            "parameters": self.sequence_model.count_parameters() if self._sequence_model else "not loaded",
        }

    def set_loop_type(self, loop_type: str) -> None:
        """Switch to a different loop configuration.

        Args:
            loop_type: One of "fast", "mid", "slow"
        """
        if loop_type == "fast":
            self.config = MambaWaveConfig.fast_loop()
        elif loop_type == "mid":
            self.config = MambaWaveConfig.mid_loop()
        elif loop_type == "slow":
            self.config = MambaWaveConfig.slow_loop()
        else:
            raise ValueError(f"Unknown loop type: {loop_type}")

        # Reset models to reload with new config
        self._sequence_model = None
        self._world_model = None
        logger.info(f"Switched to {loop_type} loop configuration")
