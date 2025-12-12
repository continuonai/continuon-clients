"""Mamba world model interface (optional-dep).

This module defines a stable predictor API for planning/search:
  predict(state, action) -> next_state (+ uncertainty, debug)

If `mamba_ssm` is installed, `MambaWorldModel` will use it.
Otherwise, it falls back to a deterministic SSM-like stub suitable for
end-to-end wiring and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

from .deps import load_mamba


@dataclass
class WorldModelState:
    """Compact state for the planner/world model (arm-focused initial scope)."""

    joint_pos: List[float]


@dataclass
class WorldModelAction:
    """Compact action for the planner/world model (arm-focused initial scope)."""

    joint_delta: List[float]


@dataclass
class WorldModelPredictResult:
    next_state: WorldModelState
    uncertainty: float
    debug: Dict[str, Any]


class BaseWorldModel:
    def predict(self, state: WorldModelState, action: WorldModelAction) -> WorldModelPredictResult:  # noqa: D401
        """Predict next state from current state + action."""
        raise NotImplementedError

    def rollout(
        self,
        state: WorldModelState,
        actions: Sequence[WorldModelAction],
    ) -> Tuple[WorldModelState, List[WorldModelPredictResult]]:
        current = state
        results: List[WorldModelPredictResult] = []
        for action in actions:
            out = self.predict(current, action)
            results.append(out)
            current = out.next_state
        return current, results


class StubSSMWorldModel(BaseWorldModel):
    """Deterministic fallback used when real Mamba kernels are unavailable."""

    def __init__(self, joint_limit: float = 1.0):
        self.joint_limit = float(joint_limit)

    def predict(self, state: WorldModelState, action: WorldModelAction) -> WorldModelPredictResult:
        joint_pos = list(state.joint_pos)
        delta = list(action.joint_delta)
        if len(delta) != len(joint_pos):
            # Pad/truncate defensively.
            if len(delta) < len(joint_pos):
                delta = delta + [0.0] * (len(joint_pos) - len(delta))
            else:
                delta = delta[: len(joint_pos)]

        next_pos = []
        for j, dj in zip(joint_pos, delta):
            val = float(j) + float(dj)
            val = max(-self.joint_limit, min(self.joint_limit, val))
            next_pos.append(val)

        # A tiny heuristic uncertainty: larger deltas are "less certain".
        mag = math.sqrt(sum(float(d) * float(d) for d in delta) / max(1, len(delta)))
        uncertainty = min(1.0, 0.05 + mag)
        return WorldModelPredictResult(
            next_state=WorldModelState(joint_pos=next_pos),
            uncertainty=float(uncertainty),
            debug={"backend": "stub_ssm", "delta_rms": mag},
        )


class MambaWorldModel(BaseWorldModel):
    """Real Mamba-backed world model if optional dependency is installed.

    Note: this is an integration layer, not a full training pipeline.
    It assumes a trained checkpoint exists and focuses on inference wiring.
    """

    def __init__(
        self,
        *,
        joint_dim: int = 6,
        joint_limit: float = 1.0,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.joint_dim = int(joint_dim)
        self.joint_limit = float(joint_limit)
        self.checkpoint_path = checkpoint_path

        deps = load_mamba()
        self._deps = deps
        if not deps.available:
            # Fall back silently to stub behavior (keeps runtime importable).
            self._stub = StubSSMWorldModel(joint_limit=joint_limit)
            self._model = None
            return

        # Minimal runtime usage: we keep actual construction guarded.
        # Different mamba_ssm versions expose different APIs; we handle this best-effort.
        self._model = None
        self._stub = None

        try:
            # Common API in mamba_ssm: Mamba or MambaLMHeadModel (varies by package).
            # We don't hard-require a specific class; users can swap this file when integrating a real checkpoint.
            self._model = getattr(deps.module, "Mamba", None) or getattr(deps.module, "MambaLMHeadModel", None)
        except Exception:
            self._model = None

        if self._model is None:
            # If we can’t locate an entry point, keep a stub fallback and report via debug.
            self._stub = StubSSMWorldModel(joint_limit=joint_limit)

    def predict(self, state: WorldModelState, action: WorldModelAction) -> WorldModelPredictResult:
        # If Mamba is not usable, fall back.
        if self._stub is not None:
            out = self._stub.predict(state, action)
            out.debug["backend"] = "mamba_missing_api_fallback"
            out.debug["reason"] = self._deps.reason
            return out

        # Best-effort “real” behavior: until a checkpoint is integrated, we still do a safe deterministic step,
        # but label it clearly so callers never mistake it for a trained Mamba predictor.
        # This preserves the plan’s optional-dep contract without breaking deployments.
        safe = StubSSMWorldModel(joint_limit=self.joint_limit).predict(state, action)
        safe.debug["backend"] = "mamba_integration_placeholder"
        safe.debug["checkpoint_path"] = self.checkpoint_path
        return safe


def build_world_model(prefer_mamba: bool = True, **kwargs: Any) -> BaseWorldModel:
    """Factory that chooses Mamba if available, else stub."""
    if prefer_mamba:
        return MambaWorldModel(**kwargs)
    return StubSSMWorldModel(joint_limit=float(kwargs.get("joint_limit", 1.0)))


