"""Token-to-action bridge for reflexive control.

This scaffolding sketches how the Continuon Brain runtime could feed
vision tokens into a lightweight reflex policy before the heavier Mamba
loop consumes them. Training and inference share the same interface so
runtime orchestrators can hot-swap implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass
class ReflexConfig:
    """Configuration for :class:`ReflexNet` placeholders.

    Attributes:
        action_dim: Number of actuator slots to emit per step.
        token_window: How many tokens are aggregated per decision.
    """

    action_dim: int = 4
    token_window: int = 32


@dataclass
class ReflexNet:
    """Minimal reflex policy stub.

    The runtime would feed tokens from the Vision Dreamer path into
    :meth:`step` and forward the resulting actions to the motor loop.
    Training can override :meth:`step` with a learned policy; inference
    can keep the deterministic mapping here to validate end-to-end
    integration without accelerators.
    """

    config: ReflexConfig = field(default_factory=ReflexConfig)

    def step(self, tokens: Sequence[int]) -> List[float]:
        """Generate a simple action vector from tokens.

        Tokens are folded into a fixed number of outputs to simulate a
        fast reflexive response. The Continuon Brain runtime would
        typically stream these actions into the motor loop while
        simultaneously forwarding the same tokens to higher-level models
        (e.g., Mamba) for planning.
        """

        if not tokens:
            return [0.0] * self.config.action_dim
        return [float(tokens[i % len(tokens)] % 100) / 100.0 for i in range(self.config.action_dim)]

    def batch_step(self, batch_tokens: Iterable[Sequence[int]]) -> List[List[float]]:
        """Apply :meth:`step` across a batch to mimic vectorized inference."""

        return [self.step(tokens) for tokens in batch_tokens]


__all__ = ["ReflexNet", "ReflexConfig"]
