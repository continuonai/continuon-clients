"""Motor loop placeholder for routing reflex outputs to actuators.

The Continuon Brain runtime would typically schedule this alongside the
vision and planning loops. Here we emulate the interaction points so
callers can observe how tokens and actions propagate without hardware
attachments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .reflex_net import ReflexNet, ReflexConfig


@dataclass
class MotorLoop:
    """Coordinate reflex inference and actuator dispatch.

    Inference mode pulls tokens from upstream vision modules and emits a
    synthetic command packet. Training mode can record the same packets
    for RLDS logging or gradient estimation when paired with differentiable
    simulators.
    """

    reflex: ReflexNet = field(default_factory=ReflexNet)

    def run_step(self, tokens: Sequence[int]) -> List[float]:
        """Produce one motor command vector from incoming tokens."""

        actions = self.reflex.step(tokens)
        # Placeholder for hardware binding: pipe actions into servo/ESC drivers.
        return actions

    def run_stream(self, token_batches: Iterable[Sequence[int]]) -> List[List[float]]:
        """Process a stream of token batches for batched inference."""

        return [self.run_step(tokens) for tokens in token_batches]

    @classmethod
    def from_config(cls, config: ReflexConfig) -> "MotorLoop":
        """Instantiate a loop with a specific reflex configuration."""

        return cls(reflex=ReflexNet(config=config))


__all__ = ["MotorLoop"]
