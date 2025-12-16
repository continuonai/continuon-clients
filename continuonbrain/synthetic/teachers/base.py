from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TeacherResult:
    """
    A teacher can provide any combination of:
    - embedding: a numeric vector to use as observation.command (preferred if present)
    - caption: a short image caption / situation description
    - planner: a lightweight plan/intention trace
    - action_command: a suggested action vector (imitation / bootstrapping)
    - extra: freeform metadata for step_metadata/tool_calls
    """

    embedding: Optional[List[float]] = None
    caption: Optional[str] = None
    planner: Optional[Dict[str, Any]] = None
    action_command: Optional[List[float]] = None
    extra: Optional[Dict[str, Any]] = None


class Teacher:
    """Interface for small helper LLM/VLA teachers used during synthetic episode generation."""

    def infer(
        self,
        *,
        rgb_bgr: np.ndarray,
        depth: Optional[np.ndarray],
        prompt: str,
        obs_dim: int,
        action_dim: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> TeacherResult:
        raise NotImplementedError


