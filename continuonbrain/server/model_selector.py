"""
Unified model selection (JAX-first) for server/startup.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import importlib.util
import os


@dataclass
class ModelChoice:
    id: str
    name: str
    backend: str  # jax | transformers | mock
    available: bool
    reason: str


def detect_models() -> List[ModelChoice]:
    """
    Detect available model backends with JAX-first preference.
    """
    choices: List[ModelChoice] = []
    jax_available = importlib.util.find_spec("jax") is not None
    prefer_jax_env = os.environ.get("CONTINUON_PREFER_JAX", "1").lower() in ("1", "true", "yes")

    if jax_available:
        choices.append(
            ModelChoice(
                id="jax-core",
                name="JAX CoreModel",
                backend="jax",
                available=True,
                reason="JAX detected; CoreModel available",
            )
        )

    # Transformers/Gemma availability (best-effort check)
    tf_available = importlib.util.find_spec("transformers") is not None
    if tf_available:
        choices.append(
            ModelChoice(
                id="gemma",
                name="Gemma/transformers",
                backend="transformers",
                available=True,
                reason="transformers detected",
            )
        )

    # Always include mock fallback
    choices.append(
        ModelChoice(
            id="mock",
            name="Mock Chat",
            backend="mock",
            available=True,
            reason="Built-in mock",
        )
    )

    # Sort: JAX-first (if preferred), then transformers, then mock
    def sort_key(c: ModelChoice):
        if c.backend == "jax":
            return 0 if prefer_jax_env else 1
        if c.backend == "transformers":
            return 1
        return 2

    choices.sort(key=sort_key)
    return choices


def select_model() -> Dict[str, Any]:
    """
    Select best available model. Returns a dict with selected model info and list of candidates.
    """
    choices = detect_models()
    selected = None
    for c in choices:
        if c.available:
            selected = c
            break

    return {
        "selected": selected.__dict__ if selected else None,
        "candidates": [c.__dict__ for c in choices],
    }

