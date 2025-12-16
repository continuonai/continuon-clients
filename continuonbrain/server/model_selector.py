"""
Unified model selection (JAX-first) for server/startup.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import importlib.util
import os
from pathlib import Path


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
        # Seed manifest (exported wavecore loops)
        manifest_path = Path("/opt/continuonos/brain/model/adapters/candidate/core_model_seed/model_manifest.json")
        if manifest_path.exists():
            choices.append(
                ModelChoice(
                    id="jax-core-seed",
                    name="JAX CoreModel Seed",
                    backend="jax",
                    available=True,
                    reason=f"Found seed manifest at {manifest_path}",
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

    # LiteRT availability
    try:
        from continuonbrain.services.chat.litert_chat import HAS_LITERT
        if HAS_LITERT:
             choices.append(
                ModelChoice(
                    id="litert-gemma",
                    name="LiteRT Gemma",
                    backend="litert",
                    available=True,
                    reason="MediaPipe GenAI detected",
                )
             )
    except ImportError:
        pass

    # Sort: JAX-first (if preferred), then LiteRT (if preferred), then transformers, then mock
    prefer_litert = os.environ.get("CONTINUON_PREFER_LITERT", "0").lower() in ("1", "true", "yes")

    def sort_key(c: ModelChoice):
        if c.backend == "jax":
            # If JAX is preferred, it's 0. If LiteRT is preferred, JAX is 1. Else 0.
            if prefer_litert:
                return 1
            return 0 if prefer_jax_env else 1
        if c.backend == "litert":
            return 0 if prefer_litert else 1
        if c.backend == "transformers":
            return 2
        return 3

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

