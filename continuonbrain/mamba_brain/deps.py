"""Backward compatibility - dependency checking now in mambawave.

MambaWave handles its own optional dependencies internally.
This module is kept for backward compatibility with code that imports:
    from continuonbrain.mamba_brain.deps import has_mamba, load_mamba
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MambaDeps:
    """Mamba dependency status."""

    available: bool
    reason: Optional[str] = None
    module: Optional[Any] = None


def has_mamba() -> bool:
    """Return True if mamba_ssm is importable."""
    return importlib.util.find_spec("mamba_ssm") is not None


def load_mamba() -> MambaDeps:
    """Load the Mamba dependency if present."""
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None:
        return MambaDeps(
            available=False,
            reason="Missing optional dependency 'mamba_ssm'. MambaWave will use spectral-only mode.",
            module=None,
        )
    try:
        import mamba_ssm

        return MambaDeps(available=True, module=mamba_ssm, reason=None)
    except Exception as exc:
        return MambaDeps(
            available=False,
            reason=f"Failed to import mamba_ssm: {exc}",
            module=None,
        )


__all__ = ["MambaDeps", "has_mamba", "load_mamba"]
