"""Optional dependency loader for Mamba/SSM implementations.

This repository must stay importable on Pi-class devices without heavy ML deps.
We therefore gate any third-party Mamba kernels behind runtime checks.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MambaDeps:
    available: bool
    reason: Optional[str] = None
    module: Optional[Any] = None


def has_mamba() -> bool:
    """Return True if a supported Mamba implementation is importable."""
    return importlib.util.find_spec("mamba_ssm") is not None


def load_mamba() -> MambaDeps:
    """Load the Mamba dependency if present."""
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None:
        return MambaDeps(
            available=False,
            reason="Missing optional dependency 'mamba_ssm'. Install it to enable the MambaWorldModel.",
            module=None,
        )
    try:
        import mamba_ssm  # type: ignore

        return MambaDeps(available=True, module=mamba_ssm, reason=None)
    except Exception as exc:  # noqa: BLE001
        return MambaDeps(available=False, reason=f"Failed to import mamba_ssm: {exc}", module=None)


