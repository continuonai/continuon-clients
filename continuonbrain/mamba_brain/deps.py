"""Import-safe loader for the Mamba dependency helper.

Loads `continuonbrain/03_mamba_brain/deps.py` under a valid module path so that
`continuonbrain.mamba_brain.world_model` can use relative imports safely.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_impl() -> ModuleType:
    here = Path(__file__).resolve()
    impl_path = here.parents[1] / "03_mamba_brain" / "deps.py"
    module_name = "continuonbrain.mamba_brain._deps_impl"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load deps impl from {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_impl = _load_impl()

MambaDeps = getattr(_impl, "MambaDeps")
has_mamba = getattr(_impl, "has_mamba")
load_mamba = getattr(_impl, "load_mamba")

__all__ = ["MambaDeps", "has_mamba", "load_mamba"]


