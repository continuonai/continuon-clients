"""Import-safe loader for the Mamba world model implementation.

Loads `continuonbrain/03_mamba_brain/world_model.py` under a valid module path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_impl() -> ModuleType:
    here = Path(__file__).resolve()
    # continuonbrain/mamba_brain/world_model.py -> continuonbrain/03_mamba_brain/world_model.py
    impl_path = here.parents[1] / "03_mamba_brain" / "world_model.py"
    module_name = "continuonbrain.mamba_brain._world_model_impl"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load world model impl from {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_impl = _load_impl()

# Re-export API surface.
BaseWorldModel = getattr(_impl, "BaseWorldModel")
MambaWorldModel = getattr(_impl, "MambaWorldModel")
StubSSMWorldModel = getattr(_impl, "StubSSMWorldModel")
WorldModelState = getattr(_impl, "WorldModelState")
WorldModelAction = getattr(_impl, "WorldModelAction")
WorldModelPredictResult = getattr(_impl, "WorldModelPredictResult")
build_world_model = getattr(_impl, "build_world_model")

__all__ = [
    "BaseWorldModel",
    "MambaWorldModel",
    "StubSSMWorldModel",
    "WorldModelState",
    "WorldModelAction",
    "WorldModelPredictResult",
    "build_world_model",
]


