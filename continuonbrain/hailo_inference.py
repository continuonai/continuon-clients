"""
Hailo-first inference helper with CPU fallback.

Usage:
- Place a compiled Hailo HEF at `/opt/continuonos/brain/model/base_model/model.hef`
  (or override via constructor arg).
- Call `HailoRunner.run(inputs)` to attempt Hailo inference; if unavailable, it
  falls back to the provided CPU callable.

This is a placeholder; a real deployment must supply a HEF compiled for the
target model and shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

try:
    import hailort  # type: ignore

    HAILO_AVAILABLE = True
except Exception:
    HAILO_AVAILABLE = False


class HailoRunner:
    def __init__(
        self,
        hef_path: Path = Path("/opt/continuonos/brain/model/base_model/model.hef"),
        cpu_fallback: Optional[Callable[[Any], Any]] = None,
    ):
        self.hef_path = hef_path
        self.cpu_fallback = cpu_fallback
        self._hailo_device = None
        self._hef = None
        self._configured = False

    def _try_init(self) -> bool:
        if not HAILO_AVAILABLE:
            return False
        if not self.hef_path.exists():
            return False
        try:
            self._hef = hailort.Hef(self.hef_path)
            self._hailo_device = hailort.Device()
            self._configured = True
            return True
        except Exception:
            self._configured = False
            return False

    def run(self, inputs: Any) -> Any:
        """Run inference on Hailo if available; otherwise CPU fallback."""
        if self._configured or self._try_init():
            try:
                # Placeholder: Real implementation must map inputs to VStreams,
                # run inference, and collect outputs. This stub only signals
                # that Hailo is selected.
                return {"hailo": True, "note": "Hailo path stub, implement VStream I/O"}
            except Exception:
                # Fall through to CPU
                pass
        if self.cpu_fallback:
            return self.cpu_fallback(inputs)
        raise RuntimeError("No Hailo available and no CPU fallback provided.")

