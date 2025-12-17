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
from typing import Any, Callable, Optional, List, Dict

try:
    import hailo_platform as hailo  # type: ignore

    HAILO_AVAILABLE = True
    HAILO_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    HAILO_AVAILABLE = False
    HAILO_IMPORT_ERROR = str(exc)


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
        self._input_infos: List[Any] = []
        self._output_infos: List[Any] = []
        self._init_error: Optional[str] = None

    def _try_init(self) -> bool:
        if not HAILO_AVAILABLE:
            return False
        if not self.hef_path.exists():
            return False
        try:
            self._hef = hailo.HEF(str(self.hef_path))
            # Capture vstream metadata for downstream wiring; we intentionally
            # avoid configuring/running here to keep initialization lightweight.
            self._input_infos = list(self._hef.get_input_vstream_infos())
            self._output_infos = list(self._hef.get_output_vstream_infos())
            self._configured = True
            return True
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)
            self._configured = False
            return False

    def run(self, inputs: Any) -> Any:
        """Run inference on Hailo if available; otherwise CPU fallback."""
        if self._configured or self._try_init():
            try:
                # Placeholder: Real implementation must map inputs to VStreams,
                # run inference, and collect outputs. We surface metadata here
                # so downstream code can wire real tensor I/O without guessing.
                result: Dict[str, Any] = {
                    "hailo": True,
                    "hef_path": str(self.hef_path),
                    "inputs": [
                        {
                            "name": info.name,
                            "shape": tuple(info.shape),
                            "format": getattr(info.format, "order", None),
                        }
                        for info in self._input_infos
                    ],
                    "outputs": [
                        {
                            "name": info.name,
                            "shape": tuple(info.shape),
                            "format": getattr(info.format, "order", None),
                        }
                        for info in self._output_infos
                    ],
                    "note": "Hailo runtime detected; tensor I/O not yet wired",
                }
                if self.cpu_fallback:
                    result["cpu_fallback_result"] = self.cpu_fallback(inputs)
                return result
            except Exception:
                # Fall through to CPU
                pass
        if self.cpu_fallback:
            return self.cpu_fallback(inputs)
        raise RuntimeError("No Hailo available and no CPU fallback provided.")


def build_hailo_first_inference(
    use_hailo: bool,
    hef_path: Path,
    cpu_fn: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """
    Return an inference callable that prefers Hailo when enabled/available,
    else falls back to the provided cpu_fn.
    """
    if not use_hailo:
        return cpu_fn
    runner = HailoRunner(hef_path=hef_path, cpu_fallback=cpu_fn)

    def _fn(inputs: Any) -> Any:
        return runner.run(inputs)

    return _fn

