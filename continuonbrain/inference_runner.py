"""
Inference runner with Hailo-first routing and CPU fallback.

Usage:
    runner = InferenceRunner(
        use_hailo=True,
        hef_path=Path("/opt/continuonos/brain/model/base_model/model.hef"),
        cpu_fn=your_cpu_forward_fn,
    )
    out = runner(inputs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .hailo_inference import build_hailo_first_inference


class InferenceRunner:
    def __init__(
        self,
        use_hailo: bool,
        hef_path: Path,
        cpu_fn: Callable[[Any], Any],
    ):
        self._infer = build_hailo_first_inference(
            use_hailo=use_hailo,
            hef_path=hef_path,
            cpu_fn=cpu_fn,
        )

    def __call__(self, inputs: Any) -> Any:
        return self._infer(inputs)

