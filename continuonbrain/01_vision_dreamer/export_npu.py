"""Export helpers for pushing Vision Dreamer graphs onto NPUs.

This file is intentionally lightweight and does not depend on the Hailo
SDK. It creates placeholder HEF artifacts and explains where hardware-
specific binding should occur so the Continuon Brain runtime can opt
into acceleration when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .model_vqgan import VQGANConfig, export_hailo_vqgan


def prepare_export_config(codebook_size: int = 1024) -> Dict[str, Any]:
    """Build an export configuration dictionary.

    Training pipelines can extend this dictionary with dataset metadata
    while inference flows can simply reuse the defaults to generate a
    placeholder HEF for integration tests.
    """

    return {
        "codebook_size": codebook_size,
        "export_note": "Populate with dataset + preprocessing details for training exports.",
    }


def run_hailo_export(output_dir: Path, config: VQGANConfig | None = None) -> Path:
    """Create a placeholder HEF file for Hailo compilation.

    In production this function would invoke Hailo's model zoo or TAPPAS
    export routines. Keeping the stub separate from the runtime allows
    CI and device bring-up to run without the SDK installed. The returned
    HEF path can be injected into :class:`~continuonbrain.01_vision_dreamer.model_vqgan.VQGANConfig`
    so inference can attempt to bind to Hailo before falling back to CPU.
    """

    config = config or VQGANConfig()
    return export_hailo_vqgan(config, output_dir)


if __name__ == "__main__":
    destination = Path("./artifacts")
    hef = run_hailo_export(destination)
    print(f"Stub HEF exported to {hef}")
