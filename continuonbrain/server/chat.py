"""
Chat service selection.

Prefers JAX-based inference when available; falls back to transformers/Gemma.
"""

import os
from pathlib import Path
from typing import Any, Optional

from continuonbrain.gemma_chat import create_gemma_chat

# TODO: integrate JAX-based chat/inference router when available.


def build_chat_service() -> Optional[Any]:
    """
    Build chat service instance.

    Respects CONTINUON_PREFER_JAX (default: 1) to skip transformers on startup.
    Returns None if JAX path is preferred or if initialization fails.
    """
    prefer_jax = os.environ.get("CONTINUON_PREFER_JAX", "1").lower() in ("1", "true", "yes")
    if prefer_jax:
        print("  CONTINUON_PREFER_JAX=1 -> skipping transformers chat; use JAX router when available.")
        return None

    # Detect Hailo accelerator for offloading
    accelerator_device = None
    try:
        hailo_devices = list(Path("/dev").glob("hailo*"))
        if hailo_devices:
            accelerator_device = "hailo8l"
            print(f"  Detected Hailo AI HAT+ ({len(hailo_devices)} device(s)) - will use for offloading")
    except Exception:
        pass

    try:
        return create_gemma_chat(use_mock=False, accelerator_device=accelerator_device)
    except Exception as exc:  # noqa: BLE001
        print(f"  Gemma chat initialization failed ({exc}); continuing without transformers.")
        return None

