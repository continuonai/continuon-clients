"""
Chat service selection.

Prefers JAX-based inference when available; falls back to transformers/Gemma.
"""

import os
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
        print("ℹ️  CONTINUON_PREFER_JAX=1 → skipping transformers chat; use JAX router when available.")
        return None

    try:
        return create_gemma_chat(use_mock=False)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Gemma chat initialization failed ({exc}); continuing without transformers.")
        return None

