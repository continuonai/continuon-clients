"""
Chat service selection.

Prefers JAX-based inference when available; falls back to transformers/Gemma.
"""

import os
from pathlib import Path
from typing import Any, Optional

from continuonbrain.gemma_chat import create_gemma_chat

# Try JAX/Flax Gemma first (multimodal, consistent with training pipeline)
try:
    from continuonbrain.gemma_chat_jax import create_gemma_chat_jax
    JAX_GEMMA_AVAILABLE = True
except ImportError:
    JAX_GEMMA_AVAILABLE = False
    create_gemma_chat_jax = None


def build_chat_service() -> Optional[Any]:
    """
    Build chat service instance.

    Prefers JAX/Flax Gemma (multimodal, consistent with training pipeline).
    Falls back to transformers/Gemma if JAX not available.
    Respects CONTINUON_PREFER_JAX (default: 1).
    """
    prefer_jax = os.environ.get("CONTINUON_PREFER_JAX", "1").lower() in ("1", "true", "yes")
    
    # Detect Hailo accelerator for offloading
    accelerator_device = None
    try:
        hailo_devices = list(Path("/dev").glob("hailo*"))
        if hailo_devices:
            accelerator_device = "hailo8l"
            print(f"  Detected Hailo AI HAT+ ({len(hailo_devices)} device(s)) - will use for offloading")
    except Exception:
        pass
    
    # Only attempt JAX/Flax chat when explicitly preferred.
    # (Most Gemma 3n multimodal checkpoints are PyTorch-first; JAX paths may differ.)
    if prefer_jax and JAX_GEMMA_AVAILABLE and create_gemma_chat_jax:
        try:
            print("  CONTINUON_PREFER_JAX=1 -> attempting JAX/Flax Gemma chat backend...")
            # Prefer text-only Flax Gemma variants if present; keep this list conservative.
            model_candidates = [
                "google/gemma-2b-it-flax",
                "google/gemma-2-2b-it",
            ]
            for model_name in model_candidates:
                try:
                    chat_jax = create_gemma_chat_jax(
                        model_name=model_name,
                        device="cpu",
                        accelerator_device=accelerator_device,
                    )
                    if chat_jax:
                        print(f"  ✅ Using {model_name} (JAX/Flax)")
                        return chat_jax
                except Exception:
                    continue
        except Exception as exc:  # noqa: BLE001
            print(f"  JAX/Flax Gemma initialization failed ({exc}); falling back to transformers...")
    
    # Fallback to transformers if JAX not available or failed
    if not prefer_jax:
        try:
            print("  Falling back to transformers Gemma...")
            return create_gemma_chat(use_mock=False, accelerator_device=accelerator_device)
        except Exception as exc:  # noqa: BLE001
            print(f"  Transformers Gemma initialization failed ({exc}); continuing without chat.")
            return None
    else:
        print("  CONTINUON_PREFER_JAX=1 and JAX Gemma unavailable -> no chat service.")
        return None

