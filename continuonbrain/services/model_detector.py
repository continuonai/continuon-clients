"""
Model Detection Service

Detects available chat models for Agent Manager selection.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
import importlib.util

logger = logging.getLogger(__name__)


class ModelDetector:
    """Detects available chat models from various sources."""
    
    def __init__(self):
        self.hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        self.jax_available = importlib.util.find_spec("jax") is not None
        self.transformers_available = importlib.util.find_spec("transformers") is not None
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available chat models.
        
        Returns:
            List of model dictionaries with keys: id, name, type, size_mb, source
        """
        models = []
        
        # Prefer JAX CoreModel if JAX is present
        if self.jax_available:
            models.append({
                "id": "jax-core",
                "name": "JAX CoreModel",
                "type": "jax",
                "size_mb": 0,
                "source": "built-in",
                "description": "JAX/Flax CoreModel (HOPE-inspired) for CPU/TPU/Hailo inference"
            })

        # Detect Gemma 3n (JAX/Flax weights) if present in HF cache
        if self.jax_available and self.hf_cache.exists():
            for model_dir in self.hf_cache.iterdir():
                if not model_dir.is_dir():
                    continue
                dir_name = model_dir.name
                if "gemma-3" in dir_name.lower():
                    models.append({
                        "id": "google/gemma-3n",
                        "name": "Gemma 3n (JAX/Flax)",
                        "type": "jax-gemma",
                        "size_mb": 0,
                        "source": "huggingface",
                        "description": "Gemma 3n JAX/Flax weights detected in HF cache"
                    })

        # Always include Mock model
        models.append({
            "id": "mock",
            "name": "Mock Gemma (Testing)",
            "type": "mock",
            "size_mb": 0,
            "source": "built-in",
            "description": "Placeholder responses for testing (no AI inference)"
        })
        
        # Check for custom VLA multimodal chat
        vla_path = Path(__file__).parent.parent / "vla_chat.py"
        if vla_path.exists():
            models.append({
                "id": "vla-custom",
                "name": "VLA Multimodal Chat",
                "type": "vla",
                "size_mb": 0,  # Size unknown
                "source": "custom",
                "description": "Custom Vision-Language-Action model"
            })
        
        # Add HOPE Logic Model (User Developed)
        models.append({
            "id": "hope-v1",
            "name": "HOPE Logic Model",
            "type": "agent",
            "size_mb": 0,
            "source": "built-in",
            "description": "Proprietary Cognitive Architecture (Continuon)"
        })
        
        # Detect downloaded HuggingFace models
        if self.hf_cache.exists():
            gemma_models = self._detect_gemma_models()
            models.extend(gemma_models)
        
        return models
    
    def _detect_gemma_models(self) -> List[Dict[str, Any]]:
        """Detect all models in HuggingFace cache."""
        models = []
        
        try:
            # List all model directories in cache
            for model_dir in self.hf_cache.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # HF cache names are like: models--google--gemma-2b-it
                dir_name = model_dir.name
                if dir_name.startswith("models--"):
                    # Parse model ID from directory name
                    # Format: models--org--model-name
                    parts = dir_name.split("--")
                    if len(parts) >= 3:
                        org = parts[1]
                        model_name = "--".join(parts[2:])
                        model_id = f"{org}/{model_name}"
                        
                        # Estimate size from snapshots directory (only largest snapshot)
                        size_mb = self._estimate_model_size(model_dir)
                        
                        models.append({
                            "id": model_id,
                            "name": model_id, # Use full ID as name
                            "type": "gemma" if "gemma" in model_id.lower() else "transformer",
                            "size_mb": size_mb,
                            "source": "huggingface",
                            "description": f"Local Model (~{size_mb}MB)"
                        })
        except Exception as e:
            logger.warning(f"Error detecting models: {e}")
            
        return models
    
    def _estimate_model_size(self, model_dir: Path) -> int:
        """Estimate model size in MB by checking ONLY the largest snapshot."""
        try:
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                return 0
            
            # Find the largest snapshot (likely the main one)
            max_size = 0
            
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    current_snap_size = 0
                    for file in snapshot.rglob("*"):
                        if file.is_file():
                            current_snap_size += file.stat().st_size
                    
                    if current_snap_size > max_size:
                        max_size = current_snap_size
            
            return int(max_size / (1024 * 1024))  # Convert to MB
        except Exception:
            return 0
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is available."""
        available = self.get_available_models()
        return any(m["id"] == model_id for m in available)
