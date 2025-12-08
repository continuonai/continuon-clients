"""
Model Detection Service

Detects available chat models for Agent Manager selection.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelDetector:
    """Detects available chat models from various sources."""
    
    def __init__(self):
        self.hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available chat models.
        
        Returns:
            List of model dictionaries with keys: id, name, type, size_mb, source
        """
        models = []
        
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
        
        # Detect downloaded HuggingFace models
        if self.hf_cache.exists():
            gemma_models = self._detect_gemma_models()
            models.extend(gemma_models)
        
        return models
    
    def _detect_gemma_models(self) -> List[Dict[str, Any]]:
        """Detect Gemma models in HuggingFace cache."""
        models = []
        
        # Common Gemma model IDs to look for
        known_gemma = {
            "google/gemma-2b-it": "Gemma 2B Instruct",
            "google/gemma-3-4b-it": "Gemma 3N 4B Instruct",
            "google/gemma-7b-it": "Gemma 7B Instruct",
        }
        
        try:
            # List all model directories in cache
            for model_dir in self.hf_cache.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # HF cache names are like: models--google--gemma-2b-it
                dir_name = model_dir.name
                if "gemma" in dir_name.lower():
                    # Try to parse model ID from directory name
                    # Format: models--org--model-name
                    parts = dir_name.split("--")
                    if len(parts) >= 3:
                        org = parts[1]
                        model_name = "--".join(parts[2:])
                        model_id = f"{org}/{model_name}"
                        
                        # Check if we know this model
                        if model_id in known_gemma:
                            # Estimate size from snapshots directory
                            size_mb = self._estimate_model_size(model_dir)
                            
                            models.append({
                                "id": model_id,
                                "name": known_gemma[model_id],
                                "type": "gemma",
                                "size_mb": size_mb,
                                "source": "huggingface",
                                "description": f"Downloaded from HuggingFace (~{size_mb}MB)"
                            })
        except Exception as e:
            logger.warning(f"Error detecting Gemma models: {e}")
        
        return models
    
    def _estimate_model_size(self, model_dir: Path) -> int:
        """Estimate model size in MB by checking snapshot files."""
        try:
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                return 0
            
            total_size = 0
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    for file in snapshot.rglob("*"):
                        if file.is_file():
                            total_size += file.stat().st_size
            
            return int(total_size / (1024 * 1024))  # Convert to MB
        except Exception:
            return 0
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is available."""
        available = self.get_available_models()
        return any(m["id"] == model_id for m in available)
