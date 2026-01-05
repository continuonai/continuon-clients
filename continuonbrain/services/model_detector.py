"""
Model Detection Service

Detects available chat and vision models for Agent Manager selection.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
import importlib.util

logger = logging.getLogger(__name__)


class ModelDetector:
    """Detects available chat and vision models from various sources."""
    
    def __init__(self):
        self.hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        self.jax_available = importlib.util.find_spec("jax") is not None
        self.transformers_available = importlib.util.find_spec("transformers") is not None
        self.torch_available = importlib.util.find_spec("torch") is not None
        
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

        # Detect Gemma 3n 2B multimodal (JAX/Flax weights) if present in HF cache
        if self.hf_cache.exists():
            for model_dir in self.hf_cache.iterdir():
                if not model_dir.is_dir():
                    continue
                dir_name = model_dir.name.lower()
                # Gemma 3n 2B multimodal (preferred for training)
                if "gemma-3n" in dir_name and "2b" in dir_name:
                    models.append({
                        "id": "google/gemma-3n-2b",
                        "name": "Gemma 3n 2B Multimodal",
                        "type": "multimodal",
                        "size_mb": self._estimate_model_size(model_dir),
                        "source": "huggingface",
                        "multimodal": True,
                        "supports_audio": True,
                        "supports_vision": True,
                        "description": "Gemma 3n 2B multimodal for vision+audio+text (ideal for HOPE training)",
                        "recommended_for": ["training", "multimodal_inference", "distillation"]
                    })
                # Other Gemma 3n variants
                elif "gemma-3" in dir_name:
                    models.append({
                        "id": "google/gemma-3n",
                        "name": "Gemma 3n (JAX/Flax)",
                        "type": "jax-gemma",
                        "size_mb": self._estimate_model_size(model_dir),
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
    
    def get_vision_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available vision models for segmentation, detection, etc.
        
        Returns:
            List of vision model dictionaries
        """
        models = []
        
        # SAM3 (Segment Anything Model 3)
        sam3_available = self._check_sam3_available()
        if sam3_available:
            models.append({
                "id": "facebook/sam3",
                "name": "SAM3 (Segment Anything Model 3)",
                "type": "segmentation",
                "size_mb": self._get_cached_model_size("facebook/sam3"),
                "source": "huggingface",
                "capabilities": ["segmentation", "detection", "tracking", "text_prompt"],
                "description": "Promptable concept segmentation with 4M+ labels",
                "recommended_for": ["object_detection", "manipulation", "scene_understanding"],
            })
        
        # SAM2 (if available)
        sam2_available = self._check_model_in_cache("facebook/sam2")
        if sam2_available:
            models.append({
                "id": "facebook/sam2",
                "name": "SAM2 (Segment Anything 2)",
                "type": "segmentation",
                "size_mb": self._get_cached_model_size("facebook/sam2"),
                "source": "huggingface",
                "capabilities": ["segmentation", "video_tracking"],
                "description": "Video-capable segmentation model",
            })
        
        # GroundingDINO (for open-vocab detection)
        grounding_dino = self._check_model_in_cache("IDEA-Research/grounding-dino")
        if grounding_dino:
            models.append({
                "id": "IDEA-Research/grounding-dino",
                "name": "Grounding DINO",
                "type": "detection",
                "size_mb": self._get_cached_model_size("IDEA-Research/grounding-dino"),
                "source": "huggingface",
                "capabilities": ["detection", "text_prompt", "open_vocabulary"],
                "description": "Open-vocabulary object detection",
            })
        
        # YOLO models (check for local .pt files)
        yolo_paths = [
            Path("/opt/continuonos/brain/models/yolo"),
            Path.home() / ".continuon" / "models" / "yolo",
        ]
        for yolo_dir in yolo_paths:
            if yolo_dir.exists():
                for pt_file in yolo_dir.glob("*.pt"):
                    models.append({
                        "id": f"yolo/{pt_file.stem}",
                        "name": f"YOLO ({pt_file.stem})",
                        "type": "detection",
                        "size_mb": int(pt_file.stat().st_size / (1024 * 1024)),
                        "source": "local",
                        "capabilities": ["detection", "fast"],
                        "path": str(pt_file),
                    })
        
        # Hailo-compiled vision models (.hef files)
        hef_paths = [
            Path("/opt/continuonos/brain/models/hailo"),
            Path("/usr/share/hailo-models"),
        ]
        for hef_dir in hef_paths:
            if hef_dir.exists():
                for hef_file in hef_dir.glob("*.hef"):
                    model_type = "detection"
                    if "seg" in hef_file.stem.lower():
                        model_type = "segmentation"
                    elif "pose" in hef_file.stem.lower():
                        model_type = "pose_estimation"
                    
                    models.append({
                        "id": f"hailo/{hef_file.stem}",
                        "name": f"Hailo {hef_file.stem}",
                        "type": model_type,
                        "size_mb": int(hef_file.stat().st_size / (1024 * 1024)),
                        "source": "hailo",
                        "capabilities": ["detection", "accelerated", "real_time"],
                        "path": str(hef_file),
                        "accelerator": "hailo",
                    })
        
        return models
    
    def _check_sam3_available(self) -> bool:
        """Check if SAM3 dependencies and model access are available."""
        if not self.torch_available or not self.transformers_available:
            return False
        
        # Check if model is in cache
        if self._check_model_in_cache("facebook/sam3"):
            return True
        
        # Check if transformers has SAM3 support
        try:
            from transformers import Sam3Model, Sam3Processor
            return True
        except ImportError:
            return False
    
    def _check_model_in_cache(self, model_id: str) -> bool:
        """Check if a model is downloaded in HuggingFace cache."""
        if not self.hf_cache.exists():
            return False
        
        # Convert model_id to cache directory name
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.hf_cache / cache_name
        
        if model_path.exists() and model_path.is_dir():
            # Check for actual model files
            snapshots = model_path / "snapshots"
            if snapshots.exists():
                return any(snapshots.iterdir())
        
        return False
    
    def _get_cached_model_size(self, model_id: str) -> int:
        """Get size of a cached model in MB."""
        if not self.hf_cache.exists():
            return 0
        
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_path = self.hf_cache / cache_name
        
        if model_path.exists():
            return self._estimate_model_size(model_path)
        return 0
    
    def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available models, categorized by type."""
        return {
            "chat": self.get_available_models(),
            "vision": self.get_vision_models(),
        }
    
    def get_recommended_vision_model(self, task: str = "segmentation") -> Dict[str, Any]:
        """
        Get recommended vision model for a specific task.
        
        Args:
            task: "segmentation", "detection", "pose_estimation"
            
        Returns:
            Best available model for the task
        """
        vision_models = self.get_vision_models()
        
        # Priority order for each task (SAM3 preferred for segmentation)
        priorities = {
            "segmentation": ["facebook/sam3", "facebook/sam2", "hailo/"],
            "detection": ["hailo/", "IDEA-Research/grounding-dino", "yolo/"],
            "pose_estimation": ["hailo/"],
        }
        
        priority_list = priorities.get(task, [])
        
        # Find best match based on priority
        for prefix in priority_list:
            for model in vision_models:
                if model["id"].startswith(prefix) or model["id"] == prefix.rstrip("/"):
                    return model
        
        # Fallback to any model of the right type
        for model in vision_models:
            if model.get("type") == task:
                return model
        
        # No model found
        return {"id": None, "name": "No suitable model found", "available": False}
