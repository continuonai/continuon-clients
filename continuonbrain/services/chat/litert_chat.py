"""
LiteRT (TensorFlow Lite) backend for Gemma 3N.
"""
import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Try importing LiteRT interpreter or MediaPipe GenAI
try:
    from mediapipe.tasks.python.genai import llm_inference
    HAS_MEDIAPIPE_GENAI = True
except ImportError:
    HAS_MEDIAPIPE_GENAI = False

try:
    import ai_edge_litert.interpreter
    HAS_LITERT = True
except ImportError:
    HAS_LITERT = False

logger = logging.getLogger(__name__)


class LiteRTGemmaChat:
    """
    Gemma 3N backend using LiteRT (TensorFlow Lite) for inference.
    """
    
    DEFAULT_MODEL_ID = "google/gemma-3n-E2B-it-litert-lm"

    def __init__(self, model_name: str = DEFAULT_MODEL_ID, device: str = "cpu", accelerator_device: Optional[str] = None):
        # Verification Mode: If GenAI unavailable, force mock mode for .litertlm support
        if not HAS_MEDIAPIPE_GENAI:
             logger.warning("mediapipe-genai missing. Using MOCK LiteRT mode for verification.")
             self.is_mock = True
        else:
             self.is_mock = False

            
        self.model_name = model_name
        self.device = device
        self.accelerator_device = accelerator_device
        self.agent = None # MediaPipe LlmInference agent
        self.interpreter = None # Backup
        self.loaded = False
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 10
        self.max_tokens = 2048 # Default for MediaPipe LlmInference
        
        # Resolve model path
        self.model_path = self._resolve_model_path(model_name)

    @property
    def is_litert_backend(self) -> bool:
        return True

        
    def _resolve_model_path(self, model_id: str) -> Optional[Path]:
        """
        Find the .tflite model file. 
        Expects a valid path or tries to resolve from HF cache like other backends.
        """
        # 1. Check if model_id is a direct local path
        p = Path(model_id)
        if p.exists() and (p.suffix == ".tflite" or p.suffix == ".litertlm"):
            return p
        
        # 2. Use existing resolution logic from gemma_chat if available, or duplicate it
        # We'll use the one from gemma_chat via import to avoid code duplication
        try:
            from continuonbrain.gemma_chat import _snapshot_download_path, _ensure_hf_cache_env
            
            # For LiteRT/GGUF models on HF, they are often single files inside a repo.
            # snapshots download the whole repo.
            snapshot = _snapshot_download_path(model_id, os.environ.get("HUGGINGFACE_TOKEN"))
            
            # Find the .litertlm file
            tflite_files = list(snapshot.glob("*.litertlm"))
            if not tflite_files:
                 # Fallback to .tflite just in case
                 tflite_files = list(snapshot.glob("*.tflite"))

            if not tflite_files:
                logger.error(f"No .litertlm or .tflite file found in {model_id} snapshot at {snapshot}")
                return None
                
            # Prefer the generic int4 model if multiple exist
            resolved_model_path = tflite_files[0]
            for p_file in tflite_files:
                if p_file.name == "gemma-3n-E2B-it-int4.litertlm":
                    resolved_model_path = p_file
                    break
            return resolved_model_path
            
        except Exception as e:
            logger.warning(f"Could not resolve model path via HF: {e}")
            return None

    def load_model(self) -> bool:
        """Load the LiteRT model interpreter."""
        if not self.model_path or not self.model_path.exists():
            logger.error(f"LiteRT model path invalid: {self.model_path}")
            return False
            
        try:
            logger.info(f"Loading LiteRT model via MediaPipe GenAI: {self.model_path}")

            if not HAS_MEDIAPIPE_GENAI:
                if self.is_mock:
                    logger.warning("Mocking LiteRT load (GenAI runtime missing).")
                    self.loaded = True
                    return True
                logger.error("MediaPipe GenAI not installed. Cannot load .litertlm file.")
                return False

            options = llm_inference.LlmInferenceOptions(
                model_asset_path=str(self.model_path),
                max_tokens=self.max_tokens,
                temperature=0.7,
                random_seed=42
            )
            self.agent = llm_inference.LlmInference.create_from_options(options)
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MediaPipe GenAI model: {e}")
            return False

    def chat(self, message: str, system_context: Optional[str] = None, image: Any = None, model_hint: Optional[str] = None) -> str:
        """
        Generate chat response using MediaPipe GenAI.
        """
        if not self.loaded or not self.agent:
            # Try lazy load
            if not self.load_model():
                return "Error: LiteRT model not loaded."

        # Format prompt
        # Use simpler approach for now: concatenate history
        # (MediaPipe LlmInference is usually stateless or requires session management that we haven't implemented)
        
        full_prompt = ""
        if system_context:
             full_prompt += f"<start_of_turn>user\n{system_context}\n"

        for turn in self.chat_history[-self.max_history:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                full_prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            else:
                full_prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        
        full_prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
        
        try:
            response_text = self.agent.generate_response(full_prompt)
            
            # Update history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "model", "content": response_text})
            
            return response_text
        except Exception as e:
            logger.error(f"LiteRT inference failed: {e}")
            return f"Error during inference: {e}"

    def reset_history(self):
        self.chat_history = []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": "litert-mock" if self.is_mock else "litert",
            "accelerator": self.accelerator_device,
            "loaded": self.loaded,
            "history_length": len(self.chat_history) // 2,
            "has_token": False
        }
