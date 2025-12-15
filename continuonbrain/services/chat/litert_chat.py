"""
LiteRT (TensorFlow Lite) backend for Gemma 3N.
"""
import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Try importing LiteRT interpreter
try:
    import ai_edge_litert.interpreter as prefix_interpreter
    Interpreter = prefix_interpreter.Interpreter
    HAS_LITERT = True
except ImportError:
    try:
        # Fallback to tflite_runtime
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        HAS_LITERT = True
    except ImportError:
        Interpreter = None
        HAS_LITERT = False

logger = logging.getLogger(__name__)


class LiteRTGemmaChat:
    """
    Gemma 3N backend using LiteRT (TensorFlow Lite) for inference.
    """
    
    DEFAULT_MODEL_ID = "google/gemma-3n-E2B-it-litert-lm"

    def __init__(self, model_name: str = DEFAULT_MODEL_ID, device: str = "cpu", accelerator_device: Optional[str] = None):
        if not HAS_LITERT:
            raise ImportError("ai-edge-litert or tflite-runtime not installed")
            
        self.model_name = model_name
        self.device = device
        self.accelerator_device = accelerator_device
        self.interpreter = None
        self.tokenizer = None
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 10
        
        # Resolve model path
        self.model_path = self._resolve_model_path(model_name)
        
    def _resolve_model_path(self, model_id: str) -> Optional[Path]:
        """
        Find the .tflite model file. 
        Expects a valid path or tries to resolve from HF cache like other backends.
        """
        # 1. Check if model_id is a direct local path
        p = Path(model_id)
        if p.exists() and p.suffix == ".tflite":
            return p
        
        # 2. Use existing resolution logic from gemma_chat if available, or duplicate it
        # We'll use the one from gemma_chat via import to avoid code duplication
        try:
            from continuonbrain.gemma_chat import _snapshot_download_path, _ensure_hf_cache_env
            
            # For LiteRT/GGUF models on HF, they are often single files inside a repo.
            # snapshots download the whole repo.
            snapshot = _snapshot_download_path(model_id, os.environ.get("HUGGINGFACE_TOKEN"))
            
            # Find the .tflite file in the snapshot
            candidates = list(snapshot.glob("*.tflite"))
            if not candidates:
                logger.error(f"No .tflite file found in {model_id} snapshot at {snapshot}")
                return None
                
            # Prefer "gpu" or "cpu" based on device?
            # For now, pick the first one or specific name if known.
            # The user linked `google/gemma-3n-E2B-it-litert-lm`.
            # Usually contains `model.tflite`.
            return candidates[0]
            
        except Exception as e:
            logger.warning(f"Could not resolve model path via HF: {e}")
            return None

    def load_model(self) -> bool:
        """Load the LiteRT model interpreter."""
        if not self.model_path or not self.model_path.exists():
            logger.error(f"LiteRT model path invalid: {self.model_path}")
            return False
            
        try:
            logger.info(f"Loading LiteRT model: {self.model_path}")
            
            # Initialize interpreter
            # TODO: Add delegate support for GPU/NPU if accelerator_device is set
            delegates = []
            if self.accelerator_device == "gpu":
                 # This requires experimental delegates support
                 pass 
                 
            self.interpreter = Interpreter(str(self.model_path), experimental_delegates=delegates)
            self.interpreter.allocate_tensors()
            
            # Get signature runner (easier API for simple models)
            # Most GenAI LiteRT models use the SignatureRunner API or "Serving" signatures.
            # However, the new "LiteRT-LM" might need a different specific API wrapper 
            # (e.g. GenerativeModel from tflite_support).
            # But the raw interpreter text-in/text-out usually relies on token IDs.
            # 
            # IMPORTANT: The raw .tflite for LLMs usually takes token IDs, not strings.
            # We need the tokenizer.
            
            # Load tokenizer (reusing transformers tokenizer for now as it's robust)
            from transformers import AutoTokenizer
            # We need the tokenizer from the *original* model or the litert repo usually has `tokenizer.json`
            # The snapshot path should have it.
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path.parent))
            
            logger.info("✅ LiteRT model and tokenizer loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LiteRT model: {e}")
            return False

    def chat(self, message: str, system_context: Optional[str] = None, image: Any = None, model_hint: Optional[str] = None) -> str:
        """
        Generate chat response using LiteRT.
        Note: This implementation assumes a standard LLM signature loop for generation, 
        which is complex to do raw with Interpreter.
        
        Ideally we would use `ai_edge_litert.llm` or similar high level API if available.
        Since we are using the raw Interpreter or `tflite_runtime`, we might need to implement the verification loop.
        
        However, for the specific `gemma-3n` LiteRT model, Google provides a higher level "Generative API" usually.
        If we are sticking to `ai-edge-litert` python pkg, let's see if we can use a higher level wrapper 
        or if we have to do the loop.
        
        For this first pass, to avoid implementing a full sampler in Python (slow), 
        we will check if the model has a custom signature that handles generation.
        """
        if not self.interpreter:
            if not self.load_model():
                return "Error: LiteRT model not loaded."

        # Prepare prompt
        prompt = message
        if system_context:
            prompt = f"System: {system_context}\nUser: {prompt}"
        
        # Tokenize
        if not self.tokenizer:
             return "Error: Tokenizer not loaded."
             
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        
        # INFERENCE LOOP (Simplified greedy)
        # This is very slow in pure Python for many tokens. 
        # But `ai-edge-litert` implies potentially better support.
        # Check signature.
        sigs = self.interpreter.get_signature_list()
        # logger.info(f"Signatures: {sigs}")
        
        # If it has a loop signature, use it. usage: runner.predict(input_ids=...)
        # Otherwise start the loop.
        
        # To remain compatible with the "Investigate" task, I will implement a basic run 
        # or a placeholder if it's too complex for raw implementation without the specific GenAI wrapper.
        
        # Hack: Validating if we can just run it once to prove it works.
        return f"[LiteRT] Model loaded from {self.model_path.name}. (Inference loop placeholder: requires GenAI wrapper)"

    def reset_history(self):
        self.chat_history = []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": "litert",
            "accelerator": self.accelerator_device,
            "loaded": self.interpreter is not None,
            "history_length": len(self.chat_history) // 2,
            "has_token": False
        }
