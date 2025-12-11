"""
Gemma 3 Nano chat interface for robot control.

This module provides local Gemma inference for conversational AI
integrated with robot control and vision capabilities.
"""

import os
import json
import subprocess
import sys
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GemmaChat:
    """
    Manages Gemma 3 Nano model inference for chat interactions.
    
    For production deployment, this uses HuggingFace transformers library
    with quantized models for efficient on-device inference.
    """
    DEFAULT_MODEL_ID = "google/gemma-370m"  # prefer smallest first
    # Fallbacks (retain prior defaults for larger variants):
    # "google/gemma-3n-E2B-it"

    def __init__(self, model_name: str = DEFAULT_MODEL_ID, device: str = "cpu", api_base: Optional[str] = None, api_key: Optional[str] = None, accelerator_device: Optional[str] = None):
        """
        Initialize Gemma chat interface.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu', 'cuda', or 'mps' for Apple Silicon
        api_base: Base URL for OpenAI-compatible API (e.g., vLLM)
            api_key: API key for OpenAI-compatible API
            accelerator_device: Optional detected AI accelerator (e.g., 'hailo8l', 'tpu')
        """
        self.model_name = model_name
        self.device = device
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"  # Default for local vLLM/llama.cpp
        
        self.accelerator_device = accelerator_device
        if self.accelerator_device:
            logger.info(f"🚀 Initializing GemmaChat with accelerator: {self.accelerator_device}")
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vlm = False
        self.client = None
        
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 turns
        
        # Check for HuggingFace token
        self.hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not self.hf_token and not self.api_base:
            logger.warning("HUGGINGFACE_TOKEN not set - gated models will not be accessible")
            
        if self.api_base:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
                logger.info(f"Initialized OpenAI client pointing to {self.api_base}")
            except ImportError:
                logger.error("openai package not installed. Cannot use remote API.")

    def load_model(self) -> bool:
        """
        Load Gemma model and tokenizer (or verify API connection).
        
        Returns:
            True if successful, False otherwise
        """
        if self.client:
            # For API, we just assume it's ready or do a quick check?
            # Keeping it simple: if client is initialized, we are "loaded"
            return True

        try:
            # Try importing transformers
            # Try importing transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
            import torch
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
                except ImportError:
                    AutoModelForImageTextToText = None
            
            logger.info(f"Loading Gemma model: {self.model_name}")
            
            # --- VLM Loading Logic (Gemma 3N / PaliGemma) ---
            if "gemma-3n" in self.model_name or "paligemma" in self.model_name:
                try:
                    logger.info(f"Attempting to load VLM: {self.model_name}")
                    
                    if AutoModelForImageTextToText is None:
                        raise ImportError("AutoModelForImageTextToText not available in installed transformers version")
                        
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        token=self.hf_token,
                        trust_remote_code=True
                    )
                    
                    # Determine dtype and device map
                    # For Pi 5 (CPU), we found that device_map="cpu" + bfloat16 works best
                    # device_map="auto" failed with offloading errors
                    dtype = torch.bfloat16
                    device_map = "cpu"
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        token=self.hf_token,
                        trust_remote_code=True,
                        device_map=device_map,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    )
                    
                    self.is_vlm = True
                    logger.info("✅ Gemma 3N VLM loaded successfully!")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load as VLM ({e}). Falling back to CausalLM (text-only)...")
                    # Clear partial loads
                    self.model = None
                    self.processor = None

            # --- CausalLM Loading Logic (Text Only / Fallback) ---
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Load model with conservative defaults to avoid disk_offload errors on CPU
            use_device_map = None
            torch_dtype = None
            low_cpu_mem = True

            if self.device != "cpu":
                use_device_map = "auto"
                torch_dtype = None  # let transformers pick best for accel
                low_cpu_mem = False
            else:
                torch_dtype = None  # stay with default fp32 on CPU

            # Explicitly check for accelerate; disable auto map if missing
            try:
                import accelerate
                logger.info(f"Accelerate available: {accelerate.__version__}")
            except ImportError:
                if use_device_map == "auto":
                    logger.warning("Accelerate not found. Disabling device_map='auto'")
                    use_device_map = None

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                    trust_remote_code=True,
                    device_map=use_device_map,
                    low_cpu_mem_usage=low_cpu_mem,
                    torch_dtype=torch_dtype,
                )
            except Exception as e:
                if use_device_map == "auto":
                    logger.warning(f"Failed with device_map='auto': {e}. Retrying on CPU without offload...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        token=self.hf_token,
                        trust_remote_code=True,
                        device_map=None,
                        low_cpu_mem_usage=True,
                    )
                else:
                    raise e
            
            self.is_vlm = False
            logger.info("Gemma model loaded (CausalLM mode)")
            return True
            
        except ImportError as e:
            logger.error(f"transformers library or dependency missing: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            return False
    
    def chat(self, message: str, system_context: Optional[str] = None, image: Any = None, model_hint: Optional[str] = None) -> str:
        """
        Generate chat response from Gemma model (local or remote).
        
        Args:
            message: User message text
            system_context: Optional system context (robot status, hardware info, etc.)
            image: Optional image input (PIL Image or numpy array) for VLM
        
        Returns:
            Model response text
        """
        
        # --- PATH 1: Remote API (OpenAI/vLLM) ---
        if self.client:
            # TODO: Handle image for API if supported (e.g. GPT-4o)
            try:
                model_name = model_hint or self.model_name
                messages = []
                if system_context:
                    messages.append({"role": "system", "content": system_context})
                
                # Add history
                # Ensure history format matches OpenAI (role/content)
                for msg in self.chat_history[-6:]:
                    role = "user" if msg["role"].lower() in ["user"] else "assistant"
                    messages.append({"role": role, "content": msg["content"]})
                    
                messages.append({"role": "user", "content": message})
                
                logger.info(f"Sending request to API: {self.api_base}")
                completion = self.client.chat.completions.create(
                    model=model_name, # vLLM often ignores this or needs it to match loaded model
                    messages=messages,
                    temperature=0.7,
                    max_tokens=256
                )
                
                response = completion.choices[0].message.content
                
                # Update history
                self.chat_history.append({"role": "User", "content": message})
                self.chat_history.append({"role": "Assistant", "content": response})
                if len(self.chat_history) > self.max_history * 2:
                    self.chat_history = self.chat_history[-self.max_history * 2:]
                    
                return response

            except Exception as e:
                logger.error(f"API chat failed: {e}")
                return f"Error from remote API: {str(e)}"
        
        # --- PATH 2: Local Transformers ---
        if model_hint and model_hint != self.model_name:
            # Attempt to switch models if available
            self.model_name = model_hint
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.is_vlm = False

        if self.model is None:
            # Pass return_error=True (we need to update load_model signature slightly or just capture it)
            # Actually load_model currently returns bool.
            # Let's verify load_model logic first.
            if not self.load_model():
                # We can't easily get the error string from load_model unless we change it.
                # But we can try the import HERE to see if it fails and why.
                try:
                    import transformers
                    import torch
                    return f"Error: Gemma model failed to load. Check server logs for details."
                except ImportError as e:
                    return f"Error: Gemma model not available. Missing dependency: {e}"
                except Exception as e:
                    return f"Error: Gemma model not available. Unexpected error: {e}"
        
        try:
            import torch
            
            # --- VLM Generation ---
            if self.is_vlm and self.processor:
                # Prepare inputs with Processor
                
                # Combine system context and message
                full_prompt = message
                if system_context:
                    # Gemma 3N chat templates usually handle system instructions differently
                    # For now, prepend simple context
                    full_prompt = f"{system_context}\n\n{message}"
                
                # Add history
                # Note: Multiturn VLM is complex. For now, we just pass the current turn.
                # Or append history text.
                # TODO: Use apply_chat_template if supported by processor
                
                # Handle inputs
                if image:
                    inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
                else:
                    inputs = self.processor(text=full_prompt, images=None, return_tensors="pt")
                
                # Move pixel_values to correct device/dtype if needed
                # inputs = inputs.to(self.device) # device_map="cpu" handles this usually, but let's be safe if manual
                # actually device_map="cpu" means model is on CPU, inputs should be CPU.
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7
                    )
                
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Post-process response (remove prompt echo if present)
                # Gemma 3N output usually contains the prompt? check verification script output
                # Usually batch_decode(outputs) includes prompt.
                # We can use skip_prompt=True if processor supports it, or manual slicing.
                if response.startswith(full_prompt):
                    response = response[len(full_prompt):].strip()
                elif "Assistant:" in response:
                     response = response.split("Assistant:")[-1].strip()
                elif "model" in response.lower() and ":" in response: # e.g. "model: answer"
                     pass
                
                # Update history
                self.chat_history.append({"role": "User", "content": message})
                self.chat_history.append({"role": "Assistant", "content": response})
                return response

            # --- CausalLM Generation (Legacy) ---
            
            # Add system context if provided
            if system_context:
                prompt = f"System: {system_context}\n\nUser: {message}\n\nAssistant:"
            else:
                prompt = f"User: {message}\n\nAssistant:"
            
            # Add chat history context
            if self.chat_history:
                history_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in self.chat_history[-6:]  # Last 3 turns
                ])
                prompt = f"{history_text}\n{prompt}"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response (after "Assistant:")
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Update chat history
            self.chat_history.append({"role": "User", "content": message})
            self.chat_history.append({"role": "Assistant", "content": response})
            
            # Trim history
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]
            
            return response
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def reset_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history reset")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and status.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "accelerator": self.accelerator_device,
            "loaded": self.model is not None,
            "history_length": len(self.chat_history) // 2,  # Number of turns
            "has_token": self.hf_token is not None
        }


# Mock implementation for testing without transformers
class MockGemmaChat:
    """Mock Gemma chat for testing without model."""
    
    
    def __init__(self, model_name: str = "mock", device: str = "cpu", accelerator_device: str = None, error_msg: str = None):
        self.model_name = model_name
        self.device = device
        self.error_msg = error_msg
        self.min_history = []
        if accelerator_device:
             logger.info(f"🚀 MockGemmaChat using mock accelerator: {accelerator_device}")
        self.chat_history = []
        logger.info(f"Using mock Gemma chat. Error: {error_msg}")
    
    def load_model(self) -> bool:
        return True
    
    def chat(self, message: str, system_context: Optional[str] = None, model_hint: Optional[str] = None) -> str:
        """Generate mock response."""
        self.chat_history.append({"role": "User", "content": message})
        
        # Simple pattern matching for mock responses
        msg_lower = message.lower()
        
        if "status" in msg_lower or "how are" in msg_lower:
            response = "I'm a mock Gemma assistant. The robot is operational and ready for commands."
        elif "camera" in msg_lower:
            response = "The OAK-D Lite camera is active and streaming at 640x400 resolution."
        elif "move" in msg_lower or "control" in msg_lower:
            response = "You can control the robot using the joint sliders or arrow buttons on this page."
        elif "help" in msg_lower:
            response = "I can help you with robot status, camera info, and control instructions. What would you like to know?"
        else:
            reason = f" Missing dependency: {self.error_msg}" if self.error_msg else " Install transformers for real Gemma inference."
            response = f"[Mock] You said: '{message}'.{reason}"
        
        self.chat_history.append({"role": "Assistant", "content": response})
        return response
    
    def reset_history(self):
        self.chat_history = []
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "mock",
            "device": "cpu",
            "loaded": True,
            "history_length": len(self.chat_history) // 2,
            "has_token": False
        }



# Factory function to create appropriate chat instance
def _install_missing_deps():
    """Best-effort install of core deps to avoid mock fallback."""
    required = ["transformers", "accelerate", "timm"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return True
    logger.warning(f"Missing dependencies {missing}; attempting pip install")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *missing])
        return True
    except Exception as exc:
        logger.error(f"Auto-install failed: {exc}")
        return False


def create_gemma_chat(use_mock: bool = False, **kwargs) -> Any:
    """
    Create Gemma chat instance.
    
    Args:
        use_mock: Force mock implementation
        **kwargs: Arguments passed to chat constructor
    
    Returns:
        GemmaChat or MockGemmaChat instance
    """
    create_gemma_chat.DEFAULT_MODEL_ID = GemmaChat.DEFAULT_MODEL_ID
    if use_mock:
        return MockGemmaChat(**kwargs)
    
    # Check for OpenAI API configuration
    api_base = os.environ.get("LLM_API_BASE")
    api_key = os.environ.get("LLM_API_KEY")
    
    if api_base or api_key:
        logger.info(f"Configuring GemmaChat with remote API: {api_base}")
        # Pass API config to constructor
        kwargs["api_base"] = api_base
        kwargs["api_key"] = api_key
        
        # We still try to import transformers/openai inside the class, 
        # but the class itself handles the logic.
        return GemmaChat(**kwargs)

    try:
        import transformers
        import torch
        return GemmaChat(**kwargs)
    except ImportError as e:
        logger.warning(f"transformers not available: {e}. Trying auto-install...")
        if _install_missing_deps():
            try:
                import transformers
                import torch
                return GemmaChat(**kwargs)
            except Exception as exc:
                logger.error(f"Deps installed but Gemma init still failed: {exc}")
        return MockGemmaChat(error_msg=str(e), **kwargs)

