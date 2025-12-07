"""
Gemma 3 Nano chat interface for robot control.

This module provides local Gemma inference for conversational AI
integrated with robot control and vision capabilities.
"""

import os
import json
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GemmaChat:
    """
    Manages Gemma 3 Nano model inference for chat interactions.
    
    For production deployment, this uses HuggingFace transformers library
    with quantized models for efficient on-device inference.
    """
    DEFAULT_MODEL_ID = "google/gemma-3-270m-it"
    # DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Backup

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
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Gemma model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Load model with quantization for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("Gemma model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"transformers library or dependency missing: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            return False
    
    def chat(self, message: str, system_context: Optional[str] = None) -> str:
        """
        Generate chat response from Gemma model (local or remote).
        
        Args:
            message: User message text
            system_context: Optional system context (robot status, hardware info, etc.)
        
        Returns:
            Model response text
        """
        
        # --- PATH 1: Remote API (OpenAI/vLLM) ---
        if self.client:
            try:
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
                    model=self.model_name, # vLLM often ignores this or needs it to match loaded model
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
        if self.model is None:
            if not self.load_model():
                return "Error: Gemma model not available. Please install transformers library."
        
        try:
            import torch
            
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
    
    
    def __init__(self, model_name: str = "mock", device: str = "cpu", accelerator_device: str = None):
        self.model_name = model_name
        self.device = device
        self.min_history = []
        if accelerator_device:
             logger.info(f"🚀 MockGemmaChat using mock accelerator: {accelerator_device}")
        self.chat_history = []
        logger.info("Using mock Gemma chat (transformers not available)")
    
    def load_model(self) -> bool:
        return True
    
    def chat(self, message: str, system_context: Optional[str] = None) -> str:
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
            response = f"[Mock] You said: '{message}'. Install transformers for real Gemma inference."
        
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
def create_gemma_chat(use_mock: bool = False, **kwargs) -> Any:
    """
    Create Gemma chat instance.
    
    Args:
        use_mock: Force mock implementation
        **kwargs: Arguments passed to chat constructor
    
    Returns:
        GemmaChat or MockGemmaChat instance
    """
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
    except ImportError:
        logger.warning("transformers not available, using mock chat")
        return MockGemmaChat(**kwargs)

