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
    
    def __init__(self, model_name: str = "google/gemma-3n-E2B-it", device: str = "cpu"):
        """
        Initialize Gemma chat interface.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu', 'cuda', or 'mps' for Apple Silicon
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 turns
        
        # Check for HuggingFace token
        self.hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not self.hf_token:
            logger.warning("HUGGINGFACE_TOKEN not set - gated models will not be accessible")
    
    def load_model(self) -> bool:
        """
        Load Gemma model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
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
            
        except ImportError:
            logger.error("transformers library not installed. Install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            return False
    
    def chat(self, message: str, system_context: Optional[str] = None) -> str:
        """
        Generate chat response from Gemma model.
        
        Args:
            message: User message text
            system_context: Optional system context (robot status, hardware info, etc.)
        
        Returns:
            Model response text
        """
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
            "loaded": self.model is not None,
            "history_length": len(self.chat_history) // 2,  # Number of turns
            "has_token": self.hf_token is not None
        }


# Mock implementation for testing without transformers
class MockGemmaChat:
    """Mock Gemma chat for testing without model."""
    
    def __init__(self, model_name: str = "mock", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
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
    
    try:
        import transformers
        import torch
        return GemmaChat(**kwargs)
    except ImportError:
        logger.warning("transformers not available, using mock chat")
        return MockGemmaChat(**kwargs)
