
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

class HOPEAgentWrapper:
    """
    Wraps HOPEAgent to provide a GemmaChat-compatible interface.
    Delegates to HOPE for cognitive responses, falls back to LLM for chit-chat.
    """
    def __init__(self, hope_agent, llm_chat):
        self.hope = hope_agent
        self.llm = llm_chat
        self.model_name = "hope-v1"
        self.device = getattr(llm_chat, "device", "cpu")
        
    def chat(self, message: str, system_context: Optional[str] = None, image: Any = None) -> str:
        # Check if HOPE should answer
        can_answer, confidence = self.hope.can_answer(message)
        
        if can_answer and confidence > self.hope.confidence_threshold:
            logger.info(f"HOPE Agent answering with confidence {confidence}")
            response = self.hope.generate_response(message)
            if response:
                 return response

        # Fallback to LLM
        logger.info(f"HOPE yielding to LLM ({confidence:.2f} confidence)")
        return self.llm.chat(message, system_context, image)

    def get_model_info(self) -> Dict[str, Any]:
        info = self.llm.get_model_info()
        info["model_name"] = "HOPE Logic Model (v1)"
        info["type"] = "agent"
        info["hope_active"] = True
        return info

    def reset_history(self):
        if hasattr(self.llm, "reset_history"):
            self.llm.reset_history()
