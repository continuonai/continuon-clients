
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
        
    def chat(self, message: str, system_context: Optional[str] = None, image: Any = None, history: list = None) -> str:
        """
        Chat with HOPE agent, with optional history for multi-turn conversations.
        
        Args:
            message: User's message
            system_context: Optional system context
            image: Optional image for vision processing
            history: Optional conversation history (list of dicts with 'role' and 'content')
        """
        # Check if HOPE should answer
        can_answer, confidence = self.hope.can_answer(message)
        
        if can_answer and confidence > self.hope.confidence_threshold:
            logger.info(f"HOPE Agent answering with confidence {confidence}")
            response = self.hope.generate_response(message)
            if response:
                 return response

        # Fallback to LLM
        logger.info(f"HOPE yielding to LLM ({confidence:.2f} confidence)")

        if not self.llm:
            logger.warning("LLM backend missing; replying with HOPE-only safety response.")
            return self._hope_only_response(message, "No LLM backend available")

        response = None
        try:
            response = self.llm.chat(message, system_context, image)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM chat failed ({exc}); using HOPE-only response")

        # If the fallback model failed to load or returned an error string, keep the agent responsive.
        if not response or "Gemma model failed to load" in str(response):
            return self._hope_only_response(message, "Gemma unavailable")

        return response

    def _hope_only_response(self, message: str, reason: str) -> str:
        """Graceful fallback when LLM backend is unavailable."""
        # Try to reuse HOPE's domain knowledge first.
        response = self.hope.generate_response(message)
        if response:
            return response

        # Final guardrail response.
        return (
            "I'm running on the HOPE seed model and don't have external LLM access right now. "
            f"I'll keep things safe and focused on the robot: {reason}. "
            "Ask about my sensors, movement, or current mode."
        )

    def get_model_info(self) -> Dict[str, Any]:
        info = self.llm.get_model_info()
        info["model_name"] = "HOPE Logic Model (v1)"
        info["type"] = "agent"
        info["hope_active"] = True
        return info

    def reset_history(self):
        if hasattr(self.llm, "reset_history"):
            self.llm.reset_history()
