"""
Chat Service

Domain service for chat/conversation functionality.
Extracted from BrainService.ChatWithGemma and related methods.
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from continuonbrain.services.container import ServiceContainer

logger = logging.getLogger(__name__)


# Template responses for fast mode
FAST_RESPONSES = {
    "hello": "Hello! I'm Continuon, your robot companion. How can I help?",
    "hi": "Hi there! Ready to assist.",
    "help": "I can help with movement, learning, and general conversation. What would you like to do?",
    "status": "All systems operational. Standing by for commands.",
    "stop": "Stopping all motion. Safety first!",
}


class ChatService:
    """
    Chat domain service implementing IChatService.

    Handles:
    - Message processing with Gemma or HOPE models
    - Session management
    - Fallback responses
    - Confidence scoring
    """

    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        container: Optional["ServiceContainer"] = None,
        model_preference: str = "auto",
        fast_mode_default: bool = False,
        max_history: int = 20,
        **kwargs,
    ):
        """
        Initialize chat service.

        Args:
            config_dir: Configuration directory
            container: Service container for dependencies
            model_preference: "gemma", "hope", or "auto"
            fast_mode_default: Use fast mode by default
            max_history: Maximum history length to maintain
        """
        self.config_dir = Path(config_dir)
        self._container = container
        self.model_preference = model_preference
        self.fast_mode_default = fast_mode_default
        self.max_history = max_history

        # State
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._chat_engine = None
        self._model_type = "none"
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of chat engine."""
        if self._initialized:
            return

        self._initialized = True

        try:
            from continuonbrain.gemma_chat import build_chat_service

            self._chat_engine = build_chat_service(config_dir=self.config_dir)
            self._model_type = "gemma"
            logger.info("Chat service initialized with Gemma")

        except ImportError as e:
            logger.warning(f"Gemma not available: {e}")
            self._try_fallback_engine()
        except Exception as e:
            logger.error(f"Failed to initialize chat engine: {e}")
            self._try_fallback_engine()

    def _try_fallback_engine(self) -> None:
        """Try to initialize a fallback chat engine."""
        try:
            from continuonbrain.hope_impl.brain import HOPEBrain

            self._chat_engine = HOPEBrain()
            self._model_type = "hope"
            logger.info("Chat service initialized with HOPE fallback")

        except ImportError:
            logger.warning("No chat engine available, using template responses")
            self._chat_engine = None
            self._model_type = "template"

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        session_id: Optional[str] = None,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response.

        Args:
            message: User message
            history: Conversation history
            session_id: Session ID (created if None)
            fast_mode: Use fast/template responses

        Returns:
            Response dictionary with response text, confidence, etc.
        """
        start_time = time.time()

        # Create session if needed
        if not session_id:
            session_id = str(uuid.uuid4())

        # Use fast mode if requested or default
        use_fast = fast_mode or self.fast_mode_default

        if use_fast:
            response = self._fast_response(message)
            return {
                "response": response,
                "confidence": 0.8 if response != self._fallback_response(message) else 0.3,
                "model": "template",
                "session_id": session_id,
                "metadata": {
                    "fast_mode": True,
                    "latency_ms": (time.time() - start_time) * 1000,
                },
            }

        # Full chat processing
        self._ensure_initialized()

        try:
            if self._chat_engine and self._model_type == "gemma":
                response, confidence = self._chat_with_gemma(message, history, session_id)
            elif self._chat_engine and self._model_type == "hope":
                response, confidence = self._chat_with_hope(message, history, session_id)
            else:
                response = self._fallback_response(message)
                confidence = 0.3

        except Exception as e:
            logger.error(f"Chat error: {e}")
            response = self._fallback_response(message)
            confidence = 0.2

        # Update session history
        self._update_session(session_id, message, response)

        return {
            "response": response,
            "confidence": confidence,
            "model": self._model_type,
            "session_id": session_id,
            "metadata": {
                "fast_mode": False,
                "latency_ms": (time.time() - start_time) * 1000,
                "history_length": len(self._sessions.get(session_id, [])),
            },
        }

    def _chat_with_gemma(
        self,
        message: str,
        history: List[Dict[str, str]],
        session_id: str,
    ) -> tuple[str, float]:
        """Chat using Gemma model."""
        try:
            result = self._chat_engine.chat(
                message=message,
                history=history,
                session_id=session_id,
            )

            if isinstance(result, dict):
                response = result.get("response", str(result))
                confidence = result.get("confidence", 0.7)
            else:
                response = str(result)
                confidence = 0.7

            return response, confidence

        except Exception as e:
            logger.error(f"Gemma chat error: {e}")
            return self._fallback_response(message), 0.3

    def _chat_with_hope(
        self,
        message: str,
        history: List[Dict[str, str]],
        session_id: str,
    ) -> tuple[str, float]:
        """Chat using HOPE model."""
        try:
            response = self._chat_engine.respond(message, history)
            return response, 0.6

        except Exception as e:
            logger.error(f"HOPE chat error: {e}")
            return self._fallback_response(message), 0.3

    def _fast_response(self, message: str) -> str:
        """Get fast template response."""
        message_lower = message.lower().strip()

        # Check for exact matches
        for key, response in FAST_RESPONSES.items():
            if key in message_lower:
                return response

        # Default to a general response
        return self._fallback_response(message)

    def _fallback_response(self, message: str) -> str:
        """Generate fallback response when models are unavailable."""
        if "?" in message:
            return "I understand you have a question. Let me think about that..."
        elif any(word in message.lower() for word in ["move", "go", "drive"]):
            return "I can help with movement commands. Please specify direction and speed."
        elif any(word in message.lower() for word in ["look", "see", "find"]):
            return "I'll use my vision system to help you. What should I look for?"
        else:
            return "I'm here to help. Could you tell me more about what you need?"

    def _update_session(self, session_id: str, message: str, response: str) -> None:
        """Update session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({"role": "user", "content": message})
        self._sessions[session_id].append({"role": "assistant", "content": response})

        # Trim history if too long
        if len(self._sessions[session_id]) > self.max_history * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.max_history * 2:]

    def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def clear_all_sessions(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the active model."""
        self._ensure_initialized()

        return {
            "model_name": self._model_type,
            "model_type": self._model_type,
            "is_loaded": self._chat_engine is not None,
            "capabilities": ["chat", "conversation"],
        }

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def is_available(self) -> bool:
        """Check if chat service is available."""
        return True  # Always available with at least fallback responses

    def shutdown(self) -> None:
        """Shutdown chat service."""
        self._sessions.clear()
        self._chat_engine = None
        self._initialized = False
