"""
Chat Service Interface

Protocol definition for chat/conversation services.
"""
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class IChatService(Protocol):
    """
    Protocol for chat/conversation services.

    Implementations handle:
    - Message processing and response generation
    - Session/conversation history management
    - Model selection and fallback logic
    - Confidence scoring
    """

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        session_id: Optional[str] = None,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response with metadata.

        Args:
            message: User message text
            history: Conversation history as list of {"role": str, "content": str}
            session_id: Optional session ID for context retrieval
            fast_mode: If True, use faster/simpler response generation

        Returns:
            Dictionary containing:
                - response: str - The generated response
                - confidence: float - Confidence score (0-1)
                - model: str - Model used for generation
                - session_id: str - Session ID (may be newly created)
                - metadata: dict - Additional metadata
        """
        ...

    def clear_session(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session ID to clear
        """
        ...

    def clear_all_sessions(self) -> None:
        """Clear all conversation sessions."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the active chat model.

        Returns:
            Dictionary containing:
                - model_name: str - Name of the model
                - model_type: str - Type (e.g., "gemma", "hope", "fallback")
                - is_loaded: bool - Whether model is loaded
                - capabilities: list - List of capabilities
        """
        ...

    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        ...

    def is_available(self) -> bool:
        """Check if the chat service is available and ready."""
        ...
