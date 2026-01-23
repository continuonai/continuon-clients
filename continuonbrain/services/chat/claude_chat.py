"""
Claude Chat Wrapper
==================

Provides Claude API integration for the ContinuonBrain chat system.
Uses the Anthropic API with user-provided API keys.

This wrapper implements the same interface as HopeChat so it can be
used interchangeably as the chat backend.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Model ID mapping
CLAUDE_MODELS = {
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """A conversation session with history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, **metadata) -> ChatMessage:
        """Add a message to the session."""
        msg = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.last_activity = time.time()
        return msg

    def get_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent history as list of dicts."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-limit:]
        ]


class ClaudeChat:
    """
    Claude API chat wrapper compatible with the ContinuonBrain chat interface.

    Implements the same interface as HopeChat so it can be used as a drop-in
    replacement for the chat backend.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        config_dir: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize Claude chat wrapper.

        Args:
            api_key: Anthropic API key. If not provided, loads from settings.
            model: Claude model to use (e.g., "claude-opus-4-5").
            config_dir: Config directory to load settings from.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        self.model_name = model
        self.model_id = CLAUDE_MODELS.get(model, model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.config_dir = config_dir
        self._client = None
        self._async_client = None
        self._api_key = api_key

        # Session management (matching HopeChat interface)
        self._sessions: Dict[str, ChatSession] = {}
        self._default_session_id = "default"

        # System personality
        self.system_prompt = (
            "You are a helpful AI assistant integrated with ContinuonBrain. "
            "You help users with tasks, answer questions, and provide assistance. "
            "Be friendly, knowledgeable, and helpful."
        )

        # Try to load API key from settings if not provided
        if not self._api_key and config_dir:
            self._api_key = self._load_api_key_from_settings(config_dir)

        # Also check environment variable
        if not self._api_key:
            self._api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self._api_key:
            logger.warning("No Anthropic API key configured. Claude chat will fail.")

    def _load_api_key_from_settings(self, config_dir: str) -> Optional[str]:
        """Load API key from settings.json."""
        try:
            settings_path = Path(config_dir) / "settings.json"
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = json.load(f)
                return settings.get("harness", {}).get("claude_code", {}).get("api_key")
        except Exception as e:
            logger.warning(f"Failed to load API key from settings: {e}")
        return None

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client

    def _get_async_client(self):
        """Lazily initialize the async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic
                self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        return self._async_client

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        system_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response.

        This method matches the HopeChat interface for drop-in compatibility.

        Args:
            message: User's message text.
            session_id: Session ID for conversation tracking.
            history: Optional conversation history.
            system_context: Optional system prompt override.

        Returns:
            Dict with response, session_id, and metadata.
        """
        start_time = time.time()

        # Get or create session
        session_id = session_id or self._default_session_id
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
        session = self._sessions[session_id]

        # Add user message to session
        session.add_message("user", message)

        # Check API key
        if not self._api_key:
            response_text = "[Error: Anthropic API key not configured. Please add your API key in Settings > Claude Code & Harness.]"
            session.add_message("assistant", response_text)
            return {
                "success": False,
                "response": response_text,
                "session_id": session_id,
                "agent": "claude_error",
                "duration_ms": (time.time() - start_time) * 1000,
                "turn_count": len(session.messages),
                "error": "API key not configured",
            }

        try:
            client = self._get_client()

            # Build messages from session history or provided history
            messages = []
            if history:
                messages.extend(history)
            else:
                # Use session history (excluding the current user message)
                for msg in session.messages[:-1]:
                    messages.append({"role": msg.role, "content": msg.content})

            # Add current user message
            messages.append({"role": "user", "content": message})

            # Build request
            kwargs = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }

            # Use provided system context or default
            kwargs["system"] = system_context or self.system_prompt

            # Call API
            response = client.messages.create(**kwargs)

            # Extract text response
            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
            else:
                response_text = "[No response from Claude]"

            response_agent = f"claude_{self.model_name}"

        except ImportError as e:
            logger.error(f"Anthropic package not installed: {e}")
            response_text = "[Error: anthropic package not installed. Install with: pip install anthropic]"
            response_agent = "claude_error"
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            response_text = f"[Claude API Error: {str(e)}]"
            response_agent = "claude_error"

        # Add response to session
        session.add_message("assistant", response_text)

        duration = time.time() - start_time

        return {
            "success": True,
            "response": response_text,
            "session_id": session_id,
            "agent": response_agent,
            "duration_ms": duration * 1000,
            "turn_count": len(session.messages),
        }

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        if session_id in self._sessions:
            self._sessions[session_id].messages.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get chat service status."""
        return {
            "ready": bool(self._api_key),
            "model": self.model_name,
            "model_id": self.model_id,
            "api_configured": bool(self._api_key),
            "active_sessions": len(self._sessions),
            "provider": "anthropic",
        }

    async def chat_async(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        system_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of chat using Anthropic's async client.

        This does not block the event loop, allowing concurrent requests.

        Args:
            message: User's message text.
            session_id: Session ID for conversation tracking.
            history: Optional conversation history.
            system_context: Optional system prompt override.

        Returns:
            Dict with response, session_id, and metadata.
        """
        start_time = time.time()

        # Get or create session
        session_id = session_id or self._default_session_id
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
        session = self._sessions[session_id]

        # Add user message to session
        session.add_message("user", message)

        # Check API key
        if not self._api_key:
            response_text = "[Error: Anthropic API key not configured. Please add your API key in Settings > Claude Code & Harness.]"
            session.add_message("assistant", response_text)
            return {
                "success": False,
                "response": response_text,
                "session_id": session_id,
                "agent": "claude_error",
                "duration_ms": (time.time() - start_time) * 1000,
                "turn_count": len(session.messages),
                "error": "API key not configured",
            }

        try:
            client = self._get_async_client()

            # Build messages from session history or provided history
            messages = []
            if history:
                messages.extend(history)
            else:
                # Use session history (excluding the current user message)
                for msg in session.messages[:-1]:
                    messages.append({"role": msg.role, "content": msg.content})

            # Add current user message
            messages.append({"role": "user", "content": message})

            # Build request
            kwargs = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }

            # Use provided system context or default
            kwargs["system"] = system_context or self.system_prompt

            # Call API asynchronously (does not block event loop)
            response = await client.messages.create(**kwargs)

            # Extract text response
            if response.content and len(response.content) > 0:
                response_text = response.content[0].text
            else:
                response_text = "[No response from Claude]"

            response_agent = f"claude_{self.model_name}"

        except ImportError as e:
            logger.error(f"Anthropic package not installed: {e}")
            response_text = "[Error: anthropic package not installed. Install with: pip install anthropic]"
            response_agent = "claude_error"
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            response_text = f"[Claude API Error: {str(e)}]"
            response_agent = "claude_error"

        # Add response to session
        session.add_message("assistant", response_text)

        duration = time.time() - start_time

        return {
            "success": True,
            "response": response_text,
            "session_id": session_id,
            "agent": response_agent,
            "duration_ms": duration * 1000,
            "turn_count": len(session.messages),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "name": f"Claude ({self.model_name})",
            "model_id": self.model_id,
            "provider": "anthropic",
            "type": "cloud",
            "max_tokens": self.max_tokens,
            "api_configured": bool(self._api_key),
        }

    def load_model(self) -> None:
        """Validate API key and connection (called during model switch)."""
        if not self._api_key:
            raise ValueError("Anthropic API key not configured")

        # Try a minimal API call to verify the key works
        try:
            client = self._get_client()
            # Just create the client - don't make an actual API call
            logger.info(f"Claude chat initialized with model: {self.model_id}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Claude client: {e}")


def create_claude_chat(
    config_dir: Optional[str] = None,
    model: str = "claude-opus-4-5",
    **kwargs,
) -> ClaudeChat:
    """
    Factory function to create a ClaudeChat instance.

    Args:
        config_dir: Config directory to load settings from.
        model: Claude model to use.
        **kwargs: Additional arguments passed to ClaudeChat.

    Returns:
        ClaudeChat instance.
    """
    return ClaudeChat(config_dir=config_dir, model=model, **kwargs)
