"""
Brain B Conversation Layer

Natural language interface for talking to your robot.
Includes Gemini/Claude integration for intelligent conversation.
"""

from .intents import Intent, IntentClassifier
from .handler import ConversationHandler
from .claude_backend import ClaudeBackend
from .gemini_backend import GeminiBackend

__all__ = ["Intent", "IntentClassifier", "ConversationHandler", "ClaudeBackend", "GeminiBackend"]
