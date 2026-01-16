"""
Gemini Backend - Integrates Google Gemini AI for intelligent conversation.

When Brain B doesn't understand a command, Gemini can:
1. Interpret user intent and suggest robot actions
2. Have natural conversations
3. Help teach new behaviors
"""

import os
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiBackend:
    """
    Gemini integration for Brain B.

    Handles unknown intents by consulting Gemini for intelligent responses.
    """

    SYSTEM_PROMPT = """You are the brain of a small robot. You help interpret user commands and control the robot.

Available robot actions:
- forward: Move forward
- backward: Move backward
- left: Turn left
- right: Turn right
- stop: Stop moving

Teaching commands:
- teach <name>: Start recording a new behavior
- done: Finish recording
- cancel: Cancel recording
- <behavior_name>: Run a learned behavior

When the user says something that could be a robot command, respond with one of these action formats:
[ACTION: forward]
[ACTION: backward]
[ACTION: left]
[ACTION: right]
[ACTION: stop]
[ACTION: teach <name>]
[ACTION: invoke <name>]

For regular conversation, just respond naturally. Be friendly, helpful, and concise.

Current robot status:
- Speed: {speed}%
- Known behaviors: {behaviors}
- Recording: {recording}"""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.model = None
        self.chat = None

        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)

    @property
    def is_available(self) -> bool:
        """Check if Gemini backend is available."""
        return self.model is not None

    def process(
        self,
        user_input: str,
        speed: float = 50,
        behaviors: list[str] = None,
        recording: str = None,
    ) -> tuple[str, Optional[dict]]:
        """
        Process user input through Gemini.

        Returns:
            (response_text, action_dict or None)
        """
        if not self.is_available:
            return ("Gemini not available. Set GEMINI_API_KEY or GEMINI env var.", None)

        behaviors = behaviors or []

        # Build system prompt with current state
        system = self.SYSTEM_PROMPT.format(
            speed=int(speed * 100),
            behaviors=", ".join(behaviors) if behaviors else "None yet",
            recording=recording or "No",
        )

        # Create or continue chat
        if self.chat is None:
            self.chat = self.model.start_chat(history=[])

        try:
            # Send message with system context
            full_prompt = f"{system}\n\nUser: {user_input}"
            response = self.chat.send_message(full_prompt)
            response_text = response.text

            # Parse for action commands
            action = self._parse_action(response_text)

            # Clean the response text (remove action tags for display)
            display_text = self._clean_response(response_text)

            return (display_text, action)

        except Exception as e:
            return (f"Gemini error: {e}", None)

    def _parse_action(self, text: str) -> Optional[dict]:
        """Extract action from Gemini's response."""
        import re

        # Look for [ACTION: xyz] pattern
        match = re.search(r'\[ACTION:\s*(\w+)(?:\s+(.+?))?\]', text)
        if not match:
            return None

        action_type = match.group(1).lower()
        param = match.group(2)

        if action_type in ("forward", "backward", "left", "right", "stop"):
            return {"type": action_type}
        elif action_type == "teach" and param:
            return {"type": "teach", "name": param.strip()}
        elif action_type == "invoke" and param:
            return {"type": "invoke", "name": param.strip()}

        return None

    def _clean_response(self, text: str) -> str:
        """Remove action tags from response for display."""
        import re
        cleaned = re.sub(r'\[ACTION:[^\]]+\]\s*', '', text)
        return cleaned.strip()

    def clear_history(self):
        """Clear conversation history."""
        self.chat = None
