"""
Claude Backend - Integrates Claude AI for intelligent conversation.

When Brain B doesn't understand a command, Claude can:
1. Interpret user intent and suggest robot actions
2. Have natural conversations
3. Help teach new behaviors
"""

import os
from typing import Optional, Callable

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class ClaudeBackend:
    """
    Claude integration for Brain B.

    Handles unknown intents by consulting Claude for intelligent responses.
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

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.client = None
        self.conversation_history = []

        if CLAUDE_AVAILABLE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def is_available(self) -> bool:
        """Check if Claude backend is available."""
        return self.client is not None

    def process(
        self,
        user_input: str,
        speed: float = 50,
        behaviors: list[str] = None,
        recording: str = None,
    ) -> tuple[str, Optional[dict]]:
        """
        Process user input through Claude.

        Returns:
            (response_text, action_dict or None)
        """
        if not self.is_available:
            return ("Claude not available. Set ANTHROPIC_API_KEY.", None)

        behaviors = behaviors or []

        # Build system prompt with current state
        system = self.SYSTEM_PROMPT.format(
            speed=int(speed * 100),
            behaviors=", ".join(behaviors) if behaviors else "None yet",
            recording=recording or "No",
        )

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
        })

        # Keep history reasonable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system,
                messages=self.conversation_history,
            )

            response_text = response.content[0].text

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text,
            })

            # Parse for action commands
            action = self._parse_action(response_text)

            # Clean the response text (remove action tags for display)
            display_text = self._clean_response(response_text)

            return (display_text, action)

        except Exception as e:
            return (f"Claude error: {e}", None)

    def _parse_action(self, text: str) -> Optional[dict]:
        """Extract action from Claude's response."""
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
        self.conversation_history = []
