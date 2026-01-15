"""
Intent Classification - Figure out what the user wants.

Simple pattern-based classification. Can upgrade to LLM later if needed.
"""

from enum import Enum, auto
from dataclasses import dataclass
import re


class Intent(Enum):
    """What the user wants to do."""

    # Movement
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    STOP = auto()

    # Speed control
    SPEED_UP = auto()
    SLOW_DOWN = auto()
    SET_SPEED = auto()

    # Teaching
    START_TEACHING = auto()
    STOP_TEACHING = auto()
    CANCEL_TEACHING = auto()

    # Invoke learned behavior
    INVOKE_BEHAVIOR = auto()

    # Memory
    LIST_BEHAVIORS = auto()
    FORGET_BEHAVIOR = auto()
    DESCRIBE_BEHAVIOR = auto()

    # System
    STATUS = auto()
    HELP = auto()
    QUIT = auto()

    # Unknown
    UNKNOWN = auto()


@dataclass
class ParsedIntent:
    """A classified intent with extracted parameters."""

    intent: Intent
    params: dict
    raw_text: str
    confidence: float = 1.0


class IntentClassifier:
    """
    Pattern-based intent classifier.

    Matches user input against regex patterns to determine intent.
    """

    # Patterns for each intent (order matters - first match wins)
    PATTERNS = [
        # Quit
        (r"^(quit|exit|bye|goodbye)$", Intent.QUIT, {}),

        # Help
        (r"^(help|\?)$", Intent.HELP, {}),

        # Status
        (r"^(status|info|state)$", Intent.STATUS, {}),

        # Teaching
        (r"^(teach|learn|record|remember)\s+(.+)$", Intent.START_TEACHING, {"name": 2}),
        (r"^(done|finished|that'?s?\s*(it|all)|end|save)$", Intent.STOP_TEACHING, {}),
        (r"^(cancel|abort|nevermind|never\s*mind)$", Intent.CANCEL_TEACHING, {}),

        # Memory
        (r"^(list|show|what)\s*(behaviors?|skills?|commands?).*$", Intent.LIST_BEHAVIORS, {}),
        (r"^(forget|delete|remove)\s+(.+)$", Intent.FORGET_BEHAVIOR, {"name": 2}),
        (r"^(describe|explain|what\s*is)\s+(.+)$", Intent.DESCRIBE_BEHAVIOR, {"name": 2}),

        # Stop (high priority)
        (r"^(stop|halt|freeze|brake|whoa|wait)!?$", Intent.STOP, {}),

        # Speed
        (r"^(faster|speed\s*up|accelerate)$", Intent.SPEED_UP, {}),
        (r"^(slower|slow\s*down|decelerate)$", Intent.SLOW_DOWN, {}),
        (r"^speed\s+(\d+)%?$", Intent.SET_SPEED, {"percent": 1}),

        # Movement with optional modifiers
        (r"^(go\s+)?(forward|ahead|straight)(\s+slow(ly)?)?$", Intent.MOVE_FORWARD, {"slow": 3}),
        (r"^(go\s+)?(back|backward|reverse)(\s+slow(ly)?)?$", Intent.MOVE_BACKWARD, {"slow": 3}),
        (r"^(turn\s+)?left$", Intent.TURN_LEFT, {}),
        (r"^(turn\s+)?right$", Intent.TURN_RIGHT, {}),

        # Shorthand
        (r"^f$", Intent.MOVE_FORWARD, {}),
        (r"^b$", Intent.MOVE_BACKWARD, {}),
        (r"^l$", Intent.TURN_LEFT, {}),
        (r"^r$", Intent.TURN_RIGHT, {}),
        (r"^s$", Intent.STOP, {}),
    ]

    def __init__(self, behaviors: list[str] | None = None):
        """
        Initialize classifier.

        Args:
            behaviors: List of known behavior names to recognize
        """
        self.behaviors = behaviors or []

    def update_behaviors(self, behaviors: list[str]):
        """Update the list of known behaviors."""
        self.behaviors = behaviors

    def classify(self, text: str) -> ParsedIntent:
        """
        Classify user input into an intent.

        Returns ParsedIntent with the detected intent and any parameters.
        """
        text = text.strip().lower()

        if not text:
            return ParsedIntent(Intent.UNKNOWN, {}, text, 0.0)

        # Check patterns
        for pattern, intent, param_groups in self.PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                params = {}
                for name, group_num in param_groups.items():
                    if isinstance(group_num, int) and group_num <= len(match.groups()):
                        value = match.group(group_num)
                        if value:
                            params[name] = value.strip()
                return ParsedIntent(intent, params, text, 1.0)

        # Check if it's a known behavior name
        for behavior in self.behaviors:
            if text == behavior.lower() or text == f"do {behavior.lower()}" or text == f"run {behavior.lower()}":
                return ParsedIntent(Intent.INVOKE_BEHAVIOR, {"name": behavior}, text, 1.0)

        # Fuzzy match behavior names
        for behavior in self.behaviors:
            if behavior.lower() in text or text in behavior.lower():
                return ParsedIntent(Intent.INVOKE_BEHAVIOR, {"name": behavior}, text, 0.7)

        return ParsedIntent(Intent.UNKNOWN, {}, text, 0.0)
