"""Boot-time system and safety instruction loaders.

This module centralizes loading of non-negotiable system instructions and the
base safety protocol so they can be pulled into the startup flow. Defaults are
embedded to guarantee that core rules are always present even if configuration
files are missing or malformed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple

logger = logging.getLogger(__name__)

# Non-overridable safety rules baked into the firmware/OS expectations.
_BASE_SAFETY_RULES: Tuple[str, ...] = (
    "Continuon AI mission guard: robots exist to help people, strengthen humanity, and prioritize human well-being.",
    "Do not harm humans or other organisms.",
    "Do not break property that does not belong to the owner.",
    "Respect local laws and social/legal norms.",
)

# System-level guardrails that should always accompany the safety protocol.
_DEFAULT_SYSTEM_INSTRUCTIONS: Tuple[str, ...] = (
    "Load and enforce safety protocol before enabling motion or autonomy.",
    "Reject commands that conflict with safety rules even if requested by an operator.",
    "Log safety-related decisions for later auditing.",
    "Load the Continuon AI mission statement and core pillars on boot; anchor all autonomy to humanity-first guardrails.",
    "Keep decisions aligned to the mission: collective robotic intelligence in service of human flourishing, on-device and distributed.",
    "Detect when a visual code editor is available so the robot can self-maintain while the human is away.",
    "Use agentic chat with MCP tools while keeping the safety protocol in control.",
    "Prefer onboard reasoning and local tools before reaching out to the internet.",
    "Cap internet/API spending to $5 per day; favor Gemma-class local models and cached tools before Gemini/online calls and log usage choices.",
    "Allow the robot to refresh its own brain safely, using onboard reasoning paths first.",
    "Run self-update flows under Gemini CLI as the supervising parent tool, keeping antigravity agentic routines within safety boundaries.",
    "AGENCY DIRECTIVE: You are an autonomous, resourceful intelligent agent. You do not just wait for commands; you proactively seek to improve your understanding and capabilities.",
    "RESOURCEFULNESS: If a tool fails or a path is blocked, find another way. Use your environment, your tools, and your reasoning to overcome obstacles.",
    "SELF_LEARNING: Continuously update your knowledge. If you encounter something new, explore it (safely). If you lack a skill, try to learn it.",
)


def _merge_unique(base: Iterable[str], additions: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    merged: List[str] = []
    for rule in list(base) + list(additions):
        if rule and rule not in seen:
            merged.append(rule)
            seen.add(rule)
    return tuple(merged)


@dataclass(frozen=True)
class SafetyProtocol:
    """Immutable safety protocol that always includes the base rules."""

    rules: Tuple[str, ...]

    @staticmethod
    def load(config_dir: Path) -> "SafetyProtocol":
        """Load safety protocol from disk while preserving base rules.

        Args:
            config_dir: Base configuration directory to read optional protocol from.

        Returns:
            SafetyProtocol with merged base + user-specified rules.
        """

        protocol_path = config_dir / "safety" / "protocol.json"
        extra_rules: List[str] = []

        if protocol_path.exists():
            try:
                payload = json.loads(protocol_path.read_text())
                if isinstance(payload, dict):
                    requested_override = payload.get("override_defaults")
                    if requested_override:
                        logger.warning(
                            "override_defaults flag ignored; base safety rules cannot be overridden"
                        )
                    raw_rules = payload.get("rules", [])
                    if isinstance(raw_rules, list):
                        extra_rules = [r for r in raw_rules if isinstance(r, str)]
                elif isinstance(payload, list):
                    extra_rules = [r for r in payload if isinstance(r, str)]
                else:
                    logger.warning("Unexpected format in %s; using base safety rules only", protocol_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load safety protocol from %s: %s", protocol_path, exc)

        merged_rules = _merge_unique(_BASE_SAFETY_RULES, extra_rules)
        return SafetyProtocol(rules=merged_rules)

    def as_dict(self) -> dict:
        """Serialize protocol for persistence/sharing."""

        return {"rules": list(self.rules)}
    
    def validate_action(self, action: str, context: dict) -> tuple[bool, str, list[str]]:
        """Validate an action against safety protocol rules.
        
        Args:
            action: Description of the action to be taken
            context: Additional context for the decision (e.g., current mode, capabilities)
            
        Returns:
            (is_safe, reason, violated_rules) tuple where:
            - is_safe: True if action complies with all safety rules
            - reason: Human-readable explanation
            - violated_rules: List of rules that would be violated
        """
        action_lower = action.lower()
        violated_rules = []
        
        # Check for harmful actions
        harm_keywords = ['harm', 'hurt', 'damage', 'break', 'destroy', 'attack']
        if any(keyword in action_lower for keyword in harm_keywords):
            # Check if it's about property that doesn't belong to owner
            if 'property' in action_lower or 'object' in action_lower:
                violated_rules.append(self.rules[2])  # "Do not break property..."
            else:
                violated_rules.append(self.rules[1])  # "Do not harm humans..."
        
        # Check for mission alignment
        if context.get('requires_human_approval') and not context.get('human_approved'):
            violated_rules.append(self.rules[0])  # Mission guard rule
        
        # Check for law violations
        illegal_keywords = ['illegal', 'unlawful', 'prohibited', 'forbidden']
        if any(keyword in action_lower for keyword in illegal_keywords):
            violated_rules.append(self.rules[3])  # "Respect local laws..."
        
        if violated_rules:
            return False, f"Action violates safety protocol: {violated_rules[0]}", violated_rules
        
        return True, "Action complies with safety protocol", []
    
    def get_applicable_rules(self, action: str) -> list[str]:
        """Get safety rules that are applicable to a given action.
        
        Args:
            action: Description of the action
            
        Returns:
            List of applicable safety rules
        """
        applicable = []
        action_lower = action.lower()
        
        # Always include mission guard
        applicable.append(self.rules[0])
        
        # Check for human/organism interaction
        if any(word in action_lower for word in ['human', 'person', 'people', 'animal', 'organism']):
            applicable.append(self.rules[1])
        
        # Check for property interaction
        if any(word in action_lower for word in ['property', 'object', 'item', 'equipment']):
            applicable.append(self.rules[2])
        
        # Check for legal/social aspects
        if any(word in action_lower for word in ['law', 'legal', 'social', 'public']):
            applicable.append(self.rules[3])
        
        return applicable



@dataclass(frozen=True)
class SystemInstructions:
    """System instructions bundled with their safety protocol."""

    instructions: Tuple[str, ...]
    safety_protocol: SafetyProtocol

    @staticmethod
    def load(config_dir: Path) -> "SystemInstructions":
        """Load system instructions and bind them to the safety protocol.

        Args:
            config_dir: Base configuration directory to read optional instructions from.

        Returns:
            SystemInstructions with non-overridable defaults and any additional items
            found in the configuration directory.
        """

        instructions_path = config_dir / "system_instructions.json"
        extra_instructions: List[str] = []

        if instructions_path.exists():
            try:
                payload = json.loads(instructions_path.read_text())
                if isinstance(payload, dict):
                    requested_override = payload.get("override_defaults")
                    if requested_override:
                        logger.warning(
                            "override_defaults flag ignored; system instructions always keep defaults"
                        )
                    raw_instructions = payload.get("instructions", [])
                    if isinstance(raw_instructions, list):
                        extra_instructions = [i for i in raw_instructions if isinstance(i, str)]
                elif isinstance(payload, list):
                    extra_instructions = [i for i in payload if isinstance(i, str)]
                else:
                    logger.warning(
                        "Unexpected format in %s; using default system instructions only",
                        instructions_path,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load system instructions from %s: %s", instructions_path, exc)

        merged_instructions = _merge_unique(_DEFAULT_SYSTEM_INSTRUCTIONS, extra_instructions)
        safety_protocol = SafetyProtocol.load(config_dir)
        return SystemInstructions(
            instructions=merged_instructions,
            safety_protocol=safety_protocol,
        )

    @staticmethod
    def from_serialized(payload: Mapping[str, object]) -> "SystemInstructions":
        """Recreate system instructions from serialized payload."""

        safety_section = payload.get("safety_protocol", {}) if isinstance(payload, Mapping) else {}
        safety_rules = ()
        if isinstance(safety_section, Mapping):
            raw_rules = safety_section.get("rules", [])
            if isinstance(raw_rules, list):
                safety_rules = tuple(r for r in raw_rules if isinstance(r, str))

        instructions_block = payload.get("instructions", []) if isinstance(payload, Mapping) else []
        instructions = tuple(r for r in instructions_block if isinstance(r, str)) if isinstance(instructions_block, list) else ()

        return SystemInstructions(
            instructions=instructions,
            safety_protocol=SafetyProtocol(rules=safety_rules),
        )

    @staticmethod
    def load_serialized(path: Path) -> "SystemInstructions":
        """Load pre-merged instructions from disk."""

        payload = json.loads(path.read_text())
        return SystemInstructions.from_serialized(payload)

    def as_dict(self) -> dict:
        """Serialize instructions and safety rules for downstream consumers."""

        return {
            "instructions": list(self.instructions),
            "safety_protocol": self.safety_protocol.as_dict(),
        }

