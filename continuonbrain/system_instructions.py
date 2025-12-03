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
    "Do not harm humans or other organisms.",
    "Do not break property that does not belong to the owner.",
    "Respect local laws and social/legal norms.",
)

# System-level guardrails that should always accompany the safety protocol.
_DEFAULT_SYSTEM_INSTRUCTIONS: Tuple[str, ...] = (
    "Load and enforce safety protocol before enabling motion or autonomy.",
    "Reject commands that conflict with safety rules even if requested by an operator.",
    "Log safety-related decisions for later auditing.",
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

