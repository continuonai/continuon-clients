"""Lightweight settings storage and validation for the robot API servers.

The settings payload is intentionally small so it can be mirrored downstream in
the production runtime. This module handles defaulting, validation, and
persistence to a JSON file located under the configured ``config_dir``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_SETTINGS: Dict[str, Any] = {
    "safety": {
        "allow_motion": True,
        "record_episodes": True,
        "require_supervision": False,
    },
    "telemetry": {"rate_hz": 2.0},
    "chat": {
        "persona": "operator",
        "temperature": 0.35,
    },
}

ALLOWED_PERSONAS = {"operator", "safety_officer", "demo_host"}


class SettingsValidationError(Exception):
    """Raised when a settings payload fails validation."""


@dataclass
class SettingsStore:
    """JSON-backed settings accessor with basic validation."""

    config_dir: Path

    @property
    def path(self) -> Path:
        return self.config_dir / "settings.json"

    def load(self) -> Dict[str, Any]:
        """Load settings from disk, falling back to defaults on error."""

        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                return self._validate(raw)
            except (json.JSONDecodeError, SettingsValidationError):
                # Malformed or invalid file; fall back to defaults and overwrite on save
                return self._defaults()

        return self._defaults()

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and persist settings to the configured path."""

        validated = self._validate(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(validated, indent=2))
        return validated

    def _validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        normalized = self._defaults()

        safety = payload.get("safety", {}) if isinstance(payload, dict) else {}
        normalized["safety"]["allow_motion"] = bool(
            safety.get("allow_motion", normalized["safety"]["allow_motion"])
        )
        normalized["safety"]["record_episodes"] = bool(
            safety.get("record_episodes", normalized["safety"]["record_episodes"])
        )
        normalized["safety"]["require_supervision"] = bool(
            safety.get("require_supervision", normalized["safety"]["require_supervision"])
        )

        telemetry = payload.get("telemetry", {}) if isinstance(payload, dict) else {}
        rate = telemetry.get("rate_hz", normalized["telemetry"]["rate_hz"])
        try:
            rate_value = float(rate)
            if rate_value <= 0 or rate_value > 30:
                errors.append("Telemetry rate must be between 0.1 and 30 Hz")
            else:
                normalized["telemetry"]["rate_hz"] = round(rate_value, 2)
        except (TypeError, ValueError):
            errors.append("Telemetry rate must be a number")

        chat = payload.get("chat", {}) if isinstance(payload, dict) else {}
        persona = str(chat.get("persona", normalized["chat"]["persona"]))
        persona_key = persona.strip() or normalized["chat"]["persona"]
        if persona_key not in ALLOWED_PERSONAS:
            errors.append(
                f"Chat persona must be one of: {', '.join(sorted(ALLOWED_PERSONAS))}"
            )
        else:
            normalized["chat"]["persona"] = persona_key

        temperature = chat.get("temperature", normalized["chat"]["temperature"])
        try:
            temperature_value = float(temperature)
            if temperature_value < 0 or temperature_value > 1:
                errors.append("Chat temperature must be between 0 and 1")
            else:
                normalized["chat"]["temperature"] = round(temperature_value, 2)
        except (TypeError, ValueError):
            errors.append("Chat temperature must be a number")

        if errors:
            raise SettingsValidationError("; ".join(errors))

        return normalized

    def _defaults(self) -> Dict[str, Any]:
        return json.loads(json.dumps(DEFAULT_SETTINGS))

