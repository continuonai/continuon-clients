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
    "agent_manager": {
        "agent_model": "hope-v1",  # Default to HOPE Agent
        "enable_thinking_indicator": True,
        "enable_intervention_prompts": True,
        "intervention_confidence_threshold": 0.5,
        "enable_status_updates": True,
        "enable_autonomous_learning": True,
        "autonomous_learning_steps_per_cycle": 100,
        "autonomous_learning_checkpoint_interval": 1000,
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
        
        # Agent Manager settings
        agent_mgr = payload.get("agent_manager", {}) if isinstance(payload, dict) else {}
        normalized["agent_manager"] = {}
        normalized["agent_manager"]["enable_thinking_indicator"] = bool(
            agent_mgr.get("enable_thinking_indicator", DEFAULT_SETTINGS["agent_manager"]["enable_thinking_indicator"])
        )
        normalized["agent_manager"]["enable_intervention_prompts"] = bool(
            agent_mgr.get("enable_intervention_prompts", DEFAULT_SETTINGS["agent_manager"]["enable_intervention_prompts"])
        )
        
        # Validate confidence threshold
        threshold = agent_mgr.get("intervention_confidence_threshold", DEFAULT_SETTINGS["agent_manager"]["intervention_confidence_threshold"])
        try:
            threshold_value = float(threshold)
            if threshold_value < 0 or threshold_value > 1:
                errors.append("Intervention confidence threshold must be between 0 and 1")
            else:
                normalized["agent_manager"]["intervention_confidence_threshold"] = round(threshold_value, 2)
        except (TypeError, ValueError):
            errors.append("Intervention confidence threshold must be a number")
        
        normalized["agent_manager"]["enable_status_updates"] = bool(
            agent_mgr.get("enable_status_updates", DEFAULT_SETTINGS["agent_manager"]["enable_status_updates"])
        )
        normalized["agent_manager"]["enable_autonomous_learning"] = bool(
            agent_mgr.get("enable_autonomous_learning", DEFAULT_SETTINGS["agent_manager"]["enable_autonomous_learning"])
        )
        
        # Validate learning parameters
        steps = agent_mgr.get("autonomous_learning_steps_per_cycle", DEFAULT_SETTINGS["agent_manager"]["autonomous_learning_steps_per_cycle"])
        try:
            steps_value = int(steps)
            if steps_value < 10 or steps_value > 1000:
                errors.append("Learning steps per cycle must be between 10 and 1000")
            else:
                normalized["agent_manager"]["autonomous_learning_steps_per_cycle"] = steps_value
        except (TypeError, ValueError):
            errors.append("Learning steps per cycle must be an integer")
        
        checkpoint = agent_mgr.get("autonomous_learning_checkpoint_interval", DEFAULT_SETTINGS["agent_manager"]["autonomous_learning_checkpoint_interval"])
        try:
            checkpoint_value = int(checkpoint)
            if checkpoint_value < 100 or checkpoint_value > 10000:
                errors.append("Checkpoint interval must be between 100 and 10000")
            else:
                normalized["agent_manager"]["autonomous_learning_checkpoint_interval"] = checkpoint_value
        except (TypeError, ValueError):
            errors.append("Checkpoint interval must be an integer")
        
        # Agent model selection
        agent_model = agent_mgr.get("agent_model", DEFAULT_SETTINGS["agent_manager"]["agent_model"])
        normalized["agent_manager"]["agent_model"] = str(agent_model).strip() or "mock"

        return normalized

    def _defaults(self) -> Dict[str, Any]:
        return json.loads(json.dumps(DEFAULT_SETTINGS))

