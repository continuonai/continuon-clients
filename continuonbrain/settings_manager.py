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
    "identity": {
        # Robot's display name (editable by owner)
        "robot_name": "ContinuonBot",
        # Human-readable, non-biometric label for the creator (not editable).
        # This is hardcoded and represents the creator of the Continuon project.
        "creator_display_name": "Craig Michael Merry",
        # Creator name is immutable - system enforced
        "_creator_immutable": True,
    },
    "safety": {
        "allow_motion": True,
        "record_episodes": True,
        "require_supervision": False,
    },
    "telemetry": {"rate_hz": 2.0},
    "chat": {
        "persona": "operator",
        "temperature": 0.35,
        # Opt-in: log chat turns to RLDS episodes for later training/eval replay.
        "log_rlds": False,
    },
    "training": {
        "enable_sidecar_trainer": False,  # Disabled by default to save resources
        "enable_sleep_learning": True,    # Enabled by default for autonomous learning
    },
    "agent_manager": {
        "agent_model": "hope-v1",  # Default to HOPE Agent. Options: hope-v1, gemma-local, claude-code
        "enable_thinking_indicator": True,
        "enable_intervention_prompts": True,
        "intervention_confidence_threshold": 0.5,
        "enable_status_updates": True,
        "enable_autonomous_learning": True,
        "autonomous_learning_steps_per_cycle": 100,
        "autonomous_learning_checkpoint_interval": 1000,
        # Offline-first: periodic "self-learning" chat turns that are logged to RLDS (if chat.log_rlds=true).
        # This does NOT call the internet; it frames questions as "internet-search style" prompts about this repo.
        "chat_learn": {
            "enabled": False,
            "interval_s": 600,  # 10 minutes
            "turns_per_cycle": 10,
            "model_hint": "google/gemma-3n-2b",
            # Optional: run a second model as a subagent (agent manager ↔ subagent) and
            # log the full dialogue to RLDS when chat.log_rlds=true.
            # Empty/None disables subagent delegation.
            "delegate_model_hint": "",
            "topic": "coding this repository (ContinuonXR/continuonbrain/continuonai)",
            # Which robot modes are allowed to run scheduled chat learning.
            # Default is idle-only; set to ["autonomous"] to learn while autonomous.
            "modes": ["idle"],
        },
        # Continuous learning orchestrator for autonomous mode.
        # Runs bounded maintenance tasks (CMS compaction, evals, WaveCore loops, tool-router refresh)
        # while respecting resource limits and pausing the learner around heavy jobs.
        "autonomy_orchestrator": {
            "enabled": False,
            "modes": ["autonomous"],
            "min_interval_s": 30,
            "cms_compact_every_s": 600,
            "hope_eval_every_s": 1800,
            "facts_eval_every_s": 3600,
            "wavecore_every_s": 1800,
            "tool_router_every_s": 3600,
            # Skip tasks when headroom drops below reserve+X MB (prevents compaction/evals from thrashing on low-memory Pi).
            "min_memory_headroom_mb": 512,
            # Bounded workloads (keep small on Pi)
            "wavecore_steps_fast": 60,
            "wavecore_steps_mid": 120,
            "wavecore_steps_slow": 180,
            "tool_router_steps": 200,
        },
    },
    # Claude Code / Agent Harness Configuration
    # Integrates with Civqo's Ralph loop for agentic workflows
    "harness": {
        # Whether harness mode is enabled
        "enabled": False,
        # Harness type: "ralph" (Ralph loop), "native" (local agent), "civqo" (Civqo cloud)
        "harness_type": "ralph",
        # Claude Code / Anthropic API configuration
        "claude_code": {
            "enabled": False,
            # API key is stored encrypted in secrets.json, not here
            # This just indicates if key is configured
            "api_key_configured": False,
            # Model preference: claude-3-opus, claude-opus-4-5, claude-3-haiku
            "model": "claude-opus-4-5",
            # Max tokens for responses
            "max_tokens": 4096,
            # Whether to show Claude thinking process
            "show_thinking": True,
            # Enable tool use (file operations, bash, etc.)
            "enable_tools": True,
        },
        # Ralph Loop configuration (from Civqo harness)
        "ralph_loop": {
            "enabled": False,
            # Max iterations per task
            "max_iterations": 50,
            # Auto-commit changes
            "auto_commit": False,
            # Require approval before execution
            "require_approval": True,
            # Export sessions as RLDS episodes
            "export_rlds": True,
            # Working directory for Ralph sessions
            "working_directory": "",
            # Task queue mode: "sequential", "parallel"
            "task_mode": "sequential",
        },
        # RLDS export configuration (for training data generation)
        "rlds_export": {
            "enabled": True,
            # Auto-export completed sessions
            "auto_export": True,
            # Output directory for RLDS episodes
            "output_dir": "/opt/continuonos/brain/rlds/episodes",
            # Anonymize exported data
            "anonymize": True,
            # Include tool call results
            "include_tool_results": True,
            # Max steps per episode
            "max_steps_per_episode": 1000,
        },
    },
}

# Valid agent model options
ALLOWED_AGENT_MODELS = {"hope-v1", "gemma-local", "claude-code", "mock"}
# Valid harness types
ALLOWED_HARNESS_TYPES = {"ralph", "native", "civqo", "claude-code"}
# Valid Claude models
ALLOWED_CLAUDE_MODELS = {"claude-opus-4-5", "claude-3-opus", "claude-opus-4-5", "claude-3-haiku", "claude-3-5-sonnet"}

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

        identity = payload.get("identity", {}) if isinstance(payload, dict) else {}
        
        # Robot name is editable by owner
        robot_name = identity.get("robot_name", normalized["identity"]["robot_name"])
        robot_name_str = str(robot_name or "").strip()
        if len(robot_name_str) > 50:
            errors.append("Robot name must be <= 50 characters")
        elif robot_name_str:
            normalized["identity"]["robot_name"] = robot_name_str
        
        # Creator display name is IMMUTABLE - always use hardcoded default
        # This represents "Craig Michael Merry" and cannot be changed
        normalized["identity"]["creator_display_name"] = DEFAULT_SETTINGS["identity"]["creator_display_name"]
        normalized["identity"]["_creator_immutable"] = True

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

        normalized["chat"]["log_rlds"] = bool(chat.get("log_rlds", normalized["chat"]["log_rlds"]))

        if errors:
            raise SettingsValidationError("; ".join(errors))
        
        # Training settings
        training = payload.get("training", {}) if isinstance(payload, dict) else {}
        normalized["training"] = {}
        normalized["training"]["enable_sidecar_trainer"] = bool(
            training.get("enable_sidecar_trainer", DEFAULT_SETTINGS["training"]["enable_sidecar_trainer"])
        )
        normalized["training"]["enable_sleep_learning"] = bool(
            training.get("enable_sleep_learning", DEFAULT_SETTINGS["training"]["enable_sleep_learning"])
        )
        
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

        # Validate chat_learn scheduler config
        chat_learn = agent_mgr.get("chat_learn", DEFAULT_SETTINGS["agent_manager"]["chat_learn"])
        if not isinstance(chat_learn, dict):
            chat_learn = DEFAULT_SETTINGS["agent_manager"]["chat_learn"]
        normalized["agent_manager"]["chat_learn"] = {}
        normalized["agent_manager"]["chat_learn"]["enabled"] = bool(
            chat_learn.get("enabled", DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["enabled"])
        )
        try:
            interval_s = int(chat_learn.get("interval_s", DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["interval_s"]))
            interval_s = max(30, min(interval_s, 3600))
        except (TypeError, ValueError):
            interval_s = int(DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["interval_s"])
        normalized["agent_manager"]["chat_learn"]["interval_s"] = interval_s
        try:
            turns = int(chat_learn.get("turns_per_cycle", DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["turns_per_cycle"]))
            turns = max(1, min(turns, 50))
        except (TypeError, ValueError):
            turns = int(DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["turns_per_cycle"])
        normalized["agent_manager"]["chat_learn"]["turns_per_cycle"] = turns
        model_hint = str(chat_learn.get("model_hint", DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["model_hint"]) or "").strip()
        normalized["agent_manager"]["chat_learn"]["model_hint"] = model_hint or DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["model_hint"]
        delegate_hint = str(
            chat_learn.get(
                "delegate_model_hint",
                DEFAULT_SETTINGS["agent_manager"]["chat_learn"].get("delegate_model_hint", ""),
            )
            or ""
        ).strip()
        # Keep it short and safe; empty disables delegation.
        normalized["agent_manager"]["chat_learn"]["delegate_model_hint"] = delegate_hint[:80]
        topic = str(chat_learn.get("topic", DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["topic"]) or "").strip()
        normalized["agent_manager"]["chat_learn"]["topic"] = topic[:200] if topic else DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["topic"]
        allowed_modes = {"idle", "autonomous", "sleep_learning", "manual_training", "manual_control"}
        modes_in = chat_learn.get("modes", DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["modes"])
        modes: list[str] = []
        if isinstance(modes_in, list):
            for m in modes_in:
                ms = str(m or "").strip().lower()
                if ms in allowed_modes and ms not in modes:
                    modes.append(ms)
        if not modes:
            modes = list(DEFAULT_SETTINGS["agent_manager"]["chat_learn"]["modes"])
        normalized["agent_manager"]["chat_learn"]["modes"] = modes

        # Autonomous orchestrator config
        orch = agent_mgr.get("autonomy_orchestrator", DEFAULT_SETTINGS["agent_manager"]["autonomy_orchestrator"])
        if not isinstance(orch, dict):
            orch = DEFAULT_SETTINGS["agent_manager"]["autonomy_orchestrator"]
        normalized["agent_manager"]["autonomy_orchestrator"] = {}
        normalized["agent_manager"]["autonomy_orchestrator"]["enabled"] = bool(
            orch.get("enabled", DEFAULT_SETTINGS["agent_manager"]["autonomy_orchestrator"]["enabled"])
        )
        modes_in2 = orch.get("modes", DEFAULT_SETTINGS["agent_manager"]["autonomy_orchestrator"]["modes"])
        modes2: list[str] = []
        if isinstance(modes_in2, list):
            for m in modes_in2:
                ms = str(m or "").strip().lower()
                if ms in allowed_modes and ms not in modes2:
                    modes2.append(ms)
        if not modes2:
            modes2 = list(DEFAULT_SETTINGS["agent_manager"]["autonomy_orchestrator"]["modes"])
        normalized["agent_manager"]["autonomy_orchestrator"]["modes"] = modes2

        def _ival(key: str, lo: int, hi: int, default: int) -> int:
            try:
                v = int(orch.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        normalized["agent_manager"]["autonomy_orchestrator"]["min_interval_s"] = _ival("min_interval_s", 10, 600, 30)
        normalized["agent_manager"]["autonomy_orchestrator"]["cms_compact_every_s"] = _ival("cms_compact_every_s", 10, 86400, 600)
        normalized["agent_manager"]["autonomy_orchestrator"]["hope_eval_every_s"] = _ival("hope_eval_every_s", 10, 86400, 1800)
        normalized["agent_manager"]["autonomy_orchestrator"]["facts_eval_every_s"] = _ival("facts_eval_every_s", 10, 86400, 3600)
        normalized["agent_manager"]["autonomy_orchestrator"]["wavecore_every_s"] = _ival("wavecore_every_s", 10, 86400, 1800)
        normalized["agent_manager"]["autonomy_orchestrator"]["tool_router_every_s"] = _ival("tool_router_every_s", 10, 86400, 3600)

        normalized["agent_manager"]["autonomy_orchestrator"]["wavecore_steps_fast"] = _ival("wavecore_steps_fast", 10, 1000, 60)
        normalized["agent_manager"]["autonomy_orchestrator"]["wavecore_steps_mid"] = _ival("wavecore_steps_mid", 10, 2000, 120)
        normalized["agent_manager"]["autonomy_orchestrator"]["wavecore_steps_slow"] = _ival("wavecore_steps_slow", 10, 3000, 180)
        normalized["agent_manager"]["autonomy_orchestrator"]["tool_router_steps"] = _ival("tool_router_steps", 50, 5000, 200)
        
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

        # Harness settings (Claude Code / Ralph Loop)
        harness = payload.get("harness", {}) if isinstance(payload, dict) else {}
        # Deep copy the defaults to avoid reference issues
        normalized["harness"] = json.loads(json.dumps(DEFAULT_SETTINGS.get("harness", {})))

        # Main harness settings
        normalized["harness"]["enabled"] = bool(harness.get("enabled", normalized["harness"].get("enabled", False)))

        harness_type = str(harness.get("harness_type", normalized["harness"].get("harness_type", "ralph"))).strip()
        if harness_type in ALLOWED_HARNESS_TYPES:
            normalized["harness"]["harness_type"] = harness_type
        else:
            normalized["harness"]["harness_type"] = "ralph"

        # Claude Code settings
        claude_code = harness.get("claude_code", {}) if isinstance(harness, dict) else {}
        if "claude_code" not in normalized["harness"]:
            normalized["harness"]["claude_code"] = {}

        normalized["harness"]["claude_code"]["enabled"] = bool(claude_code.get("enabled", normalized["harness"]["claude_code"].get("enabled", False)))

        # Preserve API key if present (don't overwrite with None)
        if claude_code.get("api_key"):
            normalized["harness"]["claude_code"]["api_key"] = claude_code["api_key"]
        elif "api_key" in normalized["harness"]["claude_code"]:
            pass  # Keep existing value

        normalized["harness"]["claude_code"]["api_key_configured"] = bool(normalized["harness"]["claude_code"].get("api_key"))

        claude_model = str(claude_code.get("model", normalized["harness"]["claude_code"].get("model", "claude-opus-4-5"))).strip()
        if claude_model in ALLOWED_CLAUDE_MODELS:
            normalized["harness"]["claude_code"]["model"] = claude_model
        else:
            normalized["harness"]["claude_code"]["model"] = "claude-opus-4-5"

        normalized["harness"]["claude_code"]["max_tokens"] = int(claude_code.get("max_tokens", normalized["harness"]["claude_code"].get("max_tokens", 4096)))
        normalized["harness"]["claude_code"]["show_thinking"] = bool(claude_code.get("show_thinking", normalized["harness"]["claude_code"].get("show_thinking", True)))
        normalized["harness"]["claude_code"]["enable_tools"] = bool(claude_code.get("enable_tools", normalized["harness"]["claude_code"].get("enable_tools", True)))

        # Ralph Loop settings
        ralph_loop = harness.get("ralph_loop", {}) if isinstance(harness, dict) else {}
        if "ralph_loop" not in normalized["harness"]:
            normalized["harness"]["ralph_loop"] = {}

        normalized["harness"]["ralph_loop"]["enabled"] = bool(ralph_loop.get("enabled", normalized["harness"]["ralph_loop"].get("enabled", False)))

        max_iter = ralph_loop.get("max_iterations", normalized["harness"]["ralph_loop"].get("max_iterations", 50))
        try:
            max_iter_val = max(1, min(200, int(max_iter)))
        except (TypeError, ValueError):
            max_iter_val = 50
        normalized["harness"]["ralph_loop"]["max_iterations"] = max_iter_val

        normalized["harness"]["ralph_loop"]["auto_commit"] = bool(ralph_loop.get("auto_commit", normalized["harness"]["ralph_loop"].get("auto_commit", False)))
        normalized["harness"]["ralph_loop"]["require_approval"] = bool(ralph_loop.get("require_approval", normalized["harness"]["ralph_loop"].get("require_approval", True)))
        normalized["harness"]["ralph_loop"]["export_rlds"] = bool(ralph_loop.get("export_rlds", normalized["harness"]["ralph_loop"].get("export_rlds", True)))
        normalized["harness"]["ralph_loop"]["working_directory"] = str(ralph_loop.get("working_directory", normalized["harness"]["ralph_loop"].get("working_directory", ""))).strip()
        normalized["harness"]["ralph_loop"]["task_mode"] = str(ralph_loop.get("task_mode", normalized["harness"]["ralph_loop"].get("task_mode", "sequential"))).strip()

        # RLDS Export settings
        rlds_export = harness.get("rlds_export", {}) if isinstance(harness, dict) else {}
        if "rlds_export" not in normalized["harness"]:
            normalized["harness"]["rlds_export"] = {}

        normalized["harness"]["rlds_export"]["enabled"] = bool(rlds_export.get("enabled", normalized["harness"]["rlds_export"].get("enabled", True)))
        normalized["harness"]["rlds_export"]["auto_export"] = bool(rlds_export.get("auto_export", normalized["harness"]["rlds_export"].get("auto_export", True)))
        normalized["harness"]["rlds_export"]["output_dir"] = str(rlds_export.get("output_dir", normalized["harness"]["rlds_export"].get("output_dir", "/opt/continuonos/brain/rlds/episodes"))).strip()
        normalized["harness"]["rlds_export"]["anonymize"] = bool(rlds_export.get("anonymize", normalized["harness"]["rlds_export"].get("anonymize", True)))
        normalized["harness"]["rlds_export"]["include_tool_results"] = bool(rlds_export.get("include_tool_results", normalized["harness"]["rlds_export"].get("include_tool_results", True)))

        max_steps = rlds_export.get("max_steps_per_episode", normalized["harness"]["rlds_export"].get("max_steps_per_episode", 1000))
        try:
            max_steps_val = max(10, min(10000, int(max_steps)))
        except (TypeError, ValueError):
            max_steps_val = 1000
        normalized["harness"]["rlds_export"]["max_steps_per_episode"] = max_steps_val

        return normalized

    def _defaults(self) -> Dict[str, Any]:
        return json.loads(json.dumps(DEFAULT_SETTINGS))

