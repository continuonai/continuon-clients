#!/usr/bin/env python3
"""
PreToolUse Hook - Validate robot actions through Brain B

This hook intercepts robot-related commands before execution and:
1. Checks against Brain B guardrails
2. Validates safety constraints
3. Can modify or block dangerous commands

Exit codes:
  0 = Allow (with optional modifications)
  2 = Block (message to Claude)
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

def load_guardrails(data_dir: Path) -> list[dict]:
    """Load guardrails from Brain B data directory."""
    guardrails_file = data_dir / "guardrails.json"
    if guardrails_file.exists():
        with open(guardrails_file) as f:
            return json.load(f).get("guardrails", [])
    return []

def check_guardrails(command: str, guardrails: list[dict]) -> tuple[bool, str]:
    """Check if command violates any guardrails."""
    command_lower = command.lower()

    for gr in guardrails:
        trigger = gr.get("trigger", "").lower()
        if trigger and trigger in command_lower:
            return False, f"Guardrail {gr.get('id', 'GR-XXX')}: {gr.get('action', 'blocked')}"

    return True, ""

def check_safety(command: str) -> tuple[bool, str, str | None]:
    """
    Check basic safety constraints.
    Returns (allowed, reason, modified_command)
    """
    command_lower = command.lower()

    # Block dangerous patterns
    dangerous = [
        ("rm -rf /", "Cannot delete root filesystem"),
        ("sudo reboot", "Cannot reboot system without confirmation"),
        ("pkill -9", "Cannot force-kill processes"),
    ]

    for pattern, reason in dangerous:
        if pattern in command_lower:
            return False, reason, None

    # Modify risky patterns
    if "motor" in command_lower and "speed" in command_lower:
        # Extract and clamp speed values
        # This is a simplified example - real implementation would parse properly
        if "1.0" in command or "100%" in command:
            modified = command.replace("1.0", "0.5").replace("100%", "50%")
            return True, "Speed clamped to 50% for safety", modified

    return True, "", None

def log_decision(data_dir: Path, tool_input: dict, decision: str, reason: str):
    """Log the validation decision for audit."""
    log_file = data_dir / "validation_log.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_input": tool_input,
        "decision": decision,
        "reason": reason,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        # No input or invalid JSON - allow by default
        sys.exit(0)

    # Get tool information
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    command = tool_input.get("command", "")

    # Get Brain B data directory
    data_dir = Path(os.environ.get("BRAIN_B_DATA_DIR", "./brain_b_data"))

    # Load guardrails
    guardrails = load_guardrails(data_dir)

    # Check guardrails
    allowed, gr_reason = check_guardrails(command, guardrails)
    if not allowed:
        log_decision(data_dir, tool_input, "blocked", gr_reason)
        print(gr_reason, file=sys.stderr)
        sys.exit(2)  # Block

    # Check safety
    safe, safety_reason, modified_command = check_safety(command)
    if not safe:
        log_decision(data_dir, tool_input, "blocked", safety_reason)
        print(safety_reason, file=sys.stderr)
        sys.exit(2)  # Block

    # If command was modified, return updated input
    if modified_command:
        log_decision(data_dir, tool_input, "modified", safety_reason)

        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": safety_reason,
                "updatedInput": {
                    "command": modified_command,
                    "description": tool_input.get("description", "") + f" (Brain B: {safety_reason})",
                }
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    # Allow without modification
    log_decision(data_dir, tool_input, "allowed", "")
    sys.exit(0)

if __name__ == "__main__":
    main()
