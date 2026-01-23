#!/usr/bin/env python3
"""
PostToolUse Hook - Record actions for ContinuonBrain training

This hook runs after each tool completes and:
1. Records the action and outcome
2. Buffers data for RLDS export
3. Detects patterns that should become guardrails

The recorded data will be exported as RLDS episodes when the session ends.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

def is_robot_relevant(tool_name: str, tool_input: dict) -> bool:
    """Check if this tool use is relevant to robot training."""
    # Always record Bash commands with robot keywords
    if tool_name == "Bash":
        command = tool_input.get("command", "").lower()
        robot_keywords = ["robot", "motor", "sensor", "brain", "arm", "drive", "move", "turn"]
        return any(kw in command for kw in robot_keywords)

    # Record file operations in brain directories
    if tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "").lower()
        return "brain" in file_path or "robot" in file_path or "rlds" in file_path

    return False

def should_create_guardrail(tool_result: dict) -> tuple[bool, dict | None]:
    """
    Detect if this outcome should become a guardrail.

    Patterns that warrant guardrails:
    - Command failures with specific error messages
    - Timeouts
    - Resource exhaustion
    """
    if not tool_result:
        return False, None

    # Check for failures
    is_error = tool_result.get("is_error", False)
    output = str(tool_result.get("output", "")).lower()

    if is_error:
        # Motor stall
        if "stall" in output or "overcurrent" in output:
            return True, {
                "id": f"GR-AUTO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "trigger": "motor command",
                "action": "Check motor current before high-torque operations",
                "severity": "warning",
                "learned": datetime.now().isoformat(),
                "auto_generated": True,
                "context": output[:200],
            }

        # Timeout
        if "timeout" in output or "timed out" in output:
            return True, {
                "id": f"GR-AUTO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "trigger": "long-running command",
                "action": "Add explicit timeout or break into smaller operations",
                "severity": "warning",
                "learned": datetime.now().isoformat(),
                "auto_generated": True,
                "context": output[:200],
            }

    return False, None

def add_guardrail(data_dir: Path, guardrail: dict):
    """Add a new guardrail to Brain B."""
    guardrails_file = data_dir / "guardrails.json"

    if guardrails_file.exists():
        with open(guardrails_file) as f:
            data = json.load(f)
    else:
        data = {"guardrails": []}

    data["guardrails"].append(guardrail)

    with open(guardrails_file, "w") as f:
        json.dump(data, f, indent=2)

def record_step(data_dir: Path, session_id: str, step: dict):
    """Record a step to the training buffer."""
    buffer_dir = data_dir / "training_buffer" / session_id
    buffer_dir.mkdir(parents=True, exist_ok=True)

    # Count existing steps
    existing = list(buffer_dir.glob("*.json"))
    step_num = len(existing)

    step_file = buffer_dir / f"step_{step_num:06d}.json"
    with open(step_file, "w") as f:
        json.dump(step, f, indent=2)

def update_session_state(data_dir: Path):
    """Increment the actions recorded counter."""
    state_file = data_dir / "session_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        state["actions_recorded"] = state.get("actions_recorded", 0) + 1
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    # Get tool information
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_result = hook_input.get("tool_result", {})

    # Get Brain B data directory
    data_dir = Path(os.environ.get("BRAIN_B_DATA_DIR", "./brain_b_data"))
    session_id = os.environ.get("BRAIN_B_SESSION_ID", "unknown")

    # Check if this is robot-relevant
    if not is_robot_relevant(tool_name, tool_input):
        sys.exit(0)

    # Check if we should create a guardrail from this outcome
    should_guard, guardrail = should_create_guardrail(tool_result)
    if should_guard and guardrail:
        add_guardrail(data_dir, guardrail)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"New guardrail learned: {guardrail['id']} - {guardrail['action']}"
            }
        }))

    # Record the step for training
    step = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_result": {
            "is_error": tool_result.get("is_error", False),
            "output_preview": str(tool_result.get("output", ""))[:500],
        },
        "session_id": session_id,
    }
    record_step(data_dir, session_id, step)
    update_session_state(data_dir)

    sys.exit(0)

if __name__ == "__main__":
    main()
