#!/usr/bin/env python3
"""
Stop Hook - Export session as RLDS episode for ContinuonBrain training

This hook runs when the Claude Code session ends and:
1. Collects all recorded steps from the session
2. Converts them to RLDS v2.0 format (no fake XR data, enriched context)
3. Exports to ContinuonBrain's training directory

Schema v2.0 changes:
- Removed fake XR placeholder data
- Added tool_schema to observations
- Added context history for temporal reasoning
- Unified action format

This creates the training pipeline: Brain B sessions -> RLDS episodes -> ContinuonBrain training
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Tool descriptions for context enrichment
TOOL_SCHEMAS = {
    "Bash": {
        "description": "Execute shell commands in the terminal",
        "parameters": {"command": "The bash command to execute"},
        "examples": ["ls -la", "git status", "python script.py"],
    },
    "Read": {
        "description": "Read contents of a file",
        "parameters": {"file_path": "Absolute path to the file"},
        "examples": ["/home/user/code/main.py"],
    },
    "Write": {
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {"file_path": "Absolute path", "content": "File content"},
        "examples": [],
    },
    "Edit": {
        "description": "Make targeted edits to a file",
        "parameters": {"file_path": "Path", "old_string": "Text to find", "new_string": "Replacement"},
        "examples": [],
    },
    "Glob": {
        "description": "Find files matching a pattern",
        "parameters": {"pattern": "Glob pattern like **/*.py"},
        "examples": ["**/*.py", "src/**/*.ts"],
    },
    "Grep": {
        "description": "Search for text patterns in files",
        "parameters": {"pattern": "Regex pattern", "path": "Directory to search"},
        "examples": [],
    },
    "Task": {
        "description": "Launch a sub-agent to handle a task",
        "parameters": {"prompt": "Task description", "subagent_type": "Agent type"},
        "examples": [],
    },
}


def collect_session_steps(buffer_dir: Path, session_id: str) -> List[Dict]:
    """Collect all steps from the session buffer."""
    session_dir = buffer_dir / session_id
    if not session_dir.exists():
        return []

    steps = []
    for step_file in sorted(session_dir.glob("step_*.json")):
        with open(step_file) as f:
            steps.append(json.load(f))

    return steps


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get schema for a tool, or None if unknown."""
    return TOOL_SCHEMAS.get(tool_name)


def build_context(
    step: Dict,
    step_idx: int,
    previous_steps: List[Dict],
    max_history: int = 3,
) -> Dict[str, Any]:
    """
    Build enriched context for a step.

    Includes tool schema and recent history for temporal reasoning.
    """
    tool_name = step.get("tool_name", "unknown")
    tool_schema = get_tool_schema(tool_name)

    # Build history from previous steps
    previous_actions = []
    previous_results = []

    for prev in previous_steps[-max_history:]:
        previous_actions.append({
            "tool_name": prev.get("tool_name", ""),
            "parameters": prev.get("tool_input", {}),
        })
        prev_result = prev.get("tool_result", {})
        previous_results.append({
            "success": not prev_result.get("is_error", False),
            "output_preview": prev_result.get("output_preview", "")[:200],
        })

    # Check for error in previous step
    last_error = None
    if previous_steps:
        last_result = previous_steps[-1].get("tool_result", {})
        if last_result.get("is_error"):
            last_error = last_result.get("output_preview", "")[:500]

    context = {
        "timestamp": step.get("timestamp", datetime.now().isoformat()),
        "step_index": step_idx,
        "tool_available": tool_name,
        "tool_schema": tool_schema,
        "previous_actions": previous_actions,
        "previous_results": previous_results,
        "last_error": last_error,
        "domain": "claude_code",
        "domain_state": {},
    }

    return context


def convert_to_rlds_step(step: Dict, step_idx: int, previous_steps: List[Dict]) -> Dict:
    """
    Convert a Brain B step to RLDS v2.0 format.

    Key changes from v1.1:
    - No fake XR placeholders
    - Enriched context with tool schema
    - History for temporal reasoning
    """
    tool_name = step.get("tool_name", "unknown")
    tool_input = step.get("tool_input", {})
    tool_result = step.get("tool_result", {})

    # Build enriched context
    context = build_context(step, step_idx, previous_steps)

    # Create observation (clean - no fake XR data)
    observation = {
        "context": context,
        # Domain-specific observation: tool result preview
        "domain_obs": {
            "result_preview": tool_result.get("output_preview", "")[:500],
            "result_type": "error" if tool_result.get("is_error") else "success",
        },
    }

    # Create action (unified format)
    success = not tool_result.get("is_error", False)
    action = {
        "action_type": "tool_call",
        "name": tool_name,
        "parameters": tool_input,
        "intent": "",  # Could be extracted from conversation context
        "raw_input": "",  # Could be extracted from user message
        "success": success,
        "error_message": tool_result.get("output_preview", "")[:200] if not success else None,
    }

    # Compute reward signal
    # More nuanced than simple success/fail
    reward = compute_reward(step, tool_result, previous_steps)

    return {
        "step_idx": step_idx,
        "timestamp_us": int(datetime.fromisoformat(
            step.get("timestamp", datetime.now().isoformat())
        ).timestamp() * 1_000_000),
        "observation": observation,
        "action": action,
        "reward": reward,
        "is_terminal": False,
        "is_truncated": False,
        "info": {
            "tool_name": tool_name,
            "session_id": step.get("session_id", "unknown"),
        },
    }


def compute_reward(
    step: Dict,
    tool_result: Dict,
    previous_steps: List[Dict],
) -> float:
    """
    Compute a more nuanced reward signal.

    Factors:
    - Success/failure of the action
    - Error recovery (success after failure = bonus)
    - Task progression signals
    """
    is_error = tool_result.get("is_error", False)

    if is_error:
        return -0.5

    # Base reward for success
    reward = 1.0

    # Bonus for recovering from error
    if previous_steps:
        prev_result = previous_steps[-1].get("tool_result", {})
        if prev_result.get("is_error"):
            reward += 0.5  # Recovery bonus

    # Check for task completion signals in output
    output = tool_result.get("output_preview", "").lower()
    if any(signal in output for signal in ["success", "complete", "done", "passed"]):
        reward += 0.25

    # Penalty for repeated identical actions (likely stuck)
    if len(previous_steps) >= 2:
        tool_name = step.get("tool_name", "")
        tool_input = step.get("tool_input", {})
        for prev in previous_steps[-2:]:
            if prev.get("tool_name") == tool_name and prev.get("tool_input") == tool_input:
                reward -= 0.25  # Repetition penalty
                break

    return reward


def create_rlds_episode(session_id: str, steps: List[Dict], state: Dict) -> Dict:
    """Create a complete RLDS v2.0 episode from session data."""
    # Generate episode ID
    episode_id = f"brainb_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Convert steps to RLDS format with history
    rlds_steps = []
    for i, step in enumerate(steps):
        previous = steps[:i]  # All previous steps
        rlds_step = convert_to_rlds_step(step, i, previous)
        rlds_steps.append(rlds_step)

    # Mark last step as terminal
    if rlds_steps:
        rlds_steps[-1]["is_terminal"] = True

    # Compute episode-level metrics
    total_reward = sum(s["reward"] for s in rlds_steps)
    success = total_reward > 0 and not any(
        s["action"].get("error_message") for s in rlds_steps[-3:]
    )  # Success if recent steps succeeded

    # Create episode metadata
    episode = {
        "metadata": {
            "schema_version": "2.0",
            "episode_id": episode_id,
            "source": "brain_b_hook",
            "domain": "claude_code",
            "session_id": session_id,
            "robot_id": "",
            "robot_model": "",
            "capabilities": ["tool_use", "code_generation", "file_operations"],
            "num_steps": len(rlds_steps),
            "total_reward": total_reward,
            "success": success,
            "start_time": state.get("started_at", 0),
            "end_time": datetime.now().timestamp(),
            "duration_s": 0,  # Will be computed
            "tags": ["brain_b", "claude_code", "v2"],
            "extra": {
                "actions_recorded": state.get("actions_recorded", 0),
                "guardrails_triggered": state.get("guardrails_triggered", 0),
            },
        },
        "steps": rlds_steps,
    }

    # Compute duration
    if episode["metadata"]["start_time"]:
        try:
            start = datetime.fromisoformat(str(episode["metadata"]["start_time"]))
            episode["metadata"]["duration_s"] = (
                datetime.now() - start
            ).total_seconds()
        except Exception:
            pass

    return episode


def write_rlds_episode(episode: Dict, output_dir: Path) -> Path:
    """Write RLDS episode to disk in the standard v2.0 format."""
    episode_id = episode["metadata"]["episode_id"]
    episode_dir = output_dir / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.json
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(episode["metadata"], f, indent=2)

    # Write steps/000000.jsonl (standard format)
    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(exist_ok=True)

    with open(steps_dir / "000000.jsonl", "w") as f:
        for step in episode["steps"]:
            f.write(json.dumps(step) + "\n")

    return episode_dir


def cleanup_buffer(buffer_dir: Path, session_id: str):
    """Clean up the session buffer after export."""
    session_dir = buffer_dir / session_id
    if session_dir.exists():
        for f in session_dir.glob("*.json"):
            f.unlink()
        try:
            session_dir.rmdir()
        except OSError:
            pass  # Directory not empty, leave it


def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        hook_input = {}

    # Get paths
    data_dir = Path(os.environ.get("BRAIN_B_DATA_DIR", "./brain_b_data"))
    rlds_dir = Path(os.environ.get("CONTINUONBRAIN_RLDS_DIR", "./continuonbrain/rlds/episodes"))
    session_id = os.environ.get("BRAIN_B_SESSION_ID", "unknown")

    # Load session state
    state_file = data_dir / "session_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {}

    # Collect steps from buffer
    buffer_dir = data_dir / "training_buffer"
    steps = collect_session_steps(buffer_dir, session_id)

    if not steps:
        print("No robot-relevant actions recorded in this session.")
        sys.exit(0)

    # Convert to RLDS episode
    episode = create_rlds_episode(session_id, steps, state)

    # Write to ContinuonBrain's RLDS directory
    rlds_dir.mkdir(parents=True, exist_ok=True)
    episode_path = write_rlds_episode(episode, rlds_dir)

    # Clean up buffer
    cleanup_buffer(buffer_dir, session_id)

    # Update session state
    state["status"] = "exported"
    state["exported_at"] = datetime.now().isoformat()
    state["episode_id"] = episode["metadata"]["episode_id"]
    state["episode_path"] = str(episode_path)
    state["schema_version"] = "2.0"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Output summary
    metadata = episode["metadata"]
    print(f"""
Brain B Session Exported to RLDS v2.0

Episode: {metadata['episode_id']}
Steps: {metadata['num_steps']}
Total Reward: {metadata['total_reward']:.2f}
Success: {metadata['success']}
Path: {episode_path}

Schema v2.0 features:
- No fake XR placeholders
- Tool schemas included
- Context history for temporal reasoning
- Nuanced reward signals

This episode is now available for ContinuonBrain training.
Run: python -m continuonbrain.trainer.auto_trainer_daemon --once
""")

    sys.exit(0)


if __name__ == "__main__":
    main()
