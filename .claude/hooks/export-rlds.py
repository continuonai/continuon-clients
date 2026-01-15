#!/usr/bin/env python3
"""
Stop Hook - Export session as RLDS episode for Brain A training

This hook runs when the Claude Code session ends and:
1. Collects all recorded steps from the session
2. Converts them to RLDS format
3. Exports to Brain A's training directory

This creates the training pipeline: Brain B sessions → RLDS episodes → Brain A training
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import hashlib

def collect_session_steps(buffer_dir: Path, session_id: str) -> list[dict]:
    """Collect all steps from the session buffer."""
    session_dir = buffer_dir / session_id
    if not session_dir.exists():
        return []

    steps = []
    for step_file in sorted(session_dir.glob("step_*.json")):
        with open(step_file) as f:
            steps.append(json.load(f))

    return steps

def convert_to_rlds_step(step: dict, step_idx: int) -> dict:
    """Convert a Brain B step to RLDS format."""
    tool_name = step.get("tool_name", "unknown")
    tool_input = step.get("tool_input", {})
    tool_result = step.get("tool_result", {})

    # Create observation (what the agent saw)
    observation = {
        "context": {
            "tool_available": tool_name,
            "timestamp": step.get("timestamp", ""),
        },
        # XR placeholders (Brain B doesn't have real XR data)
        "headset_pose": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "valid": False},
        "hand_poses": {"left": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "valid": False},
                       "right": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1], "valid": False}},
        "robot_state": {"joint_positions": [0] * 6, "gripper_position": 0, "valid": False},
    }

    # Create action (what the agent did)
    action = {
        "tool_name": tool_name,
        "parameters": tool_input,
        "success": not tool_result.get("is_error", False),
    }

    # Create reward signal
    reward = 1.0 if action["success"] else -0.5

    return {
        "step_idx": step_idx,
        "observation": observation,
        "action": action,
        "reward": reward,
        "is_terminal": False,
        "is_truncated": False,
    }

def create_rlds_episode(session_id: str, steps: list[dict], state: dict) -> dict:
    """Create a complete RLDS episode from session data."""
    # Generate episode ID
    episode_id = f"brainb_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Convert steps to RLDS format
    rlds_steps = [convert_to_rlds_step(step, i) for i, step in enumerate(steps)]

    # Mark last step as terminal
    if rlds_steps:
        rlds_steps[-1]["is_terminal"] = True

    # Create episode metadata
    episode = {
        "episode_id": episode_id,
        "environment_id": "brain_b_claude_code",
        "xr_mode": "trainer",  # Placeholder - Brain B is non-XR
        "control_role": "human_supervisor",
        "software": {
            "xr_app": "claude_code",
            "xr_app_version": "1.0",
            "brain_b_version": "0.1.0",
        },
        "metadata": {
            "source": "brain_b_hook",
            "session_id": session_id,
            "started_at": state.get("started_at", ""),
            "actions_recorded": state.get("actions_recorded", 0),
            "guardrails_triggered": state.get("guardrails_triggered", 0),
        },
        "tags": ["brain_b", "claude_code", "auto_generated"],
        "steps": rlds_steps,
        "num_steps": len(rlds_steps),
        "total_reward": sum(s["reward"] for s in rlds_steps),
    }

    return episode

def write_rlds_episode(episode: dict, output_dir: Path):
    """Write RLDS episode to disk in the standard format."""
    episode_dir = output_dir / episode["episode_id"]
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    metadata = {k: v for k, v in episode.items() if k != "steps"}
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write steps
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
        session_dir.rmdir()

def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        hook_input = {}

    # Get paths
    data_dir = Path(os.environ.get("BRAIN_B_DATA_DIR", "./brain_b_data"))
    rlds_dir = Path(os.environ.get("BRAIN_A_RLDS_DIR", "./continuonbrain/rlds/episodes"))
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

    # Write to Brain A's RLDS directory
    rlds_dir.mkdir(parents=True, exist_ok=True)
    episode_path = write_rlds_episode(episode, rlds_dir)

    # Clean up buffer
    cleanup_buffer(buffer_dir, session_id)

    # Update session state
    state["status"] = "exported"
    state["exported_at"] = datetime.now().isoformat()
    state["episode_id"] = episode["episode_id"]
    state["episode_path"] = str(episode_path)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Output summary
    print(f"""
Brain B Session Exported to RLDS

Episode: {episode['episode_id']}
Steps: {episode['num_steps']}
Total Reward: {episode['total_reward']:.2f}
Path: {episode_path}

This episode is now available for Brain A training.
Run: python -m continuonbrain.trainer.local_lora_trainer --episodes {episode_path}
""")

    sys.exit(0)

if __name__ == "__main__":
    main()
