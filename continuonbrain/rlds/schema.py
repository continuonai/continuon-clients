"""
Unified RLDS Schema for ContinuonBrain Training.

This module defines the standard RLDS episode format used across all data sources:
- Brain B Claude Code hooks
- RobotGrid simulator
- Home3D simulator
- Real robot sessions

Schema Version: 2.0

Changes from v1.1:
- Removed fake XR data (headset_pose, hand_poses with valid=false)
- Standardized on steps/000000.jsonl storage format
- Added tool_schema to observations for better learning
- Added context history for temporal reasoning
- Unified action format across all sources
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Schema version - increment when making breaking changes
SCHEMA_VERSION = "2.0"


# ============================================================================
# Core Step Schema
# ============================================================================


@dataclass
class ToolSchema:
    """Description of a tool/action available to the agent."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)


@dataclass
class Context:
    """
    Contextual information for a step.

    Provides temporal and semantic context for learning.
    """
    # Current state
    timestamp: str = ""
    step_index: int = 0

    # Tool context
    tool_available: str = ""
    tool_schema: Optional[Dict[str, Any]] = None

    # History (last N results for temporal reasoning)
    previous_actions: List[Dict[str, Any]] = field(default_factory=list)
    previous_results: List[Dict[str, Any]] = field(default_factory=list)

    # Error context (if previous step failed)
    last_error: Optional[str] = None
    error_recovery_hint: Optional[str] = None

    # Domain-specific context
    domain: str = "general"  # "claude_code", "robotgrid", "home3d", "real_robot"
    domain_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    """
    Physical robot state (optional - only for real/simulated robots).

    Only include if data is actually valid, not placeholders.
    """
    # Position
    position: Optional[List[float]] = None  # [x, y, z] or [x, y] for 2D
    rotation: Optional[List[float]] = None  # [x, y, z, w] quaternion or [yaw] for 2D

    # Joint state
    joint_positions: Optional[List[float]] = None
    joint_velocities: Optional[List[float]] = None
    gripper_position: Optional[float] = None

    # Sensor data
    camera_available: bool = False
    depth_available: bool = False

    # Validity flag
    valid: bool = False


@dataclass
class Observation:
    """
    What the agent observed at this step.

    Clean schema without fake placeholder data.
    """
    # Required: context about the step
    context: Context

    # Optional: robot state (only if valid)
    robot_state: Optional[RobotState] = None

    # Domain-specific observations
    # For RobotGrid: visible_tiles, look_ahead
    # For Home3D: visible_objects, current_room
    # For Claude Code: tool_result_preview
    domain_obs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """
    What the agent did at this step.

    Unified format across all action types.
    """
    # Action type identifier
    action_type: str  # "tool_call", "navigation", "manipulation", etc.

    # Action name/command
    name: str  # Tool name, movement command, etc.

    # Action parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Intent (natural language description if available)
    intent: str = ""

    # Raw input (if from human/LLM)
    raw_input: str = ""

    # Outcome
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class Step:
    """
    A single step in an RLDS episode.

    This is the core unit of training data.
    """
    # Step identification
    step_idx: int
    timestamp_us: int  # Microseconds since epoch

    # Core data
    observation: Observation
    action: Action
    reward: float

    # Episode state
    is_terminal: bool = False
    is_truncated: bool = False

    # Additional info (domain-specific)
    info: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Episode Schema
# ============================================================================


@dataclass
class EpisodeMetadata:
    """
    Metadata for an RLDS episode.
    """
    # Identification
    schema_version: str = SCHEMA_VERSION
    episode_id: str = ""

    # Source
    source: str = ""  # "brain_b_hook", "robotgrid", "home3d", "real_robot"
    domain: str = ""  # "claude_code", "simulation", "real_world"
    session_id: str = ""

    # Robot info (if applicable)
    robot_id: str = ""
    robot_model: str = ""
    capabilities: List[str] = field(default_factory=list)

    # Episode summary
    num_steps: int = 0
    total_reward: float = 0.0
    success: bool = False

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0

    # Tags for filtering/categorization
    tags: List[str] = field(default_factory=list)

    # Domain-specific metadata
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Episode Writer
# ============================================================================


class RLDSEpisodeWriter:
    """
    Writes episodes in the standard RLDS format.

    Usage:
        writer = RLDSEpisodeWriter(output_dir="./rlds/episodes")

        # Start episode
        episode_id = writer.start_episode(
            source="brain_b_hook",
            domain="claude_code",
        )

        # Log steps
        writer.log_step(
            observation=obs,
            action=action,
            reward=1.0,
        )

        # End episode
        writer.end_episode(success=True)
    """

    def __init__(self, output_dir: str = "./continuonbrain/rlds/episodes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_metadata: Optional[EpisodeMetadata] = None
        self.steps: List[Step] = []
        self.step_counter: int = 0

    def start_episode(
        self,
        source: str,
        domain: str,
        session_id: str = "",
        robot_id: str = "",
        robot_model: str = "",
        capabilities: List[str] = None,
        tags: List[str] = None,
        extra: Dict[str, Any] = None,
    ) -> str:
        """
        Start a new episode.

        Returns:
            episode_id: Unique ID for this episode
        """
        timestamp = int(time.time())
        unique = uuid.uuid4().hex[:8]
        episode_id = f"{source}_{timestamp}_{unique}"

        self.current_metadata = EpisodeMetadata(
            episode_id=episode_id,
            source=source,
            domain=domain,
            session_id=session_id or str(uuid.uuid4()),
            robot_id=robot_id,
            robot_model=robot_model,
            capabilities=capabilities or [],
            tags=tags or [source, domain],
            extra=extra or {},
            start_time=time.time(),
        )

        self.steps = []
        self.step_counter = 0

        return episode_id

    def log_step(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        is_terminal: bool = False,
        is_truncated: bool = False,
        info: Dict[str, Any] = None,
    ) -> None:
        """Log a single step."""
        if self.current_metadata is None:
            raise ValueError("No episode in progress. Call start_episode() first.")

        step = Step(
            step_idx=self.step_counter,
            timestamp_us=int(time.time() * 1_000_000),
            observation=observation,
            action=action,
            reward=reward,
            is_terminal=is_terminal,
            is_truncated=is_truncated,
            info=info or {},
        )

        self.steps.append(step)
        self.step_counter += 1

    def end_episode(self, success: bool = False) -> Path:
        """
        End the current episode and write to disk.

        Returns:
            Path to the episode directory
        """
        if self.current_metadata is None:
            raise ValueError("No episode in progress")

        # Mark last step as terminal if not already
        if self.steps and not self.steps[-1].is_terminal:
            self.steps[-1].is_terminal = True

        # Update metadata
        self.current_metadata.num_steps = len(self.steps)
        self.current_metadata.total_reward = sum(s.reward for s in self.steps)
        self.current_metadata.success = success
        self.current_metadata.end_time = time.time()
        self.current_metadata.duration_s = (
            self.current_metadata.end_time - self.current_metadata.start_time
        )

        # Create episode directory
        episode_dir = self.output_dir / self.current_metadata.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata.json
        metadata_dict = asdict(self.current_metadata)
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2)

        # Write steps/000000.jsonl (standard format)
        steps_dir = episode_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        with open(steps_dir / "000000.jsonl", "w") as f:
            for step in self.steps:
                step_dict = self._step_to_dict(step)
                f.write(json.dumps(step_dict) + "\n")

        # Reset state
        result_path = episode_dir
        self.current_metadata = None
        self.steps = []
        self.step_counter = 0

        return result_path

    def _step_to_dict(self, step: Step) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {
            "step_idx": step.step_idx,
            "timestamp_us": step.timestamp_us,
            "observation": self._obs_to_dict(step.observation),
            "action": asdict(step.action),
            "reward": step.reward,
            "is_terminal": step.is_terminal,
            "is_truncated": step.is_truncated,
            "info": step.info,
        }

    def _obs_to_dict(self, obs: Observation) -> Dict[str, Any]:
        """Convert observation to dictionary, excluding None values."""
        result = {
            "context": asdict(obs.context),
        }

        if obs.robot_state and obs.robot_state.valid:
            result["robot_state"] = asdict(obs.robot_state)

        if obs.domain_obs:
            result["domain_obs"] = obs.domain_obs

        return result


# ============================================================================
# Episode Reader
# ============================================================================


class RLDSEpisodeReader:
    """
    Reads episodes in the standard RLDS format.

    Handles both v1.1 (old) and v2.0 (new) formats.
    """

    def __init__(self, episodes_dir: str = "./continuonbrain/rlds/episodes"):
        self.episodes_dir = Path(episodes_dir)

    def list_episodes(self) -> List[Dict[str, Any]]:
        """List all episodes with their metadata."""
        episodes = []

        if not self.episodes_dir.exists():
            return episodes

        for ep_dir in self.episodes_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            metadata_path = ep_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    metadata["_path"] = str(ep_dir)
                    episodes.append(metadata)
                except Exception:
                    pass

        return sorted(episodes, key=lambda e: e.get("start_time", 0), reverse=True)

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """Load a specific episode."""
        ep_dir = self.episodes_dir / episode_id
        if not ep_dir.exists():
            raise ValueError(f"Episode not found: {episode_id}")

        # Load metadata
        with open(ep_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Load steps (try both formats)
        steps = []
        steps_path = ep_dir / "steps" / "000000.jsonl"
        if not steps_path.exists():
            steps_path = ep_dir / "steps.jsonl"

        if steps_path.exists():
            with open(steps_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))

        return {"metadata": metadata, "steps": steps}

    def iter_steps(self, episode_id: str):
        """Iterate over steps in an episode (memory-efficient)."""
        ep_dir = self.episodes_dir / episode_id

        steps_path = ep_dir / "steps" / "000000.jsonl"
        if not steps_path.exists():
            steps_path = ep_dir / "steps.jsonl"

        if steps_path.exists():
            with open(steps_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


# ============================================================================
# Helper Functions
# ============================================================================


def create_tool_context(
    tool_name: str,
    tool_description: str = "",
    tool_params: Dict[str, Any] = None,
    previous_actions: List[Dict[str, Any]] = None,
    previous_results: List[Dict[str, Any]] = None,
    last_error: str = None,
    domain: str = "claude_code",
    domain_state: Dict[str, Any] = None,
) -> Context:
    """
    Helper to create a Context for tool-based actions (Claude Code, etc).
    """
    tool_schema = {
        "name": tool_name,
        "description": tool_description,
        "parameters": tool_params or {},
    }

    return Context(
        timestamp=datetime.now().isoformat(),
        tool_available=tool_name,
        tool_schema=tool_schema,
        previous_actions=previous_actions or [],
        previous_results=previous_results or [],
        last_error=last_error,
        domain=domain,
        domain_state=domain_state or {},
    )


def create_tool_action(
    tool_name: str,
    parameters: Dict[str, Any],
    success: bool = True,
    error_message: str = None,
    intent: str = "",
    raw_input: str = "",
) -> Action:
    """
    Helper to create an Action for tool calls.
    """
    return Action(
        action_type="tool_call",
        name=tool_name,
        parameters=parameters,
        success=success,
        error_message=error_message,
        intent=intent,
        raw_input=raw_input,
    )


def create_navigation_action(
    command: str,
    success: bool = True,
    params: Dict[str, Any] = None,
    intent: str = "",
    raw_input: str = "",
) -> Action:
    """
    Helper to create an Action for navigation commands.
    """
    return Action(
        action_type="navigation",
        name=command,
        parameters=params or {},
        success=success,
        intent=intent,
        raw_input=raw_input,
    )
