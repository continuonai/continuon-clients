"""
RLDS Episode Logger

RLDS-compliant episode capture on Pi for JAX training pipeline.
Writes JSON/JSONL format initially (for development/debugging).
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """Single step in an RLDS episode."""
    obs: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    done: bool
    t: float  # Timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "observation": self.obs,
            "action": self.action,
            "reward": self.reward,
            "is_terminal": self.done,
            "step_metadata": {
                "timestamp": self.t,
            }
        }


@dataclass
class EpisodeMetadata:
    """Metadata for an RLDS episode."""

    @dataclass
    class ContentRating:
        audience: str = "general"
        violence: str = "none"
        language: str = "clean"

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class PiiAttestation:
        pii_present: bool = False
        faces_present: bool = False
        name_present: bool = False
        consent: bool = False

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class SafetyMetadata:
        content_rating: "EpisodeMetadata.ContentRating"
        intended_audience: str = "local"
        pii_attested: bool = False
        pii_cleared: bool = False
        pii_redacted: bool = False
        pending_review: bool = True
        pii_attestation: Optional["EpisodeMetadata.PiiAttestation"] = None

        def to_dict(self) -> Dict[str, Any]:
            data = {
                "content_rating": self.content_rating.to_dict(),
                "intended_audience": self.intended_audience,
                "pii_attested": self.pii_attested,
                "pii_cleared": self.pii_cleared,
                "pii_redacted": self.pii_redacted,
                "pending_review": self.pending_review,
            }
            if self.pii_attestation:
                data["pii_attestation"] = self.pii_attestation.to_dict()
            return data

    @dataclass
    class ShareMetadata:
        public: bool = False
        slug: Optional[str] = None
        title: Optional[str] = None
        license: Optional[str] = None
        tags: List[str] = field(default_factory=list)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "public": self.public,
                "slug": self.slug,
                "title": self.title,
                "license": self.license,
                "tags": self.tags,
            }

    episode_id: str
    environment_id: str = "pi5-dev"
    xr_mode: str = "trainer"
    control_role: str = "human_teleop"
    tags: List[str] = field(default_factory=list)
    robot_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    safety: Optional["EpisodeMetadata.SafetyMetadata"] = None
    share: Optional["EpisodeMetadata.ShareMetadata"] = None
    schema_version: str = "1.1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "episode_id": self.episode_id,
            "environment_id": self.environment_id,
            "xr_mode": self.xr_mode,
            "control_role": self.control_role,
            "tags": self.tags,
            "robot_id": self.robot_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        if self.safety:
            data["safety"] = self.safety.to_dict()
        if self.share:
            data["share"] = self.share.to_dict()
        return data


@dataclass
class Episode:
    """Complete RLDS episode."""
    metadata: EpisodeMetadata
    steps: List[Step]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


class EpisodeLogger:
    """
    RLDS-compliant episode logger for Pi data capture.
    
    Writes episodes in JSON/JSONL format for development/debugging.
    Episodes can later be converted to TFRecord for cloud TPU training.
    """
    
    def __init__(
        self,
        root_dir: str,
        format: str = "jsonl",  # "json" or "jsonl"
        auto_flush: bool = True,
    ):
        """
        Initialize episode logger.
        
        Args:
            root_dir: Root directory for episode storage
            format: Output format ("json" or "jsonl")
            auto_flush: Whether to flush after each step (for real-time logging)
        """
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.auto_flush = auto_flush
        
        # Current episode state
        self.current_episode: Optional[Episode] = None
        self.episode_file: Optional[Path] = None
    
    def start_episode(
        self,
        episode_id: Optional[str] = None,
        environment_id: str = "pi5-dev",
        xr_mode: str = "trainer",
        control_role: str = "human_teleop",
        tags: Optional[List[str]] = None,
        robot_id: Optional[str] = None,
        safety: Optional[EpisodeMetadata.SafetyMetadata] = None,
        share: Optional[EpisodeMetadata.ShareMetadata] = None,
        schema_version: str = "1.1",
    ) -> str:
        """
        Start a new episode.
        
        Args:
            episode_id: Unique episode identifier (generated if None)
            environment_id: Environment identifier
            xr_mode: XR mode ("trainer", "workstation", "observer")
            control_role: Control role ("human_teleop", "human_supervisor", etc.)
            tags: Optional tags for the episode
            robot_id: Optional robot identifier
        
        Returns:
            episode_id: The episode identifier
        """
        if self.current_episode is not None:
            self.end_episode()
        
        if episode_id is None:
            episode_id = str(uuid.uuid4())

        safety = safety or EpisodeMetadata.SafetyMetadata(
            content_rating=EpisodeMetadata.ContentRating(),
            intended_audience="local",
            pii_attested=False,
            pii_cleared=False,
            pii_redacted=False,
            pending_review=True,
            pii_attestation=EpisodeMetadata.PiiAttestation(),
        )

        metadata = EpisodeMetadata(
            episode_id=episode_id,
            environment_id=environment_id,
            xr_mode=xr_mode,
            control_role=control_role,
            tags=tags or [],
            robot_id=robot_id,
            safety=safety,
            share=share,
            schema_version=schema_version,
        )
        
        self.current_episode = Episode(metadata=metadata, steps=[])
        
        # Create episode file
        filename = f"{episode_id}.{self.format}"
        self.episode_file = self.root / filename
        
        return episode_id
    
    def log_step(
        self,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        done: bool = False,
        step_metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Log a single step in the current episode.
        
        Args:
            obs: Observation dictionary
            action: Action dictionary
            reward: Reward value
            done: Whether this is a terminal step
            step_metadata: Optional step-level metadata
        """
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")
        
        t = time.time()
        step = Step(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            t=t,
        )
        
        # Add custom metadata if provided
        if step_metadata:
            step_dict = step.to_dict()
            step_dict["step_metadata"].update(step_metadata)
            # Reconstruct step (simplified - in practice, we'd update the dict directly)
        
        self.current_episode.steps.append(step)
        
        # For JSONL format, write each step immediately
        if self.format == "jsonl" and self.auto_flush:
            self._write_step_jsonl(step)
    
    def _write_step_jsonl(self, step: Step) -> None:
        """Write a single step in JSONL format."""
        if self.episode_file is None:
            return
        
        step_dict = step.to_dict()
        # Add episode_id to step for JSONL format
        step_dict["episode_id"] = self.current_episode.metadata.episode_id
        
        with self.episode_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(step_dict) + "\n")
    
    def end_episode(self) -> Optional[Path]:
        """
        End the current episode and save it.
        
        Returns:
            Path to saved episode file, or None if no episode was active
        """
        if self.current_episode is None:
            return None
        
        if self.format == "json":
            # Write complete episode as JSON
            episode_dict = self.current_episode.to_dict()
            with self.episode_file.open("w", encoding="utf-8") as f:
                json.dump(episode_dict, f, indent=2)
        elif self.format == "jsonl":
            # For JSONL, steps were already written, but we can write metadata
            # In practice, JSONL episodes might have a separate metadata file
            # For now, we'll just ensure the file exists
            if not self.episode_file.exists():
                # Write all steps if auto_flush was False
                with self.episode_file.open("w", encoding="utf-8") as f:
                    for step in self.current_episode.steps:
                        step_dict = step.to_dict()
                        step_dict["episode_id"] = self.current_episode.metadata.episode_id
                        f.write(json.dumps(step_dict) + "\n")
        else:
            raise ValueError(f"Unknown format: {self.format}")
        
        saved_path = self.episode_file
        self.current_episode = None
        self.episode_file = None
        
        return saved_path
    
    def save_episode(self, episode: Episode) -> Path:
        """
        Save a complete episode (alternative API).
        
        Args:
            episode: Episode to save
        
        Returns:
            Path to saved episode file
        """
        filename = f"{episode.metadata.episode_id}.{self.format}"
        episode_file = self.root / filename
        
        if self.format == "json":
            episode_dict = episode.to_dict()
            with episode_file.open("w", encoding="utf-8") as f:
                json.dump(episode_dict, f, indent=2)
        elif self.format == "jsonl":
            with episode_file.open("w", encoding="utf-8") as f:
                for step in episode.steps:
                    step_dict = step.to_dict()
                    step_dict["episode_id"] = episode.metadata.episode_id
                    f.write(json.dumps(step_dict) + "\n")
        else:
            raise ValueError(f"Unknown format: {self.format}")
        
        return episode_file
    
    def list_episodes(self) -> List[Path]:
        """
        List all episode files in the root directory.
        
        Returns:
            List of episode file paths
        """
        if self.format == "json":
            return sorted(self.root.glob("*.json"))
        elif self.format == "jsonl":
            return sorted(self.root.glob("*.jsonl"))
        else:
            return []

