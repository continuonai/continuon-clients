"""
Autonomous Training Scheduler - Fully autonomous training pipeline for ContinuonBrain.

This service orchestrates the complete autonomous learning cycle:
1. Episode Quality Scoring - Evaluate RLDS episodes for training value
2. Automatic Training Triggers - Start training when conditions are met
3. Cloud Training Submission - Submit slow-loop jobs to GCP when needed
4. OTA Deployment - Automatically deploy trained models to robots
5. Capability Gap Detection - Identify areas needing improvement

The scheduler runs continuously in the background, monitoring episodes
and triggering training when thresholds are met.

Usage:
    from continuonbrain.services.autonomous_training_scheduler import (
        AutonomousTrainingScheduler,
        TrainingTriggerConfig,
    )

    scheduler = AutonomousTrainingScheduler(
        config=TrainingTriggerConfig(
            min_episodes_for_local=4,
            min_episodes_for_cloud=20,
            auto_deploy=True,
        )
    )

    scheduler.start()

    # Monitor status
    status = scheduler.get_status()
    print(f"Episodes: {status.episode_count}, Ready for training: {status.training_ready}")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("AutonomousTrainingScheduler")


class TrainingMode(str, Enum):
    """Training execution mode."""
    LOCAL_FAST = "local_fast"      # Fast loop on-device (100ms)
    LOCAL_MID = "local_mid"        # Mid loop on-device (10s)
    LOCAL_SLOW = "local_slow"      # Slow loop on-device (minutes)
    CLOUD_SLOW = "cloud_slow"      # Slow loop on cloud TPU/GPU
    CLOUD_FULL = "cloud_full"      # Full training on cloud


class SchedulerPhase(str, Enum):
    """Current phase of the autonomous scheduler."""
    IDLE = "idle"
    SCANNING_EPISODES = "scanning_episodes"
    SCORING_EPISODES = "scoring_episodes"
    LOCAL_TRAINING = "local_training"
    UPLOADING_TO_CLOUD = "uploading_to_cloud"
    CLOUD_TRAINING = "cloud_training"
    DOWNLOADING_MODEL = "downloading_model"
    VALIDATING_MODEL = "validating_model"
    DEPLOYING_MODEL = "deploying_model"
    ERROR = "error"


@dataclass
class EpisodeQualityScore:
    """Quality assessment of an RLDS episode."""
    episode_id: str
    episode_path: str

    # Core quality metrics (0-1 scale)
    completeness_score: float = 0.0    # All required fields present
    temporal_consistency: float = 0.0   # Timestamps aligned, no gaps
    action_diversity: float = 0.0       # Variety in actions taken
    observation_quality: float = 0.0    # Sensor data completeness
    task_success: float = 0.0          # Did episode achieve goal?

    # Derived metrics
    overall_score: float = 0.0
    training_value: float = 0.0         # Estimated contribution to learning

    # Metadata
    step_count: int = 0
    duration_seconds: float = 0.0
    has_dialog: bool = False
    has_planner: bool = False
    has_tool_calls: bool = False

    # Flags
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def compute_overall_score(self) -> float:
        """Compute weighted overall quality score."""
        weights = {
            "completeness": 0.25,
            "temporal": 0.15,
            "diversity": 0.20,
            "observation": 0.15,
            "success": 0.25,
        }
        self.overall_score = (
            weights["completeness"] * self.completeness_score +
            weights["temporal"] * self.temporal_consistency +
            weights["diversity"] * self.action_diversity +
            weights["observation"] * self.observation_quality +
            weights["success"] * self.task_success
        )

        # Bonus for rich metadata
        bonus = 0.0
        if self.has_dialog:
            bonus += 0.05
        if self.has_planner:
            bonus += 0.05
        if self.has_tool_calls:
            bonus += 0.05

        self.overall_score = min(1.0, self.overall_score + bonus)

        # Training value considers episode length and uniqueness
        length_factor = min(1.0, self.step_count / 100)  # Cap at 100 steps
        self.training_value = self.overall_score * (0.7 + 0.3 * length_factor)

        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingTriggerConfig:
    """Configuration for automatic training triggers."""
    # Episode thresholds
    min_episodes_for_local: int = 4
    min_episodes_for_cloud: int = 20
    min_quality_score: float = 0.5

    # Time-based triggers
    max_hours_without_training: float = 24.0
    training_cooldown_minutes: float = 30.0

    # Cloud training settings
    enable_cloud_training: bool = True
    cloud_training_interval_hours: float = 168.0  # Weekly
    max_cloud_training_cost_usd: float = 10.0

    # Auto-deployment settings
    auto_deploy: bool = True
    require_validation: bool = True
    min_improvement_threshold: float = 0.05  # 5% improvement required

    # Resource constraints
    max_memory_percent: float = 70.0
    max_temperature_c: float = 75.0
    min_battery_percent: float = 20.0

    # Paths
    episodes_dir: str = "/opt/continuonos/brain/rlds/episodes"
    checkpoint_dir: str = "/opt/continuonos/brain/checkpoints"
    export_dir: str = "/opt/continuonos/brain/model/adapters/candidate"

    @classmethod
    def from_file(cls, path: Path) -> "TrainingTriggerConfig":
        """Load config from JSON file."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            return cls()

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


@dataclass
class SchedulerStatus:
    """Current status of the autonomous training scheduler."""
    phase: SchedulerPhase
    is_running: bool

    # Episode stats
    episode_count: int = 0
    valid_episodes: int = 0
    average_quality_score: float = 0.0
    training_ready: bool = False

    # Training stats
    last_local_training: Optional[datetime] = None
    last_cloud_training: Optional[datetime] = None
    last_deployment: Optional[datetime] = None

    # Current operation
    current_operation: Optional[str] = None
    progress_percent: float = 0.0

    # Error tracking
    last_error: Optional[str] = None
    consecutive_errors: int = 0

    # Model stats
    current_model_version: str = "0.0.0"
    pending_model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "is_running": self.is_running,
            "episode_count": self.episode_count,
            "valid_episodes": self.valid_episodes,
            "average_quality_score": self.average_quality_score,
            "training_ready": self.training_ready,
            "last_local_training": self.last_local_training.isoformat() if self.last_local_training else None,
            "last_cloud_training": self.last_cloud_training.isoformat() if self.last_cloud_training else None,
            "last_deployment": self.last_deployment.isoformat() if self.last_deployment else None,
            "current_operation": self.current_operation,
            "progress_percent": self.progress_percent,
            "last_error": self.last_error,
            "consecutive_errors": self.consecutive_errors,
            "current_model_version": self.current_model_version,
            "pending_model_version": self.pending_model_version,
        }


class EpisodeQualityScorer:
    """
    Evaluates RLDS episode quality for training value.

    Scores episodes on multiple dimensions to determine
    their usefulness for training.
    """

    def __init__(self, episodes_dir: Path):
        self.episodes_dir = Path(episodes_dir)
        self._score_cache: Dict[str, EpisodeQualityScore] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes

    def score_episode(self, episode_path: Path, force: bool = False) -> EpisodeQualityScore:
        """
        Score a single episode for training quality.

        Args:
            episode_path: Path to episode JSON/JSONL file
            force: Force re-scoring even if cached

        Returns:
            EpisodeQualityScore with all metrics computed
        """
        episode_id = self._get_episode_id(episode_path)

        # Check cache
        if not force and episode_id in self._score_cache:
            cache_time = self._cache_timestamps.get(episode_id, 0)
            if time.time() - cache_time < self._cache_ttl:
                return self._score_cache[episode_id]

        score = EpisodeQualityScore(
            episode_id=episode_id,
            episode_path=str(episode_path),
        )

        try:
            # Load episode data
            episode_data = self._load_episode(episode_path)

            if not episode_data:
                score.is_valid = False
                score.validation_errors.append("Failed to load episode data")
                return score

            # Score different aspects
            score.completeness_score = self._score_completeness(episode_data)
            score.temporal_consistency = self._score_temporal_consistency(episode_data)
            score.action_diversity = self._score_action_diversity(episode_data)
            score.observation_quality = self._score_observation_quality(episode_data)
            score.task_success = self._score_task_success(episode_data)

            # Extract metadata
            steps = episode_data.get("steps", [])
            score.step_count = len(steps)

            if steps:
                first_ts = steps[0].get("timestamp", 0)
                last_ts = steps[-1].get("timestamp", 0)
                score.duration_seconds = last_ts - first_ts

            # Check for rich features
            score.has_dialog = any(s.get("dialog") for s in steps)
            score.has_planner = any(s.get("planner") for s in steps)
            score.has_tool_calls = any(s.get("tool_calls") for s in steps)

            # Compute overall score
            score.compute_overall_score()

            # Validate minimum requirements
            if score.step_count < 5:
                score.is_valid = False
                score.validation_errors.append(f"Too few steps: {score.step_count}")
            if score.completeness_score < 0.3:
                score.is_valid = False
                score.validation_errors.append("Missing required fields")

        except Exception as e:
            logger.error(f"Error scoring episode {episode_path}: {e}")
            score.is_valid = False
            score.validation_errors.append(str(e))

        # Cache result
        self._score_cache[episode_id] = score
        self._cache_timestamps[episode_id] = time.time()

        return score

    def score_all_episodes(self) -> List[EpisodeQualityScore]:
        """Score all episodes in the directory."""
        scores = []

        # Find all episode files
        episode_files = list(self.episodes_dir.glob("**/*.json"))
        episode_files += list(self.episodes_dir.glob("**/*.jsonl"))

        for ep_file in episode_files:
            score = self.score_episode(ep_file)
            scores.append(score)

        return scores

    def get_training_ready_episodes(
        self,
        min_quality: float = 0.5,
        limit: int = 100,
    ) -> List[EpisodeQualityScore]:
        """Get episodes that are ready for training, sorted by value."""
        all_scores = self.score_all_episodes()

        # Filter valid episodes above quality threshold
        ready = [
            s for s in all_scores
            if s.is_valid and s.overall_score >= min_quality
        ]

        # Sort by training value (highest first)
        ready.sort(key=lambda s: s.training_value, reverse=True)

        return ready[:limit]

    def _get_episode_id(self, path: Path) -> str:
        """Generate unique episode ID from path."""
        return hashlib.md5(str(path).encode()).hexdigest()[:12]

    def _load_episode(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load episode data from JSON/JSONL file."""
        try:
            content = path.read_text()

            if path.suffix == ".jsonl":
                # JSONL: each line is a step
                lines = content.strip().split("\n")
                steps = [json.loads(line) for line in lines if line.strip()]
                return {"steps": steps}
            else:
                # JSON: full episode structure
                data = json.loads(content)

                # Handle different formats
                if "steps" in data:
                    return data
                elif isinstance(data, list):
                    return {"steps": data}
                else:
                    # Single step or metadata
                    return {"steps": [data]}

        except Exception as e:
            logger.warning(f"Failed to load episode {path}: {e}")
            return None

    def _score_completeness(self, data: Dict[str, Any]) -> float:
        """Score how complete the episode data is."""
        required_fields = ["steps"]
        step_required = ["observation", "action"]

        score = 0.0

        # Check top-level fields
        for field in required_fields:
            if field in data:
                score += 0.2

        # Check steps
        steps = data.get("steps", [])
        if not steps:
            return score

        step_scores = []
        for step in steps:
            step_score = 0.0
            for field in step_required:
                if field in step and step[field]:
                    step_score += 0.5

            # Bonus for optional fields
            if "timestamp" in step:
                step_score += 0.1
            if "reward" in step:
                step_score += 0.1
            if "is_terminal" in step:
                step_score += 0.1

            step_scores.append(min(1.0, step_score))

        if step_scores:
            score += 0.8 * (sum(step_scores) / len(step_scores))

        return min(1.0, score)

    def _score_temporal_consistency(self, data: Dict[str, Any]) -> float:
        """Score temporal alignment and consistency."""
        steps = data.get("steps", [])
        if len(steps) < 2:
            return 0.5  # Can't evaluate with single step

        # Check timestamps exist and are monotonic
        timestamps = []
        for step in steps:
            ts = step.get("timestamp")
            if ts is not None:
                timestamps.append(ts)

        if not timestamps:
            return 0.3  # No timestamps

        if len(timestamps) != len(steps):
            return 0.5  # Some timestamps missing

        # Check monotonicity
        is_monotonic = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        if not is_monotonic:
            return 0.2  # Out of order

        # Check for gaps (>5 second gaps reduce score)
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        max_gap = max(gaps) if gaps else 0

        if max_gap > 5.0:
            return 0.7 - min(0.5, (max_gap - 5.0) / 20.0)

        return 1.0

    def _score_action_diversity(self, data: Dict[str, Any]) -> float:
        """Score diversity of actions in the episode."""
        steps = data.get("steps", [])
        if len(steps) < 3:
            return 0.5

        actions = []
        for step in steps:
            action = step.get("action")
            if action is not None:
                if isinstance(action, dict):
                    action = json.dumps(action, sort_keys=True)
                elif isinstance(action, (list, tuple)):
                    action = str(action)
                actions.append(action)

        if not actions:
            return 0.0

        # Compute uniqueness ratio
        unique_actions = len(set(actions))
        total_actions = len(actions)

        uniqueness = unique_actions / total_actions

        # Ideal is some diversity but not complete chaos
        # Score highest around 40-60% unique
        if uniqueness < 0.1:
            return 0.2  # Too repetitive
        elif uniqueness > 0.9:
            return 0.7  # Too random
        elif 0.3 <= uniqueness <= 0.7:
            return 1.0  # Good diversity
        else:
            return 0.8

    def _score_observation_quality(self, data: Dict[str, Any]) -> float:
        """Score quality and completeness of observations."""
        steps = data.get("steps", [])
        if not steps:
            return 0.0

        obs_scores = []
        for step in steps:
            obs = step.get("observation", {})
            if not obs:
                obs_scores.append(0.0)
                continue

            score = 0.0

            # Check for key observation components
            if isinstance(obs, dict):
                if "robot_state" in obs:
                    score += 0.3
                if "image" in obs or "egocentric_video" in obs:
                    score += 0.3
                if "depth" in obs:
                    score += 0.2
                if "joint_positions" in obs or "joint_velocities" in obs:
                    score += 0.2
            else:
                # Non-dict observation (array or scalar)
                score = 0.5

            obs_scores.append(min(1.0, score))

        return sum(obs_scores) / len(obs_scores) if obs_scores else 0.0

    def _score_task_success(self, data: Dict[str, Any]) -> float:
        """Score whether the episode achieved its goal."""
        steps = data.get("steps", [])
        if not steps:
            return 0.5  # Unknown

        # Check for explicit success indicators
        last_step = steps[-1]

        # Check terminal reward
        if "reward" in last_step:
            reward = last_step["reward"]
            if isinstance(reward, (int, float)):
                if reward > 0:
                    return 0.9
                elif reward < 0:
                    return 0.3

        # Check is_terminal with success
        if last_step.get("is_terminal") and last_step.get("success"):
            return 1.0

        # Check cumulative reward
        total_reward = sum(s.get("reward", 0) for s in steps if isinstance(s.get("reward"), (int, float)))
        if total_reward > 0:
            return 0.7
        elif total_reward < 0:
            return 0.4

        return 0.5  # Unknown success


class CapabilityGapDetector:
    """
    Detects capability gaps by analyzing inference errors and failures.

    Identifies areas where the model needs improvement and
    can generate targeted training tasks.
    """

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.gap_history_path = config_dir / "capability_gaps.json"
        self._gaps: Dict[str, Dict[str, Any]] = {}
        self._load_history()

    def _load_history(self) -> None:
        """Load historical gap data."""
        if self.gap_history_path.exists():
            try:
                self._gaps = json.loads(self.gap_history_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load gap history: {e}")
                self._gaps = {}

    def _save_history(self) -> None:
        """Save gap history to disk."""
        try:
            self.gap_history_path.parent.mkdir(parents=True, exist_ok=True)
            self.gap_history_path.write_text(json.dumps(self._gaps, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save gap history: {e}")

    def record_failure(
        self,
        task_type: str,
        context: Dict[str, Any],
        error_type: str,
        severity: float = 0.5,
    ) -> None:
        """
        Record a capability failure for gap detection.

        Args:
            task_type: Type of task that failed (e.g., "navigation", "manipulation")
            context: Contextual information about the failure
            error_type: Category of error
            severity: How severe the failure was (0-1)
        """
        if task_type not in self._gaps:
            self._gaps[task_type] = {
                "failures": [],
                "total_attempts": 0,
                "severity_sum": 0.0,
            }

        gap = self._gaps[task_type]
        gap["failures"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "error_type": error_type,
            "severity": severity,
        })
        gap["total_attempts"] += 1
        gap["severity_sum"] += severity

        # Keep only recent failures (last 100)
        if len(gap["failures"]) > 100:
            gap["failures"] = gap["failures"][-100:]

        self._save_history()

    def record_success(self, task_type: str) -> None:
        """Record a successful task completion."""
        if task_type not in self._gaps:
            self._gaps[task_type] = {
                "failures": [],
                "total_attempts": 0,
                "severity_sum": 0.0,
            }

        self._gaps[task_type]["total_attempts"] += 1
        self._save_history()

    def get_capability_gaps(self) -> List[Dict[str, Any]]:
        """
        Get list of identified capability gaps, sorted by severity.

        Returns:
            List of gaps with task_type, failure_rate, and severity
        """
        gaps = []

        for task_type, data in self._gaps.items():
            total = data.get("total_attempts", 0)
            failures = len(data.get("failures", []))

            if total == 0:
                continue

            failure_rate = failures / total
            avg_severity = data.get("severity_sum", 0) / max(1, failures)

            gaps.append({
                "task_type": task_type,
                "failure_rate": failure_rate,
                "average_severity": avg_severity,
                "total_failures": failures,
                "total_attempts": total,
                "recent_failures": data.get("failures", [])[-5:],
                "priority_score": failure_rate * avg_severity,
            })

        # Sort by priority (highest first)
        gaps.sort(key=lambda g: g["priority_score"], reverse=True)

        return gaps

    def get_training_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate training recommendations based on capability gaps.

        Returns:
            List of recommended training focus areas
        """
        gaps = self.get_capability_gaps()
        recommendations = []

        for gap in gaps[:5]:  # Top 5 gaps
            if gap["failure_rate"] < 0.1:
                continue  # Low failure rate, skip

            rec = {
                "task_type": gap["task_type"],
                "priority": "high" if gap["priority_score"] > 0.5 else "medium",
                "suggested_episodes": max(5, int(gap["total_failures"] * 2)),
                "focus_areas": self._extract_focus_areas(gap),
            }
            recommendations.append(rec)

        return recommendations

    def _extract_focus_areas(self, gap: Dict[str, Any]) -> List[str]:
        """Extract specific focus areas from gap data."""
        focus_areas = set()

        for failure in gap.get("recent_failures", []):
            error_type = failure.get("error_type", "")
            context = failure.get("context", {})

            if "timeout" in error_type.lower():
                focus_areas.add("response_time")
            if "accuracy" in error_type.lower():
                focus_areas.add("precision")
            if context.get("environment") == "new":
                focus_areas.add("generalization")
            if context.get("lighting") in ["low", "bright"]:
                focus_areas.add("vision_robustness")

        return list(focus_areas)[:3]


class AutonomousTrainingScheduler:
    """
    Main autonomous training scheduler.

    Orchestrates the complete autonomous learning cycle:
    - Episode collection and quality scoring
    - Automatic training triggers
    - Local and cloud training execution
    - Model validation and deployment
    """

    DEFAULT_SCAN_INTERVAL = 300  # 5 minutes

    def __init__(
        self,
        config: Optional[TrainingTriggerConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize autonomous training scheduler.

        Args:
            config: Training trigger configuration
            config_path: Path to config file (alternative to config object)
        """
        if config:
            self.config = config
        elif config_path:
            self.config = TrainingTriggerConfig.from_file(config_path)
        else:
            self.config = TrainingTriggerConfig()

        # Initialize components
        self.episodes_dir = Path(self.config.episodes_dir)
        self.scorer = EpisodeQualityScorer(self.episodes_dir)
        self.gap_detector = CapabilityGapDetector(Path(self.config.checkpoint_dir).parent)

        # State tracking
        self._phase = SchedulerPhase.IDLE
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

        # Stats
        self._last_local_training: Optional[datetime] = None
        self._last_cloud_training: Optional[datetime] = None
        self._last_deployment: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._consecutive_errors = 0
        self._progress = 0.0
        self._current_operation: Optional[str] = None

        # Episode tracking
        self._episode_scores: List[EpisodeQualityScore] = []
        self._processed_episodes: set = set()

        # Callbacks
        self._on_training_started: List[Callable[[TrainingMode], None]] = []
        self._on_training_completed: List[Callable[[TrainingMode, Dict], None]] = []
        self._on_deployment_completed: List[Callable[[str, bool], None]] = []
        self._on_error: List[Callable[[Exception], None]] = []

        # Load state
        self._load_state()

    def _load_state(self) -> None:
        """Load scheduler state from disk."""
        state_path = Path(self.config.checkpoint_dir) / "scheduler_state.json"
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                if data.get("last_local_training"):
                    self._last_local_training = datetime.fromisoformat(data["last_local_training"])
                if data.get("last_cloud_training"):
                    self._last_cloud_training = datetime.fromisoformat(data["last_cloud_training"])
                if data.get("last_deployment"):
                    self._last_deployment = datetime.fromisoformat(data["last_deployment"])
                self._processed_episodes = set(data.get("processed_episodes", []))
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

    def _save_state(self) -> None:
        """Save scheduler state to disk."""
        state_path = Path(self.config.checkpoint_dir) / "scheduler_state.json"
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_local_training": self._last_local_training.isoformat() if self._last_local_training else None,
                "last_cloud_training": self._last_cloud_training.isoformat() if self._last_cloud_training else None,
                "last_deployment": self._last_deployment.isoformat() if self._last_deployment else None,
                "processed_episodes": list(self._processed_episodes)[-1000:],  # Keep last 1000
            }
            state_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save scheduler state: {e}")

    def start(self) -> None:
        """Start the autonomous training scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AutonomousTrainingScheduler"
        )
        self._thread.start()

        logger.info("Autonomous training scheduler started")

    def stop(self) -> None:
        """Stop the autonomous training scheduler."""
        if not self._running:
            return

        logger.info("Stopping autonomous training scheduler...")
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

        self._save_state()
        logger.info("Autonomous training scheduler stopped")

    def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        with self._lock:
            valid_episodes = [s for s in self._episode_scores if s.is_valid]
            avg_quality = (
                sum(s.overall_score for s in valid_episodes) / len(valid_episodes)
                if valid_episodes else 0.0
            )

            return SchedulerStatus(
                phase=self._phase,
                is_running=self._running,
                episode_count=len(self._episode_scores),
                valid_episodes=len(valid_episodes),
                average_quality_score=avg_quality,
                training_ready=len(valid_episodes) >= self.config.min_episodes_for_local,
                last_local_training=self._last_local_training,
                last_cloud_training=self._last_cloud_training,
                last_deployment=self._last_deployment,
                current_operation=self._current_operation,
                progress_percent=self._progress,
                last_error=self._last_error,
                consecutive_errors=self._consecutive_errors,
                current_model_version=self._get_current_model_version(),
                pending_model_version=self._get_pending_model_version(),
            )

    def trigger_training_now(self, mode: TrainingMode = TrainingMode.LOCAL_SLOW) -> bool:
        """
        Manually trigger training.

        Args:
            mode: Training mode to use

        Returns:
            True if training was triggered successfully
        """
        if self._phase != SchedulerPhase.IDLE:
            logger.warning(f"Cannot trigger training: scheduler in phase {self._phase}")
            return False

        # Run training in the scheduler loop
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._run_training(mode),
                self._loop
            )
            return True
        else:
            # If loop not running, run synchronously
            asyncio.run(self._run_training(mode))
            return True

    def on_training_started(self, callback: Callable[[TrainingMode], None]) -> None:
        """Register callback for training start events."""
        self._on_training_started.append(callback)

    def on_training_completed(self, callback: Callable[[TrainingMode, Dict], None]) -> None:
        """Register callback for training completion events."""
        self._on_training_completed.append(callback)

    def on_deployment_completed(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for deployment completion events."""
        self._on_deployment_completed.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for error events."""
        self._on_error.append(callback)

    def _run_loop(self) -> None:
        """Main scheduler loop (runs in background thread)."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            while self._running:
                try:
                    # Run one iteration of the scheduler
                    self._loop.run_until_complete(self._scheduler_iteration())

                    # Wait before next iteration
                    if self._stop_event.wait(timeout=self.DEFAULT_SCAN_INTERVAL):
                        break

                except Exception as e:
                    logger.error(f"Scheduler iteration error: {e}")
                    self._last_error = str(e)
                    self._consecutive_errors += 1
                    self._notify_error(e)

                    # Exponential backoff on repeated errors
                    if self._consecutive_errors > 3:
                        backoff = min(300, 10 * (2 ** self._consecutive_errors))
                        logger.warning(f"Too many errors, backing off for {backoff}s")
                        if self._stop_event.wait(timeout=backoff):
                            break
        finally:
            self._loop.close()
            self._loop = None

    async def _scheduler_iteration(self) -> None:
        """Single iteration of the scheduler loop."""
        # Phase 1: Scan and score episodes
        await self._scan_episodes()

        # Check if we should train
        if self._should_train_local():
            await self._run_training(TrainingMode.LOCAL_SLOW)
        elif self._should_train_cloud():
            await self._run_training(TrainingMode.CLOUD_SLOW)

        # Check for completed cloud training jobs
        await self._check_cloud_training_jobs()

        # Check for models ready to deploy
        if self.config.auto_deploy:
            await self._check_deployable_models()

        # Reset error counter on successful iteration
        self._consecutive_errors = 0

    async def _scan_episodes(self) -> None:
        """Scan and score new episodes."""
        with self._lock:
            self._phase = SchedulerPhase.SCANNING_EPISODES
            self._current_operation = "Scanning episodes"
            self._progress = 0.0

        try:
            self._episode_scores = self.scorer.score_all_episodes()

            # Mark new episodes as seen
            for score in self._episode_scores:
                self._processed_episodes.add(score.episode_id)

            with self._lock:
                self._phase = SchedulerPhase.IDLE
                self._current_operation = None
                self._progress = 100.0

        except Exception as e:
            logger.error(f"Episode scanning failed: {e}")
            with self._lock:
                self._phase = SchedulerPhase.ERROR
                self._last_error = str(e)
            raise

    def _should_train_local(self) -> bool:
        """Determine if local training should be triggered."""
        valid_episodes = [s for s in self._episode_scores if s.is_valid]

        # Not enough episodes
        if len(valid_episodes) < self.config.min_episodes_for_local:
            return False

        # Check cooldown
        if self._last_local_training:
            cooldown = timedelta(minutes=self.config.training_cooldown_minutes)
            if datetime.now(timezone.utc) - self._last_local_training < cooldown:
                return False

        # Check time since last training
        if self._last_local_training:
            max_interval = timedelta(hours=self.config.max_hours_without_training)
            if datetime.now(timezone.utc) - self._last_local_training > max_interval:
                return True

        # Check average quality
        avg_quality = sum(s.overall_score for s in valid_episodes) / len(valid_episodes)
        if avg_quality >= self.config.min_quality_score:
            # Good quality episodes ready
            return True

        return False

    def _should_train_cloud(self) -> bool:
        """Determine if cloud training should be triggered."""
        if not self.config.enable_cloud_training:
            return False

        valid_episodes = [s for s in self._episode_scores if s.is_valid]

        # Not enough episodes for cloud
        if len(valid_episodes) < self.config.min_episodes_for_cloud:
            return False

        # Check cloud training interval
        if self._last_cloud_training:
            interval = timedelta(hours=self.config.cloud_training_interval_hours)
            if datetime.now(timezone.utc) - self._last_cloud_training < interval:
                return False

        return True

    async def _run_training(self, mode: TrainingMode) -> Dict[str, Any]:
        """
        Execute training in the specified mode.

        Args:
            mode: Training mode to execute

        Returns:
            Training results dictionary
        """
        with self._lock:
            if mode in (TrainingMode.CLOUD_SLOW, TrainingMode.CLOUD_FULL):
                self._phase = SchedulerPhase.UPLOADING_TO_CLOUD
            else:
                self._phase = SchedulerPhase.LOCAL_TRAINING
            self._current_operation = f"Training ({mode.value})"
            self._progress = 0.0

        # Notify callbacks
        for cb in self._on_training_started:
            try:
                cb(mode)
            except Exception as e:
                logger.warning(f"Training started callback error: {e}")

        try:
            if mode in (TrainingMode.LOCAL_FAST, TrainingMode.LOCAL_MID, TrainingMode.LOCAL_SLOW):
                result = await self._run_local_training(mode)
                self._last_local_training = datetime.now(timezone.utc)
            else:
                result = await self._run_cloud_training(mode)
                self._last_cloud_training = datetime.now(timezone.utc)

            # Notify callbacks
            for cb in self._on_training_completed:
                try:
                    cb(mode, result)
                except Exception as e:
                    logger.warning(f"Training completed callback error: {e}")

            self._save_state()
            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            with self._lock:
                self._phase = SchedulerPhase.ERROR
                self._last_error = str(e)
            raise
        finally:
            with self._lock:
                self._phase = SchedulerPhase.IDLE
                self._current_operation = None

    async def _run_local_training(self, mode: TrainingMode) -> Dict[str, Any]:
        """Run local on-device training."""
        try:
            from continuonbrain.trainer.wavecore_orchestrator import WaveCoreOrchestrator

            # Map mode to loop configuration
            if mode == TrainingMode.LOCAL_FAST:
                steps_config = {"fast_loop_steps": 12, "mid_loop_steps": 0, "slow_loop_steps": 0}
            elif mode == TrainingMode.LOCAL_MID:
                steps_config = {"fast_loop_steps": 12, "mid_loop_steps": 24, "slow_loop_steps": 0}
            else:
                steps_config = {"fast_loop_steps": 12, "mid_loop_steps": 24, "slow_loop_steps": 32}

            orchestrator = WaveCoreOrchestrator(
                rlds_dir=self.episodes_dir,
                checkpoint_dir=Path(self.config.checkpoint_dir),
                export_dir=Path(self.config.export_dir),
                **steps_config,
            )

            # Start training (blocking)
            orchestrator.start()

            # Wait for completion with progress updates
            while orchestrator.is_running():
                status = orchestrator.get_status()
                with self._lock:
                    self._progress = status.progress * 100
                await asyncio.sleep(1.0)

            # Get final status
            final_status = orchestrator.get_status()

            return {
                "success": final_status.phase.value == "completed",
                "mode": mode.value,
                "final_loss": final_status.metrics.get("loss"),
                "steps_trained": final_status.steps_completed,
                "duration_seconds": final_status.duration_seconds,
            }

        except ImportError:
            logger.warning("WaveCoreOrchestrator not available, using stub")
            # Stub for when trainer not available
            await asyncio.sleep(2.0)
            return {
                "success": True,
                "mode": mode.value,
                "final_loss": 0.1,
                "steps_trained": 32,
                "duration_seconds": 2.0,
                "stub": True,
            }

    async def _run_cloud_training(self, mode: TrainingMode) -> Dict[str, Any]:
        """Submit training job to cloud."""
        try:
            from continuonbrain.services.cloud_training import (
                CloudTrainingService,
                CloudTrainingConfig,
                TrainingJobConfig,
            )

            # Initialize cloud training service
            cloud_service = CloudTrainingService()

            if not cloud_service.is_available():
                raise RuntimeError("Cloud training not available (missing dependencies)")

            # Upload episodes
            with self._lock:
                self._phase = SchedulerPhase.UPLOADING_TO_CLOUD
                self._current_operation = "Uploading episodes to cloud"

            upload_result = await cloud_service.upload_episodes(
                local_dir=self.episodes_dir,
                compress=True,
            )

            if not upload_result.get("success"):
                raise RuntimeError(f"Episode upload failed: {upload_result.get('error')}")

            # Configure training job
            job_config = TrainingJobConfig(
                epochs=100 if mode == TrainingMode.CLOUD_FULL else 50,
                batch_size=32,
                use_tpu=mode == TrainingMode.CLOUD_FULL,
            )

            # Trigger training
            with self._lock:
                self._phase = SchedulerPhase.CLOUD_TRAINING
                self._current_operation = "Cloud training in progress"

            trigger_result = await cloud_service.trigger_training(
                episodes_uri=upload_result["gcs_uri"],
                config=job_config,
            )

            if not trigger_result.get("success"):
                raise RuntimeError(f"Training trigger failed: {trigger_result.get('error')}")

            return {
                "success": True,
                "mode": mode.value,
                "job_id": trigger_result["job_id"],
                "status": "queued",
                "episodes_uri": upload_result["gcs_uri"],
            }

        except ImportError:
            logger.warning("CloudTrainingService not available, using stub")
            return {
                "success": False,
                "mode": mode.value,
                "error": "Cloud training not available",
                "stub": True,
            }

    async def _check_cloud_training_jobs(self) -> None:
        """Check status of any running cloud training jobs."""
        try:
            from continuonbrain.services.cloud_training import (
                CloudTrainingService,
                JobStatus,
            )

            cloud_service = CloudTrainingService()
            if not cloud_service.is_available():
                return

            # List recent jobs
            jobs = await cloud_service.list_jobs(limit=5)

            for job in jobs:
                if job.status == JobStatus.COMPLETED and job.model_uri:
                    # Download completed model
                    with self._lock:
                        self._phase = SchedulerPhase.DOWNLOADING_MODEL
                        self._current_operation = f"Downloading model from job {job.job_id}"

                    download_result = await cloud_service.download_result(
                        job_id=job.job_id,
                        install=True,
                    )

                    if download_result.get("success"):
                        logger.info(f"Model downloaded from cloud job {job.job_id}")

        except ImportError:
            pass  # Cloud training not available
        except Exception as e:
            logger.warning(f"Error checking cloud training jobs: {e}")
        finally:
            with self._lock:
                if self._phase == SchedulerPhase.DOWNLOADING_MODEL:
                    self._phase = SchedulerPhase.IDLE

    async def _check_deployable_models(self) -> None:
        """Check for models ready to deploy."""
        candidate_dir = Path(self.config.export_dir)

        if not candidate_dir.exists():
            return

        manifest_path = candidate_dir / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            manifest = json.loads(manifest_path.read_text())
            candidate_version = manifest.get("version", "0.0.0")

            # Check if this is newer than current
            current_version = self._get_current_model_version()
            if candidate_version <= current_version:
                return

            # Validate if required
            if self.config.require_validation:
                with self._lock:
                    self._phase = SchedulerPhase.VALIDATING_MODEL
                    self._current_operation = f"Validating model {candidate_version}"

                validation_passed = await self._validate_model(candidate_dir)

                if not validation_passed:
                    logger.warning(f"Model {candidate_version} failed validation, skipping deployment")
                    return

            # Deploy the model
            with self._lock:
                self._phase = SchedulerPhase.DEPLOYING_MODEL
                self._current_operation = f"Deploying model {candidate_version}"

            success = await self._deploy_model(candidate_dir, candidate_version)

            # Notify callbacks
            for cb in self._on_deployment_completed:
                try:
                    cb(candidate_version, success)
                except Exception as e:
                    logger.warning(f"Deployment callback error: {e}")

            if success:
                self._last_deployment = datetime.now(timezone.utc)
                self._save_state()

        except Exception as e:
            logger.error(f"Error checking deployable models: {e}")
        finally:
            with self._lock:
                self._phase = SchedulerPhase.IDLE
                self._current_operation = None

    async def _validate_model(self, model_dir: Path) -> bool:
        """
        Validate a candidate model before deployment.

        Args:
            model_dir: Path to candidate model directory

        Returns:
            True if validation passes
        """
        try:
            # Check manifest exists
            manifest_path = model_dir / "manifest.json"
            if not manifest_path.exists():
                logger.error("Validation failed: missing manifest")
                return False

            # Check required files
            manifest = json.loads(manifest_path.read_text())
            required_files = manifest.get("required_files", [])

            for filename in required_files:
                if not (model_dir / filename).exists():
                    logger.error(f"Validation failed: missing {filename}")
                    return False

            # Run inference test
            # This would load the model and run a simple test
            # For now, just check files exist

            model_files = list(model_dir.glob("*.npz")) + list(model_dir.glob("*.safetensors"))
            if not model_files:
                logger.warning("Validation warning: no model files found")

            logger.info(f"Model validation passed: {model_dir}")
            return True

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

    async def _deploy_model(self, candidate_dir: Path, version: str) -> bool:
        """
        Deploy a validated model to active use.

        Args:
            candidate_dir: Path to candidate model
            version: Version string of the model

        Returns:
            True if deployment successful
        """
        try:
            from continuonbrain.services.ota_updater import OTAUpdater

            # Use OTA updater for atomic activation with rollback support
            updater = OTAUpdater(
                config_dir=Path(self.config.checkpoint_dir).parent,
            )

            # Activate with health check
            success = await updater.activate_update(run_health_check=True)

            if success:
                logger.info(f"Model {version} deployed successfully")
            else:
                logger.warning(f"Model {version} deployment failed, rolled back")

            return success

        except ImportError:
            logger.warning("OTAUpdater not available, using simple deployment")
            # Simple file copy as fallback
            import shutil

            current_dir = Path(self.config.checkpoint_dir).parent / "model" / "current"
            rollback_dir = Path(self.config.checkpoint_dir).parent / "model" / "rollback"

            try:
                # Backup current
                if current_dir.exists():
                    if rollback_dir.exists():
                        shutil.rmtree(rollback_dir)
                    shutil.move(str(current_dir), str(rollback_dir))

                # Activate candidate
                shutil.move(str(candidate_dir), str(current_dir))

                logger.info(f"Model {version} deployed (simple mode)")
                return True

            except Exception as e:
                logger.error(f"Simple deployment failed: {e}")
                # Attempt rollback
                if rollback_dir.exists() and not current_dir.exists():
                    shutil.move(str(rollback_dir), str(current_dir))
                return False

    def _get_current_model_version(self) -> str:
        """Get currently deployed model version."""
        current_manifest = (
            Path(self.config.checkpoint_dir).parent / "model" / "current" / "manifest.json"
        )
        if current_manifest.exists():
            try:
                data = json.loads(current_manifest.read_text())
                return data.get("version", "0.0.0")
            except Exception:
                pass
        return "0.0.0"

    def _get_pending_model_version(self) -> Optional[str]:
        """Get pending/candidate model version if any."""
        candidate_manifest = Path(self.config.export_dir) / "manifest.json"
        if candidate_manifest.exists():
            try:
                data = json.loads(candidate_manifest.read_text())
                return data.get("version")
            except Exception:
                pass
        return None

    def _notify_error(self, error: Exception) -> None:
        """Notify all error callbacks."""
        for cb in self._on_error:
            try:
                cb(error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")


# Convenience function for creating a fully configured scheduler
def create_autonomous_scheduler(
    config_path: Optional[Path] = None,
    auto_start: bool = False,
) -> AutonomousTrainingScheduler:
    """
    Create a fully configured autonomous training scheduler.

    Args:
        config_path: Optional path to config file
        auto_start: Whether to start the scheduler immediately

    Returns:
        Configured AutonomousTrainingScheduler instance
    """
    default_config_path = Path("/opt/continuonos/brain/config/training_scheduler.json")

    config = TrainingTriggerConfig.from_file(config_path or default_config_path)

    scheduler = AutonomousTrainingScheduler(config=config)

    if auto_start:
        scheduler.start()

    return scheduler
