#!/usr/bin/env python3
"""
Human Feedback Collector System

Collects and learns from human corrections and demonstrations:
1. Correction feedback - Human corrects wrong robot actions
2. Demonstration feedback - Human shows correct behavior
3. Preference feedback - Human chooses between options
4. Rating feedback - Human rates action quality

This enables continuous improvement from human interaction.

Usage:
    from human_feedback import HumanFeedbackCollector, FeedbackLearner

    collector = HumanFeedbackCollector()
    collector.record_correction(state, wrong_action, correct_action)
    collector.record_demonstration(trajectory)

    learner = FeedbackLearner()
    learner.train_from_feedback(collector.feedback)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class FeedbackType(Enum):
    """Types of human feedback."""
    CORRECTION = "correction"       # Human corrects a wrong action
    DEMONSTRATION = "demonstration" # Human demonstrates correct behavior
    PREFERENCE = "preference"       # Human chooses between options
    RATING = "rating"              # Human rates action quality (1-5)
    APPROVAL = "approval"          # Human approves/disapproves action
    ANNOTATION = "annotation"      # Human provides context/explanation


@dataclass
class StateSnapshot:
    """Snapshot of robot state at feedback time."""
    timestamp: str
    robot_position: Tuple[float, float, float]
    robot_orientation: float
    perception_summary: Dict[str, float]  # Key perception features
    task_context: str  # What task was being performed
    environment: str   # Room/location name

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeedbackRecord:
    """A single feedback record from a human."""
    id: str
    feedback_type: FeedbackType
    timestamp: str
    state: StateSnapshot

    # Correction feedback
    original_action: Optional[str] = None
    corrected_action: Optional[str] = None

    # Demonstration feedback
    demonstration_actions: List[str] = field(default_factory=list)
    demonstration_states: List[Dict] = field(default_factory=list)

    # Preference feedback
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    preferred: Optional[str] = None  # "a", "b", or "neither"

    # Rating feedback
    action_rated: Optional[str] = None
    rating: Optional[int] = None  # 1-5
    rating_reason: Optional[str] = None

    # Approval feedback
    approved: Optional[bool] = None

    # Annotation
    annotation_text: Optional[str] = None
    annotation_tags: List[str] = field(default_factory=list)

    # Metadata
    human_id: str = "default"
    session_id: str = ""
    confidence: float = 1.0  # How confident the human was

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp,
            "state": self.state.to_dict(),
            "human_id": self.human_id,
            "session_id": self.session_id,
            "confidence": self.confidence,
        }

        if self.feedback_type == FeedbackType.CORRECTION:
            d["original_action"] = self.original_action
            d["corrected_action"] = self.corrected_action
        elif self.feedback_type == FeedbackType.DEMONSTRATION:
            d["demonstration_actions"] = self.demonstration_actions
            d["demonstration_states"] = self.demonstration_states
        elif self.feedback_type == FeedbackType.PREFERENCE:
            d["option_a"] = self.option_a
            d["option_b"] = self.option_b
            d["preferred"] = self.preferred
        elif self.feedback_type == FeedbackType.RATING:
            d["action_rated"] = self.action_rated
            d["rating"] = self.rating
            d["rating_reason"] = self.rating_reason
        elif self.feedback_type == FeedbackType.APPROVAL:
            d["approved"] = self.approved
        elif self.feedback_type == FeedbackType.ANNOTATION:
            d["annotation_text"] = self.annotation_text
            d["annotation_tags"] = self.annotation_tags

        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeedbackRecord':
        state = StateSnapshot(**d["state"])
        feedback_type = FeedbackType(d["feedback_type"])

        return cls(
            id=d["id"],
            feedback_type=feedback_type,
            timestamp=d["timestamp"],
            state=state,
            original_action=d.get("original_action"),
            corrected_action=d.get("corrected_action"),
            demonstration_actions=d.get("demonstration_actions", []),
            demonstration_states=d.get("demonstration_states", []),
            option_a=d.get("option_a"),
            option_b=d.get("option_b"),
            preferred=d.get("preferred"),
            action_rated=d.get("action_rated"),
            rating=d.get("rating"),
            rating_reason=d.get("rating_reason"),
            approved=d.get("approved"),
            annotation_text=d.get("annotation_text"),
            annotation_tags=d.get("annotation_tags", []),
            human_id=d.get("human_id", "default"),
            session_id=d.get("session_id", ""),
            confidence=d.get("confidence", 1.0),
        )


class HumanFeedbackCollector:
    """Collects and stores human feedback."""

    def __init__(self, data_dir: str = "brain_b_data/human_feedback"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.feedback: List[FeedbackRecord] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.feedback_count = 0

        # Load existing feedback
        self._load_existing()

    def _load_existing(self):
        """Load existing feedback from files."""
        for f in self.data_dir.glob("feedback_*.json"):
            try:
                with open(f) as file:
                    data = json.load(file)
                    for item in data:
                        self.feedback.append(FeedbackRecord.from_dict(item))
            except Exception as e:
                print(f"Error loading {f}: {e}")

        print(f"Loaded {len(self.feedback)} existing feedback records")

    def _generate_id(self) -> str:
        """Generate unique feedback ID."""
        self.feedback_count += 1
        return f"fb_{self.session_id}_{self.feedback_count:04d}"

    def _create_state_snapshot(
        self,
        robot_pos: Tuple[float, float, float] = (0, 0, 0),
        robot_orient: float = 0.0,
        perception: Dict = None,
        task: str = "",
        environment: str = "",
    ) -> StateSnapshot:
        """Create state snapshot from current context."""
        return StateSnapshot(
            timestamp=datetime.now().isoformat(),
            robot_position=robot_pos,
            robot_orientation=robot_orient,
            perception_summary=perception or {},
            task_context=task,
            environment=environment,
        )

    def record_correction(
        self,
        original_action: str,
        correct_action: str,
        robot_pos: Tuple[float, float, float] = (0, 0, 0),
        perception: Dict = None,
        task: str = "",
        environment: str = "",
        confidence: float = 1.0,
    ) -> FeedbackRecord:
        """
        Record a correction where human corrects a wrong action.

        Args:
            original_action: The action the robot took/planned
            correct_action: The action the human says is correct
            robot_pos: Robot's position
            perception: Current perception features
            task: Current task description
            environment: Current environment/room
            confidence: How confident the human is (0-1)
        """
        state = self._create_state_snapshot(robot_pos, 0.0, perception, task, environment)

        record = FeedbackRecord(
            id=self._generate_id(),
            feedback_type=FeedbackType.CORRECTION,
            timestamp=datetime.now().isoformat(),
            state=state,
            original_action=original_action,
            corrected_action=correct_action,
            session_id=self.session_id,
            confidence=confidence,
        )

        self.feedback.append(record)
        return record

    def record_demonstration(
        self,
        actions: List[str],
        states: List[Dict] = None,
        task: str = "",
        environment: str = "",
    ) -> FeedbackRecord:
        """
        Record a demonstration where human shows correct behavior.

        Args:
            actions: Sequence of actions demonstrated
            states: Corresponding states (optional)
            task: Task being demonstrated
            environment: Environment name
        """
        state = self._create_state_snapshot(task=task, environment=environment)

        record = FeedbackRecord(
            id=self._generate_id(),
            feedback_type=FeedbackType.DEMONSTRATION,
            timestamp=datetime.now().isoformat(),
            state=state,
            demonstration_actions=actions,
            demonstration_states=states or [],
            session_id=self.session_id,
        )

        self.feedback.append(record)
        return record

    def record_preference(
        self,
        option_a: str,
        option_b: str,
        preferred: str,  # "a", "b", or "neither"
        robot_pos: Tuple[float, float, float] = (0, 0, 0),
        perception: Dict = None,
        task: str = "",
    ) -> FeedbackRecord:
        """
        Record preference where human chooses between two options.

        Args:
            option_a: First option
            option_b: Second option
            preferred: Which was preferred ("a", "b", or "neither")
        """
        state = self._create_state_snapshot(robot_pos, perception=perception, task=task)

        record = FeedbackRecord(
            id=self._generate_id(),
            feedback_type=FeedbackType.PREFERENCE,
            timestamp=datetime.now().isoformat(),
            state=state,
            option_a=option_a,
            option_b=option_b,
            preferred=preferred,
            session_id=self.session_id,
        )

        self.feedback.append(record)
        return record

    def record_rating(
        self,
        action: str,
        rating: int,  # 1-5
        reason: str = "",
        robot_pos: Tuple[float, float, float] = (0, 0, 0),
        perception: Dict = None,
        task: str = "",
    ) -> FeedbackRecord:
        """
        Record rating where human rates action quality.

        Args:
            action: Action being rated
            rating: Rating 1-5 (1=very bad, 5=excellent)
            reason: Optional reason for rating
        """
        state = self._create_state_snapshot(robot_pos, perception=perception, task=task)

        record = FeedbackRecord(
            id=self._generate_id(),
            feedback_type=FeedbackType.RATING,
            timestamp=datetime.now().isoformat(),
            state=state,
            action_rated=action,
            rating=rating,
            rating_reason=reason,
            session_id=self.session_id,
        )

        self.feedback.append(record)
        return record

    def record_approval(
        self,
        action: str,
        approved: bool,
        robot_pos: Tuple[float, float, float] = (0, 0, 0),
        perception: Dict = None,
        task: str = "",
    ) -> FeedbackRecord:
        """
        Record approval/disapproval of an action.
        """
        state = self._create_state_snapshot(robot_pos, perception=perception, task=task)

        record = FeedbackRecord(
            id=self._generate_id(),
            feedback_type=FeedbackType.APPROVAL,
            timestamp=datetime.now().isoformat(),
            state=state,
            original_action=action,
            approved=approved,
            session_id=self.session_id,
        )

        self.feedback.append(record)
        return record

    def record_annotation(
        self,
        text: str,
        tags: List[str] = None,
        robot_pos: Tuple[float, float, float] = (0, 0, 0),
        task: str = "",
        environment: str = "",
    ) -> FeedbackRecord:
        """
        Record text annotation/explanation from human.
        """
        state = self._create_state_snapshot(robot_pos, task=task, environment=environment)

        record = FeedbackRecord(
            id=self._generate_id(),
            feedback_type=FeedbackType.ANNOTATION,
            timestamp=datetime.now().isoformat(),
            state=state,
            annotation_text=text,
            annotation_tags=tags or [],
            session_id=self.session_id,
        )

        self.feedback.append(record)
        return record

    def save(self):
        """Save all feedback to file."""
        output_file = self.data_dir / f"feedback_{self.session_id}.json"
        data = [f.to_dict() for f in self.feedback]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.feedback)} feedback records to {output_file}")

    def get_corrections(self) -> List[FeedbackRecord]:
        """Get all correction feedback."""
        return [f for f in self.feedback if f.feedback_type == FeedbackType.CORRECTION]

    def get_demonstrations(self) -> List[FeedbackRecord]:
        """Get all demonstration feedback."""
        return [f for f in self.feedback if f.feedback_type == FeedbackType.DEMONSTRATION]

    def get_preferences(self) -> List[FeedbackRecord]:
        """Get all preference feedback."""
        return [f for f in self.feedback if f.feedback_type == FeedbackType.PREFERENCE]

    def get_summary(self) -> Dict:
        """Get summary of collected feedback."""
        type_counts = {}
        for f in self.feedback:
            t = f.feedback_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_records": len(self.feedback),
            "by_type": type_counts,
            "sessions": len(set(f.session_id for f in self.feedback)),
        }


class FeedbackLearner:
    """
    Learns from human feedback to improve robot behavior.

    Learning methods:
    1. Direct learning from corrections (supervised)
    2. Imitation learning from demonstrations
    3. Reward modeling from preferences
    4. Confidence weighting from ratings
    """

    def __init__(self, data_dir: str = "brain_b_data"):
        self.data_dir = Path(data_dir)

        # Action vocabulary
        self.actions = ["move_forward", "move_backward", "rotate_left", "rotate_right", "pick_up", "noop"]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

        # Simple model: correction counts per state-action
        self.correction_counts: Dict[str, Dict[str, int]] = {}

        # Preference model: pairwise rankings
        self.preference_wins: Dict[str, int] = {a: 0 for a in self.actions}
        self.preference_losses: Dict[str, int] = {a: 0 for a in self.actions}

        # Rating model: average rating per action type
        self.action_ratings: Dict[str, List[int]] = {a: [] for a in self.actions}

        # Learned reward adjustments
        self.reward_adjustments: Dict[str, float] = {a: 0.0 for a in self.actions}

    def learn_from_feedback(self, collector: HumanFeedbackCollector) -> Dict:
        """
        Learn from all feedback in collector.

        Returns metrics about what was learned.
        """
        metrics = {
            "corrections_processed": 0,
            "demonstrations_processed": 0,
            "preferences_processed": 0,
            "ratings_processed": 0,
        }

        # Process corrections
        for fb in collector.get_corrections():
            self._process_correction(fb)
            metrics["corrections_processed"] += 1

        # Process demonstrations
        for fb in collector.get_demonstrations():
            self._process_demonstration(fb)
            metrics["demonstrations_processed"] += 1

        # Process preferences
        for fb in collector.get_preferences():
            self._process_preference(fb)
            metrics["preferences_processed"] += 1

        # Process ratings
        for fb in collector.feedback:
            if fb.feedback_type == FeedbackType.RATING:
                self._process_rating(fb)
                metrics["ratings_processed"] += 1

        # Update reward adjustments
        self._update_reward_model()

        return metrics

    def _process_correction(self, fb: FeedbackRecord):
        """Learn from a correction."""
        # State key from perception summary
        state_key = self._state_to_key(fb.state)

        if state_key not in self.correction_counts:
            self.correction_counts[state_key] = {}

        # Record that original was wrong, corrected was right
        if fb.original_action:
            key = f"wrong:{fb.original_action}"
            self.correction_counts[state_key][key] = \
                self.correction_counts[state_key].get(key, 0) + 1

        if fb.corrected_action:
            key = f"correct:{fb.corrected_action}"
            self.correction_counts[state_key][key] = \
                self.correction_counts[state_key].get(key, 0) + 1

    def _process_demonstration(self, fb: FeedbackRecord):
        """Learn from a demonstration."""
        # Each action in demonstration is implicitly correct
        for i, action in enumerate(fb.demonstration_actions):
            if action in self.actions:
                # Boost this action's rating
                self.action_ratings[action].append(5)  # Demonstrated = excellent

    def _process_preference(self, fb: FeedbackRecord):
        """Learn from a preference comparison."""
        if fb.preferred == "a" and fb.option_a in self.actions:
            self.preference_wins[fb.option_a] += 1
            if fb.option_b in self.actions:
                self.preference_losses[fb.option_b] += 1

        elif fb.preferred == "b" and fb.option_b in self.actions:
            self.preference_wins[fb.option_b] += 1
            if fb.option_a in self.actions:
                self.preference_losses[fb.option_a] += 1

    def _process_rating(self, fb: FeedbackRecord):
        """Learn from a rating."""
        if fb.action_rated in self.actions and fb.rating:
            self.action_ratings[fb.action_rated].append(fb.rating)

    def _state_to_key(self, state: StateSnapshot) -> str:
        """Convert state to hashable key."""
        # Simplify perception to bins
        perc = state.perception_summary
        bins = []

        for key in sorted(perc.keys()):
            val = perc[key]
            if isinstance(val, (int, float)):
                # Bin to 0, 1, 2
                if val < 0.33:
                    bins.append(f"{key}:low")
                elif val < 0.66:
                    bins.append(f"{key}:med")
                else:
                    bins.append(f"{key}:high")

        return "|".join(bins) if bins else "default"

    def _update_reward_model(self):
        """Update reward adjustments based on feedback."""
        for action in self.actions:
            adjustment = 0.0

            # From preferences (Bradley-Terry style)
            wins = self.preference_wins[action]
            losses = self.preference_losses[action]
            if wins + losses > 0:
                win_rate = wins / (wins + losses)
                adjustment += (win_rate - 0.5) * 0.5  # [-0.25, 0.25]

            # From ratings
            ratings = self.action_ratings[action]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                adjustment += (avg_rating - 3) * 0.1  # Centered at 3

            self.reward_adjustments[action] = adjustment

    def get_action_bonus(self, action: str) -> float:
        """Get reward bonus/penalty for action based on feedback."""
        return self.reward_adjustments.get(action, 0.0)

    def get_preferred_action(self, state: Dict, candidates: List[str]) -> Tuple[str, float]:
        """
        Get preferred action for state based on learned preferences.

        Args:
            state: State features
            candidates: List of candidate actions

        Returns:
            (best_action, confidence)
        """
        scores = {}

        for action in candidates:
            score = 0.0

            # Base score from reward adjustments
            score += self.reward_adjustments.get(action, 0.0)

            # Bonus from ratings
            ratings = self.action_ratings.get(action, [])
            if ratings:
                score += (sum(ratings) / len(ratings) - 3) * 0.2

            scores[action] = score

        if not scores:
            return candidates[0] if candidates else "noop", 0.5

        best_action = max(scores, key=scores.get)
        confidence = 0.5 + scores[best_action]  # Normalize to 0-1 range
        confidence = max(0.1, min(1.0, confidence))

        return best_action, confidence

    def save(self, filepath: str = None):
        """Save learned models."""
        if filepath is None:
            filepath = self.data_dir / "feedback_learner_state.json"

        data = {
            "correction_counts": self.correction_counts,
            "preference_wins": self.preference_wins,
            "preference_losses": self.preference_losses,
            "action_ratings": self.action_ratings,
            "reward_adjustments": self.reward_adjustments,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str = None):
        """Load learned models."""
        if filepath is None:
            filepath = self.data_dir / "feedback_learner_state.json"

        if Path(filepath).exists():
            with open(filepath) as f:
                data = json.load(f)

            self.correction_counts = data.get("correction_counts", {})
            self.preference_wins = data.get("preference_wins", {a: 0 for a in self.actions})
            self.preference_losses = data.get("preference_losses", {a: 0 for a in self.actions})
            self.action_ratings = data.get("action_ratings", {a: [] for a in self.actions})
            self.reward_adjustments = data.get("reward_adjustments", {a: 0.0 for a in self.actions})


def demo():
    """Demo the human feedback system."""
    print("=" * 60)
    print("Human Feedback Collector Demo")
    print("=" * 60)

    # Create collector
    collector = HumanFeedbackCollector()

    # Simulate some feedback
    print("\nSimulating human feedback...")

    # Corrections
    collector.record_correction(
        original_action="move_forward",
        correct_action="rotate_left",
        perception={"obstacle_ahead": 0.9, "lidar_front": 0.2},
        task="navigate_to_kitchen",
    )

    collector.record_correction(
        original_action="move_forward",
        correct_action="move_backward",
        perception={"obstacle_ahead": 1.0, "lidar_front": 0.1, "lidar_left": 0.2, "lidar_right": 0.2},
        task="explore_room",
    )

    # Demonstration
    collector.record_demonstration(
        actions=["rotate_left", "move_forward", "move_forward", "rotate_right", "move_forward"],
        task="navigate_around_obstacle",
    )

    # Preferences
    collector.record_preference(
        option_a="rotate_left",
        option_b="move_backward",
        preferred="a",
        perception={"obstacle_ahead": 0.8},
    )

    # Ratings
    collector.record_rating(
        action="move_forward",
        rating=4,
        reason="Good progress towards goal",
    )

    collector.record_rating(
        action="move_backward",
        rating=2,
        reason="Unnecessary retreat",
    )

    # Summary
    print(f"\nFeedback Summary: {collector.get_summary()}")

    # Save
    collector.save()

    # Learn from feedback
    print("\nLearning from feedback...")
    learner = FeedbackLearner()
    metrics = learner.learn_from_feedback(collector)
    print(f"Learning metrics: {metrics}")

    print(f"\nReward adjustments:")
    for action, adj in learner.reward_adjustments.items():
        print(f"  {action}: {adj:+.3f}")

    # Test preferred action
    candidates = ["move_forward", "rotate_left", "rotate_right", "move_backward"]
    best, conf = learner.get_preferred_action({}, candidates)
    print(f"\nPreferred action: {best} (confidence: {conf:.2f})")

    learner.save()
    print("\nLearner state saved.")


if __name__ == "__main__":
    demo()
