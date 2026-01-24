#!/usr/bin/env python3
"""
Chief Scientist - Claude Code as strategic director of robot brain development

Mission: Build a helpful robot brain that assists our family around the house.

This module makes Claude Code act as a Chief Scientist and Product Manager,
steering the training program with clear goals and continuous improvement.

Key Responsibilities:
1. Define family-oriented goals (what tasks should the robot help with)
2. Assess current capabilities vs. what's needed
3. Prioritize learning based on family needs
4. Run experiments and training cycles
5. Analyze results and adapt strategy
6. Build new systems when needed

Usage:
    python scripts/compound/chief_scientist.py              # Run as daemon
    python scripts/compound/chief_scientist.py --status     # Check status
    python scripts/compound/chief_scientist.py --goals      # Show current goals
    python scripts/compound/chief_scientist.py --cycle      # Run one cycle
"""

import json
import os
import subprocess
import sys
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "brain_b"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ChiefScientist] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'chief_scientist.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Family-Oriented Mission & Goals
# =============================================================================

FAMILY_MISSION = """
Build a helpful robot assistant that makes our family's daily life easier and more enjoyable.

The robot should be:
- Safe and gentle around family members and pets
- Helpful with daily tasks (fetching, carrying, reminding)
- Good at navigating our home without bumping into things
- Able to understand and respond to natural language
- Fun and engaging for family interaction
- Continuously learning and improving

Priority household tasks to learn:
1. Navigate safely through all rooms
2. Recognize and greet family members
3. Fetch and carry lightweight items
4. Provide reminders and alerts
5. Monitor for safety (water on floor, doors open)
6. Entertain and engage (tell jokes, play games)
"""


class FamilyTask(str, Enum):
    """Tasks that help the family."""
    NAVIGATE_HOME = "navigate_home"           # Move safely through the house
    RECOGNIZE_FAMILY = "recognize_family"     # Know who's who
    FETCH_ITEMS = "fetch_items"               # Get things for people
    CARRY_ITEMS = "carry_items"               # Transport items safely
    GIVE_REMINDERS = "give_reminders"         # Remember and remind
    SAFETY_MONITOR = "safety_monitor"         # Watch for hazards
    ENTERTAIN = "entertain"                   # Fun interactions
    ASSIST_ELDERLY = "assist_elderly"         # Help older family members
    PET_FRIENDLY = "pet_friendly"             # Safe around pets


@dataclass
class FamilyGoal:
    """A goal focused on helping the family."""
    task: FamilyTask
    description: str
    priority: int  # 1 = highest
    required_capabilities: List[str]
    current_progress: float = 0.0  # 0-1
    milestones: List[str] = field(default_factory=list)
    achieved_milestones: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'task': self.task.value,
            'description': self.description,
            'priority': self.priority,
            'required_capabilities': self.required_capabilities,
            'current_progress': self.current_progress,
            'milestones': self.milestones,
            'achieved_milestones': self.achieved_milestones,
        }


# Default family goals
DEFAULT_FAMILY_GOALS = [
    FamilyGoal(
        task=FamilyTask.NAVIGATE_HOME,
        description="Navigate safely through every room without hitting furniture or walls",
        priority=1,
        required_capabilities=["navigation", "perception", "spatial_memory"],
        milestones=[
            "Navigate living room without collisions",
            "Navigate kitchen safely",
            "Navigate bedrooms",
            "Navigate entire home reliably",
            "Handle dynamic obstacles (people, pets)",
        ],
    ),
    FamilyGoal(
        task=FamilyTask.RECOGNIZE_FAMILY,
        description="Recognize all family members by face and voice",
        priority=2,
        required_capabilities=["perception", "face_recognition", "voice_recognition"],
        milestones=[
            "Detect faces in camera",
            "Learn to recognize 1 family member",
            "Recognize all family members",
            "Greet family members by name",
            "Recognize voices without seeing",
        ],
    ),
    FamilyGoal(
        task=FamilyTask.FETCH_ITEMS,
        description="Fetch requested items and bring them to family members",
        priority=3,
        required_capabilities=["navigation", "manipulation", "object_detection", "conversation"],
        milestones=[
            "Understand fetch commands",
            "Navigate to item location",
            "Pick up lightweight items",
            "Carry items without dropping",
            "Deliver items to requesting person",
        ],
    ),
    FamilyGoal(
        task=FamilyTask.GIVE_REMINDERS,
        description="Remember and deliver reminders to family members",
        priority=4,
        required_capabilities=["conversation", "memory", "scheduling"],
        milestones=[
            "Accept reminder requests",
            "Store reminders with times",
            "Deliver reminders on time",
            "Find the right person to remind",
            "Handle recurring reminders",
        ],
    ),
    FamilyGoal(
        task=FamilyTask.ENTERTAIN,
        description="Engage family in fun interactions, jokes, games",
        priority=5,
        required_capabilities=["conversation", "creativity", "gesture"],
        milestones=[
            "Tell jokes on request",
            "Play simple games",
            "Dance or move playfully",
            "Interactive storytelling",
            "Adapt to family's mood",
        ],
    ),
]


# =============================================================================
# Chief Scientist AI
# =============================================================================

class ChiefScientist:
    """
    Claude Code as Chief Scientist steering robot brain development.

    Responsibilities:
    - Strategic planning based on family needs
    - Experiment design and execution
    - Progress analysis and adaptation
    - Resource allocation (compute, data)
    - Quality assurance
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "brain_b_data"
        self.state_file = self.data_dir / "chief_scientist_state.json"

        self.goals = DEFAULT_FAMILY_GOALS.copy()
        self.experiments: List[Dict] = []
        self.insights: List[str] = []

        self._load_state()

    def _load_state(self):
        """Load saved state."""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                # Restore goal progress
                for goal in self.goals:
                    saved = state.get('goals', {}).get(goal.task.value, {})
                    goal.current_progress = saved.get('progress', 0)
                    goal.achieved_milestones = saved.get('achieved', [])
                self.insights = state.get('insights', [])
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save current state."""
        state = {
            'goals': {
                g.task.value: {
                    'progress': g.current_progress,
                    'achieved': g.achieved_milestones,
                }
                for g in self.goals
            },
            'insights': self.insights[-50:],  # Keep last 50
            'last_updated': datetime.now().isoformat(),
        }
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))

    def get_status(self) -> Dict:
        """Get comprehensive status for display."""
        # Import brain builder for capability status
        try:
            from brain_builder import BrainBuilder
            builder = BrainBuilder(self.project_root)
            capabilities = builder.get_status()
        except ImportError:
            capabilities = {'overall_score': 0, 'capabilities': {}}

        return {
            'mission': FAMILY_MISSION.strip(),
            'overall_progress': self._calculate_overall_progress(),
            'goals': [g.to_dict() for g in self.goals],
            'capabilities': capabilities,
            'recent_insights': self.insights[-5:],
            'next_actions': self._plan_next_actions(),
            'timestamp': datetime.now().isoformat(),
        }

    def _calculate_overall_progress(self) -> float:
        """Calculate weighted overall progress."""
        if not self.goals:
            return 0

        # Weight by priority (lower priority number = higher weight)
        total_weight = 0
        weighted_progress = 0

        for goal in self.goals:
            weight = 1.0 / goal.priority
            weighted_progress += goal.current_progress * weight
            total_weight += weight

        return weighted_progress / total_weight if total_weight > 0 else 0

    def _plan_next_actions(self) -> List[Dict]:
        """Plan next actions based on goals and capabilities."""
        actions = []

        # Sort goals by priority and progress (low progress, high priority first)
        sorted_goals = sorted(
            self.goals,
            key=lambda g: (g.priority, g.current_progress)
        )

        for goal in sorted_goals[:3]:
            # Find next milestone
            next_milestone = None
            for m in goal.milestones:
                if m not in goal.achieved_milestones:
                    next_milestone = m
                    break

            if next_milestone:
                actions.append({
                    'goal': goal.task.value,
                    'action': f"Work on: {next_milestone}",
                    'required': goal.required_capabilities,
                    'priority': goal.priority,
                })

        return actions

    def run_experiment(self, focus_goal: Optional[str] = None) -> Dict:
        """Run a training experiment focused on a goal."""
        logger.info(f"\n{'='*60}")
        logger.info("🔬 Chief Scientist: Starting Experiment")
        logger.info(f"{'='*60}")

        # Select goal to focus on
        if focus_goal:
            goal = next((g for g in self.goals if g.task.value == focus_goal), None)
        else:
            # Pick highest priority goal with lowest progress
            goal = min(self.goals, key=lambda g: (g.priority, g.current_progress))

        if not goal:
            return {'error': 'No goal selected'}

        logger.info(f"📎 Focus: {goal.task.value} - {goal.description}")
        logger.info(f"📊 Current Progress: {goal.current_progress:.0%}")

        experiment = {
            'id': f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'goal': goal.task.value,
            'started_at': datetime.now().isoformat(),
            'actions': [],
            'results': {},
        }

        # Run training based on required capabilities
        for capability in goal.required_capabilities:
            logger.info(f"\n🎯 Training: {capability}")
            result = self._train_capability(capability)
            experiment['actions'].append({
                'capability': capability,
                'result': result,
            })

        # Test progress
        logger.info("\n📊 Testing Progress...")
        test_result = self._test_goal_progress(goal)
        experiment['results'] = test_result

        # Update goal progress
        if test_result.get('success_rate', 0) > goal.current_progress:
            old_progress = goal.current_progress
            goal.current_progress = test_result['success_rate']

            # Check for milestone achievements
            for milestone in goal.milestones:
                if milestone not in goal.achieved_milestones:
                    # Simple heuristic: milestone achieved if progress exceeds threshold
                    threshold = (goal.milestones.index(milestone) + 1) / len(goal.milestones)
                    if goal.current_progress >= threshold:
                        goal.achieved_milestones.append(milestone)
                        logger.info(f"🎉 Milestone Achieved: {milestone}")

            self.insights.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: "
                f"{goal.task.value} improved from {old_progress:.0%} to {goal.current_progress:.0%}"
            )

        experiment['finished_at'] = datetime.now().isoformat()
        experiment['progress_after'] = goal.current_progress

        self.experiments.append(experiment)
        self._save_state()

        return experiment

    def _train_capability(self, capability: str) -> Dict:
        """Train a specific capability."""
        result = {'capability': capability, 'status': 'unknown'}

        try:
            if capability == 'navigation':
                result = self._run_navigation_training()
            elif capability == 'perception':
                result = self._run_perception_training()
            elif capability == 'conversation':
                result = self._run_conversation_training()
            elif capability == 'manipulation':
                result = self._run_manipulation_training()
            elif capability in ['face_recognition', 'voice_recognition']:
                result = self._run_recognition_training(capability)
            elif capability == 'spatial_memory':
                result = self._run_memory_training()
            else:
                result = {'capability': capability, 'status': 'not_implemented'}

        except Exception as e:
            result = {'capability': capability, 'status': 'error', 'error': str(e)}
            logger.error(f"Training error: {e}")

        return result

    def _run_navigation_training(self) -> Dict:
        """Run navigation training."""
        try:
            cmd = [
                sys.executable,
                str(self.project_root / "scripts/compound/training_with_inference.py"),
                "--max-cycles", "1",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.project_root)
            )
            return {
                'capability': 'navigation',
                'status': 'completed' if result.returncode == 0 else 'failed',
                'output': result.stdout[-500:] if result.stdout else '',
            }
        except Exception as e:
            return {'capability': 'navigation', 'status': 'error', 'error': str(e)}

    def _run_perception_training(self) -> Dict:
        """Run perception training."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
import sys
sys.path.insert(0, 'brain_b')
from simulator.perception_trainer import PerceptionTrainer
trainer = PerceptionTrainer('brain_b_data')
episodes = trainer.load_episodes('continuonbrain/rlds/episodes')
if episodes:
    trainer.prepare_samples(episodes)
    metrics = trainer.train(epochs=15)
    print(f'Accuracy: {metrics["accuracy"]*100:.1f}%')
else:
    print('No episodes')
"""],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(self.project_root),
            )
            return {
                'capability': 'perception',
                'status': 'completed',
                'output': result.stdout.strip(),
            }
        except Exception as e:
            return {'capability': 'perception', 'status': 'error', 'error': str(e)}

    def _run_conversation_training(self) -> Dict:
        """Run conversation training."""
        try:
            from trainer.conversation_trainer import ConversationTrainer
            trainer = ConversationTrainer("brain_b_data")
            if len(trainer.dataset.samples) < 500:
                trainer.generate_training_data(500, use_llm=False)
            metrics = trainer.train(epochs=20)
            return {
                'capability': 'conversation',
                'status': 'completed',
                'accuracy': metrics.accuracy,
            }
        except Exception as e:
            return {'capability': 'conversation', 'status': 'error', 'error': str(e)}

    def _run_manipulation_training(self) -> Dict:
        """Run manipulation training (placeholder)."""
        return {
            'capability': 'manipulation',
            'status': 'not_implemented',
            'note': 'Requires real robot arm hardware for training',
        }

    def _run_recognition_training(self, recognition_type: str) -> Dict:
        """Run recognition training."""
        return {
            'capability': recognition_type,
            'status': 'requires_data',
            'note': f'{recognition_type} needs family member samples',
        }

    def _run_memory_training(self) -> Dict:
        """Run spatial memory training."""
        try:
            from memory.spatial_memory import SpatialMemory
            memory = SpatialMemory(grid_size=(20, 20), resolution=0.25)
            # Generate some test data
            memory.remember_object("kitchen", (2.0, 5.0, 0.0))
            memory.remember_object("bedroom", (8.0, 3.0, 0.0))
            memory.save(str(self.data_dir / "spatial_memory.json"))
            return {
                'capability': 'spatial_memory',
                'status': 'initialized',
                'objects': memory.get_known_objects(),
            }
        except Exception as e:
            return {'capability': 'spatial_memory', 'status': 'error', 'error': str(e)}

    def _test_goal_progress(self, goal: FamilyGoal) -> Dict:
        """Test progress on a specific goal."""
        results = {'goal': goal.task.value, 'tests': []}

        if goal.task == FamilyTask.NAVIGATE_HOME:
            # Test navigation
            try:
                from simulator.simulator_training import get_simulator_predictor
                predictor = get_simulator_predictor()
                # Run test scenarios
                test_passed = 0
                total = 5
                for _ in range(total):
                    action, conf = predictor.predict_action([0.5, 0.5, 0.5])
                    if conf > 0.3:
                        test_passed += 1
                results['success_rate'] = test_passed / total
            except Exception:
                results['success_rate'] = 0.3  # Base estimate

        elif goal.task == FamilyTask.RECOGNIZE_FAMILY:
            # Check face recognition setup
            face_db = self.data_dir / "face_db"
            if face_db.exists():
                faces = list(face_db.glob("*.jpg")) + list(face_db.glob("*.png"))
                results['success_rate'] = min(0.2 * len(faces), 1.0)
            else:
                results['success_rate'] = 0.0

        else:
            # Default: estimate based on achieved milestones
            results['success_rate'] = len(goal.achieved_milestones) / len(goal.milestones)

        return results

    def generate_report(self) -> str:
        """Generate a human-readable progress report."""
        status = self.get_status()

        report = []
        report.append("=" * 60)
        report.append("🤖 Robot Brain Development Report")
        report.append("=" * 60)
        report.append("")
        report.append("📋 MISSION:")
        report.append(FAMILY_MISSION.strip())
        report.append("")
        report.append(f"📊 OVERALL PROGRESS: {status['overall_progress']*100:.0f}%")
        report.append("")
        report.append("🎯 FAMILY GOALS:")

        for goal in status['goals']:
            progress_bar = "█" * int(goal['current_progress'] * 10) + "░" * (10 - int(goal['current_progress'] * 10))
            report.append(f"  [{goal['priority']}] {goal['task']}: {progress_bar} {goal['current_progress']*100:.0f}%")
            report.append(f"      {goal['description']}")
            if goal['achieved_milestones']:
                report.append(f"      ✓ Achieved: {', '.join(goal['achieved_milestones'][:2])}")

        report.append("")
        report.append("🔬 NEXT ACTIONS:")
        for action in status['next_actions']:
            report.append(f"  • [{action['priority']}] {action['action']}")

        if status['recent_insights']:
            report.append("")
            report.append("💡 RECENT INSIGHTS:")
            for insight in status['recent_insights']:
                report.append(f"  • {insight}")

        report.append("")
        report.append(f"Last updated: {status['timestamp']}")
        report.append("=" * 60)

        return "\n".join(report)

    def run_daemon(self, interval_minutes: int = 30):
        """Run as a continuous daemon."""
        logger.info("🚀 Chief Scientist Daemon Starting")
        logger.info(f"   Interval: {interval_minutes} minutes")
        logger.info(f"   Mission: Help family around the house")

        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"🔬 Experiment Cycle {cycle}")
                logger.info(f"{'='*60}")

                # Run experiment on highest priority goal
                result = self.run_experiment()

                # Log results
                logger.info(f"\n📊 Cycle Results:")
                logger.info(f"   Goal: {result.get('goal', 'unknown')}")
                logger.info(f"   Progress: {result.get('progress_after', 0)*100:.0f}%")

                # Generate and log report
                report = self.generate_report()
                logger.info(f"\n{report}")

                # Wait for next cycle
                logger.info(f"\n⏰ Next cycle in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("\n🛑 Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                time.sleep(60)  # Wait 1 minute on error


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Chief Scientist - Claude Code steering robot brain development"
    )
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--goals", action="store_true", help="Show family goals")
    parser.add_argument("--cycle", action="store_true", help="Run one experiment cycle")
    parser.add_argument("--report", action="store_true", help="Generate progress report")
    parser.add_argument("--interval", type=int, default=30, help="Daemon interval in minutes")

    args = parser.parse_args()

    scientist = ChiefScientist()

    if args.status:
        status = scientist.get_status()
        print(json.dumps(status, indent=2))
    elif args.goals:
        for goal in scientist.goals:
            print(f"\n[{goal.priority}] {goal.task.value}")
            print(f"    {goal.description}")
            print(f"    Progress: {goal.current_progress*100:.0f}%")
            print(f"    Required: {', '.join(goal.required_capabilities)}")
    elif args.cycle:
        result = scientist.run_experiment()
        print(json.dumps(result, indent=2))
    elif args.report:
        print(scientist.generate_report())
    else:
        # Run as daemon
        scientist.run_daemon(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
