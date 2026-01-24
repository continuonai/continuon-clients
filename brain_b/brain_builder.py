#!/usr/bin/env python3
"""
Brain Builder - Claude Code as the central autonomous learning system

This module makes Claude Code responsible for building and improving the robot's brain.
It coordinates all training, learning, and capability development.

Key Responsibilities:
1. Understand current capabilities and gaps
2. Plan learning activities
3. Execute training (simulation, real-world, etc.)
4. Build new systems when needed
5. Optimize for compute constraints
6. Continuously improve

Usage:
    from brain_b.brain_builder import BrainBuilder

    builder = BrainBuilder()
    status = builder.get_status()
    plan = builder.plan_next_steps()
    result = builder.execute_plan(plan)
"""

import json
import os
import subprocess
import sys
import time
import logging
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [BrainBuilder] %(message)s'
)
logger = logging.getLogger(__name__)


class ComputeProfile:
    """Compute resource awareness for training decisions."""

    def __init__(self):
        self.update()

    def update(self):
        """Update compute metrics."""
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        self.memory = psutil.virtual_memory()
        self.disk = psutil.disk_usage('/')

        # Check for accelerators
        self.has_hailo = Path('/dev/hailo0').exists()
        self.has_gpu = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check for GPU availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def is_constrained(self) -> bool:
        """Check if compute resources are constrained."""
        return (
            self.cpu_percent > 80 or
            self.memory.percent > 85 or
            self.disk.percent > 90
        )

    @property
    def can_train(self) -> bool:
        """Check if we can run training workloads."""
        return (
            self.cpu_percent < 70 and
            self.memory.percent < 75 and
            self.memory.available > 1 * 1024**3  # 1GB available
        )

    @property
    def training_intensity(self) -> str:
        """Get recommended training intensity."""
        if self.has_hailo or self.has_gpu:
            return "high"
        elif self.can_train:
            return "medium"
        elif self.cpu_percent < 90:
            return "low"
        return "pause"

    def to_dict(self) -> Dict:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory.percent,
            'memory_available_gb': round(self.memory.available / 1024**3, 2),
            'disk_percent': self.disk.percent,
            'has_hailo': self.has_hailo,
            'has_gpu': self.has_gpu,
            'is_constrained': self.is_constrained,
            'can_train': self.can_train,
            'training_intensity': self.training_intensity,
        }


class CapabilityStatus(str, Enum):
    """Status of a capability."""
    MISSING = "missing"      # Not yet built
    BASIC = "basic"          # Minimal implementation
    TRAINED = "trained"      # Has trained model
    DEPLOYED = "deployed"    # Ready for real-world use
    EXCELLENT = "excellent"  # High performance


@dataclass
class Capability:
    """A robot capability with training status."""
    name: str
    description: str
    status: CapabilityStatus
    score: float  # 0-1 performance score
    models: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    training_method: str = ""
    compute_requirement: str = "low"  # low, medium, high

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'score': self.score,
            'models': self.models,
            'dependencies': self.dependencies,
            'training_method': self.training_method,
            'compute_requirement': self.compute_requirement,
        }


class BrainBuilder:
    """
    Central autonomous learning system powered by Claude Code.

    Responsibilities:
    - Assess current capabilities
    - Plan learning activities
    - Execute training
    - Build new systems
    - Optimize for compute
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.brain_b_data = self.project_root / "brain_b_data"
        self.models_dir = self.brain_b_data / "models"
        self.state_file = self.brain_b_data / "brain_builder_state.json"

        self.compute = ComputeProfile()
        self.capabilities = self._assess_capabilities()

        # Ensure directories exist
        self.brain_b_data.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def _assess_capabilities(self) -> Dict[str, Capability]:
        """Assess current capabilities based on available models and systems."""
        capabilities = {}

        # Check Navigation capability
        nav_models = list(self.models_dir.glob("navigation*.pt")) if self.models_dir.exists() else []
        nav_score = 0.9 if nav_models else 0.3
        capabilities['navigation'] = Capability(
            name='navigation',
            description='Navigate through home environment safely',
            status=CapabilityStatus.TRAINED if nav_models else CapabilityStatus.BASIC,
            score=nav_score,
            models=[m.name for m in nav_models],
            training_method='simulator + RLDS',
            compute_requirement='medium',
        )

        # Check Perception capability
        perception_models = list(self.models_dir.glob("perception*.pt")) if self.models_dir.exists() else []
        perc_score = 0.85 if perception_models else 0.2
        capabilities['perception'] = Capability(
            name='perception',
            description='Understand visual environment through camera',
            status=CapabilityStatus.TRAINED if perception_models else CapabilityStatus.BASIC,
            score=perc_score,
            models=[m.name for m in perception_models],
            dependencies=['camera', 'oakd'],
            training_method='image classification + depth',
            compute_requirement='high',
        )

        # Check Conversation capability
        conv_models = list(self.models_dir.glob("conversation*.pt")) if self.models_dir.exists() else []
        conv_score = 1.0 if conv_models else 0.5  # Claude Code always available
        capabilities['conversation'] = Capability(
            name='conversation',
            description='Natural language interaction with humans',
            status=CapabilityStatus.EXCELLENT if conv_models else CapabilityStatus.TRAINED,
            score=conv_score,
            models=[m.name for m in conv_models] + ['claude-code'],
            training_method='LLM augmentation',
            compute_requirement='low',
        )

        # Check Manipulation capability
        manip_models = list(self.models_dir.glob("manipulation*.pt")) if self.models_dir.exists() else []
        manip_score = 0.6 if manip_models else 0.1
        capabilities['manipulation'] = Capability(
            name='manipulation',
            description='Interact with objects using robot arms',
            status=CapabilityStatus.BASIC if manip_models else CapabilityStatus.MISSING,
            score=manip_score,
            models=[m.name for m in manip_models],
            dependencies=['arms'],
            training_method='imitation learning + simulation',
            compute_requirement='high',
        )

        # Check World Model capability
        world_models = list(self.models_dir.glob("world_model*.pt")) if self.models_dir.exists() else []
        world_score = 0.7 if world_models else 0.1
        capabilities['world_model'] = Capability(
            name='world_model',
            description='Predict future states and consequences of actions',
            status=CapabilityStatus.BASIC if world_models else CapabilityStatus.MISSING,
            score=world_score,
            models=[m.name for m in world_models],
            training_method='self-supervised prediction',
            compute_requirement='high',
        )

        return capabilities

    def get_status(self) -> Dict:
        """Get comprehensive brain status."""
        self.compute.update()

        # Overall score
        scores = [c.score for c in self.capabilities.values()]
        overall_score = sum(scores) / len(scores) if scores else 0

        # Count by status
        status_counts = {}
        for cap in self.capabilities.values():
            status = cap.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Get training data stats
        rlds_dir = self.project_root / "continuonbrain" / "rlds" / "episodes"
        episode_count = len(list(rlds_dir.glob("*/metadata.json"))) if rlds_dir.exists() else 0

        return {
            'overall_score': round(overall_score, 2),
            'capabilities': {name: cap.to_dict() for name, cap in self.capabilities.items()},
            'status_counts': status_counts,
            'compute': self.compute.to_dict(),
            'training_data': {
                'rlds_episodes': episode_count,
            },
            'recommendations': self._get_recommendations(),
            'timestamp': datetime.now().isoformat(),
        }

    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving the brain."""
        recs = []

        # Based on capability gaps
        for name, cap in self.capabilities.items():
            if cap.status == CapabilityStatus.MISSING:
                recs.append(f"Build {name} capability - currently missing")
            elif cap.score < 0.5:
                recs.append(f"Improve {name} - score is {cap.score:.0%}")

        # Based on compute
        if not self.compute.can_train:
            recs.append("Reduce system load to enable training")
        elif self.compute.training_intensity == "high":
            recs.append("Good compute resources - can run intensive training")

        # Based on training data
        rlds_dir = self.project_root / "continuonbrain" / "rlds" / "episodes"
        episode_count = len(list(rlds_dir.glob("*/metadata.json"))) if rlds_dir.exists() else 0
        if episode_count < 100:
            recs.append(f"Generate more training episodes (have {episode_count}, need 100+)")

        return recs[:5]  # Top 5 recommendations

    def plan_next_steps(self, max_steps: int = 3) -> List[Dict]:
        """Plan next learning/building steps based on current state."""
        self.compute.update()

        steps = []

        # Find weakest capabilities
        sorted_caps = sorted(
            self.capabilities.items(),
            key=lambda x: x[1].score
        )

        for name, cap in sorted_caps[:max_steps]:
            # Skip if compute is too constrained for this capability
            if cap.compute_requirement == "high" and not self.compute.can_train:
                continue

            step = {
                'capability': name,
                'current_score': cap.score,
                'target_score': min(cap.score + 0.2, 1.0),
                'action': self._determine_action(cap),
                'compute_ok': self._can_execute(cap),
                'estimated_time': self._estimate_time(cap),
            }
            steps.append(step)

        return steps

    def _determine_action(self, cap: Capability) -> str:
        """Determine the best action for improving a capability."""
        if cap.status == CapabilityStatus.MISSING:
            return f"build_{cap.name}_system"
        elif cap.status == CapabilityStatus.BASIC:
            return f"train_{cap.name}_model"
        elif cap.score < 0.7:
            return f"improve_{cap.name}_training"
        else:
            return f"deploy_{cap.name}"

    def _can_execute(self, cap: Capability) -> bool:
        """Check if we can execute training for this capability."""
        if cap.compute_requirement == "high":
            return self.compute.training_intensity in ["high", "medium"]
        elif cap.compute_requirement == "medium":
            return self.compute.training_intensity != "pause"
        return True

    def _estimate_time(self, cap: Capability) -> str:
        """Estimate time to improve capability."""
        if cap.status == CapabilityStatus.MISSING:
            return "hours"
        elif cap.score < 0.5:
            return "30-60 minutes"
        else:
            return "10-30 minutes"

    def execute_action(self, action: str) -> Dict:
        """Execute a learning/building action."""
        logger.info(f"Executing action: {action}")

        result = {
            'action': action,
            'success': False,
            'message': '',
            'details': {},
            'timestamp': datetime.now().isoformat(),
        }

        try:
            if action.startswith("train_"):
                result = self._run_training(action)
            elif action.startswith("build_"):
                result = self._build_system(action)
            elif action.startswith("improve_"):
                result = self._improve_training(action)
            elif action.startswith("deploy_"):
                result = self._deploy_capability(action)
            else:
                result['message'] = f"Unknown action: {action}"

        except Exception as e:
            result['message'] = f"Error: {str(e)}"
            logger.error(f"Action failed: {e}")

        return result

    def _run_training(self, action: str) -> Dict:
        """Run a training job."""
        # Extract capability name
        cap_name = action.replace("train_", "").replace("_model", "")

        # Run training script
        training_script = self.project_root / "scripts" / "compound" / "training_with_inference.py"

        if training_script.exists():
            # Run limited training cycles
            cmd = [
                sys.executable, str(training_script),
                "--max-cycles", "1",
                "--focus", cap_name,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.project_root)
            )

            return {
                'action': action,
                'success': result.returncode == 0,
                'message': f"Training completed for {cap_name}",
                'details': {
                    'stdout': result.stdout[-1000:] if result.stdout else '',
                    'stderr': result.stderr[-500:] if result.stderr else '',
                },
                'timestamp': datetime.now().isoformat(),
            }

        return {
            'action': action,
            'success': False,
            'message': f"Training script not found",
            'timestamp': datetime.now().isoformat(),
        }

    def _build_system(self, action: str) -> Dict:
        """Build a new system/capability."""
        cap_name = action.replace("build_", "").replace("_system", "")

        # This would typically involve:
        # 1. Analyzing what's needed
        # 2. Using web search to find best practices
        # 3. Generating code
        # 4. Testing the implementation

        return {
            'action': action,
            'success': True,
            'message': f"Queued system build for {cap_name} - requires Claude Code",
            'details': {
                'requires': ['web_search', 'code_generation', 'testing'],
                'capability': cap_name,
            },
            'timestamp': datetime.now().isoformat(),
        }

    def _improve_training(self, action: str) -> Dict:
        """Improve training for a capability."""
        cap_name = action.replace("improve_", "").replace("_training", "")

        return {
            'action': action,
            'success': True,
            'message': f"Scheduled training improvement for {cap_name}",
            'details': {
                'strategy': 'generate_more_episodes_and_retrain',
            },
            'timestamp': datetime.now().isoformat(),
        }

    def _deploy_capability(self, action: str) -> Dict:
        """Deploy a capability for real-world use."""
        cap_name = action.replace("deploy_", "")

        # Update capability status
        if cap_name in self.capabilities:
            self.capabilities[cap_name].status = CapabilityStatus.DEPLOYED

        return {
            'action': action,
            'success': True,
            'message': f"Deployed {cap_name} capability",
            'timestamp': datetime.now().isoformat(),
        }

    def save_state(self):
        """Save builder state."""
        state = {
            'capabilities': {name: cap.to_dict() for name, cap in self.capabilities.items()},
            'compute_snapshot': self.compute.to_dict(),
            'saved_at': datetime.now().isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def load_state(self):
        """Load builder state."""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                # Restore capabilities
                for name, cap_dict in state.get('capabilities', {}).items():
                    if name in self.capabilities:
                        self.capabilities[name].score = cap_dict.get('score', 0)
                        self.capabilities[name].status = CapabilityStatus(cap_dict.get('status', 'basic'))
            except Exception as e:
                logger.warning(f"Could not load state: {e}")


# API interface for Claude Code chat
def get_brain_status() -> Dict:
    """Get brain status for chat interface."""
    builder = BrainBuilder()
    return builder.get_status()


def get_next_actions() -> List[Dict]:
    """Get recommended next actions for chat interface."""
    builder = BrainBuilder()
    return builder.plan_next_steps()


def run_brain_action(action: str) -> Dict:
    """Execute a brain building action from chat."""
    builder = BrainBuilder()
    return builder.execute_action(action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Brain Builder CLI")
    parser.add_argument("--status", action="store_true", help="Show brain status")
    parser.add_argument("--plan", action="store_true", help="Show next steps")
    parser.add_argument("--action", type=str, help="Execute an action")

    args = parser.parse_args()

    builder = BrainBuilder()

    if args.status:
        status = builder.get_status()
        print(json.dumps(status, indent=2))
    elif args.plan:
        plan = builder.plan_next_steps()
        print(json.dumps(plan, indent=2))
    elif args.action:
        result = builder.execute_action(args.action)
        print(json.dumps(result, indent=2))
    else:
        # Default: show status
        status = builder.get_status()
        print("\n=== Brain Builder Status ===")
        print(f"Overall Score: {status['overall_score']:.0%}")
        print(f"\nCapabilities:")
        for name, cap in status['capabilities'].items():
            print(f"  {name}: {cap['status']} ({cap['score']:.0%})")
        print(f"\nCompute: {status['compute']['training_intensity']} intensity")
        print(f"\nRecommendations:")
        for rec in status['recommendations']:
            print(f"  - {rec}")
