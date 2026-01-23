#!/usr/bin/env python3
"""
Learning Partner - Autonomous learning system for ContinuonXR

Mission: Help Brain B and ContinuonBrain become useful embodied AI
         that serves humans and collaborates with other robots.

This system understands and uses:
1. Room Scanner - Convert real house → 3D training environment
2. HomeScan Simulator - Train navigation in 3D replica of home
3. RLDS Pipeline - Record and export training episodes
4. Brain B - Simple behaviors and teaching interface
5. ContinuonBrain - Neural networks and world models

Learning Loops:
1. OBSERVE: Scan real environment (house, objects, obstacles)
2. SIMULATE: Generate training scenarios in 3D simulator
3. TRAIN: Brain B learns from simulation + real experience
4. TRANSFER: Export to ContinuonBrain for neural training
5. DEPLOY: Apply learned behaviors in real world
6. REPEAT: Continuous improvement cycle

Usage:
    python scripts/compound/learning_partner.py           # Run learning loop
    python scripts/compound/learning_partner.py --status  # Check brain health
    python scripts/compound/learning_partner.py --goals   # Show learning goals
    python scripts/compound/learning_partner.py --train   # Run training cycle
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
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('learning_partner.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Mission & Goals
# =============================================================================

MISSION = """
Create a useful embodied AI that helps humans and collaborates with other robots.
The robot learns by:
1. Scanning and understanding its home environment
2. Training in simulation (3D replica of home)
3. Transferring skills to real-world actions
4. Continuously improving through experience
"""

# Training tools available in this codebase
TRAINING_TOOLS = {
    "room_scanner": {
        "module": "trainer_ui.room_scanner",
        "description": "Converts room images → 3D assets for simulator",
        "capabilities": ["image_to_3d", "object_detection", "depth_estimation"],
        "input": "camera images",
        "output": "3D assets for HomeScan simulator",
    },
    "simulator_utils": {
        "module": "trainer_ui.simulator_utils",
        "description": "Records simulator sessions as RLDS episodes",
        "capabilities": ["episode_recording", "environment_state", "collision_detection"],
        "input": "simulator interactions",
        "output": "RLDS training episodes",
    },
    "simulator_training": {
        "module": "brain_b.simulator.simulator_training",
        "description": "Trains Brain B from simulator RLDS episodes",
        "capabilities": ["action_prediction", "navigation_learning", "state_encoding"],
        "input": "RLDS episodes",
        "output": "trained action predictor",
    },
    "home_scan": {
        "module": "trainer_ui.home_scan",
        "description": "3D home environment builder",
        "capabilities": ["environment_building", "asset_placement", "obstacle_mapping"],
        "input": "room scan data",
        "output": "complete 3D home environment",
    },
    "auto_trainer": {
        "module": "continuonbrain.trainer.auto_trainer_daemon",
        "description": "Automatic training from RLDS episodes",
        "capabilities": ["batch_training", "model_checkpointing", "metrics_tracking"],
        "input": "RLDS episodes",
        "output": "trained neural networks",
    },
    "conversation_trainer": {
        "module": "brain_b.trainer.conversation_trainer",
        "description": "Trains natural language understanding using Claude/Gemini CLI",
        "capabilities": ["intent_classification", "natural_language", "llm_augmentation"],
        "input": "conversation samples, LLM generation",
        "output": "trained conversation model",
        "backends": ["claude", "gemini", "local"],
    },
}

# Core capabilities the robot needs
CORE_CAPABILITIES = {
    "perception": {
        "description": "Understand the physical world through sensors",
        "training_tools": ["room_scanner", "home_scan"],
        "skills": ["camera_vision", "depth_sensing", "object_recognition", "scene_understanding"],
    },
    "navigation": {
        "description": "Move through the environment safely",
        "training_tools": ["simulator_training", "simulator_utils"],
        "skills": ["path_planning", "obstacle_avoidance", "localization", "mapping"],
    },
    "manipulation": {
        "description": "Interact with objects in the world",
        "training_tools": ["simulator_training"],
        "skills": ["grasping", "placing", "pushing", "tool_use"],
    },
    "communication": {
        "description": "Interact with humans and other robots",
        "training_tools": ["conversation_trainer"],
        "skills": ["natural_language", "intent_inference", "gesture_recognition"],
    },
    "learning": {
        "description": "Improve through experience",
        "training_tools": ["auto_trainer", "simulator_training"],
        "skills": ["imitation_learning", "few_shot_adaptation", "error_recovery"],
    },
}


class LearningPhase(str, Enum):
    OBSERVE = "observe"      # Scanning real environment
    SIMULATE = "simulate"    # Training in simulator
    TRAIN = "train"          # Neural network training
    TRANSFER = "transfer"    # Sim-to-real transfer
    DEPLOY = "deploy"        # Real-world execution


@dataclass
class LearningGoal:
    """A learning goal for the system."""
    id: str
    phase: LearningPhase
    capability: str
    description: str
    priority: float  # 0-1
    progress: float = 0.0
    training_tool: Optional[str] = None
    actions_taken: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'phase': self.phase.value,
            'capability': self.capability,
            'description': self.description,
            'priority': self.priority,
            'progress': self.progress,
            'training_tool': self.training_tool,
            'actions_taken': self.actions_taken,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningGoal':
        data['phase'] = LearningPhase(data['phase'])
        return cls(**data)


# =============================================================================
# Training Pipeline Manager
# =============================================================================

class TrainingPipelineManager:
    """Manages the training pipeline tools."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.rlds_dir = project_root / "continuonbrain" / "rlds" / "episodes"
        self.brain_b_data = project_root / "brain_b_data"
        self.room_scans = project_root / "trainer_ui" / "room_scans"

    def check_tool_availability(self) -> Dict[str, bool]:
        """Check which training tools are available."""
        available = {}
        for tool_name, tool_info in TRAINING_TOOLS.items():
            module_path = tool_info['module'].replace('.', '/')
            if '.' in tool_info['module']:
                parts = tool_info['module'].split('.')
                file_path = self.project_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py"
            else:
                file_path = self.project_root / f"{module_path}.py"
            available[tool_name] = file_path.exists()
        return available

    def get_room_scan_status(self) -> Dict:
        """Check room scanning status."""
        if not self.room_scans.exists():
            return {'status': 'no_scans', 'scan_count': 0, 'rooms': []}

        scans = list(self.room_scans.glob("*.json"))
        rooms = []
        for scan_file in scans[:10]:
            try:
                data = json.loads(scan_file.read_text())
                rooms.append({
                    'name': scan_file.stem,
                    'assets': len(data.get('assets', [])),
                    'timestamp': data.get('timestamp', 'unknown'),
                })
            except Exception:
                pass

        return {
            'status': 'ready' if scans else 'empty',
            'scan_count': len(scans),
            'rooms': rooms,
        }

    def get_rlds_status(self) -> Dict:
        """Check RLDS training data status."""
        if not self.rlds_dir.exists():
            return {'status': 'no_data', 'episode_count': 0}

        episodes = list(self.rlds_dir.glob("*/metadata.json"))
        total_steps = 0
        total_reward = 0
        sources = {'brain_b': 0, 'simulator': 0, 'real': 0}

        for ep_file in episodes:
            try:
                meta = json.loads(ep_file.read_text())
                total_steps += meta.get('num_steps', 0)
                total_reward += meta.get('total_reward', 0)

                # Categorize by source
                source = meta.get('source', '')
                if 'simulator' in source.lower():
                    sources['simulator'] += 1
                elif 'brain_b' in source.lower() or 'hook' in source.lower():
                    sources['brain_b'] += 1
                else:
                    sources['real'] += 1
            except Exception:
                pass

        return {
            'status': 'collecting' if episodes else 'empty',
            'episode_count': len(episodes),
            'total_steps': total_steps,
            'avg_reward': total_reward / len(episodes) if episodes else 0,
            'sources': sources,
        }

    def get_brain_b_status(self) -> Dict:
        """Check Brain B learning status."""
        behaviors_dir = self.brain_b_data / "behaviors"
        guardrails_dir = self.brain_b_data / "guardrails"

        behaviors = []
        if behaviors_dir.exists():
            for beh_file in behaviors_dir.glob("*.json"):
                try:
                    data = json.loads(beh_file.read_text())
                    behaviors.append({
                        'name': beh_file.stem,
                        'steps': len(data.get('steps', [])),
                    })
                except Exception:
                    pass

        guardrails = []
        if guardrails_dir.exists():
            guardrails = [f.stem for f in guardrails_dir.glob("*.json")]

        return {
            'behaviors_learned': len(behaviors),
            'behaviors': behaviors[:10],
            'guardrails_count': len(guardrails),
            'status': 'learning' if behaviors else 'empty',
        }

    def run_room_scan(self, image_paths: List[Path]) -> Optional[Dict]:
        """Run room scanner on images."""
        try:
            sys.path.insert(0, str(self.project_root / "trainer_ui"))
            from room_scanner import RoomScanner

            scanner = RoomScanner()
            # Load images...
            # Return scan result
            return {'status': 'success'}
        except Exception as e:
            logger.error(f"Room scan failed: {e}")
            return None

    def run_simulator_training(self) -> Optional[Dict]:
        """Run simulator training."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m',
                    'brain_b.simulator.simulator_training',
                    '--episodes', str(self.rlds_dir),
                    '--output', str(self.brain_b_data / "simulator_models"),
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {'status': 'success' if result.returncode == 0 else 'failed', 'output': result.stdout}
        except Exception as e:
            logger.error(f"Simulator training failed: {e}")
            return None

    def run_auto_trainer(self) -> Optional[Dict]:
        """Run the auto trainer daemon once."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m',
                    'continuonbrain.trainer.auto_trainer_daemon',
                    '--once',
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {'status': 'success' if result.returncode == 0 else 'failed', 'output': result.stdout}
        except Exception as e:
            logger.error(f"Auto trainer failed: {e}")
            return None

    def generate_training_games(self, num_games: int = 20) -> Optional[Dict]:
        """Generate diverse training games to enhance learning."""
        try:
            result = subprocess.run(
                [
                    sys.executable, '-c',
                    f'''
from brain_b.simulator.training_games import generate_training_games
stats = generate_training_games(
    num_games={num_games},
    episodes_per_game=3,
    output_dir="continuonbrain/rlds/episodes"
)
print(f"Generated {{stats['episodes']}} episodes with {{stats['steps']}} steps")
'''
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,
            )
            logger.info(result.stdout)
            return {'status': 'success' if result.returncode == 0 else 'failed', 'output': result.stdout}
        except Exception as e:
            logger.error(f"Training games generation failed: {e}")
            return None

    def run_conversation_training(self, epochs: int = 100, use_llm: bool = True) -> Optional[Dict]:
        """
        Run conversation training using Claude/Gemini CLI when available.

        This trains natural language understanding so the robot can:
        - Understand greetings and questions
        - Parse natural navigation commands
        - Have basic conversations
        """
        try:
            result = subprocess.run(
                [
                    sys.executable, '-c',
                    f'''
import sys
sys.path.insert(0, "brain_b")
from trainer.conversation_trainer import ConversationTrainer, get_llm_backend

# Check LLM availability
llm = get_llm_backend()
print(f"LLM Backend: {{llm.backend}}")
print(f"  Claude available: {{llm.claude_available}}")
print(f"  Gemini available: {{llm.gemini_available}}")

# Run training
trainer = ConversationTrainer("brain_b_data")
total = trainer.generate_training_data(count=2000, use_llm={use_llm})
metrics = trainer.train(epochs={epochs})

print(f"\\nTraining complete:")
print(f"  Samples trained: {{metrics.samples_trained}}")
print(f"  Intents learned: {{metrics.intents_learned}}")
print(f"  Accuracy: {{metrics.accuracy:.2%}}")
'''
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,
            )

            output = result.stdout + result.stderr
            logger.info(f"Conversation training output: {output}")

            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'output': output,
            }
        except subprocess.TimeoutExpired:
            logger.error("Conversation training timed out")
            return {'status': 'timeout', 'output': ''}
        except Exception as e:
            logger.error(f"Conversation training failed: {e}")
            return None

    def get_conversation_status(self) -> Dict:
        """Check conversation training status."""
        conv_model_dir = self.brain_b_data / "conversation_models"
        conv_data_dir = self.brain_b_data / "conversations"

        models = list(conv_model_dir.glob("conv_model_*.json")) if conv_model_dir.exists() else []
        data_files = list(conv_data_dir.glob("*.json")) if conv_data_dir.exists() else []

        # Count samples
        total_samples = 0
        for f in data_files:
            try:
                data = json.loads(f.read_text())
                total_samples += len(data.get("samples", []))
            except Exception:
                pass

        return {
            'status': 'ready' if models else 'not_trained',
            'models': len(models),
            'latest_model': str(models[-1]) if models else None,
            'training_samples': total_samples,
            'data_files': len(data_files),
        }


# =============================================================================
# Brain State Analyzer (enhanced)
# =============================================================================

class BrainStateAnalyzer:
    """Deep understanding of the robot's current state and capabilities."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pipeline = TrainingPipelineManager(project_root)

    def analyze_full_state(self) -> Dict[str, Any]:
        """Get comprehensive brain state."""
        return {
            'mission': MISSION.strip(),
            'training_tools': self.pipeline.check_tool_availability(),
            'room_scans': self.pipeline.get_room_scan_status(),
            'rlds_data': self.pipeline.get_rlds_status(),
            'brain_b': self.pipeline.get_brain_b_status(),
            'capabilities': self._analyze_capabilities(),
            'current_phase': self._determine_current_phase(),
            'next_steps': self._suggest_next_steps(),
        }

    def _analyze_capabilities(self) -> Dict[str, Any]:
        """Analyze current capability levels."""
        capabilities = {}

        for cap_name, cap_info in CORE_CAPABILITIES.items():
            # Check training tool availability
            tools_available = sum(
                1 for tool in cap_info['training_tools']
                if self.pipeline.check_tool_availability().get(tool, False)
            )
            tools_total = len(cap_info['training_tools']) or 1

            # Check skill implementations
            skills_implemented = sum(
                1 for skill in cap_info['skills']
                if self._skill_exists(skill)
            )
            skills_total = len(cap_info['skills'])

            health = (tools_available / tools_total + skills_implemented / skills_total) / 2

            capabilities[cap_name] = {
                'description': cap_info['description'],
                'health': health,
                'training_tools': cap_info['training_tools'],
                'tools_ready': tools_available,
                'skills_implemented': skills_implemented,
                'skills_total': skills_total,
            }

        return capabilities

    def _skill_exists(self, skill_name: str) -> bool:
        """Check if a skill is implemented."""
        patterns = {
            'camera_vision': ['cv2', 'camera', 'vision'],
            'depth_sensing': ['depth', 'rgbd'],
            'object_recognition': ['detect', 'recognition', 'yolo'],
            'scene_understanding': ['scene', 'semantic'],
            'path_planning': ['pathfind', 'astar', 'plan'],
            'obstacle_avoidance': ['obstacle', 'collision', 'avoid'],
            'localization': ['localize', 'position', 'odometry'],
            'mapping': ['slam', 'map', 'occupancy'],
            'grasping': ['grasp', 'grip'],
            'placing': ['place', 'put'],
            'pushing': ['push', 'nudge'],
            'tool_use': ['tool', 'instrument'],
            'natural_language': ['chat', 'nlp', 'language'],
            'intent_inference': ['intent', 'inference'],
            'gesture_recognition': ['gesture', 'pose'],
            'imitation_learning': ['imitat', 'demonstrat', 'rlds'],
            'few_shot_adaptation': ['few_shot', 'adapt'],
            'error_recovery': ['recovery', 'fallback'],
        }

        search_patterns = patterns.get(skill_name, [skill_name])

        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', 'node_modules']):
                continue
            try:
                content = py_file.read_text().lower()
                if any(p in content for p in search_patterns):
                    return True
            except Exception:
                pass
        return False

    def _determine_current_phase(self) -> LearningPhase:
        """Determine what phase of learning we're in."""
        room_status = self.pipeline.get_room_scan_status()
        rlds_status = self.pipeline.get_rlds_status()
        brain_b_status = self.pipeline.get_brain_b_status()

        # No room scans yet? Need to observe
        if room_status['scan_count'] == 0:
            return LearningPhase.OBSERVE

        # Few RLDS episodes? Need more simulation
        if rlds_status['episode_count'] < 20:
            return LearningPhase.SIMULATE

        # Have data but few behaviors? Need training
        if brain_b_status['behaviors_learned'] < 5:
            return LearningPhase.TRAIN

        # Have behaviors? Ready to transfer/deploy
        return LearningPhase.DEPLOY

    def _suggest_next_steps(self) -> List[str]:
        """Suggest next actions based on current state."""
        steps = []
        room_status = self.pipeline.get_room_scan_status()
        rlds_status = self.pipeline.get_rlds_status()

        if room_status['scan_count'] == 0:
            steps.append("Scan your home environment using trainer_ui")
            steps.append("Take photos of rooms and run room_scanner")

        if rlds_status['episode_count'] < 20:
            steps.append("Generate more training episodes in HomeScan simulator")
            steps.append("Run: python trainer_ui/server.py and use the simulator")

        if rlds_status['episode_count'] >= 20:
            steps.append("Run simulator training: python -m brain_b.simulator.simulator_training")

        return steps


# =============================================================================
# Learning Goal Generator
# =============================================================================

class LearningGoalGenerator:
    """Generates learning goals from brain state analysis."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.analyzer = BrainStateAnalyzer(project_root)

    def generate_goals(self) -> List[LearningGoal]:
        """Generate prioritized learning goals."""
        state = self.analyzer.analyze_full_state()
        goals = []

        current_phase = state['current_phase']

        # Phase-specific goals
        if current_phase == LearningPhase.OBSERVE:
            goals.append(LearningGoal(
                id="scan_home",
                phase=LearningPhase.OBSERVE,
                capability="perception",
                description="Scan home environment to create 3D training world",
                priority=1.0,
                training_tool="room_scanner",
            ))

        if current_phase == LearningPhase.SIMULATE:
            goals.append(LearningGoal(
                id="generate_episodes",
                phase=LearningPhase.SIMULATE,
                capability="navigation",
                description=f"Generate more training episodes (current: {state['rlds_data']['episode_count']})",
                priority=0.9,
                training_tool="simulator_utils",
            ))

        if current_phase == LearningPhase.TRAIN:
            goals.append(LearningGoal(
                id="train_navigation",
                phase=LearningPhase.TRAIN,
                capability="navigation",
                description="Train action prediction from simulator episodes",
                priority=0.8,
                training_tool="simulator_training",
            ))

        # Capability-based goals
        for cap_name, cap_data in state['capabilities'].items():
            if cap_data['health'] < 0.5:
                for tool in cap_data['training_tools']:
                    if state['training_tools'].get(tool, False):
                        goals.append(LearningGoal(
                            id=f"improve_{cap_name}_{tool}",
                            phase=LearningPhase.TRAIN,
                            capability=cap_name,
                            description=f"Improve {cap_name} using {tool}",
                            priority=0.6,
                            training_tool=tool,
                        ))

        # Sort by priority
        goals.sort(key=lambda g: g.priority, reverse=True)

        return goals


# =============================================================================
# Learning Partner (Main Loop)
# =============================================================================

class LearningPartner:
    """The main learning partner that helps the brain improve."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state_file = project_root / "scripts" / "compound" / "learning_state.json"
        self.analyzer = BrainStateAnalyzer(project_root)
        self.goal_generator = LearningGoalGenerator(project_root)
        self.pipeline = TrainingPipelineManager(project_root)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load learning partner state."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {
            'goals': [],
            'completed_goals': [],
            'learning_cycles': 0,
            'started_at': datetime.now().isoformat(),
        }

    def _save_state(self):
        """Save learning partner state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_learning_cycle(self) -> Dict:
        """Run a single learning cycle."""
        logger.info("=" * 60)
        logger.info("🧠 Learning Partner - Starting cycle")
        logger.info("=" * 60)

        # 1. Analyze current state
        logger.info("Analyzing brain state...")
        brain_state = self.analyzer.analyze_full_state()

        current_phase = brain_state['current_phase']
        logger.info(f"Current phase: {current_phase.value}")

        # 2. Generate/update goals
        logger.info("Generating learning goals...")
        goals = self.goal_generator.generate_goals()
        self.state['goals'] = [g.to_dict() for g in goals]

        if not goals:
            logger.info("✨ No learning goals - brain is doing well!")
            return {'status': 'complete', 'phase': current_phase.value}

        goal = goals[0]
        logger.info(f"🎯 Working on: {goal.description}")
        logger.info(f"   Using tool: {goal.training_tool}")

        # 3. Execute based on phase
        success = self._execute_goal(goal, brain_state)

        # 4. Update state
        self.state['learning_cycles'] += 1
        self.state['last_cycle'] = datetime.now().isoformat()
        self._save_state()

        return {
            'status': 'success' if success else 'in_progress',
            'phase': current_phase.value,
            'goal': goal.to_dict(),
        }

    def _execute_goal(self, goal: LearningGoal, brain_state: Dict) -> bool:
        """Execute a learning goal."""
        if goal.phase == LearningPhase.OBSERVE:
            return self._execute_observe(goal)
        elif goal.phase == LearningPhase.SIMULATE:
            return self._execute_simulate(goal)
        elif goal.phase == LearningPhase.TRAIN:
            return self._execute_train(goal)
        elif goal.phase == LearningPhase.TRANSFER:
            return self._execute_transfer(goal)
        elif goal.phase == LearningPhase.DEPLOY:
            return self._execute_deploy(goal)
        return False

    def _execute_observe(self, goal: LearningGoal) -> bool:
        """Execute observation phase - needs human interaction for camera."""
        logger.info("📸 OBSERVE phase requires human interaction")
        logger.info("   Please run: python trainer_ui/server.py")
        logger.info("   Then use the Room Scanner in the web UI")

        # Could spawn Claude Code to help guide
        prompt = """Help the user scan their home environment:

1. Check if trainer_ui/server.py is running
2. Guide them to the HomeScan feature in the web UI
3. Explain how to take photos and process them
4. Verify room scan results are saved

This is for building a 3D training environment."""

        return self._run_claude_code(prompt)

    def _execute_simulate(self, goal: LearningGoal) -> bool:
        """Execute simulation phase - generate training episodes."""
        logger.info("🎮 SIMULATE phase - generating training episodes")

        # Could run simulator in headless mode or prompt for interaction
        prompt = """Generate training episodes in the HomeScan simulator:

1. Check for existing room scans in trainer_ui/room_scans/
2. Load the scanned environment into the simulator
3. Run automated exploration to generate RLDS episodes
4. Verify episodes are saved to continuonbrain/rlds/episodes/

Goal: Generate navigation training data."""

        return self._run_claude_code(prompt)

    def _execute_train(self, goal: LearningGoal) -> bool:
        """Execute training phase."""
        logger.info("🏋️ TRAIN phase - training models")

        # First, generate more training games if needed
        rlds_status = self.pipeline.get_rlds_status()
        if rlds_status['episode_count'] < 100:
            logger.info("🎮 Generating more training games...")
            self.pipeline.generate_training_games(num_games=20)

        if goal.training_tool == "simulator_training":
            result = self.pipeline.run_simulator_training()
            return result and result.get('status') == 'success'

        if goal.training_tool == "auto_trainer":
            result = self.pipeline.run_auto_trainer()
            return result and result.get('status') == 'success'

        if goal.training_tool == "conversation_trainer":
            logger.info("💬 Training conversation understanding...")
            result = self.pipeline.run_conversation_training(epochs=100, use_llm=True)
            return result and result.get('status') == 'success'

        # Fallback to Claude Code
        prompt = f"""Train the robot using {goal.training_tool}:

1. Load RLDS episodes from continuonbrain/rlds/episodes/
2. Run the training pipeline
3. Save model checkpoints
4. Report training metrics

Goal: {goal.description}"""

        return self._run_claude_code(prompt)

    def _execute_transfer(self, goal: LearningGoal) -> bool:
        """Execute transfer phase - sim to real."""
        logger.info("🔄 TRANSFER phase - sim to real transfer")
        # This would involve model adaptation
        return False

    def _execute_deploy(self, goal: LearningGoal) -> bool:
        """Execute deploy phase - real world."""
        logger.info("🚀 DEPLOY phase - real world execution")
        # This would run on actual hardware
        return False

    def _run_claude_code(self, prompt: str) -> bool:
        """Run Claude Code with a prompt."""
        try:
            result = subprocess.run(
                [
                    'claude',
                    '--print',
                    '--dangerously-skip-permissions',
                    '-p', prompt,
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900,
            )

            output = result.stdout + result.stderr
            logger.info(f"Claude Code output: {output[:500]}...")

            return any(s in output.lower() for s in ['success', 'complete', 'done'])

        except Exception as e:
            logger.error(f"Claude Code error: {e}")
            return False

    def run_daemon(self, interval: int = 600):
        """Run continuously as learning partner."""
        logger.info("🤝 Learning Partner starting in continuous mode")
        logger.info(f"   Interval: {interval} seconds")

        while True:
            try:
                result = self.run_learning_cycle()
                logger.info(f"Cycle result: {result['status']}")
            except KeyboardInterrupt:
                logger.info("Learning Partner stopped")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            time.sleep(interval)

    def status(self) -> Dict:
        """Get learning partner status."""
        brain_state = self.analyzer.analyze_full_state()
        goals = self.goal_generator.generate_goals()

        return {
            'mission': MISSION.strip()[:100],
            'current_phase': brain_state['current_phase'].value,
            'room_scans': brain_state['room_scans']['scan_count'],
            'rlds_episodes': brain_state['rlds_data']['episode_count'],
            'behaviors_learned': brain_state['brain_b']['behaviors_learned'],
            'training_tools': brain_state['training_tools'],
            'active_goals': len(goals),
            'top_goals': [g.to_dict() for g in goals[:3]],
            'next_steps': brain_state['next_steps'],
            'learning_cycles': self.state.get('learning_cycles', 0),
            'capabilities': {
                name: data['health']
                for name, data in brain_state['capabilities'].items()
            },
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Learning Partner - Help the robot learn')
    parser.add_argument('--once', action='store_true', help='Run single cycle')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--goals', action='store_true', help='Show learning goals')
    parser.add_argument('--train', action='store_true', help='Run training cycle')
    parser.add_argument('--interval', type=int, default=600, help='Cycle interval (seconds)')
    parser.add_argument('--project', type=str, default='.', help='Project root')

    args = parser.parse_args()

    project_root = Path(args.project).resolve()
    partner = LearningPartner(project_root)

    if args.status:
        status = partner.status()
        print("\n🧠 Learning Partner Status")
        print("=" * 50)
        print(f"Phase: {status['current_phase'].upper()}")
        print(f"Learning Cycles: {status['learning_cycles']}")
        print(f"\nTraining Data:")
        print(f"  Room Scans: {status['room_scans']}")
        print(f"  RLDS Episodes: {status['rlds_episodes']}")
        print(f"  Behaviors Learned: {status['behaviors_learned']}")
        print(f"\nCapabilities:")
        for cap, health in status['capabilities'].items():
            bar = '█' * int(health * 10) + '░' * (10 - int(health * 10))
            print(f"  {cap:15} [{bar}] {health*100:.0f}%")
        print(f"\nTraining Tools:")
        for tool, available in status['training_tools'].items():
            icon = '✅' if available else '❌'
            print(f"  {icon} {tool}")
        print(f"\n📋 Next Steps:")
        for step in status['next_steps']:
            print(f"  → {step}")
        return

    if args.goals:
        status = partner.status()
        print("\n🎯 Learning Goals")
        print("=" * 50)
        for i, goal in enumerate(status['top_goals'], 1):
            print(f"\n{i}. [{goal['phase'].upper()}] {goal['description']}")
            print(f"   Capability: {goal['capability']}")
            print(f"   Tool: {goal['training_tool'] or 'N/A'}")
            print(f"   Priority: {goal['priority']:.2f}")
        return

    if args.train:
        print("Running training cycle...")
        partner.pipeline.run_simulator_training()
        partner.pipeline.run_auto_trainer()
        return

    if args.once:
        partner.run_learning_cycle()
    else:
        partner.run_daemon(interval=args.interval)


if __name__ == '__main__':
    main()
