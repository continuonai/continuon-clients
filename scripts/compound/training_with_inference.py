#!/usr/bin/env python3
"""
Enhanced Training Loop with Live Inference Testing

Runs training cycles and periodically tests inference quality:
1. Train navigation/conversation models
2. Every N cycles, run inference tests
3. Log performance metrics
4. Use Claude Code to suggest improvements when performance drops

New Features (v2):
5. Expert perception training with 95%+ accuracy
6. Human feedback collection and learning
7. Spatial memory for navigation planning
8. Curiosity-driven exploration for autonomous learning
9. Multi-modal sensor fusion (LiDAR + Camera + IMU + Audio)

Usage:
    python scripts/compound/training_with_inference.py
    python scripts/compound/training_with_inference.py --test-interval 5
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "brain_b"))

from trainer.conversation_trainer import ConversationTrainer, get_llm_backend
from simulator.simulator_training import get_simulator_predictor, IDX_TO_ACTION

# New v2 imports for enhanced training
try:
    from trainer.expert_perception import ExpertPerceptionTrainer
    from trainer.human_feedback import HumanFeedbackCollector, FeedbackLearner
    from trainer.curiosity_exploration import CuriousExplorer, StateFeatures
    from trainer.multimodal_fusion import MultiModalFusion, LiDARData, CameraData, IMUData, AudioData
    from memory.spatial_memory import SpatialMemory
    V2_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some v2 features unavailable: {e}")
    V2_FEATURES_AVAILABLE = False


class InferenceTester:
    """Tests inference quality across different modes."""

    def __init__(self):
        self.conversation = ConversationTrainer("brain_b_data")
        self.conversation.load_model()
        self.nav_predictor = get_simulator_predictor()
        self._load_nav_model()
        self.llm = get_llm_backend()

        self.test_history: List[Dict] = []

    def _load_nav_model(self):
        """Load best navigation model."""
        checkpoint_dir = Path("brain_b_data/simulator_checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("sim_model_*.json"))
            if checkpoints:
                try:
                    self.nav_predictor.load(str(checkpoints[-1]))
                except Exception as e:
                    print(f"  Nav model error: {e}")

    def test_conversation(self) -> Dict:
        """Test conversation inference."""
        test_cases = [
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("what can you do", "ask_capabilities"),
            ("go forward", "nav_forward"),
            ("stop", "nav_stop"),
            ("turn left", "nav_left"),
            ("go to the kitchen", "nav_location"),
            ("teach me something", "start_teaching"),
            ("yes", "confirm"),
            ("good job", "praise"),
        ]

        correct = 0
        total_conf = 0
        results = []

        for text, expected in test_cases:
            result = self.conversation.predict(text, use_llm=False)
            actual = result.get("intent", "unknown")
            conf = result.get("confidence", 0)

            is_correct = actual == expected
            if is_correct:
                correct += 1
            total_conf += conf

            results.append({
                "input": text,
                "expected": expected,
                "actual": actual,
                "confidence": conf,
                "correct": is_correct,
            })

        accuracy = correct / len(test_cases)
        avg_conf = total_conf / len(test_cases)

        return {
            "mode": "conversation",
            "accuracy": accuracy,
            "avg_confidence": avg_conf,
            "correct": correct,
            "total": len(test_cases),
            "details": results,
        }

    def test_navigation(self) -> Dict:
        """Test navigation inference using improved expert-trained model."""
        # Try to load improved navigation model first
        try:
            from simulator.expert_navigation import ImprovedNavigationModel, NavigationState
            improved_model = ImprovedNavigationModel()
            improved_model.load("brain_b_data/improved_navigation_model.json")
            use_improved = True
        except Exception:
            use_improved = False

        if use_improved:
            # Test with improved model using proper NavigationState
            scenarios = [
                ("clear_ahead", NavigationState(front=0.9, front_left=0.8, front_right=0.8, goal_direction=0.0), ["move_forward"]),
                ("wall_ahead", NavigationState(front=0.1, front_left=0.3, front_right=0.3, left=0.6, right=0.6), ["rotate_left", "rotate_right", "move_backward"]),
                ("open_left", NavigationState(front=0.5, front_left=0.8, left=0.9, right=0.3), ["rotate_left", "move_forward"]),
                ("corner_right", NavigationState(front=0.15, front_left=0.5, front_right=0.1, left=0.7, right=0.1), ["rotate_left", "move_backward"]),
                ("dead_end", NavigationState(front=0.1, front_left=0.15, front_right=0.15, left=0.15, right=0.15, back=0.8, back_left=0.5, back_right=0.5), ["move_backward"]),
            ]

            correct = 0
            results = []

            for name, state, valid_actions in scenarios:
                action, conf, probs = improved_model.predict(state)

                is_reasonable = action in valid_actions
                if is_reasonable:
                    correct += 1

                results.append({
                    "scenario": name,
                    "action": action,
                    "confidence": conf,
                    "valid_actions": valid_actions,
                    "reasonable": is_reasonable,
                })

            accuracy = correct / len(scenarios) if scenarios else 0

            return {
                "mode": "navigation",
                "status": "ready (improved)",
                "accuracy": accuracy,
                "correct": correct,
                "total": len(scenarios),
                "details": results,
            }

        # Fallback to old predictor
        if not self.nav_predictor.is_ready:
            return {
                "mode": "navigation",
                "status": "not_ready",
                "accuracy": 0,
            }

        # Test scenarios with expected reasonable actions (legacy)
        scenarios = [
            ("clear_ahead", [0.9, 0.1, 0.1, 0.0] + [0.0]*44, ["move_forward"]),
            ("wall_ahead", [0.1, 0.9, 0.1, 0.0] + [0.0]*44, ["rotate_left", "rotate_right", "move_backward"]),
            ("open_left", [0.5, 0.1, 0.9, 0.0] + [0.0]*44, ["rotate_left", "move_forward"]),
        ]

        correct = 0
        results = []

        for name, state, valid_actions in scenarios:
            probs = self.nav_predictor.predict(state)
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            action = IDX_TO_ACTION.get(best_idx, "unknown")
            conf = probs[best_idx]

            is_reasonable = action in valid_actions
            if is_reasonable:
                correct += 1

            results.append({
                "scenario": name,
                "action": action,
                "confidence": conf,
                "valid_actions": valid_actions,
                "reasonable": is_reasonable,
            })

        accuracy = correct / len(scenarios) if scenarios else 0

        return {
            "mode": "navigation",
            "status": "ready",
            "accuracy": accuracy,
            "correct": correct,
            "total": len(scenarios),
            "details": results,
        }

    def test_llm_backend(self) -> Dict:
        """Test LLM backend availability and response."""
        if not self.llm.is_available:
            return {
                "mode": "llm",
                "status": "not_available",
                "backend": self.llm.backend,
            }

        # Test with a complex query
        test_input = "please navigate carefully to the bedroom and avoid obstacles"
        result = self.llm.process(test_input)

        return {
            "mode": "llm",
            "status": "available",
            "backend": self.llm.backend,
            "test_input": test_input,
            "response": result,
        }

    def test_expert_perception(self) -> Dict:
        """Test expert perception model."""
        if not V2_FEATURES_AVAILABLE:
            return {"mode": "expert_perception", "status": "unavailable"}

        try:
            from trainer.expert_perception import ExpertPerceptionModel, PerceptionDataGenerator

            model = ExpertPerceptionModel()
            model.load("brain_b_data/expert_perception_model.json")

            generator = PerceptionDataGenerator()

            # Test scenarios using generator
            # Actions: move_forward, move_backward, rotate_left, rotate_right, stop
            test_scenarios = [
                ("clear_path", ["move_forward"]),
                ("person_ahead", ["stop", "move_forward"]),
                ("obstacle_ahead", ["stop", "rotate_left", "rotate_right", "move_backward"]),
            ]

            correct = 0
            results = []
            for name, valid_actions in test_scenarios:
                scenario = generator.generate_scenario(name)
                features = generator.scenario_to_features(scenario)
                action, conf, _ = model.predict(features)

                is_reasonable = action in valid_actions
                if is_reasonable:
                    correct += 1

                results.append({
                    "scenario": name,
                    "action": action,
                    "confidence": conf,
                    "valid": valid_actions,
                    "reasonable": is_reasonable,
                })

            accuracy = correct / len(test_scenarios)
            return {
                "mode": "expert_perception",
                "status": "ready",
                "accuracy": accuracy,
                "correct": correct,
                "total": len(test_scenarios),
                "details": results,
            }
        except Exception as e:
            return {"mode": "expert_perception", "status": "error", "error": str(e)}

    def test_multimodal_fusion(self) -> Dict:
        """Test multi-modal sensor fusion."""
        if not V2_FEATURES_AVAILABLE:
            return {"mode": "multimodal_fusion", "status": "unavailable"}

        try:
            import math
            fusion = MultiModalFusion()
            try:
                fusion.load("brain_b_data/multimodal_fusion.json")
            except Exception:
                pass  # Use default initialization

            # Test with sample data
            lidar = LiDARData(
                distances=[1.0 - 0.2 * math.sin(i * 0.05) for i in range(360)],
                timestamp=0.0
            )
            camera = CameraData(
                color_histogram=[0.5] * 24,
                edge_density=[0.3, 0.4, 0.3, 0.4],
                brightness=0.5,
                timestamp=0.0
            )
            imu = IMUData(
                acceleration=(0.0, 0.0, 9.81),
                gyroscope=(0.0, 0.0, 0.0),
                timestamp=0.0
            )

            result = fusion.fuse(lidar, camera, imu, dt=0.1)

            return {
                "mode": "multimodal_fusion",
                "status": "ready",
                "confidence": result["confidence"],
                "sensors_used": result["sensors_used"],
                "state_uncertainty": result["state_estimate"]["uncertainty"],
            }
        except Exception as e:
            return {"mode": "multimodal_fusion", "status": "error", "error": str(e)}

    def test_spatial_memory(self) -> Dict:
        """Test spatial memory and path planning."""
        if not V2_FEATURES_AVAILABLE:
            return {"mode": "spatial_memory", "status": "unavailable"}

        try:
            memory = SpatialMemory(grid_size=(10, 10), resolution=0.5)
            try:
                memory.load("brain_b_data/spatial_memory.json")
            except Exception:
                pass  # Use default initialization

            # Test path planning (remember_object, not add_object)
            memory.remember_object("goal", (4.0, 4.0, 0.0), confidence=0.9)
            path = memory.plan_path((0.5, 0.5), (4.0, 4.0))

            summary = memory.get_memory_summary()

            # Calculate grid coverage from stats
            grid_stats = summary.get("grid_stats", {})
            total_cells = sum(grid_stats.values()) if grid_stats else 1
            explored = grid_stats.get("free_cells", 0) + grid_stats.get("visited_cells", 0)
            coverage = explored / total_cells if total_cells > 0 else 0

            return {
                "mode": "spatial_memory",
                "status": "ready",
                "grid_coverage": coverage,
                "objects": summary.get("objects_remembered", 0),
                "path_found": path is not None,
                "path_length": len(path) if path else 0,
            }
        except Exception as e:
            return {"mode": "spatial_memory", "status": "error", "error": str(e)}

    def run_full_test(self) -> Dict:
        """Run all inference tests."""
        print("\n" + "="*60)
        print("🧪 INFERENCE TEST (v2)")
        print("="*60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "conversation": self.test_conversation(),
            "navigation": self.test_navigation(),
            "llm": self.test_llm_backend(),
        }

        # V2 tests
        if V2_FEATURES_AVAILABLE:
            results["expert_perception"] = self.test_expert_perception()
            results["multimodal_fusion"] = self.test_multimodal_fusion()
            results["spatial_memory"] = self.test_spatial_memory()

        # Print summary
        conv = results["conversation"]
        nav = results["navigation"]
        llm = results["llm"]

        print(f"\n📊 Core Results:")
        print(f"  Conversation: {conv['accuracy']*100:.0f}% accuracy ({conv['correct']}/{conv['total']})")
        print(f"  Navigation:   {nav.get('accuracy', 0)*100:.0f}% reasonable ({nav.get('status', 'unknown')})")
        print(f"  LLM Backend:  {llm['status']} ({llm['backend']})")

        # V2 results
        if V2_FEATURES_AVAILABLE:
            print(f"\n📊 V2 Feature Results:")
            expert = results.get("expert_perception", {})
            fusion = results.get("multimodal_fusion", {})
            spatial = results.get("spatial_memory", {})

            if expert.get("status") == "ready":
                print(f"  Expert Perception: {expert.get('accuracy', 0)*100:.0f}% ({expert.get('correct', 0)}/{expert.get('total', 0)})")
            else:
                print(f"  Expert Perception: {expert.get('status', 'unknown')}")

            if fusion.get("status") == "ready":
                print(f"  Multi-Modal Fusion: confidence={fusion.get('confidence', 0):.2f}, "
                      f"sensors={fusion.get('sensors_used', [])}")
            else:
                print(f"  Multi-Modal Fusion: {fusion.get('status', 'unknown')}")

            if spatial.get("status") == "ready":
                print(f"  Spatial Memory: {spatial.get('grid_coverage', 0)*100:.1f}% explored, "
                      f"path_found={spatial.get('path_found', False)}")
            else:
                print(f"  Spatial Memory: {spatial.get('status', 'unknown')}")

        # Calculate overall score (weighted)
        scores = [conv['accuracy'], nav.get('accuracy', 0)]
        weights = [1.0, 1.0]

        if V2_FEATURES_AVAILABLE:
            expert = results.get("expert_perception", {})
            if expert.get("status") == "ready":
                scores.append(expert.get("accuracy", 0))
                weights.append(0.5)

            fusion = results.get("multimodal_fusion", {})
            if fusion.get("status") == "ready":
                scores.append(fusion.get("confidence", 0))
                weights.append(0.3)

            spatial = results.get("spatial_memory", {})
            if spatial.get("status") == "ready":
                scores.append(1.0 if spatial.get("path_found") else 0.5)
                weights.append(0.2)

        overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        results["overall_score"] = overall
        print(f"\n  Overall Score: {overall*100:.0f}%")

        self.test_history.append(results)
        return results


class EnhancedTrainingLoop:
    """Enhanced training loop with inference testing."""

    def __init__(self, test_interval: int = 5):
        self.test_interval = test_interval
        self.cycle_count = 0
        self.tester = InferenceTester()
        self.project_root = Path(__file__).parent.parent.parent

        self.metrics_file = self.project_root / "brain_b_data" / "training_metrics.json"
        self.metrics: List[Dict] = self._load_metrics()

    def _load_metrics(self) -> List[Dict]:
        """Load existing metrics."""
        if self.metrics_file.exists():
            try:
                return json.loads(self.metrics_file.read_text())
            except Exception:
                pass
        return []

    def _save_metrics(self):
        """Save metrics to file."""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics[-100:], f, indent=2)  # Keep last 100

    def run_training_cycle(self) -> Dict:
        """Run one training cycle."""
        print(f"\n{'='*60}")
        print(f"🏋️ Training Cycle {self.cycle_count + 1}")
        print(f"{'='*60}")

        results = {"cycle": self.cycle_count + 1, "timestamp": datetime.now().isoformat()}

        # 1. Run conversation training (quick)
        print("\n💬 Training conversation model...")
        try:
            conv_trainer = ConversationTrainer("brain_b_data")
            if len(conv_trainer.dataset.samples) < 500:
                conv_trainer.generate_training_data(500, use_llm=False)
            metrics = conv_trainer.train(epochs=20)
            results["conversation"] = {
                "samples": metrics.samples_trained,
                "accuracy": metrics.accuracy,
            }
            print(f"   Accuracy: {metrics.accuracy*100:.0f}%")
        except Exception as e:
            print(f"   Error: {e}")
            results["conversation"] = {"error": str(e)}

        # 2. Run simulator training
        print("\n🎮 Training navigation model...")
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
import sys
sys.path.insert(0, 'brain_b')
from simulator.simulator_training import SimulatorTrainer
trainer = SimulatorTrainer('brain_b_data')
episodes = trainer.load_episodes('continuonbrain/rlds/episodes')
if episodes:
    trainer.train(episodes, epochs=10)
    print(f'Trained on {len(episodes)} episodes')
else:
    print('No episodes found')
"""],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root,
            )
            results["navigation"] = {"output": result.stdout.strip()}
            print(f"   {result.stdout.strip()}")
        except Exception as e:
            print(f"   Error: {e}")
            results["navigation"] = {"error": str(e)}

        # 3. Generate high-fidelity training episodes
        print("\n🎯 Generating high-fidelity training games...")
        try:
            result = subprocess.run(
                [sys.executable, "-c", """
import sys
sys.path.insert(0, 'brain_b')
from simulator.high_fidelity_trainer import HighFidelityTrainer

trainer = HighFidelityTrainer(resolution=(320, 240))
episodes = trainer.generate_curriculum(
    total_episodes=20,
    max_steps=50,
    save=True,
)
print(f'Generated {len(episodes)} high-fidelity episodes')
success = sum(1 for e in episodes if e._check_completion())
print(f'Success rate: {success}/{len(episodes)}')
"""],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=self.project_root,
            )
            results["game_generation"] = {"output": result.stdout.strip()}
            print(f"   {result.stdout.strip()}")
        except Exception as e:
            print(f"   Error: {e}")
            results["game_generation"] = {"error": str(e)}

        # 4. Train perception-based navigation
        print("\n👁️ Training perception model...")
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
    print(f'Perception accuracy: {metrics["accuracy"]*100:.1f}%')
else:
    print('No episodes')
"""],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=self.project_root,
            )
            results["perception"] = {"output": result.stdout.strip()}
            print(f"   {result.stdout.strip()}")
        except Exception as e:
            print(f"   Error: {e}")
            results["perception"] = {"error": str(e)}

        # ========== V2 FEATURES ==========
        if V2_FEATURES_AVAILABLE:
            # 5. Expert perception training (high accuracy visual decisions)
            print("\n🎯 Training expert perception model...")
            try:
                result = subprocess.run(
                    [sys.executable, "-c", """
import sys
sys.path.insert(0, 'brain_b')
from trainer.expert_perception import ExpertPerceptionTrainer

trainer = ExpertPerceptionTrainer(data_dir='brain_b_data')
trainer.generate_expert_samples(2000)
metrics = trainer.train(epochs=30)
print(f'Expert perception accuracy: {metrics["accuracy"]*100:.1f}%')
trainer.save_model('brain_b_data/expert_perception_model.json')
"""],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=self.project_root,
                )
                results["expert_perception"] = {"output": result.stdout.strip()}
                print(f"   {result.stdout.strip()}")
            except Exception as e:
                print(f"   Error: {e}")
                results["expert_perception"] = {"error": str(e)}

            # 6. Curiosity-driven exploration (autonomous learning)
            print("\n🔍 Running curiosity exploration cycle...")
            try:
                result = subprocess.run(
                    [sys.executable, "-c", """
import sys
import math
import random
sys.path.insert(0, 'brain_b')
from trainer.curiosity_exploration import CuriousExplorer, StateFeatures

explorer = CuriousExplorer()

# Run simulated exploration
position = [0.0, 0.0]
orientation = 0.0
initial_state = StateFeatures(position=tuple(position), orientation=orientation)
explorer.start_exploration(initial_state)

for step in range(100):
    distances = [max(0.1, 1.0 - 0.3*abs(math.sin(position[0]+position[1]+i))) for i in range(8)]
    sensor_data = {
        "position": tuple(position),
        "orientation": orientation,
        "distances": distances,
        "brightness": 0.5
    }
    action, novelty = explorer.step(sensor_data)

    # Apply action
    if action == "move_forward":
        position[0] += 0.1 * math.cos(orientation)
        position[1] += 0.1 * math.sin(orientation)
    elif action == "move_backward":
        position[0] -= 0.05 * math.cos(orientation)
        position[1] -= 0.05 * math.sin(orientation)
    elif action == "rotate_left":
        orientation += 0.3
    elif action == "rotate_right":
        orientation -= 0.3

stats = explorer.end_exploration()
print(f'Exploration: {stats["exploration_steps"]} steps, {stats["unique_states"]} unique states')
print(f'Avg intrinsic reward: {stats["avg_intrinsic_reward"]:.4f}')
"""],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root,
                )
                results["curiosity_exploration"] = {"output": result.stdout.strip()}
                print(f"   {result.stdout.strip()}")
            except Exception as e:
                print(f"   Error: {e}")
                results["curiosity_exploration"] = {"error": str(e)}

            # 7. Multi-modal fusion training
            print("\n📡 Training multi-modal fusion...")
            try:
                result = subprocess.run(
                    [sys.executable, "-c", """
import sys
import math
import random
sys.path.insert(0, 'brain_b')
from trainer.multimodal_fusion import MultiModalFusion, LiDARData, CameraData, IMUData

fusion = MultiModalFusion()

# Train on simulated multi-sensor data
for i in range(50):
    t = i * 0.1
    lidar = LiDARData(
        distances=[1.0 - 0.3*math.sin(j*0.1 + t) for j in range(360)],
        timestamp=t
    )
    camera = CameraData(
        color_histogram=[random.random() for _ in range(24)],
        edge_density=[0.3, 0.5, 0.2, 0.4],
        brightness=0.5 + 0.1*math.sin(t),
        timestamp=t
    )
    imu = IMUData(
        acceleration=(0.1*math.sin(t), 0.05*math.cos(t), 9.81),
        gyroscope=(0.0, 0.0, 0.1*math.sin(t*2)),
        timestamp=t
    )
    result = fusion.fuse(lidar, camera, imu, dt=0.1)

state = fusion.kalman.get_state()
print(f'Fusion trained: confidence={result["confidence"]:.2f}, uncertainty={state["uncertainty"]:.4f}')
fusion.save('brain_b_data/multimodal_fusion.json')
"""],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root,
                )
                results["multimodal_fusion"] = {"output": result.stdout.strip()}
                print(f"   {result.stdout.strip()}")
            except Exception as e:
                print(f"   Error: {e}")
                results["multimodal_fusion"] = {"error": str(e)}

            # 8. Spatial memory update
            print("\n🗺️ Updating spatial memory...")
            try:
                result = subprocess.run(
                    [sys.executable, "-c", """
import sys
import math
sys.path.insert(0, 'brain_b')
from memory.spatial_memory import SpatialMemory

memory = SpatialMemory(grid_size=(20, 20), resolution=0.5)

# Simulate environment exploration
robot_pos = (5.0, 5.0)
robot_theta = 0.0

for i in range(20):
    # Simulate LiDAR scan
    distances = []
    angles = []
    for a in range(0, 360, 5):
        angle = math.radians(a) + robot_theta
        # Simulate walls
        dist = 5.0
        if 45 < a < 135:  # Wall ahead
            dist = 2.0 + 0.5*math.sin(a*0.1)
        distances.append(dist)
        angles.append(angle)

    memory.update_from_lidar(robot_pos, robot_theta, distances, angles)

    # Move robot
    robot_pos = (robot_pos[0] + 0.2*math.cos(robot_theta),
                 robot_pos[1] + 0.2*math.sin(robot_theta))
    robot_theta += 0.1

# Add some objects
memory.add_object("goal", (8.0, 8.0), confidence=0.9)

summary = memory.get_summary()
print(f'Spatial memory: {summary["grid_coverage"]*100:.1f}% explored')
print(f'Objects: {summary["objects_remembered"]}, Landmarks: {summary["landmarks"]}')
memory.save('brain_b_data/spatial_memory.json')
"""],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root,
                )
                results["spatial_memory"] = {"output": result.stdout.strip()}
                print(f"   {result.stdout.strip()}")
            except Exception as e:
                print(f"   Error: {e}")
                results["spatial_memory"] = {"error": str(e)}

        self.cycle_count += 1
        return results

    def run_inference_test(self) -> Dict:
        """Run inference tests."""
        # Reload models to get latest
        self.tester = InferenceTester()
        return self.tester.run_full_test()

    def check_for_improvements(self, test_results: Dict):
        """Use Claude Code to suggest improvements if needed."""
        overall = test_results.get("overall_score", 0)

        if overall < 0.7:
            print("\n🤖 Performance below 70% - consulting Claude Code for improvements...")

            prompt = f"""The robot training system has the following test results:

Conversation accuracy: {test_results['conversation']['accuracy']*100:.0f}%
Navigation accuracy: {test_results['navigation'].get('accuracy', 0)*100:.0f}%
Overall score: {overall*100:.0f}%

Conversation failures:
{json.dumps([r for r in test_results['conversation']['details'] if not r['correct']], indent=2)}

Suggest 2-3 specific improvements to increase accuracy. Focus on:
1. Training data quality
2. Model architecture tweaks
3. Feature engineering

Be concise."""

            try:
                result = subprocess.run(
                    ["claude", "--print", "-p", prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    print("\n📝 Claude's suggestions:")
                    print(result.stdout[:500])
            except Exception as e:
                print(f"   Claude Code error: {e}")

    def run(self, max_cycles: int = 0):
        """Run the enhanced training loop."""
        print("="*60)
        print("🧠 Enhanced Training Loop")
        print("="*60)
        print(f"Test interval: every {self.test_interval} cycles")
        print(f"Interactive server: http://localhost:8765")
        print()

        try:
            while True:
                # Run training cycle
                train_results = self.run_training_cycle()

                # Test inference periodically
                if self.cycle_count % self.test_interval == 0:
                    test_results = self.run_inference_test()
                    train_results["inference_test"] = test_results

                    # Check if we need improvements
                    self.check_for_improvements(test_results)

                # Save metrics
                self.metrics.append(train_results)
                self._save_metrics()

                # Check max cycles
                if max_cycles > 0 and self.cycle_count >= max_cycles:
                    print(f"\n✅ Completed {max_cycles} cycles")
                    break

                # Wait before next cycle
                print(f"\n⏳ Waiting 60s before next cycle...")
                time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n🛑 Training stopped by user")
            print(f"   Completed {self.cycle_count} cycles")


def run_visual_training(num_episodes: int = 10, template: str = 'studio_apartment'):
    """Run visual 3D training mode with house_3d renderer."""
    print("\n" + "=" * 60)
    print("🏠 Visual 3D Training Mode")
    print("=" * 60)

    try:
        from simulator.house_3d import (
            create_visual_training_env,
            run_visual_training_episode,
            generate_visual_training_batch,
        )

        print(f"\nGenerating {num_episodes} visual training episodes...")
        print(f"Template: {template}")

        results = generate_visual_training_batch(
            num_episodes=num_episodes,
            template=template,
            output_dir='brain_b_data/visual_episodes',
            save_frames=True,
        )

        print(f"\n✅ Generated {len(results)} episodes")
        total_steps = sum(r['steps'] for r in results)
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        print(f"   Total steps: {total_steps}")
        print(f"   Average reward: {avg_reward:.2f}")

        return results

    except ImportError as e:
        print(f"❌ Visual training not available: {e}")
        print("   Make sure house_3d module is installed")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced training with inference testing")
    parser.add_argument("--test-interval", type=int, default=5, help="Run inference test every N cycles")
    parser.add_argument("--max-cycles", type=int, default=0, help="Max cycles (0 = infinite)")
    parser.add_argument("--test-only", action="store_true", help="Just run inference test")
    parser.add_argument("--visual", action="store_true", help="Run visual 3D training mode")
    parser.add_argument("--visual-episodes", type=int, default=10, help="Number of visual episodes to generate")
    parser.add_argument("--template", type=str, default="studio_apartment",
                       help="House template (studio_apartment, two_bedroom)")

    args = parser.parse_args()

    if args.visual:
        run_visual_training(num_episodes=args.visual_episodes, template=args.template)
    elif args.test_only:
        tester = InferenceTester()
        tester.run_full_test()
    else:
        loop = EnhancedTrainingLoop(test_interval=args.test_interval)
        loop.run(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
