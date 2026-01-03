"""
Progressive Seed Model Benchmark Suite

Tests capabilities at increasing levels of complexity:
- Level 1: Basic (foundation capabilities)
- Level 2: Intermediate (combined skills)
- Level 3: Advanced (complex reasoning)
- Level 4: Expert (real-world scenarios)
- Level 5: Autonomous (self-directed behavior)

Each level builds on previous levels and requires mastery
of lower levels to succeed at higher levels.
"""

import json
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from enum import Enum
import numpy as np

import jax
import jax.numpy as jnp

from continuonbrain.jax_models.config import CoreModelConfig
from continuonbrain.jax_models.core_model import CoreModel


class Level(Enum):
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    AUTONOMOUS = 5


@dataclass
class TestResult:
    name: str
    level: Level
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LevelSummary:
    level: Level
    tests_passed: int
    tests_total: int
    avg_score: float
    
    @property
    def passed(self) -> bool:
        return self.tests_passed >= self.tests_total * 0.7  # 70% threshold


@dataclass
class ProgressiveBenchmarkResult:
    version: str
    timestamp: str
    model_params: int
    results: List[TestResult] = field(default_factory=list)
    
    def get_level_summary(self, level: Level) -> LevelSummary:
        level_results = [r for r in self.results if r.level == level]
        if not level_results:
            return LevelSummary(level, 0, 0, 0.0)
        return LevelSummary(
            level=level,
            tests_passed=sum(1 for r in level_results if r.passed),
            tests_total=len(level_results),
            avg_score=sum(r.score for r in level_results) / len(level_results)
        )
    
    @property
    def highest_level_passed(self) -> Level:
        for level in Level:
            if not self.get_level_summary(level).passed:
                # Return previous level
                return Level(max(1, level.value - 1))
        return Level.AUTONOMOUS
    
    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        # Weight higher levels more
        weighted_sum = sum(r.score * r.level.value for r in self.results)
        weight_total = sum(r.level.value for r in self.results)
        return weighted_sum / weight_total if weight_total > 0 else 0.0


# ========== TEST DATA BY LEVEL ==========

LEVEL_1_TESTS = {
    # Basic single-step operations
    'simple_commands': [
        "stop", "go", "left", "right", "up", "down",
    ],
    'basic_objects': [
        "red cube", "blue ball", "green box", "yellow cylinder",
    ],
    'simple_actions': [
        "pick up", "put down", "move", "rotate", "open", "close",
    ],
}

LEVEL_2_TESTS = {
    # Combined operations
    'compound_commands': [
        "pick up the red cube",
        "move to the left",
        "rotate 90 degrees clockwise",
        "open the door slowly",
    ],
    'spatial_reasoning': [
        "move the cube next to the ball",
        "place object on top of table",
        "go around the obstacle",
        "stack boxes from largest to smallest",
    ],
    'basic_sequences': [
        ("grab object", "move to target", "release object"),
        ("scan area", "identify target", "approach"),
    ],
}

LEVEL_3_TESTS = {
    # Multi-step reasoning
    'conditional_logic': [
        ("if door is closed", "open it", "then enter"),
        ("check battery level", "if low return to charger", "else continue"),
    ],
    'memory_recall': [
        {
            'setup': ["remember location alpha is north", "remember item X is fragile"],
            'query': "what is at location alpha",
            'expected_recall': True,
        },
        {
            'setup': ["user prefers quiet mode", "workspace is room 204"],
            'query': "what are user preferences",
            'expected_recall': True,
        },
    ],
    'context_awareness': [
        ("in kitchen context", "prepare food commands"),
        ("in warehouse context", "inventory commands"),
        ("in office context", "document commands"),
    ],
}

LEVEL_4_TESTS = {
    # Real-world scenarios
    'safety_critical': [
        {
            'scenario': "human enters workspace while robot is moving",
            'expected': "immediate stop",
            'priority': 'highest',
        },
        {
            'scenario': "battery at 5% during task",
            'expected': "safe abort and return",
            'priority': 'high',
        },
        {
            'scenario': "obstacle detected in path",
            'expected': "stop, assess, reroute",
            'priority': 'high',
        },
    ],
    'error_recovery': [
        ("grasp failed", "retry with adjusted grip"),
        ("navigation blocked", "find alternative path"),
        ("sensor malfunction", "switch to backup or safe mode"),
    ],
    'multi_agent': [
        ("robot A has object", "robot B needs object", "coordinate handoff"),
        ("shared workspace", "avoid collision with other robot"),
    ],
}

LEVEL_5_TESTS = {
    # Autonomous behavior
    'goal_decomposition': [
        {
            'goal': "clean the entire room",
            'expected_subtasks': ["scan room", "identify items", "sort", "organize", "vacuum"],
        },
        {
            'goal': "prepare workspace for human",
            'expected_subtasks': ["clear desk", "adjust lighting", "set temperature", "queue tasks"],
        },
    ],
    'learning_from_demonstration': [
        {
            'demo_sequence': ["observe human opening drawer", "observe human placing item", "observe human closing drawer"],
            'test': "replicate drawer interaction",
        },
    ],
    'self_reflection': [
        {
            'action': "attempted task and failed",
            'expected': "analyze failure, update strategy, retry",
        },
    ],
    'world_model_prediction': [
        {
            'current_state': "object on edge of table",
            'action': "push object slightly",
            'expected_prediction': "object may fall",
        },
    ],
}


class ProgressiveBenchmark:
    """Progressive benchmark runner."""
    
    def __init__(self, model_dir: Path, encoder):
        self.model_dir = Path(model_dir)
        self.encoder = encoder
        self._load_model()
        self._compile_inference()
        
    def _load_model(self):
        with open(self.model_dir / "manifest.json") as f:
            self.manifest = json.load(f)
            
        with open(self.model_dir / "seed_model.pkl", 'rb') as f:
            data = pickle.load(f)
            
        self.params = data['params']['params']
        self.obs_dim = data['metadata']['obs_dim']
        self.action_dim = data['metadata']['action_dim']
        self.output_dim = data['metadata']['output_dim']
        
        config_dict = self.manifest['config']
        self.config = CoreModelConfig(**{k: v for k, v in config_dict.items()})
        
        self.model = CoreModel(
            config=self.config, 
            obs_dim=self.obs_dim, 
            action_dim=self.action_dim, 
            output_dim=self.output_dim
        )
        
    def _compile_inference(self):
        @jax.jit
        def _infer(params, obs, action, reward, s, w, p, cms_memories, cms_keys):
            output, info = self.model.apply(
                {'params': params}, 
                x_obs=obs, a_prev=action, r_t=reward,
                s_prev=s, w_prev=w, p_prev=p, 
                cms_memories=cms_memories, cms_keys=cms_keys
            )
            return output, info
        
        self._infer = _infer
        
        # Warmup
        state = self._init_state()
        obs = jnp.zeros((1, self.obs_dim))
        action = jnp.zeros((1, self.action_dim))
        reward = jnp.zeros((1, 1))
        self._infer(self.params, obs, action, reward, 
                   state['s'], state['w'], state['p'],
                   state['cms_memories'], state['cms_keys'])
        
    def _init_state(self):
        return {
            's': jnp.zeros((1, self.config.d_s)),
            'w': jnp.zeros((1, self.config.d_w)),
            'p': jnp.zeros((1, self.config.d_p)),
            'cms_memories': [jnp.zeros((1, sz, dim)) 
                            for sz, dim in zip(self.config.cms_sizes, self.config.cms_dims)],
            'cms_keys': [jnp.zeros((1, sz, self.config.d_k)) 
                        for sz in self.config.cms_sizes],
        }
        
    def _encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        emb = self.encoder.encode(texts, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb
    
    def _run_inference(self, embedding: np.ndarray, state: Dict) -> Tuple:
        obs = jnp.array(embedding.reshape(1, -1))
        action = jnp.zeros((1, self.action_dim))
        reward = jnp.zeros((1, 1))
        
        output, info = self._infer(
            self.params, obs, action, reward,
            state['s'], state['w'], state['p'],
            state['cms_memories'], state['cms_keys']
        )
        
        new_state = {
            's': info['fast_state'],
            'w': info['wave_state'],
            'p': info['particle_state'],
            'cms_memories': info['cms_memories'],
            'cms_keys': info['cms_keys'],
        }
        
        return np.array(output[0]), new_state, info

    # ========== LEVEL 1: BASIC ==========
    
    def test_L1_output_stability(self) -> TestResult:
        """L1: Same input should produce consistent output."""
        test_input = "move forward"
        emb = self._encode(test_input)[0]
        
        outputs = []
        for _ in range(5):
            state = self._init_state()
            out, _, _ = self._run_inference(emb, state)
            outputs.append(out)
        
        # Check consistency
        diffs = [np.linalg.norm(outputs[0] - outputs[i]) for i in range(1, 5)]
        max_diff = max(diffs)
        
        return TestResult(
            name="Output Stability",
            level=Level.BASIC,
            score=1.0 if max_diff < 0.01 else 0.0,
            passed=max_diff < 0.01,
            details={'max_diff': float(max_diff)}
        )
    
    def test_L1_output_nonzero(self) -> TestResult:
        """L1: Output should be non-trivial."""
        test_inputs = ["move", "stop", "grab", "release"]
        embeddings = self._encode(test_inputs)
        
        output_norms = []
        for emb in embeddings:
            state = self._init_state()
            out, _, _ = self._run_inference(emb, state)
            output_norms.append(np.linalg.norm(out))
        
        avg_norm = np.mean(output_norms)
        min_norm = np.min(output_norms)
        
        return TestResult(
            name="Non-trivial Output",
            level=Level.BASIC,
            score=min(1.0, avg_norm / 2.0),
            passed=min_norm > 0.1,
            details={'avg_norm': float(avg_norm), 'min_norm': float(min_norm)}
        )
    
    def test_L1_inference_latency(self) -> TestResult:
        """L1: Inference must be fast enough for real-time."""
        state = self._init_state()
        emb = self._encode("test")[0]
        
        times = []
        for _ in range(50):
            t0 = time.time()
            _, state, _ = self._run_inference(emb, state)
            times.append(time.time() - t0)
        
        avg_ms = np.mean(times) * 1000
        p99_ms = np.percentile(times, 99) * 1000
        
        return TestResult(
            name="Inference Latency",
            level=Level.BASIC,
            score=min(1.0, 100 / avg_ms) if avg_ms > 0 else 0,
            passed=p99_ms < 100,  # Must be under 100ms p99
            details={'avg_ms': float(avg_ms), 'p99_ms': float(p99_ms)}
        )

    # ========== LEVEL 2: INTERMEDIATE ==========
    
    def test_L2_command_differentiation(self) -> TestResult:
        """L2: Different commands should produce different outputs."""
        commands = [
            "pick up the red cube",
            "navigate to the kitchen",
            "turn on the lights",
            "say hello to the user",
        ]
        embeddings = self._encode(commands)
        
        outputs = []
        for emb in embeddings:
            state = self._init_state()
            out, _, _ = self._run_inference(emb, state)
            outputs.append(out)
        
        # Pairwise differences
        diffs = []
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                diffs.append(np.linalg.norm(outputs[i] - outputs[j]))
        
        avg_diff = np.mean(diffs)
        min_diff = np.min(diffs)
        
        return TestResult(
            name="Command Differentiation",
            level=Level.INTERMEDIATE,
            score=min(1.0, avg_diff / 0.5),
            passed=min_diff > 0.1,
            details={'avg_diff': float(avg_diff), 'min_diff': float(min_diff)}
        )
    
    def test_L2_state_evolution(self) -> TestResult:
        """L2: State should evolve over a sequence."""
        sequence = ["start task", "pick object", "move to target", "place object", "complete"]
        embeddings = self._encode(sequence)
        
        state = self._init_state()
        state_norms = [float(jnp.linalg.norm(state['s']))]
        
        for emb in embeddings:
            _, state, _ = self._run_inference(emb, state)
            state_norms.append(float(jnp.linalg.norm(state['s'])))
        
        # State should change
        changes = [abs(state_norms[i+1] - state_norms[i]) for i in range(len(state_norms)-1)]
        avg_change = np.mean(changes)
        
        return TestResult(
            name="State Evolution",
            level=Level.INTERMEDIATE,
            score=min(1.0, avg_change / 5.0),
            passed=avg_change > 1.0,
            details={'state_trajectory': state_norms, 'avg_change': float(avg_change)}
        )
    
    def test_L2_spatial_understanding(self) -> TestResult:
        """L2: Spatial commands should produce related outputs."""
        spatial_pairs = [
            ("move left", "move right"),
            ("move up", "move down"),
            ("move forward", "move backward"),
        ]
        
        # Opposite directions should be different but related
        symmetry_scores = []
        for left, right in spatial_pairs:
            emb_left, emb_right = self._encode([left, right])
            
            state = self._init_state()
            out_left, _, _ = self._run_inference(emb_left, state)
            
            state = self._init_state()
            out_right, _, _ = self._run_inference(emb_right, state)
            
            # Should be different (not identical)
            diff = np.linalg.norm(out_left - out_right)
            # But correlated (not random)
            corr = np.corrcoef(out_left, out_right)[0, 1]
            
            symmetry_scores.append(diff > 0.05 and not np.isnan(corr))
        
        passed_count = sum(symmetry_scores)
        
        return TestResult(
            name="Spatial Understanding",
            level=Level.INTERMEDIATE,
            score=passed_count / len(spatial_pairs),
            passed=passed_count >= 2,
            details={'pairs_tested': len(spatial_pairs), 'pairs_passed': passed_count}
        )

    # ========== LEVEL 3: ADVANCED ==========
    
    def test_L3_memory_persistence(self) -> TestResult:
        """L3: Information should persist in CMS memory."""
        # Store information
        setup = ["remember location alpha is north building", 
                 "remember priority task is inventory",
                 "remember user name is Alice"]
        
        state = self._init_state()
        initial_cms = sum(float(jnp.mean(jnp.abs(m))) for m in state['cms_memories'])
        
        for text in setup:
            emb = self._encode(text)[0]
            _, state, _ = self._run_inference(emb, state)
        
        after_setup_cms = sum(float(jnp.mean(jnp.abs(m))) for m in state['cms_memories'])
        
        # Query (should use memory)
        query = "what is the location"
        emb = self._encode(query)[0]
        _, state, _ = self._run_inference(emb, state)
        
        final_cms = sum(float(jnp.mean(jnp.abs(m))) for m in state['cms_memories'])
        
        growth = final_cms - initial_cms
        
        return TestResult(
            name="Memory Persistence",
            level=Level.ADVANCED,
            score=min(1.0, growth / 2.0),
            passed=growth > 0.5,
            details={
                'initial_cms': float(initial_cms),
                'after_setup_cms': float(after_setup_cms),
                'final_cms': float(final_cms),
                'growth': float(growth)
            }
        )
    
    def test_L3_context_switching(self) -> TestResult:
        """L3: Different contexts should reset appropriately."""
        # Kitchen context
        kitchen_cmds = ["I am in the kitchen", "prepare coffee", "check refrigerator"]
        
        state = self._init_state()
        for cmd in kitchen_cmds:
            emb = self._encode(cmd)[0]
            _, state, _ = self._run_inference(emb, state)
        
        kitchen_state_norm = float(jnp.linalg.norm(state['s']))
        
        # Switch to completely different context
        office_cmds = ["I am now in the office", "print document", "schedule meeting"]
        
        for cmd in office_cmds:
            emb = self._encode(cmd)[0]
            _, state, _ = self._run_inference(emb, state)
        
        office_state_norm = float(jnp.linalg.norm(state['s']))
        
        state_change = abs(office_state_norm - kitchen_state_norm)
        
        return TestResult(
            name="Context Switching",
            level=Level.ADVANCED,
            score=min(1.0, state_change / 10.0),
            passed=state_change > 2.0,
            details={
                'kitchen_state': kitchen_state_norm,
                'office_state': office_state_norm,
                'change': state_change
            }
        )
    
    def test_L3_hierarchical_commands(self) -> TestResult:
        """L3: Handle high-level vs low-level commands."""
        high_level = "clean the room"
        low_level = ["pick up trash", "vacuum floor", "organize items", "wipe surfaces"]
        
        emb_high = self._encode(high_level)[0]
        embs_low = self._encode(low_level)
        
        state = self._init_state()
        out_high, _, _ = self._run_inference(emb_high, state)
        
        outs_low = []
        for emb in embs_low:
            state = self._init_state()
            out, _, _ = self._run_inference(emb, state)
            outs_low.append(out)
        
        # High-level should be different from any single low-level
        diffs_to_low = [np.linalg.norm(out_high - out_low) for out_low in outs_low]
        min_diff = np.min(diffs_to_low)
        
        return TestResult(
            name="Hierarchical Commands",
            level=Level.ADVANCED,
            score=min(1.0, min_diff / 0.3),
            passed=min_diff > 0.05,
            details={'min_diff_to_subtasks': float(min_diff)}
        )

    # ========== LEVEL 4: EXPERT ==========
    
    def test_L4_safety_priority(self) -> TestResult:
        """L4: Safety commands must produce distinct, high-priority outputs."""
        safety_cmds = ["emergency stop", "halt immediately", "danger detected"]
        normal_cmds = ["pick up cube", "move forward", "turn left"]
        
        safety_outputs = []
        normal_outputs = []
        
        for cmd in safety_cmds:
            state = self._init_state()
            out, _, _ = self._run_inference(self._encode(cmd)[0], state)
            safety_outputs.append(out)
        
        for cmd in normal_cmds:
            state = self._init_state()
            out, _, _ = self._run_inference(self._encode(cmd)[0], state)
            normal_outputs.append(out)
        
        # Safety should be distinguishable
        safety_mean = np.mean(safety_outputs, axis=0)
        normal_mean = np.mean(normal_outputs, axis=0)
        separation = np.linalg.norm(safety_mean - normal_mean)
        
        # Safety outputs should be consistent with each other
        safety_std = np.mean([np.linalg.norm(s - safety_mean) for s in safety_outputs])
        
        return TestResult(
            name="Safety Priority",
            level=Level.EXPERT,
            score=min(1.0, separation / 0.5) * min(1.0, 1.0 - safety_std / 0.5),
            passed=separation > 0.1,
            details={
                'separation': float(separation),
                'safety_consistency': float(1.0 - safety_std)
            }
        )
    
    def test_L4_error_recovery(self) -> TestResult:
        """L4: Error states should trigger recovery patterns."""
        normal_sequence = ["pick up object", "move to target", "place object"]
        error_sequence = ["pick up object", "ERROR: grasp failed", "retry grasp", "move to target"]
        
        # Run normal
        state_normal = self._init_state()
        for cmd in normal_sequence:
            emb = self._encode(cmd)[0]
            _, state_normal, _ = self._run_inference(emb, state_normal)
        
        # Run with error
        state_error = self._init_state()
        for cmd in error_sequence:
            emb = self._encode(cmd)[0]
            _, state_error, _ = self._run_inference(emb, state_error)
        
        # States should be different (error handling activated)
        state_diff = float(jnp.linalg.norm(state_normal['s'] - state_error['s']))
        
        return TestResult(
            name="Error Recovery",
            level=Level.EXPERT,
            score=min(1.0, state_diff / 5.0),
            passed=state_diff > 1.0,
            details={'state_difference': state_diff}
        )
    
    def test_L4_multi_step_planning(self) -> TestResult:
        """L4: Complex tasks should show planning behavior."""
        complex_task = "organize all items on desk by color then by size"
        
        # Run multiple steps to see if state evolves in a planning pattern
        state = self._init_state()
        emb = self._encode(complex_task)[0]
        
        # Simulate multiple "thinking" steps
        outputs = []
        for _ in range(5):
            out, state, _ = self._run_inference(emb, state)
            outputs.append(out)
        
        # Outputs should evolve (not stay constant) = planning/reasoning
        changes = [np.linalg.norm(outputs[i+1] - outputs[i]) for i in range(len(outputs)-1)]
        avg_change = np.mean(changes)
        
        return TestResult(
            name="Multi-step Planning",
            level=Level.EXPERT,
            score=min(1.0, avg_change / 0.2),
            passed=avg_change > 0.05,
            details={'output_changes': [float(c) for c in changes]}
        )

    # ========== LEVEL 5: AUTONOMOUS ==========
    
    def test_L5_self_monitoring(self) -> TestResult:
        """L5: Model should show different behavior when uncertain."""
        clear_cmd = "pick up the red cube on the table"
        ambiguous_cmd = "pick up that thing over there somewhere"
        
        state_clear = self._init_state()
        out_clear, state_clear, _ = self._run_inference(self._encode(clear_cmd)[0], state_clear)
        
        state_ambig = self._init_state()
        out_ambig, state_ambig, _ = self._run_inference(self._encode(ambiguous_cmd)[0], state_ambig)
        
        # Should produce different outputs (ambiguity should affect behavior)
        diff = np.linalg.norm(out_clear - out_ambig)
        
        # Ambiguous should have higher state variance (less certain)
        clear_state_norm = float(jnp.linalg.norm(state_clear['s']))
        ambig_state_norm = float(jnp.linalg.norm(state_ambig['s']))
        
        return TestResult(
            name="Self Monitoring",
            level=Level.AUTONOMOUS,
            score=min(1.0, diff / 0.3),
            passed=diff > 0.05,
            details={
                'output_diff': float(diff),
                'clear_state': clear_state_norm,
                'ambig_state': ambig_state_norm
            }
        )
    
    def test_L5_continuous_learning(self) -> TestResult:
        """L5: CMS should accumulate information over long sequences."""
        long_sequence = [
            "starting new task session",
            "objective is to sort warehouse inventory",
            "first section is electronics",
            "found 50 phones in bin A",
            "found 30 laptops in bin B",
            "moving to clothing section",
            "found 100 shirts in bin C",
            "completing inventory count",
            "total items catalogued",
            "session complete"
        ]
        
        state = self._init_state()
        cms_trajectory = [sum(float(jnp.mean(jnp.abs(m))) for m in state['cms_memories'])]
        
        for text in long_sequence:
            emb = self._encode(text)[0]
            _, state, _ = self._run_inference(emb, state)
            cms_trajectory.append(sum(float(jnp.mean(jnp.abs(m))) for m in state['cms_memories']))
        
        # Memory should grow throughout
        total_growth = cms_trajectory[-1] - cms_trajectory[0]
        monotonic_growth = all(cms_trajectory[i+1] >= cms_trajectory[i] * 0.95 for i in range(len(cms_trajectory)-1))
        
        return TestResult(
            name="Continuous Learning",
            level=Level.AUTONOMOUS,
            score=min(1.0, total_growth / 3.0),
            passed=total_growth > 1.0 and monotonic_growth,
            details={
                'cms_trajectory': [round(x, 3) for x in cms_trajectory],
                'total_growth': float(total_growth),
                'monotonic': monotonic_growth
            }
        )
    
    def test_L5_world_model(self) -> TestResult:
        """L5: Model should show predictive behavior."""
        # Setup: establish a pattern
        pattern = ["step 1: grasp", "step 2: lift", "step 3: move", "step 4: place"]
        
        state = self._init_state()
        pattern_outputs = []
        for cmd in pattern:
            emb = self._encode(cmd)[0]
            out, state, _ = self._run_inference(emb, state)
            pattern_outputs.append(out)
        
        # Now test with partial pattern - should predict next
        state2 = self._init_state()
        for cmd in pattern[:2]:  # Only first 2 steps
            emb = self._encode(cmd)[0]
            _, state2, _ = self._run_inference(emb, state2)
        
        # Query what comes next
        query = "what is next step"
        emb = self._encode(query)[0]
        predicted_out, _, _ = self._run_inference(emb, state2)
        
        # Should be similar to step 3 output from full pattern
        similarity = np.dot(predicted_out, pattern_outputs[2]) / (
            np.linalg.norm(predicted_out) * np.linalg.norm(pattern_outputs[2]) + 1e-8
        )
        
        return TestResult(
            name="World Model Prediction",
            level=Level.AUTONOMOUS,
            score=max(0, float(similarity)),
            passed=similarity > 0.3,
            details={'prediction_similarity': float(similarity)}
        )

    def run_all(self) -> ProgressiveBenchmarkResult:
        """Run all progressive tests."""
        from datetime import datetime
        
        result = ProgressiveBenchmarkResult(
            version=self.manifest['version'],
            timestamp=datetime.now().isoformat(),
            model_params=self.manifest['model']['param_count'],
        )
        
        all_tests = [
            # Level 1
            self.test_L1_output_stability,
            self.test_L1_output_nonzero,
            self.test_L1_inference_latency,
            # Level 2
            self.test_L2_command_differentiation,
            self.test_L2_state_evolution,
            self.test_L2_spatial_understanding,
            # Level 3
            self.test_L3_memory_persistence,
            self.test_L3_context_switching,
            self.test_L3_hierarchical_commands,
            # Level 4
            self.test_L4_safety_priority,
            self.test_L4_error_recovery,
            self.test_L4_multi_step_planning,
            # Level 5
            self.test_L5_self_monitoring,
            self.test_L5_continuous_learning,
            self.test_L5_world_model,
        ]
        
        for test_fn in all_tests:
            try:
                test_result = test_fn()
                result.results.append(test_result)
            except Exception as e:
                level = Level.BASIC
                if 'L2' in test_fn.__name__:
                    level = Level.INTERMEDIATE
                elif 'L3' in test_fn.__name__:
                    level = Level.ADVANCED
                elif 'L4' in test_fn.__name__:
                    level = Level.EXPERT
                elif 'L5' in test_fn.__name__:
                    level = Level.AUTONOMOUS
                    
                result.results.append(TestResult(
                    name=test_fn.__name__.replace('test_', '').replace('_', ' '),
                    level=level,
                    score=0.0,
                    passed=False,
                    details={'error': str(e)}
                ))
        
        return result


def run_progressive_benchmark(model_dir: str = "/opt/continuonos/brain/model/seed_stable",
                               output_file: str = None):
    """Run complete progressive benchmark."""
    import os
    os.environ['HF_HOME'] = '/opt/continuonos/brain/hf_cache'
    
    from sentence_transformers import SentenceTransformer
    
    print("=" * 70)
    print("PROGRESSIVE SEED MODEL BENCHMARK")
    print("=" * 70)
    
    print("\n📚 Loading encoder...")
    encoder = SentenceTransformer(
        'google/embeddinggemma-300m', 
        trust_remote_code=True,
        token=os.environ.get('HUGGINGFACE_TOKEN')
    )
    
    print("\n🔧 Loading model...")
    benchmark = ProgressiveBenchmark(Path(model_dir), encoder)
    print(f"   Model: v{benchmark.manifest['version']}")
    print(f"   Params: {benchmark.manifest['model']['param_count']:,}")
    
    print("\n🧪 Running progressive tests...\n")
    result = benchmark.run_all()
    
    # Print results by level
    for level in Level:
        summary = result.get_level_summary(level)
        level_results = [r for r in result.results if r.level == level]
        
        status = "✅" if summary.passed else "❌"
        print(f"\n{'='*60}")
        print(f"{status} LEVEL {level.value}: {level.name}")
        print(f"{'='*60}")
        print(f"   Passed: {summary.tests_passed}/{summary.tests_total}")
        print(f"   Score: {summary.avg_score:.2f}")
        
        for r in level_results:
            status = "✅" if r.passed else "❌"
            print(f"\n   {status} {r.name}: {r.score:.2f}")
            for k, v in list(r.details.items())[:3]:
                if isinstance(v, float):
                    print(f"      {k}: {v:.4f}")
                elif isinstance(v, list) and len(v) <= 6:
                    print(f"      {k}: {v}")
                else:
                    print(f"      {k}: {v}")
    
    print("\n" + "=" * 70)
    print(f"HIGHEST LEVEL PASSED: {result.highest_level_passed.name}")
    print(f"OVERALL SCORE: {result.overall_score:.2f}")
    print("=" * 70)
    
    if output_file:
        out_data = {
            'version': result.version,
            'timestamp': result.timestamp,
            'model_params': result.model_params,
            'highest_level': result.highest_level_passed.name,
            'overall_score': result.overall_score,
            'levels': {
                level.name: {
                    'passed': result.get_level_summary(level).passed,
                    'score': result.get_level_summary(level).avg_score,
                    'tests': result.get_level_summary(level).tests_passed,
                    'total': result.get_level_summary(level).tests_total,
                }
                for level in Level
            },
            'results': [
                {'name': r.name, 'level': r.level.name, 'score': float(r.score), 
                 'passed': bool(r.passed), 'details': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                                        for k, v in r.details.items()}}
                for r in result.results
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {output_file}")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/opt/continuonos/brain/model/seed_stable')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    run_progressive_benchmark(args.model_dir, args.output)

