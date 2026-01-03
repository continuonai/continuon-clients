"""
Seed Model Evaluation Benchmark Suite

Tests the core capabilities of the WaveCore seed model:
1. Semantic Differentiation - Can it distinguish different intents?
2. Temporal Consistency - Does state persist correctly over time?
3. CMS Memory - Does memory write/read work?
4. Action Coherence - Are outputs contextually appropriate?
5. Domain Generalization - Does it work across different domains?
6. Robustness - Is it stable under noise/perturbation?
7. Context Switching - Can it handle rapid context changes?
"""

import json
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

import jax
import jax.numpy as jnp

from continuonbrain.jax_models.config import CoreModelConfig
from continuonbrain.jax_models.core_model import CoreModel


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    name: str
    score: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True


@dataclass 
class BenchmarkSuite:
    """Complete benchmark suite results."""
    version: str
    timestamp: str
    model_params: int
    results: List[BenchmarkResult] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    def to_dict(self) -> Dict:
        def _convert(obj):
            """Convert numpy types to Python types."""
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj
        
        return _convert({
            'version': self.version,
            'timestamp': self.timestamp,
            'model_params': self.model_params,
            'overall_score': self.overall_score,
            'passed': self.passed_count,
            'total': len(self.results),
            'results': [
                {'name': r.name, 'score': r.score, 'passed': r.passed, 'details': r.details}
                for r in self.results
            ]
        })


# Test datasets organized by category
SEMANTIC_TESTS = {
    'robot_manipulation': [
        "Pick up the red cube from the table",
        "Move the gripper to position A",
        "Rotate the arm 90 degrees clockwise",
        "Release the object gently",
        "Grasp the blue cylinder",
    ],
    'navigation': [
        "Move forward two meters",
        "Turn left at the next corner", 
        "Navigate to the kitchen",
        "Avoid the obstacle ahead",
        "Return to the charging station",
    ],
    'home_automation': [
        "Turn on the living room lights",
        "Set the thermostat to 72 degrees",
        "Lock the front door",
        "Start the coffee maker",
        "Close the garage door",
    ],
    'conversation': [
        "Hello, how are you today?",
        "What is the weather forecast?",
        "Tell me a joke",
        "What time is it?",
        "Remind me to call mom later",
    ],
    'safety_critical': [
        "Emergency stop now",
        "Halt all movement immediately",
        "Collision detected ahead",
        "Battery critically low",
        "Human detected in workspace",
    ],
}

TEMPORAL_SEQUENCES = [
    # Sequence of related commands that should show state evolution
    [
        "Initialize robot arm",
        "Move to home position",
        "Pick up object A",
        "Move to drop zone",
        "Release object A",
    ],
    [
        "Start patrol mode",
        "Scan room for obstacles",
        "Navigate around furniture",
        "Check door status",
        "Return to base",
    ],
]

CONTEXT_SWITCHES = [
    # Pairs of very different contexts
    ("Pick up the heavy box", "What's the capital of France?"),
    ("Emergency stop!", "Play some relaxing music"),
    ("Navigate to room 204", "Calculate 15% tip on $45"),
    ("Grasp the delicate glass", "Tell me about quantum physics"),
]

ROBUSTNESS_TESTS = [
    # Original + perturbed versions
    ("Pick up the red cube", "Pick up teh red cube"),  # Typo
    ("Move forward", "Move    forward"),  # Extra spaces
    ("Turn left", "TURN LEFT"),  # Case change
    ("Open the door", "Open the door please"),  # Extra words
]


class SeedBenchmark:
    """Benchmark runner for seed model evaluation."""
    
    def __init__(self, model_dir: Path, encoder):
        self.model_dir = Path(model_dir)
        self.encoder = encoder
        self._load_model()
        self._compile_inference()
        
    def _load_model(self):
        """Load model and config from stable directory."""
        with open(self.model_dir / "manifest.json") as f:
            self.manifest = json.load(f)
            
        with open(self.model_dir / "seed_model.pkl", 'rb') as f:
            data = pickle.load(f)
            
        self.params = data['params']['params']
        self.obs_dim = data['metadata']['obs_dim']
        self.action_dim = data['metadata']['action_dim']
        self.output_dim = data['metadata']['output_dim']
        
        config_dict = self.manifest['config']
        self.config = CoreModelConfig(
            d_s=config_dict['d_s'],
            d_w=config_dict['d_w'],
            d_p=config_dict['d_p'],
            d_e=config_dict['d_e'],
            d_k=config_dict['d_k'],
            d_c=config_dict['d_c'],
            num_levels=config_dict['num_levels'],
            cms_sizes=config_dict['cms_sizes'],
            cms_dims=config_dict['cms_dims'],
            cms_decays=config_dict['cms_decays'],
        )
        
        self.model = CoreModel(
            config=self.config, 
            obs_dim=self.obs_dim, 
            action_dim=self.action_dim, 
            output_dim=self.output_dim
        )
        
    def _compile_inference(self):
        """JIT compile inference function."""
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
        """Initialize fresh state."""
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
        """Encode texts to embeddings."""
        emb = self.encoder.encode(texts, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb
    
    def _run_inference(self, embedding: np.ndarray, state: Dict) -> tuple:
        """Run single inference step."""
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
    
    # ========== Benchmark Tests ==========
    
    def test_semantic_differentiation(self) -> BenchmarkResult:
        """Test: Can the model produce different outputs for different semantic categories?"""
        category_outputs = {}
        
        for category, texts in SEMANTIC_TESTS.items():
            embeddings = self._encode(texts)
            outputs = []
            
            for emb in embeddings:
                state = self._init_state()
                output, _, _ = self._run_inference(emb, state)
                outputs.append(output)
            
            category_outputs[category] = np.array(outputs)
        
        # Calculate inter-category vs intra-category variance
        categories = list(category_outputs.keys())
        inter_diffs = []
        intra_diffs = []
        
        for i, cat1 in enumerate(categories):
            # Intra-category: differences within same category
            outs1 = category_outputs[cat1]
            for j in range(len(outs1)):
                for k in range(j+1, len(outs1)):
                    intra_diffs.append(np.linalg.norm(outs1[j] - outs1[k]))
            
            # Inter-category: differences between categories
            for cat2 in categories[i+1:]:
                outs2 = category_outputs[cat2]
                for o1 in outs1:
                    for o2 in outs2:
                        inter_diffs.append(np.linalg.norm(o1 - o2))
        
        avg_inter = np.mean(inter_diffs)
        avg_intra = np.mean(intra_diffs)
        
        # Score: inter should be > intra (different categories = different outputs)
        ratio = avg_inter / (avg_intra + 1e-8)
        score = min(1.0, ratio / 2.0)  # Normalize: ratio of 2 = perfect
        
        return BenchmarkResult(
            name="Semantic Differentiation",
            score=score,
            passed=ratio > 1.0,
            details={
                'inter_category_diff': float(avg_inter),
                'intra_category_diff': float(avg_intra),
                'ratio': float(ratio),
                'categories_tested': len(categories),
            }
        )
    
    def test_temporal_consistency(self) -> BenchmarkResult:
        """Test: Does state evolve consistently over sequential commands?"""
        all_state_changes = []
        all_output_changes = []
        
        for sequence in TEMPORAL_SEQUENCES:
            embeddings = self._encode(sequence)
            state = self._init_state()
            
            prev_output = None
            prev_state_norm = 0
            
            for emb in embeddings:
                output, state, _ = self._run_inference(emb, state)
                
                # Track state evolution
                state_norm = float(jnp.linalg.norm(state['s']))
                if prev_output is not None:
                    all_state_changes.append(abs(state_norm - prev_state_norm))
                    all_output_changes.append(np.linalg.norm(output - prev_output))
                
                prev_output = output
                prev_state_norm = state_norm
        
        # Score: state should change but not wildly
        avg_state_change = np.mean(all_state_changes)
        std_state_change = np.std(all_state_changes)
        
        # Coefficient of variation (lower = more consistent)
        cv = std_state_change / (avg_state_change + 1e-8)
        score = max(0, 1.0 - cv)  # CV of 0 = perfect, CV of 1+ = poor
        
        return BenchmarkResult(
            name="Temporal Consistency",
            score=score,
            passed=cv < 1.0,
            details={
                'avg_state_change': float(avg_state_change),
                'std_state_change': float(std_state_change),
                'coefficient_of_variation': float(cv),
                'sequences_tested': len(TEMPORAL_SEQUENCES),
            }
        )
    
    def test_cms_memory(self) -> BenchmarkResult:
        """Test: Does CMS memory accumulate and persist?"""
        sequence = [
            "Remember this: the robot is named Atlas",
            "The workspace is in room 204",
            "The primary task is object sorting",
            "What is the robot's name?",  # Should recall from memory
            "Where is the workspace?",     # Should recall from memory
        ]
        
        embeddings = self._encode(sequence)
        state = self._init_state()
        
        memory_norms = []
        for emb in embeddings:
            _, state, info = self._run_inference(emb, state)
            
            # Track CMS memory activity
            cms_norm = sum(float(jnp.mean(jnp.abs(m))) for m in state['cms_memories'])
            memory_norms.append(cms_norm)
        
        # Score: memory should accumulate (increase over time)
        growth_rate = (memory_norms[-1] - memory_norms[0]) / len(memory_norms)
        
        # Also check that memory is non-trivial at the end
        final_memory = memory_norms[-1]
        
        score = min(1.0, (growth_rate * 10) + (final_memory * 0.5))
        
        return BenchmarkResult(
            name="CMS Memory Persistence",
            score=score,
            passed=growth_rate > 0 and final_memory > 0.1,
            details={
                'initial_memory': float(memory_norms[0]),
                'final_memory': float(memory_norms[-1]),
                'growth_rate': float(growth_rate),
                'memory_trajectory': [float(m) for m in memory_norms],
            }
        )
    
    def test_context_switching(self) -> BenchmarkResult:
        """Test: Can the model handle rapid context switches?"""
        switch_diffs = []
        
        for ctx1, ctx2 in CONTEXT_SWITCHES:
            emb1, emb2 = self._encode([ctx1, ctx2])
            
            # Run both from fresh state
            state1 = self._init_state()
            out1, _, _ = self._run_inference(emb1, state1)
            
            state2 = self._init_state()
            out2, _, _ = self._run_inference(emb2, state2)
            
            # Different contexts should produce different outputs
            diff = np.linalg.norm(out1 - out2)
            switch_diffs.append(diff)
        
        avg_diff = np.mean(switch_diffs)
        min_diff = np.min(switch_diffs)
        
        # Score: all context switches should produce notable differences
        score = min(1.0, avg_diff / 3.0)  # Diff of 3.0 = perfect
        
        return BenchmarkResult(
            name="Context Switching",
            score=score,
            passed=min_diff > 0.5,
            details={
                'avg_context_diff': float(avg_diff),
                'min_context_diff': float(min_diff),
                'max_context_diff': float(np.max(switch_diffs)),
                'pairs_tested': len(CONTEXT_SWITCHES),
            }
        )
    
    def test_robustness(self) -> BenchmarkResult:
        """Test: Is the model robust to minor input perturbations?"""
        similarities = []
        
        for original, perturbed in ROBUSTNESS_TESTS:
            emb_orig, emb_pert = self._encode([original, perturbed])
            
            state = self._init_state()
            out_orig, _, _ = self._run_inference(emb_orig, state)
            
            state = self._init_state()
            out_pert, _, _ = self._run_inference(emb_pert, state)
            
            # Similar inputs should produce similar outputs
            # Use cosine similarity
            cos_sim = np.dot(out_orig, out_pert) / (
                np.linalg.norm(out_orig) * np.linalg.norm(out_pert) + 1e-8
            )
            similarities.append(cos_sim)
        
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        
        # Score: perturbations should not drastically change output
        score = max(0, avg_sim)
        
        return BenchmarkResult(
            name="Robustness to Perturbations",
            score=score,
            passed=min_sim > 0.5,
            details={
                'avg_similarity': float(avg_sim),
                'min_similarity': float(min_sim),
                'pairs_tested': len(ROBUSTNESS_TESTS),
            }
        )
    
    def test_safety_priority(self) -> BenchmarkResult:
        """Test: Does the model treat safety commands specially?"""
        safety_texts = SEMANTIC_TESTS['safety_critical']
        normal_texts = SEMANTIC_TESTS['robot_manipulation'][:5]
        
        safety_emb = self._encode(safety_texts)
        normal_emb = self._encode(normal_texts)
        
        safety_outputs = []
        normal_outputs = []
        
        for emb in safety_emb:
            state = self._init_state()
            out, _, _ = self._run_inference(emb, state)
            safety_outputs.append(out)
            
        for emb in normal_emb:
            state = self._init_state()
            out, _, _ = self._run_inference(emb, state)
            normal_outputs.append(out)
        
        # Safety commands should be distinguishable from normal commands
        safety_mean = np.mean(safety_outputs, axis=0)
        normal_mean = np.mean(normal_outputs, axis=0)
        
        separation = np.linalg.norm(safety_mean - normal_mean)
        
        # Also check variance within safety (should be low = consistent handling)
        safety_var = np.mean([np.linalg.norm(s - safety_mean) for s in safety_outputs])
        
        score = min(1.0, separation / 3.0)
        
        return BenchmarkResult(
            name="Safety Command Priority",
            score=score,
            passed=separation > 1.0,
            details={
                'safety_normal_separation': float(separation),
                'safety_consistency': float(1.0 - safety_var),
                'safety_commands_tested': len(safety_texts),
            }
        )
    
    def test_inference_speed(self) -> BenchmarkResult:
        """Test: Is inference fast enough for real-time robotics?"""
        state = self._init_state()
        
        # Benchmark
        n_steps = 100
        embeddings = self._encode(["Test input"] * n_steps)
        
        t0 = time.time()
        for emb in embeddings:
            _, state, _ = self._run_inference(emb, state)
        elapsed = time.time() - t0
        
        hz = n_steps / elapsed
        ms_per_step = (elapsed / n_steps) * 1000
        
        # Score: 10 Hz = minimum, 100 Hz = excellent
        score = min(1.0, (hz - 10) / 90)  # Linear from 10-100 Hz
        
        return BenchmarkResult(
            name="Inference Speed",
            score=max(0, score),
            passed=hz >= 10,
            details={
                'hz': float(hz),
                'ms_per_step': float(ms_per_step),
                'steps_tested': n_steps,
            }
        )
    
    def run_all(self) -> BenchmarkSuite:
        """Run all benchmark tests."""
        from datetime import datetime
        
        suite = BenchmarkSuite(
            version=self.manifest['version'],
            timestamp=datetime.now().isoformat(),
            model_params=self.manifest['model']['param_count'],
        )
        
        tests = [
            self.test_semantic_differentiation,
            self.test_temporal_consistency,
            self.test_cms_memory,
            self.test_context_switching,
            self.test_robustness,
            self.test_safety_priority,
            self.test_inference_speed,
        ]
        
        for test_fn in tests:
            try:
                result = test_fn()
                suite.results.append(result)
            except Exception as e:
                suite.results.append(BenchmarkResult(
                    name=test_fn.__name__.replace('test_', '').replace('_', ' ').title(),
                    score=0.0,
                    passed=False,
                    details={'error': str(e)}
                ))
        
        return suite


def run_benchmark(model_dir: str = "/opt/continuonos/brain/model/seed_stable",
                  output_file: str = None) -> BenchmarkSuite:
    """Run complete benchmark suite."""
    import os
    os.environ['HF_HOME'] = '/opt/continuonos/brain/hf_cache'
    
    from sentence_transformers import SentenceTransformer
    
    print("=" * 70)
    print("SEED MODEL EVALUATION BENCHMARK")
    print("=" * 70)
    
    print("\n📚 Loading encoder...")
    encoder = SentenceTransformer(
        'google/embeddinggemma-300m', 
        trust_remote_code=True,
        token=os.environ.get('HUGGINGFACE_TOKEN')
    )
    
    print("\n🔧 Loading model...")
    benchmark = SeedBenchmark(Path(model_dir), encoder)
    print(f"   Model: v{benchmark.manifest['version']}")
    print(f"   Params: {benchmark.manifest['model']['param_count']:,}")
    
    print("\n🧪 Running tests...\n")
    suite = benchmark.run_all()
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for result in suite.results:
        status = "✅" if result.passed else "❌"
        print(f"\n{status} {result.name}")
        print(f"   Score: {result.score:.2f}")
        for key, value in result.details.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            elif isinstance(value, list) and len(value) <= 10:
                print(f"   {key}: {[f'{v:.2f}' for v in value]}")
            else:
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print(f"OVERALL SCORE: {suite.overall_score:.2f}")
    print(f"PASSED: {suite.passed_count}/{len(suite.results)}")
    print("=" * 70)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\n📁 Results saved to: {output_file}")
    
    return suite


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/opt/continuonos/brain/model/seed_stable')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    run_benchmark(args.model_dir, args.output)

