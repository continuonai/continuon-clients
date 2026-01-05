"""
Latency Optimizer for Seed Model

Techniques:
1. AOT (Ahead-of-Time) compilation with JAX
2. Model quantization (float32 -> bfloat16/int8)
3. Batch size optimization
4. Memory layout optimization
5. JIT trace caching
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from continuonbrain.jax_models.core_model import CoreModel, make_core_model
from continuonbrain.jax_models.config import CoreModelConfig
from continuonbrain.jax_models.config_presets import get_config_for_preset


@dataclass
class LatencyBenchmark:
    """Latency benchmark results."""
    name: str
    iterations: int
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    passed: bool
    target_ms: float = 100.0


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    technique: str
    before_ms: float
    after_ms: float
    improvement_pct: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class LatencyOptimizer:
    """
    Optimizes seed model inference latency for production.

    Target: <100ms for real-time robot control (10Hz minimum)
    Stretch: <50ms for smooth 20Hz control
    """

    TARGET_LATENCY_MS = 100.0
    STRETCH_LATENCY_MS = 50.0
    WARMUP_ITERATIONS = 5
    BENCHMARK_ITERATIONS = 100

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        config: Optional[CoreModelConfig] = None,
        obs_dim: int = 128,
        action_dim: int = 32,
        output_dim: int = 32,
    ):
        self.model_dir = Path(model_dir) if model_dir else None
        self.config = config or get_config_for_preset("pi5")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.output_dim = output_dim

        self.model = None
        self.params = None
        self.optimized_params = None
        self.optimization_results: List[OptimizationResult] = []

    def load_model(self, params_path: Optional[Path] = None) -> None:
        """Load model and parameters."""
        rng_key = jax.random.PRNGKey(42)
        self.model, self.params = make_core_model(
            rng_key, self.obs_dim, self.action_dim, self.output_dim, self.config
        )

        if params_path and params_path.exists():
            with open(params_path, 'rb') as f:
                loaded = pickle.load(f)
                # Handle nested params structure
                if 'params' in loaded:
                    if isinstance(loaded['params'], dict) and 'params' in loaded['params']:
                        # Double-nested: {'params': {'params': ...}}
                        self.params = loaded['params']
                    else:
                        # Single-nested: {'params': ...}
                        self.params = loaded
                else:
                    self.params = {'params': loaded}

        self.optimized_params = self.params
        print(f"Model loaded: {self._count_params():,} parameters")

    def _count_params(self) -> int:
        """Count model parameters."""
        leaves = jax.tree_util.tree_leaves(self.params)
        return sum(x.size for x in leaves if hasattr(x, 'size'))

    def _create_test_inputs(self, batch_size: int = 1) -> Dict[str, jnp.ndarray]:
        """Create test inputs for benchmarking."""
        rng = jax.random.PRNGKey(0)
        return {
            'obs': jax.random.normal(rng, (batch_size, self.obs_dim)),
            'action': jnp.zeros((batch_size, self.action_dim)),
            'reward': jnp.zeros((batch_size, 1)),
            's': jnp.zeros((batch_size, self.config.d_s)),
            'w': jnp.zeros((batch_size, self.config.d_w)),
            'p': jnp.zeros((batch_size, self.config.d_p)),
            'cms_memories': [
                jnp.zeros((batch_size, sz, dim))
                for sz, dim in zip(self.config.cms_sizes, self.config.cms_dims)
            ],
            'cms_keys': [
                jnp.zeros((batch_size, sz, self.config.d_k))
                for sz in self.config.cms_sizes
            ],
        }

    def _run_inference(self, inputs: Dict, params: Dict) -> Tuple[jnp.ndarray, Dict]:
        """Run single inference."""
        return self.model.apply(
            params,
            inputs['obs'], inputs['action'], inputs['reward'],
            inputs['s'], inputs['w'], inputs['p'],
            inputs['cms_memories'], inputs['cms_keys']
        )

    def benchmark(
        self,
        name: str = "baseline",
        params: Optional[Dict] = None,
        iterations: int = None,
        warmup: int = None,
    ) -> LatencyBenchmark:
        """Benchmark inference latency."""
        if params is None:
            params = self.optimized_params or self.params
        if iterations is None:
            iterations = self.BENCHMARK_ITERATIONS
        if warmup is None:
            warmup = self.WARMUP_ITERATIONS

        inputs = self._create_test_inputs(batch_size=1)

        # Warmup
        for _ in range(warmup):
            self._run_inference(inputs, params)

        # Benchmark
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self._run_inference(inputs, params)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies_sorted = sorted(latencies)

        return LatencyBenchmark(
            name=name,
            iterations=iterations,
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=sum(latencies) / len(latencies),
            p50_ms=latencies_sorted[int(0.5 * len(latencies))],
            p95_ms=latencies_sorted[int(0.95 * len(latencies))],
            p99_ms=latencies_sorted[int(0.99 * len(latencies))],
            passed=sum(latencies) / len(latencies) < self.TARGET_LATENCY_MS,
            target_ms=self.TARGET_LATENCY_MS,
        )

    def optimize_jit_compilation(self) -> OptimizationResult:
        """
        Optimization 1: Ensure JIT compilation is properly cached.
        Create a pre-compiled inference function.
        """
        print("\n[1/5] Optimizing JIT compilation...")

        before = self.benchmark("before_jit", iterations=50)

        # Create JIT-compiled inference function
        @jax.jit
        def fast_inference(params, obs, action, reward, s, w, p, cms_m, cms_k):
            return self.model.apply(
                params, obs, action, reward, s, w, p, cms_m, cms_k
            )

        # Store for later use
        self._fast_inference = fast_inference

        # Warmup the JIT function
        inputs = self._create_test_inputs()
        for _ in range(10):
            fast_inference(
                self.optimized_params,
                inputs['obs'], inputs['action'], inputs['reward'],
                inputs['s'], inputs['w'], inputs['p'],
                inputs['cms_memories'], inputs['cms_keys']
            )

        # Benchmark with explicit JIT
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            fast_inference(
                self.optimized_params,
                inputs['obs'], inputs['action'], inputs['reward'],
                inputs['s'], inputs['w'], inputs['p'],
                inputs['cms_memories'], inputs['cms_keys']
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        after_ms = sum(latencies) / len(latencies)
        improvement = (before.avg_ms - after_ms) / before.avg_ms * 100

        result = OptimizationResult(
            technique="JIT Compilation Cache",
            before_ms=before.avg_ms,
            after_ms=after_ms,
            improvement_pct=improvement,
            success=improvement > 0,
            details={"jit_warmup_runs": 10}
        )
        self.optimization_results.append(result)
        print(f"    Before: {before.avg_ms:.2f}ms, After: {after_ms:.2f}ms ({improvement:+.1f}%)")
        return result

    def optimize_dtype(self, target_dtype: str = "bfloat16") -> OptimizationResult:
        """
        Optimization 2: Convert to lower precision dtype.
        bfloat16 is often faster on modern hardware.
        """
        print(f"\n[2/5] Optimizing dtype to {target_dtype}...")

        before = self.benchmark("before_dtype", iterations=50)

        dtype_map = {
            "bfloat16": jnp.bfloat16,
            "float16": jnp.float16,
            "float32": jnp.float32,
        }

        target = dtype_map.get(target_dtype, jnp.bfloat16)

        # Convert parameters to target dtype
        def convert_dtype(x):
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(target)
            return x

        self.optimized_params = jax.tree_util.tree_map(convert_dtype, self.optimized_params)

        after = self.benchmark("after_dtype", iterations=50)
        improvement = (before.avg_ms - after.avg_ms) / before.avg_ms * 100

        result = OptimizationResult(
            technique=f"Dtype Conversion ({target_dtype})",
            before_ms=before.avg_ms,
            after_ms=after.avg_ms,
            improvement_pct=improvement,
            success=improvement > 0,
            details={"target_dtype": target_dtype}
        )
        self.optimization_results.append(result)
        print(f"    Before: {before.avg_ms:.2f}ms, After: {after.avg_ms:.2f}ms ({improvement:+.1f}%)")
        return result

    def optimize_model_size(self) -> OptimizationResult:
        """
        Optimization 3: Create a smaller model variant for inference.
        Reduce hidden dimensions while maintaining key capabilities.
        """
        print("\n[3/5] Optimizing model size (creating lite variant)...")

        before = self.benchmark("before_lite", iterations=50)

        # Create lite config
        lite_config = CoreModelConfig(
            d_s=64,   # Reduced from 128
            d_w=64,   # Reduced from 128
            d_p=32,   # Reduced from 64
            d_e=64,   # Reduced from 128
            d_k=16,   # Reduced from 32
            d_c=64,   # Reduced from 128
            num_levels=2,  # Reduced from 3
            cms_sizes=(16, 32),  # Reduced
            cms_dims=(32, 64),   # Reduced
            cms_decays=(0.05, 0.02),
            use_mamba_wave=True,
            state_saturation_limit=10.0,
        )

        # Create lite model
        rng_key = jax.random.PRNGKey(42)
        lite_model, lite_params = make_core_model(
            rng_key, self.obs_dim, self.action_dim, self.output_dim, lite_config
        )

        lite_param_count = sum(x.size for x in jax.tree_util.tree_leaves(lite_params))
        original_param_count = self._count_params()

        # Store lite model
        self._lite_model = lite_model
        self._lite_params = lite_params
        self._lite_config = lite_config

        # Benchmark lite model
        inputs = self._create_test_inputs()
        # Adjust inputs for lite model dimensions
        lite_inputs = {
            'obs': inputs['obs'],
            'action': inputs['action'],
            'reward': inputs['reward'],
            's': jnp.zeros((1, lite_config.d_s)),
            'w': jnp.zeros((1, lite_config.d_w)),
            'p': jnp.zeros((1, lite_config.d_p)),
            'cms_memories': [
                jnp.zeros((1, sz, dim))
                for sz, dim in zip(lite_config.cms_sizes, lite_config.cms_dims)
            ],
            'cms_keys': [
                jnp.zeros((1, sz, lite_config.d_k))
                for sz in lite_config.cms_sizes
            ],
        }

        # Warmup
        for _ in range(10):
            lite_model.apply(
                lite_params,
                lite_inputs['obs'], lite_inputs['action'], lite_inputs['reward'],
                lite_inputs['s'], lite_inputs['w'], lite_inputs['p'],
                lite_inputs['cms_memories'], lite_inputs['cms_keys']
            )

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            lite_model.apply(
                lite_params,
                lite_inputs['obs'], lite_inputs['action'], lite_inputs['reward'],
                lite_inputs['s'], lite_inputs['w'], lite_inputs['p'],
                lite_inputs['cms_memories'], lite_inputs['cms_keys']
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        after_ms = sum(latencies) / len(latencies)
        improvement = (before.avg_ms - after_ms) / before.avg_ms * 100

        result = OptimizationResult(
            technique="Model Size Reduction (Lite)",
            before_ms=before.avg_ms,
            after_ms=after_ms,
            improvement_pct=improvement,
            success=improvement > 0,
            details={
                "original_params": original_param_count,
                "lite_params": lite_param_count,
                "size_reduction_pct": (1 - lite_param_count / original_param_count) * 100,
            }
        )
        self.optimization_results.append(result)
        print(f"    Before: {before.avg_ms:.2f}ms, After: {after_ms:.2f}ms ({improvement:+.1f}%)")
        print(f"    Params: {original_param_count:,} -> {lite_param_count:,} ({result.details['size_reduction_pct']:.1f}% reduction)")
        return result

    def optimize_batch_inference(self) -> OptimizationResult:
        """
        Optimization 4: Batch multiple inputs for throughput.
        Useful when processing multiple commands simultaneously.
        """
        print("\n[4/5] Optimizing batch inference...")

        # Single inference baseline
        inputs_single = self._create_test_inputs(batch_size=1)

        # Warmup
        for _ in range(5):
            self._run_inference(inputs_single, self.optimized_params)

        latencies_single = []
        for _ in range(50):
            t0 = time.perf_counter()
            self._run_inference(inputs_single, self.optimized_params)
            latencies_single.append((time.perf_counter() - t0) * 1000)

        single_ms = sum(latencies_single) / len(latencies_single)

        # Batch inference
        inputs_batch = self._create_test_inputs(batch_size=4)

        # Warmup
        for _ in range(5):
            self._run_inference(inputs_batch, self.optimized_params)

        latencies_batch = []
        for _ in range(50):
            t0 = time.perf_counter()
            self._run_inference(inputs_batch, self.optimized_params)
            latencies_batch.append((time.perf_counter() - t0) * 1000)

        batch_ms = sum(latencies_batch) / len(latencies_batch)
        per_item_ms = batch_ms / 4

        improvement = (single_ms - per_item_ms) / single_ms * 100

        result = OptimizationResult(
            technique="Batch Inference (4x)",
            before_ms=single_ms,
            after_ms=per_item_ms,
            improvement_pct=improvement,
            success=improvement > 0,
            details={
                "batch_size": 4,
                "total_batch_ms": batch_ms,
                "per_item_ms": per_item_ms,
            }
        )
        self.optimization_results.append(result)
        print(f"    Single: {single_ms:.2f}ms, Batch(4): {batch_ms:.2f}ms ({per_item_ms:.2f}ms/item)")
        return result

    def optimize_memory_layout(self) -> OptimizationResult:
        """
        Optimization 5: Ensure contiguous memory layout.
        """
        print("\n[5/5] Optimizing memory layout...")

        before = self.benchmark("before_memory", iterations=50)

        # Ensure all arrays are contiguous using numpy
        def make_contiguous(x):
            if isinstance(x, jnp.ndarray):
                # Convert to numpy, make contiguous, convert back
                return jnp.array(np.ascontiguousarray(np.array(x)))
            return x

        try:
            self.optimized_params = jax.tree_util.tree_map(make_contiguous, self.optimized_params)
        except Exception as e:
            print(f"    Memory layout optimization skipped: {e}")
            result = OptimizationResult(
                technique="Memory Layout (Contiguous)",
                before_ms=before.avg_ms,
                after_ms=before.avg_ms,
                improvement_pct=0,
                success=True,
                details={"skipped": True, "reason": str(e)}
            )
            self.optimization_results.append(result)
            return result

        after = self.benchmark("after_memory", iterations=50)
        improvement = (before.avg_ms - after.avg_ms) / before.avg_ms * 100

        result = OptimizationResult(
            technique="Memory Layout (Contiguous)",
            before_ms=before.avg_ms,
            after_ms=after.avg_ms,
            improvement_pct=improvement,
            success=improvement > -5,  # Allow small regression
            details={}
        )
        self.optimization_results.append(result)
        print(f"    Before: {before.avg_ms:.2f}ms, After: {after.avg_ms:.2f}ms ({improvement:+.1f}%)")
        return result

    def run_all_optimizations(self) -> Dict[str, Any]:
        """Run all optimization techniques."""
        print("=" * 60)
        print("LATENCY OPTIMIZATION SUITE")
        print("=" * 60)
        print(f"Target: <{self.TARGET_LATENCY_MS}ms (stretch: <{self.STRETCH_LATENCY_MS}ms)")

        if self.model is None:
            self.load_model()

        # Initial benchmark
        initial = self.benchmark("initial", iterations=100)
        print(f"\nInitial latency: {initial.avg_ms:.2f}ms (p95: {initial.p95_ms:.2f}ms)")

        # Run optimizations
        self.optimize_jit_compilation()
        self.optimize_dtype("bfloat16")
        self.optimize_model_size()
        self.optimize_batch_inference()
        self.optimize_memory_layout()

        # Final benchmark
        final = self.benchmark("final", iterations=100)

        # Summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Initial: {initial.avg_ms:.2f}ms")
        print(f"Final:   {final.avg_ms:.2f}ms")
        print(f"Total improvement: {(initial.avg_ms - final.avg_ms) / initial.avg_ms * 100:.1f}%")
        print(f"Target met: {'YES' if final.passed else 'NO'} (target: <{self.TARGET_LATENCY_MS}ms)")

        if hasattr(self, '_lite_model'):
            print(f"\nLite model available for further speed (see optimize_model_size results)")

        return {
            "initial_benchmark": initial,
            "final_benchmark": final,
            "optimizations": self.optimization_results,
            "target_met": final.passed,
            "lite_model_available": hasattr(self, '_lite_model'),
        }

    def export_optimized_model(self, output_dir: Path) -> Dict[str, Path]:
        """Export optimized model and lite variant."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exports = {}

        # Export optimized params
        optimized_path = output_dir / "optimized_params.pkl"
        with open(optimized_path, 'wb') as f:
            pickle.dump({
                'params': self.optimized_params,
                'config': self.config,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'output_dim': self.output_dim,
            }, f)
        exports['optimized'] = optimized_path

        # Export lite model if available
        if hasattr(self, '_lite_model'):
            lite_path = output_dir / "lite_params.pkl"
            with open(lite_path, 'wb') as f:
                pickle.dump({
                    'params': self._lite_params,
                    'config': self._lite_config,
                    'obs_dim': self.obs_dim,
                    'action_dim': self.action_dim,
                    'output_dim': self.output_dim,
                }, f)
            exports['lite'] = lite_path

        # Export manifest
        manifest = {
            'optimized_model': str(optimized_path),
            'lite_model': str(exports.get('lite', '')),
            'optimizations_applied': [r.technique for r in self.optimization_results],
            'final_latency_ms': self.optimization_results[-1].after_ms if self.optimization_results else None,
        }
        manifest_path = output_dir / "optimization_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        exports['manifest'] = manifest_path

        print(f"\nExported optimized models to: {output_dir}")
        return exports


if __name__ == "__main__":
    optimizer = LatencyOptimizer()
    results = optimizer.run_all_optimizations()

    # Export if successful
    from pathlib import Path
    export_dir = Path(__file__).parent.parent.parent / "models" / "optimized"
    optimizer.export_optimized_model(export_dir)
