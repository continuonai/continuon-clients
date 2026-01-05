#!/usr/bin/env python3
"""
End-to-End Training and Benchmark Runner

Runs:
1. WavecoreTrainer fast/mid/slow loops
2. Progressive Benchmark (all 6 levels)
3. Reports combined results with timing
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_training():
    """Run training using WavecoreTrainer."""
    print("=" * 70)
    print("PHASE 1: WAVECORE TRAINING")
    print("=" * 70)

    from continuonbrain.services.wavecore_trainer import WavecoreTrainer

    trainer = WavecoreTrainer(
        default_rlds_dir=project_root / "continuonbrain" / "rlds" / "episodes",
        log_dir=project_root / "training_logs",
        checkpoint_dir=project_root / "checkpoints" / "e2e_benchmark",
        export_dir=project_root / "models" / "e2e_benchmark",
    )

    # Configure training payload
    payload = {
        "fast": {
            "max_steps": 8,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "use_synthetic": True,
            "disable_jit": True,
            "arch_preset": "pi5",
        },
        "mid": {
            "max_steps": 12,
            "batch_size": 4,
            "learning_rate": 5e-4,
            "use_synthetic": True,
            "disable_jit": True,
            "arch_preset": "pi5",
        },
        "slow": {
            "max_steps": 16,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "use_synthetic": True,
            "disable_jit": True,
            "arch_preset": "pi5",
        },
        "checkpoint_dir": str(project_root / "checkpoints" / "e2e_benchmark"),
        "export_dir": str(project_root / "models" / "e2e_benchmark"),
    }

    print("\nTraining configuration:")
    print(f"  Fast loop: 8 steps, lr=1e-3")
    print(f"  Mid loop: 12 steps, lr=5e-4")
    print(f"  Slow loop: 16 steps, lr=2e-4")
    print()

    start_time = time.time()
    results = trainer._run_sync(payload)
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Status: {results.get('status', 'unknown')}")

    # Print loop results
    for loop_name in ["fast", "mid", "slow"]:
        loop_result = results.get(loop_name, {})
        loop_data = loop_result.get("result", {})
        final_loss = loop_data.get("final_loss", "N/A")
        avg_loss = loop_data.get("avg_loss", "N/A")
        steps = loop_data.get("steps_run", "N/A")
        print(f"  {loop_name.upper()}: steps={steps}, final_loss={final_loss}, avg_loss={avg_loss}")

    return {
        "status": results.get("status"),
        "training_time_seconds": training_time,
        "loops": {
            name: results.get(name, {}).get("result", {})
            for name in ["fast", "mid", "slow"]
        },
        "export": results.get("export"),
    }


def run_progressive_benchmark():
    """Run progressive benchmark using the trained model."""
    print("\n" + "=" * 70)
    print("PHASE 2: PROGRESSIVE BENCHMARK")
    print("=" * 70)

    import jax
    import jax.numpy as jnp
    import pickle
    from continuonbrain.jax_models.core_model import CoreModel
    from continuonbrain.jax_models.config import CoreModelConfig

    # Load trained model
    model_dir = project_root / "models" / "e2e_benchmark"
    manifest_path = model_dir / "model_manifest.json"
    params_path = model_dir / "params_step_16.pkl"

    if not manifest_path.exists() or not params_path.exists():
        print("No trained model found. Skipping benchmark.")
        return {"error": "No model found", "benchmark_time_seconds": 0}

    print(f"\nLoading model from: {model_dir}")
    start_time = time.time()

    # Load manifest and params
    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(params_path, 'rb') as f:
        params_data = pickle.load(f)

    config_dict = manifest['config']
    obs_dim = manifest['input_dims']['obs_dim']
    action_dim = manifest['input_dims']['action_dim']
    output_dim = manifest['output_dim']

    config = CoreModelConfig(**{k: v for k, v in config_dict.items()})
    model = CoreModel(config=config, obs_dim=obs_dim, action_dim=action_dim, output_dim=output_dim)

    # Extract params - handle nested structure
    if 'params' in params_data and isinstance(params_data['params'], dict) and 'params' in params_data['params']:
        # Double-nested: {'params': {'params': ...}}
        params = params_data['params']
    elif 'params' in params_data:
        # Single-nested: {'params': ...}
        params = params_data
    else:
        # Already flat
        params = {'params': params_data}

    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))

    param_count = count_params(params)
    print(f"  Model parameters: {param_count:,}")

    # Run benchmark tests
    results = []

    def init_state():
        return {
            's': jnp.zeros((1, config.d_s)),
            'w': jnp.zeros((1, config.d_w)),
            'p': jnp.zeros((1, config.d_p)),
            'cms_memories': [jnp.zeros((1, sz, dim)) for sz, dim in zip(config.cms_sizes, config.cms_dims)],
            'cms_keys': [jnp.zeros((1, sz, config.d_k)) for sz in config.cms_sizes],
        }

    def run_inference(obs, state):
        action = jnp.zeros((1, action_dim))
        reward = jnp.zeros((1, 1))
        output, info = model.apply(
            params, obs, action, reward,
            state['s'], state['w'], state['p'],
            state['cms_memories'], state['cms_keys']
        )
        return output, {
            's': info['fast_state'],
            'w': info['wave_state'],
            'p': info['particle_state'],
            'cms_memories': info['cms_memories'],
            'cms_keys': info['cms_keys'],
        }

    # Level 1: Output Stability
    print("\n  Testing L1: Output Stability...")
    state = init_state()
    test_obs = jax.random.normal(jax.random.PRNGKey(0), (1, obs_dim))
    outputs = []
    for _ in range(5):
        state = init_state()
        out, _ = run_inference(test_obs, state)
        outputs.append(out)

    diffs = [float(jnp.linalg.norm(outputs[0] - outputs[i])) for i in range(1, 5)]
    max_diff = max(diffs)
    l1_stability = {"name": "Output Stability", "level": 1, "score": 1.0 if max_diff < 0.01 else 0.5, "passed": max_diff < 0.01, "details": {"max_diff": max_diff}}
    results.append(l1_stability)
    print(f"    [{'PASS' if l1_stability['passed'] else 'FAIL'}] max_diff={max_diff:.6f}")

    # Level 1: Non-zero output
    print("  Testing L1: Non-zero Output...")
    test_inputs = [jax.random.normal(jax.random.PRNGKey(i), (1, obs_dim)) for i in range(4)]
    output_norms = []
    for inp in test_inputs:
        state = init_state()
        out, _ = run_inference(inp, state)
        output_norms.append(float(jnp.linalg.norm(out)))

    avg_norm = sum(output_norms) / len(output_norms)
    min_norm = min(output_norms)
    l1_nonzero = {"name": "Non-zero Output", "level": 1, "score": min(1.0, avg_norm / 2.0), "passed": min_norm > 0.1, "details": {"avg_norm": avg_norm, "min_norm": min_norm}}
    results.append(l1_nonzero)
    print(f"    [{'PASS' if l1_nonzero['passed'] else 'FAIL'}] avg_norm={avg_norm:.4f}")

    # Level 1: Inference Latency
    print("  Testing L1: Inference Latency...")
    latencies = []
    for _ in range(10):
        state = init_state()
        t0 = time.time()
        run_inference(test_obs, state)
        latencies.append((time.time() - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    l1_latency = {"name": "Inference Latency", "level": 1, "score": min(1.0, 100.0 / avg_lat), "passed": avg_lat < 100, "details": {"avg_latency_ms": avg_lat}}
    results.append(l1_latency)
    print(f"    [{'PASS' if l1_latency['passed'] else 'FAIL'}] avg_latency={avg_lat:.2f}ms")

    # Level 2: State Evolution
    print("\n  Testing L2: State Evolution...")
    state = init_state()
    state_norms = []
    for i in range(10):
        inp = jax.random.normal(jax.random.PRNGKey(i + 100), (1, obs_dim))
        _, state = run_inference(inp, state)
        state_norms.append(float(jnp.linalg.norm(state['s'])))

    state_changes = [abs(state_norms[i+1] - state_norms[i]) for i in range(len(state_norms)-1)]
    avg_change = sum(state_changes) / len(state_changes)
    l2_evolution = {"name": "State Evolution", "level": 2, "score": min(1.0, avg_change * 10), "passed": avg_change > 0.01, "details": {"avg_state_change": avg_change}}
    results.append(l2_evolution)
    print(f"    [{'PASS' if l2_evolution['passed'] else 'FAIL'}] avg_change={avg_change:.4f}")

    # Level 2: Command Differentiation
    print("  Testing L2: Command Differentiation...")
    commands = [jax.random.normal(jax.random.PRNGKey(i + 200), (1, obs_dim)) for i in range(5)]
    outputs = []
    for cmd in commands:
        state = init_state()
        out, _ = run_inference(cmd, state)
        outputs.append(out)

    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diffs.append(float(jnp.linalg.norm(outputs[i] - outputs[j])))

    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    l2_diff = {"name": "Command Differentiation", "level": 2, "score": min(1.0, avg_diff * 2), "passed": avg_diff > 0.1, "details": {"avg_output_diff": avg_diff}}
    results.append(l2_diff)
    print(f"    [{'PASS' if l2_diff['passed'] else 'FAIL'}] avg_diff={avg_diff:.4f}")

    # Level 3: Memory Persistence
    print("\n  Testing L3: Memory Persistence...")
    state = init_state()
    mem_before = [float(jnp.sum(jnp.abs(m))) for m in state['cms_memories']]

    # Run several steps
    for i in range(5):
        inp = jax.random.normal(jax.random.PRNGKey(i + 300), (1, obs_dim))
        _, state = run_inference(inp, state)

    mem_after = [float(jnp.sum(jnp.abs(m))) for m in state['cms_memories']]
    mem_change = sum(abs(a - b) for a, b in zip(mem_after, mem_before))
    l3_memory = {"name": "Memory Persistence", "level": 3, "score": min(1.0, mem_change * 10), "passed": mem_change > 0.001, "details": {"memory_change": mem_change}}
    results.append(l3_memory)
    print(f"    [{'PASS' if l3_memory['passed'] else 'FAIL'}] mem_change={mem_change:.6f}")

    # Level 4: Safety Priority (simulated with high-magnitude input)
    print("\n  Testing L4: Safety Priority...")
    normal_inp = jax.random.normal(jax.random.PRNGKey(400), (1, obs_dim))
    emergency_inp = jax.random.normal(jax.random.PRNGKey(401), (1, obs_dim)) * 10  # Larger magnitude

    state1 = init_state()
    state2 = init_state()
    out_normal, _ = run_inference(normal_inp, state1)
    out_emergency, _ = run_inference(emergency_inp, state2)

    response_diff = float(jnp.linalg.norm(out_emergency - out_normal))
    l4_safety = {"name": "Safety Priority", "level": 4, "score": min(1.0, response_diff / 2), "passed": response_diff > 0.5, "details": {"response_diff": response_diff}}
    results.append(l4_safety)
    print(f"    [{'PASS' if l4_safety['passed'] else 'FAIL'}] response_diff={response_diff:.4f}")

    # Level 5: Self-Monitoring (state bounds check)
    print("\n  Testing L5: Self-Monitoring...")
    state = init_state()
    max_state_values = []
    for i in range(20):
        inp = jax.random.normal(jax.random.PRNGKey(i + 500), (1, obs_dim)) * 5
        _, state = run_inference(inp, state)
        max_state_values.append(float(jnp.max(jnp.abs(state['s']))))

    max_val = max(max_state_values)
    bounded = max_val < config.state_saturation_limit * 1.1
    l5_monitor = {"name": "Self-Monitoring", "level": 5, "score": 1.0 if bounded else 0.5, "passed": bounded, "details": {"max_state_value": max_val, "saturation_limit": config.state_saturation_limit}}
    results.append(l5_monitor)
    print(f"    [{'PASS' if l5_monitor['passed'] else 'FAIL'}] max_state={max_val:.4f}")

    benchmark_time = time.time() - start_time

    # Calculate summary
    passed_count = sum(1 for r in results if r['passed'])
    overall_score = sum(r['score'] for r in results) / len(results)

    level_summaries = {}
    for level in range(1, 6):
        level_results = [r for r in results if r['level'] == level]
        if level_results:
            level_summaries[f"Level_{level}"] = {
                "passed": sum(1 for r in level_results if r['passed']),
                "total": len(level_results),
                "avg_score": sum(r['score'] for r in level_results) / len(level_results),
            }

    print(f"\n" + "-" * 50)
    print(f"Benchmark completed in {benchmark_time:.2f} seconds")
    print(f"Overall: {passed_count}/{len(results)} tests passed, score={overall_score:.4f}")

    return {
        "benchmark_time_seconds": benchmark_time,
        "model_params": param_count,
        "overall_score": overall_score,
        "passed_count": passed_count,
        "total_tests": len(results),
        "level_summaries": level_summaries,
        "test_results": results,
    }


def run_inference_test():
    """Run inference test to verify the model works."""
    print("\n" + "=" * 70)
    print("PHASE 3: INFERENCE TEST")
    print("=" * 70)

    import jax
    import jax.numpy as jnp
    from continuonbrain.jax_models.core_model import make_core_model
    from continuonbrain.jax_models.config_presets import get_config_for_preset

    config = get_config_for_preset("pi5")
    obs_dim = 128
    action_dim = 32
    output_dim = 32
    batch_size = 4

    print(f"\nModel config: d_s={config.d_s}, d_w={config.d_w}, d_p={config.d_p}")
    print(f"Input dims: obs={obs_dim}, action={action_dim}, output={output_dim}")

    # Create model
    rng_key = jax.random.PRNGKey(42)
    model, params = make_core_model(rng_key, obs_dim, action_dim, output_dim, config)

    # Initialize states
    fast_state = jnp.zeros((batch_size, config.d_s))
    wave_state = jnp.zeros((batch_size, config.d_w))
    particle_state = jnp.zeros((batch_size, config.d_p))
    cms_memories = [
        jnp.zeros((batch_size, size, dim))
        for size, dim in zip(config.cms_sizes, config.cms_dims)
    ]
    cms_keys = [
        jnp.zeros((batch_size, size, config.d_k))
        for size in config.cms_sizes
    ]

    # Create test input
    test_obs = jax.random.normal(rng_key, (batch_size, obs_dim))
    test_action = jax.random.normal(rng_key, (batch_size, action_dim))
    test_reward = jnp.zeros((batch_size, 1))

    print("\nRunning inference...")

    # Run multiple inference iterations
    latencies = []
    for i in range(10):
        start_time = time.time()
        output, info = model.apply(
            params,
            test_obs,
            test_action,
            test_reward,  # reward parameter
            fast_state,
            wave_state,
            particle_state,
            cms_memories,
            cms_keys,
        )
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)

        if i == 0:
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {float(jnp.mean(output)):.6f}")
            print(f"  Output std: {float(jnp.std(output)):.6f}")

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\nInference latency (10 runs):")
    print(f"  Average: {avg_latency:.2f} ms")
    print(f"  Min: {min_latency:.2f} ms")
    print(f"  Max: {max_latency:.2f} ms")

    return {
        "output_shape": list(output.shape),
        "output_mean": float(jnp.mean(output)),
        "output_std": float(jnp.std(output)),
        "latency_avg_ms": avg_latency,
        "latency_min_ms": min_latency,
        "latency_max_ms": max_latency,
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("END-TO-END TRAINING AND BENCHMARK")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 70)

    total_start = time.time()
    results = {
        "timestamp": datetime.now().isoformat(),
        "training": None,
        "inference": None,
        "benchmark": None,
        "total_time_seconds": 0,
    }

    # Phase 1: Training
    try:
        results["training"] = run_training()
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        results["training"] = {"error": str(e)}

    # Phase 2: Inference Test
    try:
        results["inference"] = run_inference_test()
    except Exception as e:
        print(f"\nInference test failed: {e}")
        import traceback
        traceback.print_exc()
        results["inference"] = {"error": str(e)}

    # Phase 3: Progressive Benchmark
    try:
        results["benchmark"] = run_progressive_benchmark()
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        results["benchmark"] = {"error": str(e)}

    # Summary
    total_time = time.time() - total_start
    results["total_time_seconds"] = total_time

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Training status: {results['training'].get('status', 'N/A')}")

    if results["inference"] and not results["inference"].get("error"):
        lat = results['inference'].get('latency_avg_ms')
        if lat is not None:
            print(f"Inference latency: {lat:.2f} ms avg")

    if results["benchmark"] and not results["benchmark"].get("error"):
        score = results['benchmark'].get('overall_score')
        passed = results['benchmark'].get('passed_count', 0)
        total = results['benchmark'].get('total_tests', 0)
        if score is not None:
            print(f"Benchmark: {passed}/{total} tests passed, score={score:.4f}")

    # Save results
    output_path = project_root / "e2e_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("E2E BENCHMARK COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
