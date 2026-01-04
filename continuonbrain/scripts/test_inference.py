#!/usr/bin/env python3
"""
Inference testing for ContinuonBrain WaveCore model.

Tests:
1. Output stability and consistency
2. Response to different input patterns
3. State evolution over sequences
4. Inference speed/latency
5. Action differentiation
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

import jax
import jax.numpy as jnp

from continuonbrain.jax_models.core_model import make_core_model
from continuonbrain.jax_models.config import CoreModelConfig
from continuonbrain.jax_models.config_presets import get_config_for_preset


def create_test_model(config_preset: str = "pi5"):
    """Create and initialize a test model."""
    config = get_config_for_preset(config_preset)
    key = jax.random.PRNGKey(42)
    model, params = make_core_model(
        key, obs_dim=128, action_dim=32, output_dim=32, config=config
    )
    return model, params, config


def create_initial_state(config: CoreModelConfig, batch_size: int = 1):
    """Create initial model state."""
    return {
        "s": jnp.zeros((batch_size, config.d_s)),
        "w": jnp.zeros((batch_size, config.d_w)),
        "p": jnp.zeros((batch_size, config.d_p)),
        "cms_memories": [
            jnp.zeros((batch_size, n, d))
            for n, d in zip(config.cms_sizes, config.cms_dims)
        ],
        "cms_keys": [
            jnp.zeros((batch_size, n, config.d_k))
            for n in config.cms_sizes
        ],
    }


def test_output_stability(model, params, config):
    """Test that same input produces same output (deterministic)."""
    print("\n=== Test 1: Output Stability ===")

    state = create_initial_state(config)
    obs = jnp.ones((1, 128)) * 0.5
    action = jnp.zeros((1, 32))
    reward = jnp.zeros((1, 1))

    # Run twice with same input
    out1, info1 = model.apply(
        params, obs, action, reward,
        state["s"], state["w"], state["p"],
        state["cms_memories"], state["cms_keys"]
    )

    out2, info2 = model.apply(
        params, obs, action, reward,
        state["s"], state["w"], state["p"],
        state["cms_memories"], state["cms_keys"]
    )

    diff = jnp.abs(out1 - out2).max()
    passed = diff < 1e-6

    print(f"  Max difference: {diff:.2e}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_input_differentiation(model, params, config):
    """Test that different inputs produce different outputs."""
    print("\n=== Test 2: Input Differentiation ===")

    state = create_initial_state(config)
    reward = jnp.zeros((1, 1))

    # Different observation patterns
    obs_forward = jnp.zeros((1, 128)).at[0, 0].set(1.0)  # "forward" signal
    obs_left = jnp.zeros((1, 128)).at[0, 1].set(1.0)     # "left" signal
    obs_stop = jnp.zeros((1, 128))                        # "stop" signal

    action = jnp.zeros((1, 32))

    out_forward, _ = model.apply(
        params, obs_forward, action, reward,
        state["s"], state["w"], state["p"],
        state["cms_memories"], state["cms_keys"]
    )

    out_left, _ = model.apply(
        params, obs_left, action, reward,
        state["s"], state["w"], state["p"],
        state["cms_memories"], state["cms_keys"]
    )

    out_stop, _ = model.apply(
        params, obs_stop, action, reward,
        state["s"], state["w"], state["p"],
        state["cms_memories"], state["cms_keys"]
    )

    # Outputs should be different
    diff_forward_left = jnp.abs(out_forward - out_left).mean()
    diff_forward_stop = jnp.abs(out_forward - out_stop).mean()
    diff_left_stop = jnp.abs(out_left - out_stop).mean()

    all_different = diff_forward_left > 0.01 and diff_forward_stop > 0.01 and diff_left_stop > 0.01

    print(f"  Forward vs Left diff: {diff_forward_left:.4f}")
    print(f"  Forward vs Stop diff: {diff_forward_stop:.4f}")
    print(f"  Left vs Stop diff: {diff_left_stop:.4f}")
    print(f"  Status: {'PASSED' if all_different else 'FAILED'}")

    return all_different


def test_state_evolution(model, params, config):
    """Test that state evolves over time (memory works)."""
    print("\n=== Test 3: State Evolution ===")

    state = create_initial_state(config)
    reward = jnp.zeros((1, 1))
    action = jnp.zeros((1, 32))

    states_history = []
    outputs_history = []

    # Run 10 steps with varying input
    for i in range(10):
        obs = jnp.sin(jnp.arange(128) * 0.1 + i * 0.5).reshape(1, -1)

        out, info = model.apply(
            params, obs, action, reward,
            state["s"], state["w"], state["p"],
            state["cms_memories"], state["cms_keys"]
        )

        # Update state
        state["s"] = info["fast_state"]
        state["w"] = info["wave_state"]
        state["p"] = info["particle_state"]
        state["cms_memories"] = info["cms_memories"]
        state["cms_keys"] = info["cms_keys"]

        states_history.append(float(jnp.abs(state["s"]).mean()))
        outputs_history.append(float(jnp.abs(out).mean()))

    # State should have non-zero values and evolve
    state_mean = np.mean(states_history)
    state_std = np.std(states_history)
    output_std = np.std(outputs_history)

    passed = state_mean > 0.001 and state_std > 0.0001

    print(f"  State mean activation: {state_mean:.6f}")
    print(f"  State std over time: {state_std:.6f}")
    print(f"  Output std over time: {output_std:.6f}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_inference_speed(model, params, config, num_iterations: int = 100):
    """Test inference latency."""
    print("\n=== Test 4: Inference Speed ===")

    state = create_initial_state(config)
    obs = jnp.ones((1, 128)) * 0.5
    action = jnp.zeros((1, 32))
    reward = jnp.zeros((1, 1))

    # Warmup
    for _ in range(5):
        _ = model.apply(
            params, obs, action, reward,
            state["s"], state["w"], state["p"],
            state["cms_memories"], state["cms_keys"]
        )

    # Timed iterations
    start = time.time()
    for _ in range(num_iterations):
        out, info = model.apply(
            params, obs, action, reward,
            state["s"], state["w"], state["p"],
            state["cms_memories"], state["cms_keys"]
        )
        # Update state
        state["s"] = info["fast_state"]
        state["w"] = info["wave_state"]
        state["p"] = info["particle_state"]

    elapsed = time.time() - start
    avg_ms = (elapsed / num_iterations) * 1000
    steps_per_sec = num_iterations / elapsed

    # Target: <20ms per step for real-time control
    passed = avg_ms < 20

    print(f"  Total time: {elapsed:.3f}s for {num_iterations} iterations")
    print(f"  Average latency: {avg_ms:.2f}ms")
    print(f"  Steps/second: {steps_per_sec:.1f}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'} (target <20ms)")

    return passed, avg_ms, steps_per_sec


def test_action_range(model, params, config):
    """Test that outputs are in reasonable range."""
    print("\n=== Test 5: Action Range ===")

    state = create_initial_state(config)
    reward = jnp.zeros((1, 1))
    action = jnp.zeros((1, 32))

    outputs = []
    for _ in range(20):
        obs = jax.random.normal(jax.random.PRNGKey(_), (1, 128))
        out, info = model.apply(
            params, obs, action, reward,
            state["s"], state["w"], state["p"],
            state["cms_memories"], state["cms_keys"]
        )
        outputs.append(out)
        state["s"] = info["fast_state"]
        state["w"] = info["wave_state"]
        state["p"] = info["particle_state"]

    all_outputs = jnp.concatenate(outputs, axis=0)
    min_val = float(all_outputs.min())
    max_val = float(all_outputs.max())
    mean_val = float(all_outputs.mean())
    std_val = float(all_outputs.std())

    # Outputs should be bounded and not exploding
    passed = abs(max_val) < 10 and abs(min_val) < 10

    print(f"  Output range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Output mean: {mean_val:.4f}")
    print(f"  Output std: {std_val:.4f}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def test_batch_consistency(model, params, config):
    """Test that batched inference matches sequential."""
    print("\n=== Test 6: Batch Consistency ===")

    # Single sample
    state1 = create_initial_state(config, batch_size=1)
    obs1 = jnp.ones((1, 128)) * 0.3
    action1 = jnp.zeros((1, 32))
    reward1 = jnp.zeros((1, 1))

    out1, _ = model.apply(
        params, obs1, action1, reward1,
        state1["s"], state1["w"], state1["p"],
        state1["cms_memories"], state1["cms_keys"]
    )

    # Batched (same input repeated)
    state4 = create_initial_state(config, batch_size=4)
    obs4 = jnp.ones((4, 128)) * 0.3
    action4 = jnp.zeros((4, 32))
    reward4 = jnp.zeros((4, 1))

    out4, _ = model.apply(
        params, obs4, action4, reward4,
        state4["s"], state4["w"], state4["p"],
        state4["cms_memories"], state4["cms_keys"]
    )

    # First element of batch should match single
    diff = jnp.abs(out1[0] - out4[0]).max()
    passed = diff < 1e-5

    print(f"  Single vs Batch[0] diff: {diff:.2e}")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def run_all_tests():
    """Run all inference tests."""
    print("=" * 60)
    print("CONTINUONBRAIN INFERENCE TESTS")
    print("=" * 60)

    print("\nInitializing model...")
    model, params, config = create_test_model("pi5")
    print(f"  Config: d_s={config.d_s}, d_w={config.d_w}, d_p={config.d_p}")

    results = {}

    results["output_stability"] = test_output_stability(model, params, config)
    results["input_differentiation"] = test_input_differentiation(model, params, config)
    results["state_evolution"] = test_state_evolution(model, params, config)

    passed, latency, throughput = test_inference_speed(model, params, config)
    results["inference_speed"] = passed
    results["latency_ms"] = latency
    results["throughput"] = throughput

    results["action_range"] = test_action_range(model, params, config)
    results["batch_consistency"] = test_batch_consistency(model, params, config)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    test_results = [
        ("Output Stability", results["output_stability"]),
        ("Input Differentiation", results["input_differentiation"]),
        ("State Evolution", results["state_evolution"]),
        ("Inference Speed", results["inference_speed"]),
        ("Action Range", results["action_range"]),
        ("Batch Consistency", results["batch_consistency"]),
    ]

    passed_count = sum(1 for _, p in test_results if p)
    total_count = len(test_results)

    for name, passed in test_results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    print()
    print(f"Tests passed: {passed_count}/{total_count}")
    print(f"Latency: {results['latency_ms']:.2f}ms")
    print(f"Throughput: {results['throughput']:.1f} steps/sec")

    return results


if __name__ == "__main__":
    results = run_all_tests()
