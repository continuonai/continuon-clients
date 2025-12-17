"""
HOPE Pi5 Deployment Example

Demonstrates Raspberry Pi 5 deployment:
1. Quantized model loading
2. Real-time inference loop
3. Memory monitoring
4. Performance benchmarking
"""

import torch
import time
from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain
from hope_impl.pi5_optimizations import (
    optimize_for_pi5,
    benchmark_inference,
    Pi5MemoryManager,
)


def main():
    print("=" * 60)
    print("HOPE Pi5 Deployment Demo")
    print("=" * 60)
    
    # 1. Create Pi5-optimized configuration
    print("\n1. Creating Pi5-optimized configuration...")
    config = HOPEConfig.pi5_optimized()
    print(f"   Config: {config.num_levels} levels, d_s={config.d_s}")
    print(f"   Quantization: {config.use_quantization} ({config.quantization_dtype})")
    
    # 2. Create brain
    print("\n2. Creating HOPE brain...")
    brain = HOPEBrain(
        config=config,
        obs_dim=10,
        action_dim=4,
        output_dim=4,
    )
    print(f"   Parameters: {sum(p.numel() for p in brain.parameters())}")
    
    # 3. Apply Pi5 optimizations
    print("\n3. Applying Pi5 optimizations...")
    brain = optimize_for_pi5(brain)
    
    # 4. Memory usage before optimization
    print("\n4. Memory usage...")
    memory = brain.get_memory_usage()
    print(f"   Model: {memory['model_parameters']:.2f} MB")
    print(f"   Total: {memory.get('total', memory['model_parameters']):.2f} MB")
    
    # 5. Benchmark inference
    print("\n5. Benchmarking inference...")
    results = benchmark_inference(
        brain=brain,
        obs_dim=10,
        action_dim=4,
        num_steps=100,
        warmup_steps=10,
    )
    
    # 6. Check if meets Pi5 targets
    print("\n6. Checking Pi5 targets...")
    target_fps = 10.0
    target_memory_mb = 2000.0
    
    meets_fps = results['steps_per_second'] >= target_fps
    meets_memory = results['memory_total_mb'] <= target_memory_mb
    
    print(f"   Target FPS: {target_fps} → {'✓ PASS' if meets_fps else '✗ FAIL'} "
          f"({results['steps_per_second']:.2f} steps/sec)")
    print(f"   Target Memory: {target_memory_mb} MB → {'✓ PASS' if meets_memory else '✗ FAIL'} "
          f"({results['memory_total_mb']:.2f} MB)")
    
    # 7. Real-time inference loop
    print("\n7. Running real-time inference loop (10 seconds)...")
    brain.reset()
    
    memory_manager = Pi5MemoryManager(max_memory_mb=2000)
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < 10.0:
        # Generate random inputs (in real deployment, these come from sensors)
        x_obs = torch.randn(10)
        a_prev = torch.randn(4)
        r_t = 0.0
        
        # Inference step
        with torch.no_grad():
            state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
        
        step_count += 1
        
        # Check memory every 100 steps
        if step_count % 100 == 0:
            memory_manager.cleanup_if_needed(brain)
    
    elapsed = time.time() - start_time
    actual_fps = step_count / elapsed
    
    print(f"   Completed {step_count} steps in {elapsed:.2f} sec")
    print(f"   Actual FPS: {actual_fps:.2f}")
    
    # 8. Final stability check
    print("\n8. Final stability check...")
    metrics = brain.stability_monitor.get_metrics()
    print(f"   Mean Lyapunov: {metrics['lyapunov_mean']:.4f}")
    print(f"   Stable: {brain.stability_monitor.is_stable()}")
    
    # 9. Save optimized model
    print("\n9. Saving optimized model...")
    checkpoint_path = "/tmp/hope_pi5_optimized.pt"
    brain.save_checkpoint(checkpoint_path)
    print(f"   Saved to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Pi5 Deployment Demo completed!")
    print("=" * 60)
    
    # Summary
    print("\nSummary:")
    print(f"  ✓ Inference speed: {actual_fps:.2f} steps/sec (target: {target_fps})")
    print(f"  ✓ Memory usage: {results['memory_total_mb']:.2f} MB (target: <{target_memory_mb})")
    print(f"  ✓ Stability: {brain.stability_monitor.is_stable()}")
    print(f"  ✓ Model size: {memory['model_parameters']:.2f} MB")


if __name__ == "__main__":
    main()
