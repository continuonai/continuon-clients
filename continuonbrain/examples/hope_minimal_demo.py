"""
Minimal HOPE Demo

Demonstrates basic HOPE brain usage:
1. Brain initialization
2. Single step execution
3. Multi-step rollout
4. Checkpoint save/load
"""

import torch
from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain


def main():
    print("=" * 60)
    print("HOPE Minimal Demo")
    print("=" * 60)
    
    # 1. Create configuration
    print("\n1. Creating HOPE configuration...")
    config = HOPEConfig.development()  # Small config for demo
    print(f"   Config: {config.num_levels} levels, d_s={config.d_s}")
    
    # 2. Create brain
    print("\n2. Creating HOPE brain...")
    brain = HOPEBrain(
        config=config,
        obs_dim=10,  # Example: 10D observation
        action_dim=4,  # Example: 4D action
        output_dim=4,  # Example: 4D output
        obs_type="vector",
        output_type="continuous",
    )
    print(f"   Brain created with {sum(p.numel() for p in brain.parameters())} parameters")
    
    # 3. Initialize brain state
    print("\n3. Initializing brain state...")
    state = brain.reset()
    print(f"   Fast state shape: s={state.fast_state.s.shape}")
    print(f"   CMS levels: {state.cms.num_levels}")
    
    # 4. Single step execution
    print("\n4. Executing single step...")
    x_obs = torch.randn(10)
    a_prev = torch.zeros(4)
    r_t = 0.0
    
    state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
    print(f"   Output shape: {y_t.shape}")
    print(f"   Lyapunov energy: {info['lyapunov']:.4f}")
    
    # 5. Multi-step rollout
    print("\n5. Running 100-step rollout...")
    for i in range(100):
        x_obs = torch.randn(10)
        a_prev = y_t  # Use previous output as action
        r_t = torch.randn(1).item()  # Random reward
        
        state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
        
        if (i + 1) % 20 == 0:
            print(f"   Step {i+1}: Lyapunov={info['lyapunov']:.4f}, "
                  f"State norm={info['stability_metrics']['state_norm']:.4f}")
    
    # 6. Check stability
    print("\n6. Checking stability...")
    metrics = brain.stability_monitor.get_metrics()
    print(f"   Mean Lyapunov: {metrics['lyapunov_mean']:.4f}")
    print(f"   Stable: {brain.stability_monitor.is_stable()}")
    
    # 7. Memory usage
    print("\n7. Memory usage...")
    memory = brain.get_memory_usage()
    print(f"   Model parameters: {memory['model_parameters']:.2f} MB")
    print(f"   Total state: {memory['total_state']:.2f} MB")
    print(f"   Total: {memory['total']:.2f} MB")
    
    # 8. Save checkpoint
    print("\n8. Saving checkpoint...")
    checkpoint_path = "/tmp/hope_demo.pt"
    brain.save_checkpoint(checkpoint_path)
    print(f"   Saved to {checkpoint_path}")
    
    # 9. Load checkpoint
    print("\n9. Loading checkpoint...")
    brain_loaded = HOPEBrain.load_checkpoint(checkpoint_path)
    print(f"   Loaded successfully")
    
    # 10. Verify loaded brain works
    print("\n10. Testing loaded brain...")
    x_obs = torch.randn(10)
    a_prev = torch.zeros(4)
    r_t = 0.0
    
    state_next, y_t, info = brain_loaded.step(x_obs, a_prev, r_t)
    print(f"   Output shape: {y_t.shape}")
    print(f"   Lyapunov energy: {info['lyapunov']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
