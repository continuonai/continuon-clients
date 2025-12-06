"""
HOPE Brain Monitoring Integration Demo

Demonstrates how to integrate a HOPE brain with the web monitoring interface.
"""

import torch
import time
from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain

# Import API routes for integration
try:
    from api.routes import hope_routes
except ImportError:
    print("Warning: API routes not available. Run from continuonbrain directory.")
    hope_routes = None


def demo_hope_monitoring():
    """
    Demo: Create HOPE brain and run it while monitoring via web interface.
    
    To use:
    1. Start the web server: python api/server.py --port 8080
    2. Run this script in a separate terminal
    3. Navigate to http://localhost:8080/ui/hope/training
    """
    
    print("=" * 60)
    print("HOPE Brain Monitoring Integration Demo")
    print("=" * 60)
    
    # 1. Create HOPE brain
    print("\n1. Creating HOPE brain...")
    config = HOPEConfig.development()
    brain = HOPEBrain(
        config=config,
        obs_dim=10,
        action_dim=4,
        output_dim=4,
    )
    print(f"   Brain created with {sum(p.numel() for p in brain.parameters())} parameters")
    
    # 2. Register brain with monitoring API
    if hope_routes:
        print("\n2. Registering brain with monitoring API...")
        hope_routes.set_hope_brain(brain)
        print("   ✓ Brain registered")
    else:
        print("\n2. Skipping API registration (not available)")
    
    # 3. Initialize brain
    print("\n3. Initializing brain state...")
    brain.reset()
    print("   ✓ State initialized")
    
    # 4. Run training loop
    print("\n4. Running training loop...")
    print("   Navigate to http://localhost:8080/ui/hope/training to monitor")
    print("   Press Ctrl+C to stop\n")
    
    try:
        step = 0
        while True:
            # Generate random inputs
            x_obs = torch.randn(10)
            a_prev = torch.randn(4)
            r_t = torch.randn(1).item()
            
            # Execute brain step
            state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
            
            step += 1
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"   Step {step:4d}: "
                      f"Lyapunov={info['lyapunov']:.2f}, "
                      f"State norm={info['stability_metrics']['state_norm']:.2f}")
            
            # Sleep to simulate real-time processing
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        print("\n\n5. Stopping training loop...")
    
    # 5. Final stats
    print("\n6. Final statistics:")
    metrics = brain.stability_monitor.get_metrics()
    print(f"   Total steps: {metrics['steps']}")
    print(f"   Mean Lyapunov: {metrics['lyapunov_mean']:.2f}")
    print(f"   Stable: {brain.stability_monitor.is_stable()}")
    
    memory = brain.get_memory_usage()
    print(f"   Memory usage: {memory['total']:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_hope_monitoring()
