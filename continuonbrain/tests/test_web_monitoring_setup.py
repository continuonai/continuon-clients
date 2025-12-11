"""
Test Web Monitoring Setup

Integration test for HOPE web monitoring interface.
Verifies that monitoring pages work correctly with a live HOPE brain.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain

# Skip unless HOPE tests are explicitly enabled
if os.environ.get("CONTINUON_ENABLE_HOPE_TESTS", "0").lower() not in ("1", "true", "yes"):
    import pytest
    pytest.skip(
        "HOPE monitoring test disabled (set CONTINUON_ENABLE_HOPE_TESTS=1 to run)",
        allow_module_level=True,
    )


def test_web_monitoring_setup():
    """
    Test that web monitoring integration works.
    
    This script:
    1. Creates a HOPE brain
    2. Registers it with the monitoring API
    3. Runs a training loop
    4. Verifies metrics are collected
    
    Run this alongside the web server:
        Terminal 1: python api/server.py --port 8080
        Terminal 2: python tests/test_web_monitoring_setup.py
        Browser: http://localhost:8080/ui/hope/training
    """
    
    print("=" * 70)
    print("HOPE Web Monitoring Integration Test")
    print("=" * 70)
    
    # 1. Create HOPE brain
    print("\n[1/5] Creating HOPE brain...")
    config = HOPEConfig.development()
    brain = HOPEBrain(
        config=config,
        obs_dim=10,
        action_dim=4,
        output_dim=4,
    )
    print(f"   ✓ Brain created: {sum(p.numel() for p in brain.parameters())} parameters")
    
    # 2. Register with monitoring API
    print("\n[2/5] Registering brain with monitoring API...")
    try:
        # Test setting brain
        from continuonbrain.api.routes import hope_routes
        hope_routes.set_hope_brain(brain)
        print("   ✓ Brain registered with hope_routes")
    except ImportError as e:
        print(f"   ✗ Failed to import hope_routes: {e}")
        print("   Note: This is expected if running outside the server context")
        print("   The monitoring API will be available when the server is running")
    
    # 3. Initialize brain
    print("\n[3/5] Initializing brain state...")
    brain.reset()
    print("   ✓ State initialized")
    print(f"   - Fast state shape: s={brain.get_state().fast_state.s.shape}")
    print(f"   - CMS levels: {brain.get_state().cms.num_levels}")
    
    # 4. Run training loop
    print("\n[4/5] Running training loop (100 steps)...")
    print("   If web server is running, navigate to:")
    print("   - http://localhost:8080/ui/hope/training")
    print("   - http://localhost:8080/ui/hope/memory")
    print("   - http://localhost:8080/ui/hope/stability")
    print()
    
    for step in range(100):
        # Generate random inputs
        x_obs = torch.randn(10)
        a_prev = torch.randn(4)
        r_t = torch.randn(1).item()
        
        # Execute brain step
        state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
        
        # Print progress
        if (step + 1) % 20 == 0:
            print(f"   Step {step+1:3d}: "
                  f"V={info['lyapunov']:8.2f}, "
                  f"||s||={info['stability_metrics']['state_norm']:6.2f}, "
                  f"stable={brain.stability_monitor.is_stable()}")
        
        # Small delay to simulate real-time
        time.sleep(0.05)  # 20 Hz
    
    # 5. Verify metrics collection
    print("\n[5/5] Verifying metrics collection...")
    metrics = brain.stability_monitor.get_metrics()
    
    checks = {
        "Steps recorded": metrics['steps'] == 100,
        "Lyapunov computed": metrics.get('lyapunov_current', 0) > 0,
        "State norm tracked": metrics.get('state_norm', 0) > 0,
        "System stable": brain.stability_monitor.is_stable(),
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {check}")
    
    # Memory usage
    memory = brain.get_memory_usage()
    print(f"\n   Memory usage: {memory['total']:.2f} MB")
    
    # Final summary
    print("\n" + "=" * 70)
    if all(checks.values()):
        print("✓ All checks passed! Web monitoring integration is working.")
    else:
        print("✗ Some checks failed. Review the output above.")
    print("=" * 70)
    
    return all(checks.values())


def test_api_endpoints():
    """
    Test API endpoints directly (requires server to be running).
    """
    print("\n" + "=" * 70)
    print("Testing API Endpoints")
    print("=" * 70)
    print("\nNote: This requires the web server to be running.")
    print("Start with: python api/server.py --port 8080\n")
    
    import requests
    
    base_url = "http://localhost:8080"
    endpoints = [
        "/api/hope/metrics",
        "/api/hope/stability",
        "/api/hope/config",
        "/api/hope/cms/level/0",
        "/api/hope/history?window=10",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(base_url + endpoint, timeout=2)
            if response.status_code == 200:
                print(f"   ✓ {endpoint}")
            else:
                print(f"   ✗ {endpoint} (status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"   ⚠ {endpoint} (server not running)")
        except Exception as e:
            print(f"   ✗ {endpoint} (error: {e})")


if __name__ == "__main__":
    # Run main test
    success = test_web_monitoring_setup()
    
    # Optionally test API endpoints
    print("\n")
    try:
        import requests
        test_api_endpoints()
    except ImportError:
        print("Skipping API endpoint tests (requests library not installed)")
        print("Install with: pip install requests")
    
    sys.exit(0 if success else 1)
