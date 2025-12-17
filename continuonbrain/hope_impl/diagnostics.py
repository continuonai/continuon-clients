"""
HOPE Brain Diagnostics

Tools to verify training health and parameter updates.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


def verify_parameter_updates(brain, num_steps: int = 100) -> Dict[str, any]:
    """
    Verify that parameters actually change during training.
    
    Args:
        brain: HOPEBrain instance
        num_steps: Number of training steps to run
    
    Returns:
        Dict with verification results
    """
    print(f"🔍 Verifying parameter updates over {num_steps} steps...")
    
    # Capture initial parameters
    params_before = {
        name: param.clone().detach()
        for name, param in brain.named_parameters()
    }
    
    # Run training steps
    brain.reset()
    obs_dim = brain.obs_dim
    action_dim = brain.action_dim
    
    for step in range(num_steps):
        x_obs = torch.randn(obs_dim)
        a_prev = torch.randn(action_dim)
        r_t = np.random.randn()
        
        # Perform step with parameter updates enabled
        state, output, info = brain.step(
            x_obs, a_prev, r_t,
            perform_param_update=True,
            log_stability=True
        )
    
    # Capture final parameters
    params_after = {
        name: param.clone().detach()
        for name, param in brain.named_parameters()
    }
    
    # Check for changes
    changed_params = []
    unchanged_params = []
    total_change = 0.0
    
    for name in params_before:
        if name in params_after:
            diff = torch.norm(params_after[name] - params_before[name]).item()
            total_change += diff
            
            if diff > 1e-6:
                changed_params.append((name, diff))
            else:
                unchanged_params.append(name)
    
    success = len(changed_params) > 0
    
    results = {
        'success': success,
        'num_changed': len(changed_params),
        'num_unchanged': len(unchanged_params),
        'total_change': total_change,
        'changed_params': changed_params[:10],  # Top 10
        'message': '✅ Parameters are updating!' if success else '❌ Parameters NOT updating!'
    }
    
    print(f"  {results['message']}")
    print(f"  Changed: {len(changed_params)} params")
    print(f"  Unchanged: {len(unchanged_params)} params")
    print(f"  Total change magnitude: {total_change:.6f}")
    
    return results


def check_cms_memory_retention(brain) -> Dict[str, any]:
    """
    Verify CMS memory is being stored and retained.
    
    Args:
        brain: HOPEBrain instance
    
    Returns:
        Dict with verification results
    """
    print("🔍 Checking CMS memory retention...")
    
    brain.reset()
    state = brain.get_state()
    
    # Check initial memory (should be near zero)
    initial_norms = []
    for level in state.cms.levels:
        norm = torch.norm(level.M).item()
        initial_norms.append(norm)
    
    # Run some steps
    obs_dim = brain.obs_dim
    action_dim = brain.action_dim
    
    for _ in range(50):
        x_obs = torch.randn(obs_dim)
        a_prev = torch.randn(action_dim)
        r_t = np.random.randn()
        
        brain.step(x_obs, a_prev, r_t, perform_cms_write=True)
    
    # Check final memory (should have increased)
    state = brain.get_state()
    final_norms = []
    for level in state.cms.levels:
        norm = torch.norm(level.M).item()
        final_norms.append(norm)
    
    # Verify memory increased
    increases = [final > initial for initial, final in zip(initial_norms, final_norms)]
    success = any(increases)
    
    results = {
        'success': success,
        'initial_norms': initial_norms,
        'final_norms': final_norms,
        'increases': increases,
        'message': '✅ CMS memory is being stored!' if success else '❌ CMS memory NOT being stored!'
    }
    
    print(f"  {results['message']}")
    for i, (init, final, increased) in enumerate(zip(initial_norms, final_norms, increases)):
        status = '✓' if increased else '✗'
        print(f"  Level {i}: {init:.4f} → {final:.4f} {status}")
    
    return results


def validate_stability_metrics(brain, num_steps: int = 100) -> Dict[str, any]:
    """
    Check Lyapunov energy and gradient stability.
    
    Args:
        brain: HOPEBrain instance
        num_steps: Number of steps to monitor
    
    Returns:
        Dict with validation results
    """
    print(f"🔍 Validating stability metrics over {num_steps} steps...")
    
    brain.reset()
    brain.stability_monitor.reset()
    
    obs_dim = brain.obs_dim
    action_dim = brain.action_dim
    
    has_nan = False
    has_inf = False
    max_lyapunov = 0.0
    
    for step in range(num_steps):
        x_obs = torch.randn(obs_dim)
        a_prev = torch.randn(action_dim)
        r_t = np.random.randn()
        
        state, output, info = brain.step(
            x_obs, a_prev, r_t,
            perform_param_update=True,
            log_stability=True
        )
        
        # Check for NaN/Inf
        if torch.isnan(state.fast_state.s).any():
            has_nan = True
        if torch.isinf(state.fast_state.s).any():
            has_inf = True
        
        max_lyapunov = max(max_lyapunov, info['lyapunov'])
    
    metrics = brain.stability_monitor.get_metrics()
    is_stable = brain.stability_monitor.is_stable()
    
    success = is_stable and not has_nan and not has_inf
    
    results = {
        'success': success,
        'is_stable': is_stable,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'max_lyapunov': max_lyapunov,
        'final_metrics': metrics,
        'message': '✅ System is stable!' if success else '❌ Stability issues detected!'
    }
    
    print(f"  {results['message']}")
    print(f"  Stable: {is_stable}")
    print(f"  NaN detected: {has_nan}")
    print(f"  Inf detected: {has_inf}")
    print(f"  Max Lyapunov: {max_lyapunov:.2f}")
    print(f"  Dissipation rate: {metrics.get('dissipation_rate', 0):.6f}")
    
    return results


def run_full_diagnostic(brain) -> Dict[str, any]:
    """
    Run complete diagnostic suite.
    
    Args:
        brain: HOPEBrain instance
    
    Returns:
        Dict with all diagnostic results
    """
    print("=" * 70)
    print("HOPE Brain Health Diagnostic")
    print("=" * 70)
    print()
    
    results = {}
    
    # 1. Parameter updates
    results['parameter_updates'] = verify_parameter_updates(brain, num_steps=100)
    print()
    
    # 2. CMS memory retention
    results['cms_memory'] = check_cms_memory_retention(brain)
    print()
    
    # 3. Stability metrics
    results['stability'] = validate_stability_metrics(brain, num_steps=100)
    print()
    
    # Overall status
    all_success = all(r['success'] for r in results.values())
    
    print("=" * 70)
    if all_success:
        print("✅ ALL DIAGNOSTICS PASSED - System is healthy!")
    else:
        print("❌ SOME DIAGNOSTICS FAILED - Review issues above")
    print("=" * 70)
    
    results['overall_success'] = all_success
    
    return results


if __name__ == "__main__":
    # Example usage
    from hope_impl.config import HOPEConfig
    from hope_impl.brain import HOPEBrain
    
    print("Creating HOPE brain for diagnostics...")
    config = HOPEConfig.pi5_optimized()
    brain = HOPEBrain(
        config=config,
        obs_dim=10,
        action_dim=4,
        output_dim=4,
    )
    
    # Run diagnostics
    results = run_full_diagnostic(brain)
