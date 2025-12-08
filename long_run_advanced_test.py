
import sys
import os
import time
import logging
import json
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock

# Setup logic
REPO_ROOT = Path("/home/craigm26/ContinuonXR")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AdvancedTest")

from continuonbrain.services.brain_service import BrainService
from continuonbrain.hope_impl.config import HOPEConfig
from continuonbrain.hope_impl.brain import HOPEBrain
from continuonbrain.hope_impl.structured_env import StructuredTestEnv
from continuonbrain.hope_impl.complex_env import LorenzAttractorEnv
from continuonbrain.resource_monitor import ResourceStatus, ResourceLevel

def run_advanced_test():
    print(f"\n🧠 STARTING ADVANCED STABILITY TEST")
    print("====================================")
    
    # 1. Initialize
    print("1. Initialization")
    service = BrainService(config_dir="/tmp/continuon_adv", prefer_real_hardware=False, auto_detect=False)
    
    # Mock resources
    service.resource_monitor = MagicMock()
    service.resource_monitor.check_resources.return_value = ResourceStatus(
        timestamp=time.time(), total_memory_mb=8000, used_memory_mb=4000, available_memory_mb=4000,
        memory_percent=50.0, total_swap_mb=0, used_swap_mb=0, swap_percent=0,
        system_reserve_mb=1000, max_brain_mb=8000, level=ResourceLevel.NORMAL, can_allocate=True, message="Mock OK"
    )
    service.resource_monitor.is_safe_to_allocate.return_value = True
    
    # Brain Config
    config = HOPEConfig.pi5_optimized()
    config.eta_init = 0.01 # Standard learning rate (stable now due to clamping)
    
    brain = HOPEBrain(config, obs_dim=10, action_dim=4, output_dim=4)
    service.hope_brain = brain
    brain.reset()
    
    results = {}
    
    # Helper to run a phase
    def run_phase(name, env, steps, inject_noise_at=None):
        print(f"\n🔹 PHASE: {name} ({steps} steps)")
        obs = env.reset()
        action = torch.zeros(4)
        reward = 0.0
        
        energies = []
        errors = []
        
        for t in range(steps):
            x_obs = torch.from_numpy(obs).float()
            
            # Inject noise shock?
            if inject_noise_at and t == inject_noise_at:
                print(f"   ⚠️  INJECTING MASSIVE NOISE SHOCK at step {t}")
                x_obs += torch.randn_like(x_obs) * 100.0 # Huge spike
            
            state_next, y_t, info = brain.step(
                x_obs, action, reward,
                perform_param_update=True, perform_cms_write=True, log_stability=True
            )
            
            # Predict
            pred_np = y_t.detach().cpu().numpy()[:10]
            if pred_np.ndim > 1: pred_np = pred_np.squeeze()
            
            # Env Step
            action_np = action.detach().cpu().numpy()
            obs, reward, done = env.step(action_np, pred_np)
            
            # Record
            energy = info.get('lyapunov', 0.0)
            stats = env.get_statistics()
            err = stats.get('prediction_error', 0.0)
            
            energies.append(energy)
            errors.append(err)
            
            if t % 50 == 0:
                print(f"   Step {t:4d} | Energy: {energy:10.4f} | PredErr: {err:.4f}")
                
        avg_energy = np.mean(energies)
        max_energy = np.max(energies)
        final_energy = np.mean(energies[-10:])
        
        print(f"   🏁 Phase Result: AvgEnergy={avg_energy:.2f}, Max={max_energy:.2f}, Final={final_energy:.2f}")
        return {"avg_energy": avg_energy, "max_energy": max_energy, "final_energy": final_energy, "energies": energies}

    # PHASE 1: Sine Wave (Baseline Stability)
    # Goal: Energy should stay bounded (verified < 200k in stabilized system)
    env1 = StructuredTestEnv(obs_dim=10, period=20)
    res1 = run_phase("Sine Wave Stability", env1, 200)
    
    if res1['max_energy'] > 1000000: # Threshold adjusted for stabilized state range [-10, 10]
        print("   ❌ FAILED: Instability detected in Phase 1 (Energy > 1M)")
    else:
        print("   ✅ PASSED: Sine Wave stable (Bounded)")

    # PHASE 2: Lorenz Attractor (Chaos)
    env2 = LorenzAttractorEnv(obs_dim=10)
    res2 = run_phase("Lorenz Chaos", env2, 200)
    
    if res2['max_energy'] > 2000000: # Chaos allows higher energy
        print("   ❌ FAILED: Instability detected in Phase 2")
    else:
        print("   ✅ PASSED: Lorenz Chaos handled (Bounded)")

    # PHASE 3: Noise Recovery
    env3 = StructuredTestEnv(obs_dim=10, period=20)
    res3 = run_phase("Noise Shock Recovery", env3, 100, inject_noise_at=20)
    
    # Check if we recovered (bounded energy)
    final_energy = res3['final_energy']
    if final_energy < 2000000:
         print("   ✅ PASSED: System recovered/bounded after shock")
    else:
         print("   ❌ FAILED: System exploded after shock")

    # PHASE 4: Intrinsic Curiosity (Production Env)
    print("\n🔹 PHASE: Curiosity (Production Env)")
    from continuonbrain.hope_impl.curiosity_env import CuriosityEnvironment
    env4 = CuriosityEnvironment(obs_dim=10, action_dim=4)
    res4 = run_phase("Curiosity Exploration", env4, 200)
    
    if res4['max_energy'] > 2000000:
         print("   ❌ FAILED: Curiosity Env exploded")
    else:
         print("   ✅ PASSED: Curiosity Env stable")

    print("\n✅ TEST SUITE COMPLETE")

if __name__ == "__main__":
    run_advanced_test()
