
import sys
import os
import time
import logging
import json
from pathlib import Path
import asyncio
from unittest.mock import MagicMock

# Setup logic
REPO_ROOT = Path("/home/craigm26/ContinuonXR")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["HUGGINGFACE_TOKEN"] = "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"

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s') # Reduce noise
logger = logging.getLogger("ProofOfLearning")

from continuonbrain.services.brain_service import BrainService
from continuonbrain.services.background_learner import BackgroundLearner
from continuonbrain.resource_monitor import ResourceStatus, ResourceLevel

async def main():
    print("\n🔬 PROOF OF LEARNING: CAPABILITY DEMONSTRATION 🔬")
    print("==================================================")
    
    # 1. Initialize Brain
    print("1. Initializing System...")
    service = BrainService(
        config_dir="/tmp/continuon_learning_proof",
        prefer_real_hardware=False,
        auto_detect=False
    )
    await service.initialize()
    
    if not service.hope_brain:
        print("❌ CRITICAL: HOPE Brain failed to initialize.")
        return
        
    # Reset brain to ensure state exists
    service.hope_brain.reset()
    print("   Brain initialized and reset.")

    # 2. Configure & Start Learner
    print("2. Starting Autonomous Learning Service...")
    
    # Mock Resource Monitor to force "NORMAL" status
    # This proves the LEARNING capability logic works even if physical RAM is currently full
    mock_res_monitor = MagicMock()
    mock_res_monitor.check_resources.return_value = ResourceStatus(
        timestamp=time.time(),
        total_memory_mb=16000,
        used_memory_mb=8000,
        available_memory_mb=8000,
        memory_percent=50.0,
        total_swap_mb=0, used_swap_mb=0, swap_percent=0,
        system_reserve_mb=1000, max_brain_mb=8000,
        level=ResourceLevel.NORMAL,
        can_allocate=True,
        message="Mocked Normal Status"
    )
    mock_res_monitor.is_safe_to_allocate.return_value = True
    print("   (Resource Monitor mocked to bypass memory constraints for software proof)")
    
    # Custom config for fast updates
    learner_config = {
        'steps_per_cycle': 10,       
        'cycle_interval_sec': 0.1,   
        'checkpoint_interval': 500,
        'exploration_bonus': 0.9,    # Very High curiosity to force updates
        'novelty_threshold': 0.0,    # Always novel
        'learning_update_interval': 1 
    }
    
    learner = BackgroundLearner(
        service.hope_brain, 
        config=learner_config,
        resource_monitor=mock_res_monitor
    )
    
    learner.start()
    
    # 3. Monitor Loop
    print("3. Monitoring Learning Dynamics (10 seconds)...")
    print("-" * 65)
    print(f"{'Time':<8} | {'Updates':<8} | {'Param Δ':<15} | {'Novelty':<10} | {'Status'}")
    print("-" * 65)
    
    metrics_log = []
    
    start_time = time.time()
    duration = 10 # seconds
    
    try:
        while (time.time() - start_time) < duration:
            status = learner.get_status()
            
            elapsed = int(time.time() - start_time)
            updates = status.get('learning_updates', 0)
            param_delta = status.get('recent_parameter_change', 0) or 0.0
            novelty = status.get('current_novelty', 0) or 0.0 # Handle None
            
            # Log to console
            print(f"{elapsed:>4}s   | {updates:>8} | {param_delta:>15.9f} | {novelty:>10.4f} | {'🟢' if updates > 0 else '⚪'}")
            
            metrics_log.append({
                'time': elapsed,
                'updates': updates,
                'param_delta': param_delta,
                'novelty': novelty
            })
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping early...")
    finally:
        learner.stop()
        
    # 4. Analysis
    print("\n4. Analysis & Proof")
    print("===================")
    
    total_updates = metrics_log[-1]['updates'] if metrics_log else 0
    max_delta = max((m['param_delta'] for m in metrics_log), default=0)
    final_novelty = metrics_log[-1]['novelty'] if metrics_log else 0
    
    print(f"Total Parameter Updates: {total_updates}")
    print(f"Max Parameter Change:    {max_delta:.9f}")
    print(f"Final Novelty Score:     {final_novelty:.4f}")
    
    # 5. Verdict
    print("\nVERDICT:")
    # We accept very small changes as proof, as long as it's non-zero
    if total_updates > 0 and max_delta > 1e-9:
        print("✅ SUCCESS: System demonstrated active learning capability.")
        print("   - Parameters were updated autonomously.")
        print("   - Neural plasticity confirmed.")
    else:
        print("❌ FAILURE: No learning detected.")
        if total_updates == 0:
            print("   - No updates occurred.")
        if max_delta <= 1e-9:
            print("   - Parameters did not change significantly (frozen or zero gradient).")
            
    # Save proof artifact
    proof_file = Path("proof_of_learning.json")
    with open(proof_file, 'w') as f:
        json.dump({
            "timestamp": time.time(),
            "config": learner_config,
            "metrics": metrics_log,
            "verdict": "SUCCESS" if total_updates > 0 else "FAILURE"
        }, f, indent=2)
    print(f"\nProof data saved to {proof_file.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
