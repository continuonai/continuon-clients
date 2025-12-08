
import sys
import os
import torch

# Ensure repo root on path
sys.path.insert(0, os.getcwd())

from continuonbrain.hope_impl.brain import HOPEBrain, HOPEConfig

def verify_toggle():
    print("Verifying HOPEBrain toggle functionality...")
    
    # 1. Setup Config
    config = HOPEConfig(
        use_hybrid_mode=False,
        num_columns=1,
        device="cpu"
    )
    
    # 2. Instantiate Brain (Standard Mode)
    brain = HOPEBrain(config, obs_dim=10, action_dim=4, output_dim=4)
    print(f"Initial State: Columns={len(brain.columns)}, Hybrid={brain.config.use_hybrid_mode}")
    
    if len(brain.columns) != 1:
        print("FAIL: Expected 1 column initially")
        return False
        
    # 3. Toggle to Hybrid Mode (4 columns)
    print("Switching to Hybrid Mode (4 columns)...")
    try:
        brain.initialize(num_columns=4)
    except AttributeError:
        print("FAIL: initialize method not found!")
        return False
    except Exception as e:
        print(f"FAIL: initialize raised exception: {e}")
        return False
        
    print(f"New State: Columns={len(brain.columns)}, Hybrid={brain.config.use_hybrid_mode}")
    
    if len(brain.columns) != 4:
        print(f"FAIL: Expected 4 columns, got {len(brain.columns)}")
        return False
        
    if not brain.config.use_hybrid_mode:
        print("FAIL: Expected use_hybrid_mode=True")
        return False

    # 4. Toggle back to Standard Mode
    print("Switching back to Standard Mode (1 column)...")
    brain.initialize(num_columns=1)
    print(f"Final State: Columns={len(brain.columns)}, Hybrid={brain.config.use_hybrid_mode}")

    if len(brain.columns) != 1:
        print(f"FAIL: Expected 1 column, got {len(brain.columns)}")
        return False
        
    if brain.config.use_hybrid_mode:
        print("FAIL: Expected use_hybrid_mode=False")
        return False
        
    print("SUCCESS: HOPEBrain toggle verified.")
    return True

if __name__ == "__main__":
    success = verify_toggle()
    if not success:
        sys.exit(1)
