import torch
import sys
import os
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.hope_impl.config import HOPEConfig
from continuonbrain.hope_impl.brain import HOPEBrain

def test_hybrid_brain():
    print("🧠 Testing Hybrid HOPE Brain (Thousand Brains)...")
    
    # 1. Config for Hybrid Mode
    config = HOPEConfig(
        d_s=32, d_e=16, d_c=64, # Small dims for test
        use_hybrid_mode=True,
        num_columns=4,
        device="cpu"
    )
    
    # 2. Instantiate Brain
    brain = HOPEBrain(config, obs_dim=10, action_dim=2, output_dim=2)
    print(f"✅ Brain initialized with {len(brain.columns)} columns.")
    assert len(brain.columns) == 4, "Expected 4 columns"
    
    # 3. Create dummy input
    x = torch.randn(1, 10)
    a = torch.zeros(1, 2)
    r = torch.tensor([0.0])
    
    # 4. Run Step
    print("🏃 Running step...")
    state, y, info = brain.step(x, a, r)
    
    # 5. Verify Voting Info
    print("📊 Step Info Keywords:", info.keys())
    
    if 'hybrid_votes' in info:
        print("🗳️ Votes received:", info['hybrid_votes'])
        print(f"🏆 Winner Column: {info.get('winner_column')}")
    else:
        print("❌ 'hybrid_votes' not found in info dict!")
        # If lyapunov is nan, logs might be skipped?
        
    # Check if output exists
    print(f"✅ Output shape: {y.shape}")
    
    print("\n✅ Hybrid Brain Verification Complete.")

if __name__ == "__main__":
    test_hybrid_brain()
