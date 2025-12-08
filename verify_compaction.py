
import unittest
import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from continuonbrain.hope_impl.config import HOPEConfig
from continuonbrain.hope_impl.brain import HOPEBrain
from continuonbrain.services.brain_service import BrainService

class TestMemoryCompaction(unittest.TestCase):
    def setUp(self):
        self.config = HOPEConfig(
            d_s=32,    # Reduced for fast test
            d_w=32,
            d_p=16,
            d_e=32,
            d_k=16,
            d_c=32,
            device='cpu',
            use_hybrid_mode=False
        )
        self.brain = HOPEBrain(self.config, obs_dim=10, action_dim=4, output_dim=4)
        print("\nInitialized HOPEBrain for Compaction Test")

    def test_compaction_cycle(self):
        # 1. Fill Memory (Wake Phase)
        print("Phase 1: Wake - Filling Memory...")
        x = torch.randn(1, 10)
        a = torch.randn(1, 4)
        r = torch.tensor([0.1])
        
        # Run 50 steps to build up memory
        for _ in range(50):
            self.brain.step(x, a, r)
            
        # 2. Measure Pre-Compaction State
        state_pre = self.brain.get_state()
        
        # Calculate CMS Energy (Norm of M matrices)
        cms_energy_pre = sum(level.M.norm().item() for level in state_pre.cms.levels)
        print(f"Pre-Compaction CMS Energy: {cms_energy_pre:.4f}")
        
        # Calculate Parameter Norm (Theta)
        param_norm_pre = sum(p.norm().item() for p in state_pre.params.theta.values())
        print(f"Pre-Compaction Param Norm: {param_norm_pre:.4f}")
        
        # 3. Trigger Compaction (Sleep Phase)
        print("Phase 2: Sleep - triggering compaction...")
        results = self.brain.compact_memory()
        print(f"Compaction Results: {results}")
        
        # 4. Measure Post-Compaction State
        state_post = self.brain.get_state()
        
        cms_energy_post = sum(level.M.norm().item() for level in state_post.cms.levels)
        print(f"Post-Compaction CMS Energy: {cms_energy_post:.4f}")
        
        param_norm_post = sum(p.norm().item() for p in state_post.params.theta.values())
        print(f"Post-Compaction Param Norm: {param_norm_post:.4f}")
        
        # 5. Assertions
        # CMS Energy should decrease significantly (Flush)
        self.assertLess(cms_energy_post, cms_energy_pre * 0.8, "CMS Energy did not decrease significantly (flush failed)")
        
        # Parameters should change (Consolidated Learning)
        # Note: Depending on the update, norm might increase or decrease, but it should CHANGE.
        # But specifically, we expect theta to absorb information.
        # Check if params are identical?
        self.assertNotAlmostEqual(param_norm_pre, param_norm_post, places=5, msg="Parameters did not change (learning failed)")
        
        print("✅ Compaction Verification Passed: Memory flushed, Parameters updated.")

if __name__ == "__main__":
    unittest.main()
