
import unittest
import numpy as np
import sys
import os

# Ensure proper path
sys.path.append("/home/craigm26/ContinuonXR")

from continuonbrain.aina_impl.compensation import SensorFusionModule

class TestSensorFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = SensorFusionModule(confidence_threshold=0.5)
        
    def test_memory_persistence(self):
        print("\n--- Testing Occlusion Memory ---")
        # 1. Good Observation
        pos1 = np.array([1.0, 1.0, 0.5])
        conf1 = 0.9
        fused, status = self.fusion.fuse_inputs(pos1, conf1)
        
        self.assertTrue(np.allclose(fused, pos1))
        self.assertEqual(status, "LIVE_FUSION")
        print("✅ High Confidence Update accepted")
        
        # 2. Bad Observation (Occlusion)
        pos2 = np.array([99.0, 99.0, 99.0]) # Garbage noise
        conf2 = 0.1 # Detected as bad
        fused, status = self.fusion.fuse_inputs(pos2, conf2)
        
        # Should return pos1 (Memory), not pos2 (Noise)
        self.assertTrue(np.allclose(fused, pos1))
        self.assertEqual(status, "MEMORY_RECALL")
        print("✅ Low Confidence rejected, Memory recalled")
        
    def test_active_depth_fusion(self):
        print("\n--- Testing Active Depth Fusion ---")
        # Inferred (Passive) says 1.0
        inferred = np.array([1.0, 1.0, 1.0])
        # Active (OAK-D) says 1.05 (Close agreement)
        active = np.array([1.05, 1.05, 1.05])
        
        fused, status = self.fusion.fuse_inputs(inferred, 0.9, active_depth_pos=active)
        
        # Should be weighted covariance -> roughly 1.04
        # Our logic: 0.2 * 1.0 + 0.8 * 1.05 = 0.2 + 0.84 = 1.04
        expected = inferred * 0.2 + active * 0.8
        
        self.assertTrue(np.allclose(fused, expected))
        print(f"✅ Fused Result {fused} matches expected weighted avg")

if __name__ == "__main__":
    unittest.main()
