
import unittest
import torch
import numpy as np
import sys
import os

# Adapt path
sys.path.append("/home/craigm26/ContinuonXR")

from continuonbrain.aina_impl.policy import AINAPolicy, VectorNeuronMLP
from continuonbrain.aina_impl.data_processing import kabsch_alignment

class TestAINAImplementation(unittest.TestCase):
    
    def test_vector_neuron_mlp(self):
        print("\n--- Testing Vector Neuron MLP ---")
        # Input: [B, N=10, 3, C=1]
        x = torch.randn(2, 10, 3, 1)
        model = VectorNeuronMLP(1, 16, 32)
        y = model(x)
        # Expected: [B, N, 3, 32]
        self.assertEqual(y.shape, (2, 10, 3, 32))
        print("✅ VN-MLP Shape Correct")
        
    def test_aina_policy_forward(self):
        print("\n--- Testing AINA Policy Forward Pass ---")
        # [B, T=10, N=5, 3]
        fingertips = torch.randn(2, 10, 5, 3)
        # [B, T=10, P=50, 3]
        objects = torch.randn(2, 10, 50, 3)
        
        model = AINAPolicy(n_fingers=5, n_obj_points=50, pred_horizon=5)
        out = model(fingertips, objects)
        
        # Expected: [B, T_pred=5, N=5, 3]
        self.assertEqual(out.shape, (2, 5, 5, 3))
        print("✅ AINA Policy Output Shape Correct")
        
    def test_kabsch_alignment(self):
        print("\n--- Testing Kabsch Alignment ---")
        # Create a known rotation
        theta = np.radians(90) # 90 deg around Z
        c, s = np.cos(theta), np.sin(theta)
        R_true = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Points P (cube corners)
        P = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ], dtype=np.float32)
        
        # Points Q = R*P (centered)
        Q = np.dot(P, R_true.T)
        
        # Shift both random amounts
        t_p = np.array([1, 2, 3])
        t_q = np.array([-1, -5, 10])
        
        P_shifted = P + t_p
        Q_shifted = Q + t_q
        
        # Recover Rotation
        R_est, t_est = kabsch_alignment(P_shifted, Q_shifted)
        
        # Check Rotation
        np.testing.assert_allclose(R_est, R_true, atol=1e-5)
        print("✅ Kabsch Rotation Recovery Correct")
        
        # Check alignment
        P_aligned = np.dot(P_shifted, R_est.T) + t_est
        np.testing.assert_allclose(P_aligned, Q_shifted, atol=1e-5)
        print("✅ Kabsch Point Alignment Correct")

if __name__ == "__main__":
    unittest.main()
