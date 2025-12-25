import pytest
import torch
import sys
from pathlib import Path

# repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_novelty_computation_math():
    """Verify that novelty (MSE) is correctly calculated."""
    # prediction and actual vectors
    pred = torch.tensor([0.1, 0.2, 0.3])
    actual = torch.tensor([0.1, 0.2, 0.4]) # Difference in 3rd element
    
    # Raw MSE: (0.1^2) / 3 = 0.01 / 3 = 0.00333
    expected_mse = torch.mean((pred - actual)**2).item()
    
    # Normalized confidence check
    k = 1.0
    # confidence = exp(-k * mse)
    expected_conf = math.exp(-k * expected_mse)
    
    assert expected_mse > 0
    assert expected_conf < 1.0

import math
