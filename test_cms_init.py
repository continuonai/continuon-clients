
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from continuonbrain.hope_impl.cms import CMSRead
from continuonbrain.hope_impl.state import CMSMemory, MemoryLevel

def test():
    print("Testing CMSRead...", file=sys.stderr)
    d_c = 64
    num_levels = 2
    cms_dims = [128, 64]
    
    cms = CMSRead(
        d_s=256,
        d_e=256,
        d_k=16,
        d_c=d_c,
        num_levels=num_levels,
        cms_dims=cms_dims
    )
    
    print("CMSRead initialized.", file=sys.stderr)
    
    # Create fake inputs
    batch_size = 64
    s_t = torch.randn(batch_size, 256)
    e_t = torch.randn(batch_size, 256)
    
    # Create fake memory
    levels = []
    for ell in range(num_levels):
        M = torch.randn(10, cms_dims[ell])
        K = torch.randn(10, 16)
        levels.append(MemoryLevel(M, K, 0.1))
    
    memory = CMSMemory(levels)
    
    print("Running forward...", file=sys.stderr)
    cms(memory, s_t, e_t)

if __name__ == "__main__":
    test()
