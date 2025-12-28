#!/usr/bin/env python3
"""
Verify GPU setup for JAX training.
Run this after setup to confirm GPU acceleration is working.
"""

import sys

def main():
    print("=" * 60)
    print("  ContinuonBrain GPU Verification")
    print("=" * 60)
    print()

    # Check JAX
    print("1. Checking JAX installation...")
    try:
        import jax
        import jax.numpy as jnp
        print(f"   ✓ JAX version: {jax.__version__}")
    except ImportError as e:
        print(f"   ✗ JAX not installed: {e}")
        sys.exit(1)

    # Check devices
    print()
    print("2. Checking JAX devices...")
    devices = jax.devices()
    print(f"   Found {len(devices)} device(s):")
    for d in devices:
        print(f"   - {d}")
    
    gpu_available = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    if gpu_available:
        print("   ✓ GPU detected!")
    else:
        print("   ⚠ No GPU detected - will use CPU (slower)")

    # Check Flax
    print()
    print("3. Checking Flax installation...")
    try:
        import flax
        print(f"   ✓ Flax version: {flax.__version__}")
    except ImportError:
        print("   ✗ Flax not installed")
        sys.exit(1)

    # Check Optax
    print()
    print("4. Checking Optax installation...")
    try:
        import optax
        print(f"   ✓ Optax version: {optax.__version__}")
    except ImportError:
        print("   ✗ Optax not installed")
        sys.exit(1)

    # Run a simple GPU test
    print()
    print("5. Running GPU computation test...")
    try:
        # Create a simple computation
        x = jnp.ones((1000, 1000))
        
        @jax.jit
        def matmul_test(a):
            return jnp.dot(a, a)
        
        # Warm up JIT
        _ = matmul_test(x)
        
        # Time it
        import time
        start = time.time()
        for _ in range(10):
            result = matmul_test(x)
            result.block_until_ready()
        elapsed = time.time() - start
        
        print(f"   ✓ 10x matrix multiply (1000x1000): {elapsed*1000:.1f}ms")
        if elapsed < 0.1:
            print("   ✓ GPU acceleration confirmed (fast!)")
        elif elapsed < 1.0:
            print("   ~ Moderate speed - GPU may be working")
        else:
            print("   ⚠ Slow - likely using CPU fallback")
            
    except Exception as e:
        print(f"   ✗ Computation test failed: {e}")
        sys.exit(1)

    # Memory info
    print()
    print("6. GPU Memory Info...")
    try:
        # Try to get GPU memory using nvidia-smi
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            total, free, used = result.stdout.strip().split(", ")
            print(f"   Total: {total} MB")
            print(f"   Used:  {used} MB")
            print(f"   Free:  {free} MB")
        else:
            print("   Could not query GPU memory")
    except Exception:
        print("   nvidia-smi not available")

    # Summary
    print()
    print("=" * 60)
    if gpu_available:
        print("  ✓ GPU SETUP VERIFIED - Ready for training!")
    else:
        print("  ⚠ CPU ONLY - Training will be slower")
    print("=" * 60)
    print()
    print("Next: python scripts/gpu_training/train_gpu.py --help")

if __name__ == "__main__":
    main()

