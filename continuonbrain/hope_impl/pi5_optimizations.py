"""
Raspberry Pi 5 Optimizations

Optimizations specific to Raspberry Pi 5 deployment:
    - Quantization (INT8/FP16)
    - Model compression
    - Memory optimization
    - Inference benchmarking
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import time

from .brain import HOPEBrain
from .config import HOPEConfig


def quantize_brain(brain: HOPEBrain, dtype: str = "int8") -> HOPEBrain:
    """
    Quantize HOPE brain for deployment.
    
    Args:
        brain: HOPE brain to quantize
        dtype: "int8" or "fp16"
    
    Returns:
        Quantized brain
    """
    brain.eval()  # Set to eval mode
    
    if dtype == "int8":
        # Dynamic quantization for linear layers
        quantized = torch.quantization.quantize_dynamic(
            brain,
            {nn.Linear},
            dtype=torch.qint8,
        )
        print(f"Quantized to INT8")
    elif dtype == "fp16":
        # Half precision
        quantized = brain.half()
        print(f"Converted to FP16")
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    
    return quantized


def optimize_for_pi5(brain: HOPEBrain) -> HOPEBrain:
    """
    Apply Pi5-specific optimizations.
    
    Args:
        brain: HOPE brain to optimize
    
    Returns:
        Optimized brain
    """
    # 1. Quantize to INT8
    brain = quantize_brain(brain, dtype="int8")
    
    # 2. Fuse operations where possible
    # (PyTorch will automatically fuse some ops in eval mode)
    brain.eval()
    
    # 3. Set to CPU (Pi5 doesn't have CUDA)
    brain = brain.cpu()
    
    # 4. Enable inference optimizations
    torch.set_num_threads(4)  # Pi5 has 4 cores
    
    print("Applied Pi5 optimizations")
    return brain


def benchmark_inference(
    brain: HOPEBrain,
    obs_dim: int,
    action_dim: int,
    num_steps: int = 100,
    warmup_steps: int = 10,
) -> Dict[str, float]:
    """
    Benchmark inference performance.
    
    Args:
        brain: HOPE brain to benchmark
        obs_dim: Observation dimension
        action_dim: Action dimension
        num_steps: Number of steps to benchmark
        warmup_steps: Number of warmup steps
    
    Returns:
        Dict with benchmark results
    """
    brain.eval()
    brain.reset()
    
    # Create dummy inputs
    device = next(brain.parameters()).device
    dtype = next(brain.parameters()).dtype
    
    # Warmup
    print(f"Warming up for {warmup_steps} steps...")
    for _ in range(warmup_steps):
        x_obs = torch.randn(obs_dim, device=device, dtype=dtype)
        a_prev = torch.randn(action_dim, device=device, dtype=dtype)
        r_t = 0.0
        
        with torch.no_grad():
            _, _, _ = brain.step(x_obs, a_prev, r_t)
    
    # Benchmark
    print(f"Benchmarking for {num_steps} steps...")
    start_time = time.time()
    
    for _ in range(num_steps):
        x_obs = torch.randn(obs_dim, device=device, dtype=dtype)
        a_prev = torch.randn(action_dim, device=device, dtype=dtype)
        r_t = 0.0
        
        with torch.no_grad():
            _, _, _ = brain.step(x_obs, a_prev, r_t)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Compute metrics
    steps_per_sec = num_steps / elapsed
    ms_per_step = (elapsed / num_steps) * 1000
    
    # Memory usage
    memory_usage = brain.get_memory_usage()
    
    results = {
        'steps_per_second': steps_per_sec,
        'ms_per_step': ms_per_step,
        'total_time_sec': elapsed,
        'num_steps': num_steps,
        **{f'memory_{k}_mb': v for k, v in memory_usage.items()},
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Steps/sec: {steps_per_sec:.2f}")
    print(f"  ms/step: {ms_per_step:.2f}")
    print(f"  Total memory: {memory_usage['total']:.2f} MB")
    
    return results


def compress_brain(
    brain: HOPEBrain,
    pruning_amount: float = 0.3,
) -> HOPEBrain:
    """
    Compress brain using pruning.
    
    Args:
        brain: HOPE brain to compress
        pruning_amount: Fraction of weights to prune (0.0 to 1.0)
    
    Returns:
        Compressed brain
    """
    import torch.nn.utils.prune as prune
    
    # Prune linear layers
    for name, module in brain.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
    
    print(f"Pruned {pruning_amount * 100:.1f}% of weights")
    return brain


def export_to_onnx(
    brain: HOPEBrain,
    obs_dim: int,
    action_dim: int,
    output_path: str,
):
    """
    Export brain to ONNX format for optimized inference.
    
    Args:
        brain: HOPE brain to export
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_path: Path to save ONNX model
    """
    brain.eval()
    
    # Create dummy inputs
    device = next(brain.parameters()).device
    dtype = next(brain.parameters()).dtype
    
    x_obs = torch.randn(obs_dim, device=device, dtype=dtype)
    a_prev = torch.randn(action_dim, device=device, dtype=dtype)
    r_t = torch.tensor(0.0, device=device, dtype=dtype)
    
    # Export
    torch.onnx.export(
        brain,
        (x_obs, a_prev, r_t),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation', 'action', 'reward'],
        output_names=['output'],
        dynamic_axes={
            'observation': {0: 'obs_dim'},
            'action': {0: 'action_dim'},
        },
    )
    
    print(f"Exported to ONNX: {output_path}")


class Pi5MemoryManager:
    """
    Memory manager for Pi5 deployment.
    
    Monitors memory usage and triggers cleanup when needed.
    """
    
    def __init__(self, max_memory_mb: float = 2000):
        """
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = 0.8 * max_memory_mb
    
    def check_memory(self, brain: HOPEBrain) -> Dict[str, any]:
        """
        Check current memory usage.
        
        Args:
            brain: HOPE brain
        
        Returns:
            Dict with memory status
        """
        usage = brain.get_memory_usage()
        total_mb = usage['total']
        
        status = {
            'total_mb': total_mb,
            'max_mb': self.max_memory_mb,
            'usage_percent': (total_mb / self.max_memory_mb) * 100,
            'warning': total_mb > self.warning_threshold,
            'critical': total_mb > self.max_memory_mb,
        }
        
        return status
    
    def cleanup_if_needed(self, brain: HOPEBrain):
        """
        Cleanup memory if usage is too high.
        
        Args:
            brain: HOPE brain
        """
        status = self.check_memory(brain)
        
        if status['critical']:
            print(f"CRITICAL: Memory usage {status['total_mb']:.1f} MB exceeds limit {self.max_memory_mb} MB")
            # Reset brain state to free memory
            brain.reset()
            print("Reset brain state to free memory")
        elif status['warning']:
            print(f"WARNING: Memory usage {status['total_mb']:.1f} MB approaching limit")
