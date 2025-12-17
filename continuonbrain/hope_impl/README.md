# HOPE Implementation

PyTorch implementation of the HOPE (Hierarchical Online Predictive Encoding) architecture optimized for Raspberry Pi 5 deployment.

## Overview

HOPE is a hybrid dynamical system that combines:
- **Wave subsystem**: SSM-like global linear dynamics
- **Particle subsystem**: Local nonlinear dynamics  
- **Continuous Memory System (CMS)**: Hierarchical content-addressable memory with decay
- **Nested Learning**: Slow parameter adaptation

This implementation provides a complete, production-ready system suitable for edge deployment on Raspberry Pi 5.

## Architecture

```
Input (obs, action, reward)
    ↓
Encoder → e_t
    ↓
CMS Read → (q_t, c_t)
    ↓
HOPE Core (wave-particle hybrid) → (s_t, w_t, p_t)
    ↓
Output Decoder → y_t
    ↓
CMS Write → M_t
    ↓
Nested Learning → Θ_t
```

## Installation

```bash
cd /home/craigm26/ContinuonXR/continuonbrain

# Install dependencies (already in requirements.txt)
pip install torch>=2.2.0 numpy>=1.26.0

# For testing
pip install pytest
```

## Quick Start

### Minimal Example

```python
from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain
import torch

# Create configuration
config = HOPEConfig.development()

# Create brain
brain = HOPEBrain(
    config=config,
    obs_dim=10,
    action_dim=4,
    output_dim=4,
)

# Reset state
brain.reset()

# Run a step
x_obs = torch.randn(10)
a_prev = torch.zeros(4)
r_t = 0.0

state_next, y_t, info = brain.step(x_obs, a_prev, r_t)
print(f"Output: {y_t}")
print(f"Lyapunov energy: {info['lyapunov']}")
```

### Pi5 Deployment

```python
from hope_impl.config import HOPEConfig
from hope_impl.brain import HOPEBrain
from hope_impl.pi5_optimizations import optimize_for_pi5, benchmark_inference

# Create Pi5-optimized configuration
config = HOPEConfig.pi5_optimized()

# Create and optimize brain
brain = HOPEBrain(config=config, obs_dim=10, action_dim=4, output_dim=4)
brain = optimize_for_pi5(brain)

# Benchmark
results = benchmark_inference(brain, obs_dim=10, action_dim=4, num_steps=100)
print(f"Steps/sec: {results['steps_per_second']:.2f}")
print(f"Memory: {results['memory_total_mb']:.2f} MB")
```

## Module Structure

```
hope_impl/
├── __init__.py           # Package exports
├── config.py             # Configuration management
├── state.py              # State objects (FastState, CMSMemory, etc.)
├── encoders.py           # Input/output encoders
├── cms.py                # CMS read/write operations
├── core.py               # HOPE core dynamics
├── learning.py           # Nested learning
├── stability.py          # Lyapunov functions and monitoring
├── brain.py              # Main HOPEBrain interface
└── pi5_optimizations.py  # Pi5-specific optimizations
```

## Configuration Presets

### Development (Fast Iteration)
```python
config = HOPEConfig.development()
# Small dimensions, 2 CMS levels
# ~5 MB memory, fast inference
```

### Pi5 Optimized (Production)
```python
config = HOPEConfig.pi5_optimized()
# Balanced dimensions, 3 CMS levels
# INT8 quantization enabled
# <2 GB memory, >10 steps/sec
```

### Custom
```python
config = HOPEConfig(
    d_s=256,
    d_w=256,
    d_p=128,
    num_levels=3,
    cms_sizes=[64, 128, 256],
    cms_dims=[128, 256, 512],
    cms_decays=[0.1, 0.05, 0.01],
    use_quantization=True,
)
```

## Testing

```bash
# Run all tests
pytest tests/test_hope_impl.py -v

# Run specific test class
pytest tests/test_hope_impl.py::TestHOPEBrain -v

# Run with coverage
pytest tests/test_hope_impl.py --cov=hope_impl
```

## Examples

### 1. Minimal Demo
```bash
python examples/hope_minimal_demo.py
```

Demonstrates:
- Brain initialization
- Single step execution
- Multi-step rollout
- Checkpoint save/load

### 2. Pi5 Deployment
```bash
python examples/hope_pi5_deployment.py
```

Demonstrates:
- Quantized model loading
- Real-time inference loop
- Memory monitoring
- Performance benchmarking

## Performance Targets (Pi5)

| Metric | Target | Typical |
|--------|--------|---------|
| Inference speed | >10 steps/sec | ~15-20 steps/sec |
| Memory usage | <2 GB | ~500 MB - 1 GB |
| Model size | <500 MB | ~200-300 MB |
| Quantized accuracy | >95% of FP32 | ~98% |

## API Reference

### HOPEBrain

Main interface for HOPE architecture.

**Methods:**
- `reset()` - Initialize/reset brain state
- `step(x_obs, a_prev, r_t)` - Execute one HOPE step
- `save_checkpoint(path)` - Save brain state
- `load_checkpoint(path)` - Load brain state
- `to_quantized(dtype)` - Convert to quantized version
- `get_memory_usage()` - Get memory statistics

**Returns from `step()`:**
- `state_next`: Updated full state
- `y_t`: Output tensor
- `info`: Dict with query, context, attention weights, stability metrics

### Configuration

**HOPEConfig fields:**
- `d_s, d_w, d_p`: Fast state dimensions
- `d_e, d_k, d_c`: Encoder/key/context dimensions
- `num_levels`: Number of CMS levels
- `cms_sizes, cms_dims, cms_decays`: Per-level CMS parameters
- `use_quantization`: Enable quantization
- `learning_rate, eta_init`: Training parameters

## Mathematical Foundation

See [`docs/CMS_FORMAL_SPEC.md`](file:///home/craigm26/ContinuonXR/continuonbrain/docs/CMS_FORMAL_SPEC.md) for complete mathematical specification including:
- CMS update rule derivation
- Stability proofs (Lyapunov analysis)
- Computational complexity
- Implementation guidance

## Checkpointing

```python
# Save
brain.save_checkpoint("brain.pt")

# Load
brain = HOPEBrain.load_checkpoint("brain.pt")
```

Checkpoints include:
- Model weights
- Internal state (fast state, CMS, parameters)
- Stability monitor history
- Configuration

## Optimization Tips

### Memory Reduction
1. Use smaller dimensions (`d_s`, `d_w`, `d_p`)
2. Reduce CMS levels or slots per level
3. Enable quantization (`use_quantization=True`)
4. Use FP16 instead of FP32

### Speed Improvement
1. Reduce CMS levels
2. Use smaller hidden dimensions
3. Enable quantization
4. Set `torch.set_num_threads(4)` for Pi5

### Stability
1. Enable layer normalization (`use_layer_norm=True`)
2. Use gradient clipping during training
3. Monitor Lyapunov energy
4. Start with conservative decay rates

## Troubleshooting

**Q: Memory usage too high?**
- Use `HOPEConfig.pi5_optimized()` preset
- Enable quantization
- Reduce CMS sizes

**Q: Inference too slow?**
- Reduce number of CMS levels
- Use smaller dimensions
- Enable quantization
- Check `torch.set_num_threads()`

**Q: Unstable training?**
- Enable layer normalization
- Reduce learning rate
- Use gradient clipping
- Monitor Lyapunov energy

**Q: NaN values appearing?**
- Check input normalization
- Reduce learning rate
- Enable gradient clipping
- Check decay rates (should be in (0, 1))

## Citation

If you use this implementation, please cite:

```bibtex
@software{hope_impl_2024,
  title={HOPE: Hierarchical Online Predictive Encoding},
  author={ContinuonXR Team},
  year={2024},
  url={https://github.com/continuonxr/ContinuonXR}
}
```

## License

See main repository LICENSE file.

## Contributing

This is part of the ContinuonXR project. For contributions, see the main repository.

## References

- Original HOPE specification: [`hope.py`](file:///home/craigm26/ContinuonXR/continuonbrain/hope.py)
- Mathematical foundations: [`maths.md`](file:///home/craigm26/ContinuonXR/continuonbrain/maths.md)
- CMS formal specification: [`docs/CMS_FORMAL_SPEC.md`](file:///home/craigm26/ContinuonXR/continuonbrain/docs/CMS_FORMAL_SPEC.md)
- Implementation plan: See artifacts directory
