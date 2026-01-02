# Seed Model: Universal Robot Initialization

The **Seed Model** is the universal initialization point for every robot in the Continuon ecosystem. It runs on any hardware platform and provides foundational cognitive capabilities.

## Key Principles

| Principle | Description |
|-----------|-------------|
| **Universal** | Every new robot starts from the same seed |
| **Hardware-Agnostic** | Runs on ARM, x64, RISC-V, quantum, neuromorphic |
| **Permanent** | Core foundation—never deprecated |
| **Evolvable** | Continuous learning builds on seed capabilities |

---

## Quick Start

```python
from continuonbrain.seed import SeedModel

# Auto-detect hardware and initialize
seed = SeedModel()

# Or specify target platform
seed = SeedModel(target='pi5')      # Raspberry Pi 5
seed = SeedModel(target='jetson')   # NVIDIA Jetson
seed = SeedModel(target='cloud')    # Cloud/TPU

# Get model info
print(seed.get_info())
# {
#   "version": "1.0.0",
#   "param_count": 172202,
#   "hardware": {"architecture": "arm64", "device_name": "Raspberry Pi 5"},
#   "capabilities": ["world_model", "context_graph", "semantic_search", ...],
#   "portability": ["arm64", "x86_64", "riscv64", "quantum (future)", ...]
# }

# Forward pass
output, state = seed.forward(
    observation=obs,
    action_prev=prev_action,
    reward=0.5,
)

# Save checkpoint
seed.save('/opt/continuonos/brain/model/seed/')

# Load existing checkpoint
seed2 = SeedModel(checkpoint_path='/opt/continuonos/brain/model/seed/')
```

---

## Hardware Portability

The seed model runs on any platform:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARDWARE PORTABILITY MATRIX                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Architecture      │ Runtime        │ Accelerator      │ Status            │
│  ──────────────────┼────────────────┼──────────────────┼─────────────────  │
│  ARM64 (Pi5)       │ JAX CPU        │ Hailo-8 NPU      │ ✅ Primary        │
│  ARM64 (Jetson)    │ JAX CUDA       │ Tensor Cores     │ ✅ Supported      │
│  x86_64 (PC)       │ JAX CPU/CUDA   │ NVIDIA GPU       │ ✅ Supported      │
│  x86_64 (Cloud)    │ JAX TPU        │ TPU v4/v5        │ ✅ Supported      │
│  RISC-V            │ Portable C     │ Custom NPU       │ 🔶 Planned        │
│  Apple Silicon     │ JAX Metal      │ ANE              │ 🔶 Planned        │
│  Quantum           │ Pennylane/JAX  │ QPU              │ 🔮 Research       │
│  Neuromorphic      │ Lava/Loihi     │ Intel Loihi 2    │ 🔮 Research       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Presets

Configs automatically scale based on hardware:

| Preset | RAM | Params | Target |
|--------|-----|--------|--------|
| `embedded` | <2GB | ~50K | MCU, low-power edge |
| `minimal` | 2-4GB | ~100K | Pi 4, constrained |
| `edge` (default) | 4-16GB | ~172K | Pi 5, Jetson Nano |
| `workstation` | 16-64GB | ~500K | Jetson Orin, PC |
| `cloud` | 64GB+ | ~2M | TPU, Server |

```python
from continuonbrain.seed import SeedConfig

# Auto-detect
config = SeedConfig.auto()

# Or manual
config = SeedConfig.edge()      # Pi5 default
config = SeedConfig.cloud()     # TPU
config = SeedConfig.embedded()  # MCU

# Custom
config = SeedConfig(
    d_s=128, d_w=128, d_p=64,
    num_levels=4,
)

# Use with model
seed = SeedModel(config=config)
```

---

## Capabilities

The seed model provides these foundational capabilities:

### 1. World Model (Next-Token Prediction)

Predicts future states given actions:

```
s_{t+1} = WaveCore(s_t, action_t)
```

### 2. Context Graph (Relational Reasoning)

Tracks entities and relationships:

```
cup --on--> table --near--> user
```

### 3. Semantic Search (768-dim)

EmbeddingGemma-300m for meaning-based retrieval:

```python
query = "Where is the red cup?"
# → Similar memories ranked by cosine similarity
```

### 4. Decision Traces (Explainability)

Every decision logged with provenance:

```yaml
trace_id: dt_20260102_100523
reasoning_steps:
  - goal_parsing: "Pick up cup" → GRASP
  - object_grounding: "red cup" → cup_42
  - safety_check: PASSED
decision: {action: GRASP, target: cup_42, confidence: 0.94}
```

### 5. CMS Memory (Multi-Timescale)

3-level hierarchical memory:

| Level | Timescale | Content |
|-------|-----------|---------|
| Fast | 100ms | Raw sensory, motor |
| Mid | 10s | Task context, goals |
| Slow | ∞ | Skills, knowledge |

---

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `model.py` | `SeedModel` main class |
| `config.py` | `SeedConfig` with hardware presets |
| `hardware.py` | Hardware detection utilities |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEED MODEL ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  New Robot (Any Hardware)                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Hardware Detection                                                       ││
│  │ ARM64? x86? RISC-V? Quantum?                                            ││
│  └───────────────────────────────────┬─────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Config Selection                                                         ││
│  │ embedded / minimal / edge / workstation / cloud                         ││
│  └───────────────────────────────────┬─────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ WaveCore (Mamba SSM)                                                     ││
│  │ O(n) complexity, runs on any backend                                    ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ CMS Memory (3-level)                                                     ││
│  │ Fast (100ms) | Mid (10s) | Slow (∞)                                     ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ Context Graph + Semantic Search + Decision Traces                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Robot Operational                                                        ││
│  │ Experience Collection → Local Learning → Evolution                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Lifecycle

```
1. Robot boots for first time
   ↓
2. Hardware detection
   ↓
3. Download/verify seed model from cloud
   ↓
4. Select optimal config
   ↓
5. Initialize WaveCore + CMS + Context Graph
   ↓
6. Robot operational with full capabilities
   ↓
7. Experience collection → Local learning → Cloud aggregation → OTA updates
```

---

## Next Steps

See:
- [Full Architecture](../../docs/seed-to-hope-evolution.md)
- [WaveCore Implementation](../jax_models/README.md)
- [CMS Memory](../hope_impl/README.md)
- [Context Graph](../core/README.md)

