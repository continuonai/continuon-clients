# WaveCore Module Spec (ContinuonOS Integration)

WaveCore is the spectral core inside ContinuonOS used to process continuous and discrete streams on-device (Pi 5, Orin, phones) and in the cloud.

## Module Structure
- **WaveCore** (top-level orchestrator)
  - **WaveEncoder**: modality adapters (text, audio, sensors) that convert inputs into WaveTensor.
  - **WaveRouter**: routes fused spectral features to planner/policy heads or logging.
  - **WaveMemory**: manages streaming state and decay schedules across ticks.

## WaveTensor
```
WaveTensor shape: [batch, channels, time]
dtype: float32
semantics: resampled to global tick Δt
```
Examples: audio channels, IMU axes, joint positions/velocities, token embeddings projected to continuous channels.

## Core Algorithm (per tick)
1. **Detail path**: depthwise 1D conv over time for local context.
2. **Global spectral path**: stack of SpectralBlocks with causal masking and learnable per-channel decay.
3. **Gated fusion**: `gate = sigmoid(linear(cat(detail, global)))`; `fused = gate * global + (1 - gate) * detail`.
4. **State handling**: optional recurrent state for streaming windows; decay schedule controls long-range retention.

### JAX seed implementation note (Mamba-like selective SSM)
The current JAX seed core uses a **Mamba-like selective SSM** (stable diagonal A, input-dependent Δ/B/C) implemented with `jax.lax.scan` (no custom kernels) so the same code runs on GPU/TPU and Pi-class CPUs.

## Inbound APIs
- `WaveCore.ingest_text(tokens, modality_id)` → embeds tokens, aligns to Δt, produces WaveTensor.
- `WaveCore.ingest_audio(waveform, sample_rate)` → resamples, optional STFT, produces WaveTensor.
- `WaveCore.ingest_sensor(sensor_tensor, metadata)` → normalizes sensor streams (IMU, joints) into WaveTensor.

## Outbound APIs
- `encode_for_planner()` → compressed representation for high-level LLM/planner.
- `encode_for_policy()` → low-latency features for control/VLA heads.
- `export_rlds_chunk()` → RLDS-compatible continuous trajectories (aligns with `docs/rlds-schema.md`).

## Deployment Tiers
- **Edge (Pi 5 / Orin / phone)**: shallow stack (2–4 blocks, 64–256 channels), small windows; latency target <10 ms/control tick, <50 ms perception.
- **Cloud (Continuon Cloud / TPU)**: deeper stack (12–48 blocks, wider channels) used for offline training/distillation into edge configs; specs live in `continuonai/continuon-cloud/` (Google Cloud path).

## Design Principles
- Sub-quadratic long-range modeling via FFT (O(N log N)).
- Stable spectral decay to prevent gradient explosions in deep stacks.
- Unified handling of continuous and discrete streams in a single spectral representation.
