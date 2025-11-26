# PRD: WaveCore – Spectral Sequence Engine for ContinuonOS

## 1. Problem Statement
ContinuonOS needs long-horizon temporal modeling that runs efficiently on modest hardware (Pi 5, phones, Orin) while unifying continuous (audio, IMU, joints) and discrete (text tokens) data. Full Transformers are quadratic in context length, strain memory/latency budgets, and require aggressive truncation. WaveCore targets an FFT-based, sub-quadratic alternative that remains stable for deep stacks and aligns with RLDS logging.

## 2. Users
- Robotics Platform team running ContinuonOS on Pi/Orin robots.
- XR Trainer team streaming multimodal data from LiveCaptionsXR/trainer apps.
- Research/Nested Learning team experimenting with RLDS datasets and new skills.

## 3. Goals (v1–v3)
- **v1 – Toy & Hybrid (Pi 5 Prototype):** Ship Pi-runnable HybridBlock demo with benchmarks (tokens/sec, memory, latency) on synthetic tasks; document install steps.
- **v2 – Wave-first Encoder for One Modality:** Replace audio or sensor encoder with WaveCore; integrate with RLDS logger/cloud training; target 30–60% compute reduction at equal task quality.
- **v3 – Transformer Replacement:** Make WaveCore default for long-sequence workloads; keep Transformers for planner/high-level symbolic reasoning and small local blocks.

## 4. Functional Requirements
1. **WaveCore Block Library**
   - PyTorch modules: SpectralBlock, HybridBlock, WaveEncoder.
   - Causal streaming support with rolling windows and state carryover.
2. **Modality Adapters**
   - Text → embedding → WaveTensor.
   - Audio → resampled waveform or STFT → WaveTensor.
   - Sensor/robot state → normalized floats → WaveTensor.
3. **Training Pipeline**
   - Integrate with RLDS logging (continuous trajectories).
   - Support supervised next-step prediction and RL/imitation on WaveCore outputs.
4. **Evaluation Harness**
   - Long-sequence copy/add tasks, simple control tasks, and tiny language corpora.
   - Automated plots: loss vs iteration, gradient norms, spectral decay curves.

## 5. Non-Goals (for now)
- Removing Transformers entirely (planner/high-level LLM can stay Transformer-based).
- Custom kernels (C++/Rust) in v1; PyTorch FFT suffices.
- Internet-scale corpus training; focus on robot-scale, local-first workloads.

## 6. Constraints
- **Hardware:** Raspberry Pi 5 (8GB), Orin Nano-class boards; CPU-only or light NPU offload.
- **Memory:** WaveCore v1 < 2GB for model + activations at 64–128 context.
- **Latency:** Control loop ≤10 ms per step; WaveCore share ≤5 ms.

## 7. Milestones & Deliverables
- **M1 (2 weeks):** Implement SpectralBlock + SpectralLanguageModel; run on Pi 5; document install steps, timings, memory; produce report with diagrams/results.
- **M2 (4 weeks):** Implement HybridBlock and long-range benchmarks (Transformer-only vs Wave-only vs Hybrid); choose default fusion.
- **M3 (6–8 weeks):** Wrap WaveCore APIs (`ingest_*`), replace one real encoder, integrate RLDS logging, run 24h stress tests.
- **M4 (3–4 months):** Wave-first ContinuonOS release; distillation pipeline TPU → device; show energy/performance wins vs Transformer baseline.
