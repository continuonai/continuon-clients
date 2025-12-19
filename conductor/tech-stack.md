# Technology Stack: Continuon AI

## Edge Runtime (The Brain)
- **Language:** Python 3.11+
- **Deep Learning Frameworks:** 
  - **JAX:** Primary for high-performance training and inference loops (WaveCore).
  - **PyTorch/Transformers:** Used for model development, certain LLM hooks, and the sequential training orchestrator.
- **AI Models:**
  - **Mamba (SSM):** Core architecture for the "System 2" world model and symbolic search.
  - **Gemma-3n:** Base LLM for reasoning and planning (optimized for edge).
  - **VQ-VAE:** Used for latent tokenization of visual data on accelerators.
- **Edge Acceleration:**
  - **Hailo-8L / Coral Edge TPU:** Support for dedicated NPU/TPU offloading.
  - **XNNPACK:** Optimized CPU inference for ARM (Pi 5).
- **Hardware Interaction:** CircuitPython (adafruit-servokit), smbus2, and lgpio for Raspberry Pi 5 hardware/servo control.

## Spatial & Mobile Interfaces (The Shells)
- **Android XR:** 
  - **Language:** Kotlin
  - **UI Framework:** Jetpack XR (SceneCore, Compose for XR).
- **Companion App:** 
  - **Framework:** Flutter (Dart) for iOS, Android, and Web support.
- **Communication:**
  - **gRPC / WebRTC:** Low-latency bidirectional streaming between Brain and Shells.
  - **Protobuf:** Universal data serialization.
  - **BLE:** High-speed telemetry for the Continuon Glove.

## Data & Infrastructure
- **Data Format:** RLDS (Reinforcement Learning Dataset) using JSONL/TFRecord.
- **Architecture:** Monorepo managing all products and shared contracts.
- **Cloud Interface:** HTTPS/WebSockets for manual RLDS ingestion and automated OTA updates.
