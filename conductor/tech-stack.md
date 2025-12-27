# Technology Stack: Continuon AI

## Edge Runtime (The Brain)
- **Language:** Python 3.11+
- **Unified Entry Points:** `startup_manager.py` (system boot/HAL/API) and `run_trainer.py` (multi-backend training).
- **Process Management:** Declarative `ServiceRegistry` for orchestrating background services (API, Safety Kernel, Trainer) with prioritized startup and robust teardown.
- **Deep Learning Frameworks:** 
  - **JAX:** Primary for high-performance training and inference loops (WaveCore).
  - **PyTorch/Transformers:** Used for model development, certain LLM hooks, and the sequential training orchestrator.
- **AI Models:**
  - **Mamba (SSM):** Core architecture for the "System 2" world model and symbolic search.
  - **Gemma-3n:** Base LLM for reasoning and planning (optimized for edge).
  - **WaveCore (SSM + FFT):** Matatured modular library for high-frequency control and tactical planning loops, including a synthetic logic generator and distillation hooks.
  - **Hybrid Architectures:** Fused SSM and Sliding-Window Attention for efficient long-context edge reasoning.
  - **VQ-VAE:** Used for latent tokenization of visual data on accelerators.
  - **Context Graphs:** Hybrid symbolic/dense retrieval system using `sentence-transformers` (`all-MiniLM-L6-v2`) bridging RLDS/CMS and the HOPE planner.
  - **Memory Synthesizer:** LLM-driven process for merging redundant conversation clusters into high-fidelity semantic anchors with dynamic confidence decay.
  - **Brain Tools:** Pluggable capability modules (Calculator, Wikipedia, FileSystem) with standardized interfaces.
- **Edge Acceleration:**
  - **Hailo-8L / Coral Edge TPU:** Support for dedicated NPU/TPU offloading.
  - **XNNPACK:** Optimized CPU inference for ARM (Pi 5).
- **Hardware Interaction:** CircuitPython (adafruit-servokit), smbus2, and lgpio for Raspberry Pi 5 hardware/servo control.
- **Safety Enforcement:** Ring 0 Safety Kernel (Python) using Unix Sockets (Linux) or TCP (Windows) for IPC and a "Constitution" for deterministic kinematic and health validation.
- **Process & Lifecycle:** `psutil` for instance locking and platform-agnostic process monitoring.
- **Testing:** `pytest` (Unit/Integration) and `playwright` (End-to-End Browser Automation) for full-stack verification.
- **Authentication & Security:**
  - **Firebase Auth:** JWT-based identity management and token verification.
  - **RBAC:** Role-Based Access Control (Creator, Developer, Consumer) enforced via middleware.
- **Graph & Context:**
  - **SQLite:** Persistent storage for the context graph (nodes, edges).
  - **Feedback Store:** Dedicated SQLite persistence for user validation and "Gold Data" prioritization, featuring automated threshold-based consolidation and synthesis hooks.
  - **Session Store:** Persistent SQLite storage for multi-turn conversation context.
  - **NetworkX:** In-memory graph traversal and analysis.
  - **Visualization:** D3.js for graph rendering, Server-Sent Events (SSE) for real-time cognitive streams.

## Spatial & Mobile Interfaces (The Shells)
- **Android XR:** 
  - **Language:** Kotlin
  - **UI Framework:** Jetpack XR (SceneCore, Compose for XR).
- **Companion App:** 
  - **Framework:** Flutter (Dart) with BLoC architecture for iOS, Android, and Web support.
- **Communication:**
  - **gRPC / WebRTC:** Low-latency bidirectional streaming between Brain and Shells.
  - **Protobuf:** Universal data serialization.
  - **BLE:** High-speed telemetry for the Continuon Glove.
  - **mDNS / Avahi:** Automatic service discovery on local networks using `zeroconf` (Python) and `nsd` (Flutter).
  - **SSH / Fabric:** Remote Conductor interface for command orchestration and automated key management.
  - **rsync / Watchdog:** High-performance file synchronization with real-time file system monitoring.
  - **SSE (Server-Sent Events):** Real-time streaming of cognitive states and metrics to web interfaces.

## Data & Infrastructure
- **Data Format:** RLDS (Reinforcement Learning Dataset) using JSONL/TFRecord.
- **Architecture:** Monorepo managing all products and shared contracts.
- **Cloud Interface:** HTTPS/WebSockets for manual RLDS ingestion and automated OTA updates.
