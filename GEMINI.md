# ContinuonXR Context

## Project Overview
**ContinuonXR** is a monorepo for the **Continuon AI** ecosystem, a self-learning robotics platform built on the "One Brain, Many Shells" architecture. It decouples robot intelligence (the "Brain") from physical hardware (the "Shells"), enabling continuous learning through a standardized data loop.

### Core Philosophy
- **One Brain, Many Shells:** A single AI runtime adapts to different robots (Raspberry Pi 5, arms, rovers).
- **HOPE (Hierarchical Optimizer, Perpetual Engine):** A nested learning architecture using Fast (reflex), Mid (tactical), and Slow (strategic/cloud) loops.
- **RLDS (Reinforcement Learning Dataset):** The universal data format for all training data (human demos, telemetry, internet video).
- **Offline-First:** Critical operations (control, basic learning) happen on-edge; cloud uploads are opt-in.

## Monorepo Structure

| Directory | Component | Technology | Description |
| :--- | :--- | :--- | :--- |
| **`apps/continuonxr/`** | **Android XR Shell** | Kotlin, Jetpack XR | The spatial workstation and teleop interface for Galaxy XR devices. Captures "Gold Data" via human demonstrations. |
| **`continuonbrain/`** | **Brain Runtime** | Python, TFLite, JAX | The on-device runtime for robots (e.g., Pi 5). Handles HAL, Robot API (gRPC/WebRTC), and local training. |
| **`continuonai/`** | **Companion App** | Flutter (Dart) | Mobile/Web app for robot discovery, ownership, and RLDS data exploration (WorldTape). |
| **`proto/`** | **Contracts** | Protobuf | Shared definitions for Robot API (`continuonbrain_link.proto`) and RLDS schemas. |
| **`docs/`** | **Documentation** | Markdown | Architecture specs (`hope-cms-vla.md`), contracts (`rlds-schema.md`), and roadmaps. |

## Development & Usage

### 1. Android XR App (`apps/continuonxr`)
*   **Build:** `./gradlew :apps:continuonxr:assembleDebug`
*   **Test:** `./gradlew :apps:continuonxr:testDebugUnitTest`
*   **Proto Gen:** `./gradlew :apps:continuonxr:generateDebugProto`
*   **Key Files:** `src/main/java/com/continuonxr/app/` (Logic), `app/glove/` (BLE parsing).

### 2. Continuon Brain Runtime (`continuonbrain`)
Targeting Raspberry Pi 5 / Jetson / Linux SBCs.
*   **Setup:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -r continuonbrain/requirements.txt
    ```
*   **Run Server (Robot API):**
    ```bash
    python -m continuonbrain.robot_api_server
    ```
*   **Run Trainer (Simulation/Local):**
    ```bash
    python -m continuonbrain.trainer.examples.pi5_integration --config continuonbrain/configs/pi5-donkey.json
    ```
*   **Key Files:** `robot_api_server.py`, `startup_manager.py`, `hal/` (Hardware Abstraction).

### 3. Continuon AI Companion (`continuonai`)
*   **Run:** `flutter run`
*   **Test:** `flutter test`
*   **Analyze:** `flutter analyze`

## Key Conventions & Constraints

### Data & Contracts
*   **RLDS is Law:** All training data **MUST** adhere to the schema in `docs/rlds-schema.md`.
*   **Proto Consistency:** Changes to `proto/` require regenerating stubs for all languages (`gradlew generateProtoKotlin`, `buf lint`).
*   **Offline-First:** Default to local logging. Cloud uploads require explicit user opt-in and PII checks.

### AI Architecture (HOPE/CMS)
*   **Fast Loop (50-100ms):** On-device reflexes (SafetyHead, motor control).
*   **Mid Loop (1-10s):** On-device planning and short-term adaptation (LoRA adapters).
*   **Slow Loop (Hrs/Days):** Cloud-based retraining of the core World Model (Mamba/Transformer).

### Contribution Guidelines
*   **Boundaries:** Respect module separation. Do not import `continuonbrain` code into `apps/continuonxr` directly (use Protos/gRPC).
*   **Testing:** Run the relevant test suite for the module being modified before committing.
*   **Hardware:** Code must be resilient to missing hardware (use Mock HALs if sensors are absent).
