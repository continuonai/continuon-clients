**Product Requirements Document (PRD): ContinuonXR**
**Repository Owner:** @ContinuonAI
**Repository Name:** `ContinuonXR`
**Version:** 1.0 (Initial Development)

***

## I. Strategic Foundations & Goals

### 1.1 Product Vision
ContinuonXR is the human-facing application that serves as the secure, spatial workstation, trainer interface, and the **primary data ingestion point** for the Continuon self-learning ecosystem. It overcomes the "static" nature of traditional models by continually supplying high-quality, human-generated **RLDS episodes** (gold data) to the **Continuon Cloud Factory** (staging docs live under `continuonai/continuon-cloud/` for the Google Cloud-only ingest/train/package path). All Continuon products and services (XR, Cloud, Continuon Brain runtime, web explorer, and orchestration) are built and shipped from this repository to keep architecture and contracts aligned.

### 1.2 Unique Selling Proposition (USP)
ContinuonXR is the unified **PC replacement, robot controller, and data collection rig**. It integrates the developer’s workspace with the robot’s physical presence, ensuring that **every interaction becomes training data**.

### 1.3 Target Audience
1.  **Robot Operators/End Users:** Consumers or small teams managing 1–3 robots, requiring simple teleoperation and OTA updates (Continuon Home subscription).
2.  **Developers/Researchers:** Individuals and enterprise teams needing spatial tools for debugging, planning, and generating high-quality demonstrations for specific skill policies (Continuon Fleet subscription).

***

## II. Product Capabilities and User Modes

ContinuonXR supports distinct operational modes, each generating specific types of training data while utilizing the same underlying **RLDS schema**.

### 2.1 Spatial Workstation Mode (Mode B: PC Replacement)
This mode transforms the XR headset into the developer's primary computer interface, linking workflow context directly to robot operations.

| Capability | Features | Data Logged (RLDS Episode Metadata) |
| :--- | :--- | :--- |
| **Workspace** | Floating IDE (e.g., VS Code / JetBrains), terminals, dashboards, logs, Git interfaces. | `continuon.xr_mode = "workstation"` |
| **Workflow Logging** | Edit policies, trigger builds and deployments, inspect logs after failures, label runs as "good" vs "bad". | Focus context (which file/line, which dashboard panel), UI actions (`open panel`, `run command`, `commit`, `rollout`), tagged `source = "human_dev_xr"`. |
| **Goal** | Provide training data for a **Continuon assistant** that understands development workflows and debugging patterns. | |

### 2.2 XR Trainer Mode (Mode A: Direct Robot Control & Demonstration)
This mode facilitates **human-in-the-loop demonstrations**, capturing gold data for imitation learning.

| Capability | Features | Data Logged (RLDS Step Observation) |
| :--- | :--- | :--- |
| **Control Interface** | 3D robot teleop view, hand/pose mapping to End-Effector (EE) goals or joystick commands, gaze/point control, voice goals ("pick up the blue bin"). | `steps[*].observation`: XR head & hand poses, Egocentric video + depth, Robot state (joints, EE pose). |
| **Action Logging** | Records robot command in a normalized space (e.g., EE velocity, joint position delta). | `steps[*].action` tagged as `human_teleop_xr`. |
| **Goal** | Generate high-fidelity, synchronous **RLDS episodes** for co-training the robot's VLA stack. | |

### 2.3 XR Observer Mode (Mode C: Annotation & Supervision)
This mode provides tools for passive supervision, crucial for defining safety boundaries and refining reward functions.

| Capability | Features | Data Logged (RLDS Step Action) |
| :--- | :--- | :--- |
| **Supervision** | Live state visualizations, overlays of predicted trajectories and safety zones. | `steps[*].action` includes Annotations (polygons, masks, flags like `good/bad`), commands to tweak thresholds or safe zones. |
| **Goal** | Provide supervision for **perception, safety models, and reward shaping** in Continuon Cloud. | |

### 2.4 Companion Apps and Visualization
ContinuonXR includes applications for non-XR devices:
*   **Flutter Apps:** For phones/desktop, providing setup, quick teleop, logs, and a mobile RLDS browser.
*   **Visualization:** Optional WebXR viewers and Unity/OpenXR simulators for visualization and offline simulations.

***

## III. Technical Requirements & Data Contracts

### 3.1 Core Platform & Architecture
*   **Platform:** Native Android XR app built with **Kotlin + Jetpack XR (Compose for XR + SceneCore)** on Galaxy XR hardware.
*   **Communication:** Must establish a local **gRPC/WebRTC link** to the **Continuon Brain runtime** (formerly ContinuonBrain/OS) hosted on the paired edge device or dock for real-time state and command exchange. Must use HTTPS/WebSockets to communicate with **Continuon Cloud**.
*   **Edge Integration:** Must respect and use the stable **Robot API** exposed by the Continuon Brain runtime.

### 3.2 RLDS Data Contract (Must be Strictly Implemented)
ContinuonXR must log all experiences into a standardized RLDS-style schema.

| RLDS Field | Requirement | Source |
| :--- | :--- | :--- |
| **`episode_metadata`** | Must include `continuon.xr_mode` (`"trainer"`, `"workstation"`, etc.) and `continuon.control_role` (`"human_teleop"`) for cloud processing. | |
| **`steps[*].observation`** | Must include `xr_headset_pose` and `xr_hand_right_pose` (pose/quat data). Must synchronize egocentric video/depth with robot state. | |
| **`steps[*].action`** | Must record human command in a normalized space, tagged with `source = "human_teleop_xr"`. | |

### 3.3 Sensorized Glove Integration (Continuon Glove v0)
ContinuonXR must integrate the sensorized glove as a dedicated input stream for highly dexterous training.

*   **Connectivity:** Must connect to the glove (e.g., XIAO nRF52840 Sense) via **BLE** and negotiate MTU ≥ 64 bytes for frame transfer.
*   **Data Structure:** Must parse the raw byte array stream (e.g., 45 bytes) into a Kotlin data class containing normalized sensor readings.
*   **RLDS Fields:** The application must fuse the glove data into the `observation` block using specific fields:
    *   `glove.flex`: 5 normalized floats (finger bend).
    *   `glove.fsr`: 8 normalized floats (contact/pressure pads).
    *   `glove.orientation_quat`: 4 floats (hand orientation).
    *   `glove.accel`: 3 floats (acceleration).

***

## IV. Execution, Metrics, and Validation

### 4.1 Development Methodology (MVP Focus)
The project should follow a **Lean methodology** in its initial stages to maximize learning and reduce waste, transitioning to a hybrid Lean-Scrum structure as the team scales.

### 4.2 Minimum Viable Product (MVP) Definition (Phase 1)
The initial MVP must achieve the core function of data capture and interface.

| Phase | Goal | Key Deliverable |
| :--- | :--- | :--- |
| **Phase 0** | **Contracts** | Define [RLDS schema](docs/rlds-schema.md) and [XR app contract](docs/xr-app-spec.md) in the documentation folder. |
| **Phase 1 (MVP)** | **Lab Prototype** | Jetpack XR MVP with basic panels + functional teleop (Mode A) to a mock Continuon Brain runtime instance. Single local service saving RLDS episodes. |

See [`docs/unified-roadmap.md`](docs/unified-roadmap.md) for the authoritative owners, dates, and current phase status that align these MVP targets with the Pi 5 lifecycle milestones.

### 4.3 Success Metrics (KPIs)

| Metric Category | Target KPI | Rationale |
| :--- | :--- | :--- |
| **Data Quality/Retention** | **95%** of all completed XR Trainer Mode (Mode A) sessions are logged as valid, clean RLDS episodes. | Ensures sufficient **gold data** is produced for imitation learning (high data retention rate). |
| **Performance** | **100 Hz** reliable BLE data streaming from Continuon Glove to the XR app. | Guarantees low latency and fidelity for dexterous training inputs. |
| **Integration** | 100% successful bidirectional communication (gRPC/WebRTC) between ContinuonXR and the Continuon Brain runtime module in this repo. | Verifies the core link in the data loop is robust. |
| **Engagement** | Daily/Weekly Active Users (DAU/WAU) of the XR Trainer Mode (Mode A). | Measures developer/operator engagement in creating new training data. |

### 4.4 Ingestion and Upload Safety (Offline-First)
- Default to offline/local RLDS logging across XR and the Continuon Brain runtime; uploads are manual/opt-in only and must follow the gating rules in [`continuon-lifecycle-plan.md`](continuon-lifecycle-plan.md).
- Before enabling uploads, apply the [Upload Readiness Checklist](docs/upload-readiness-checklist.md): capture explicit consent, curate/redact episodes, package zips with provenance manifests, and verify checksums/signatures over TLS before marking data as exported.
- Cloud-facing ingestion features in MVP and later phases must inherit these controls so provenance and operator intent remain aligned with the lifecycle plan.

***

## V. Continuon Brain Runtime on Raspberry Pi 5 (ContinuonBrain)

The Pi 5-based Continuon Brain runtime is the edge executor that ingests XR commands, runs the robot control stack, and handles local learning before synchronizing with Continuon Cloud. This section extends the runtime contract to keep hardware, control loops, HAL boundaries, and OTA flows consistent with the XR and cloud products.

### 5.1 Target Hardware & Accelerators
* **Compute**: Raspberry Pi 5 (8 GB) with onboard VideoCore VII GPU and PCIe-attached Coral Edge TPU (preferred) or Intel NPU hats as optional accelerators. Thermal budget must allow sustained 50–100 ms control loops without CPU/GPU throttling.
* **AI Runtime**: TFLite primary executor (Edge TPU delegate when available) with ONNX Runtime fallback for non-delegable ops.
* **Connectivity**: Dual-band Wi-Fi for OTA, wired Ethernet for lab reliability, GPIO/UART/CAN for actuator buses, and CSI for vision sensors.

### 5.2 HAL Interfaces for Sensors and Actuators
* **Sensor HAL**: Standardized modules for RGB-D cameras (CSI/USB), IMU (I2C/SPI), force/torque sensors (CAN/USB), joint encoders, and glove telemetry bridged from ContinuonXR. Each HAL exposes timestamped packets normalized to SI units and device health signals (`temp`, `crc_ok`, `dropped_frames`).
* **Actuator HAL**: Abstracted drivers for differential drive, robotic arms (ROS2-compatible topics), and grippers. Commands are expressed as target velocity/torque or EE twists; HAL enforces rate limiting, torque bounds, and safety interlocks (E-stop, watchdog).
* **Contract**: All HAL modules publish/consume Protobuf messages defined in `proto/` and surfaced via the runtime’s gRPC server to keep XR and cloud simulators aligned.

### 5.3 Control Loop Budget and Scheduling
* **Primary Control Loop**: 50–100 ms (10–20 Hz) deterministic tick for sensor fusion, inference, and actuator commands. Scheduler prioritizes safety interlocks > state estimation > policy inference > logging.
* **Clock Discipline**: Monotonic timebase with NTP drift correction; dropped ticks trigger a safe deceleration profile and flag the RLDS episode.
* **CMS Fast/Mid/Slow Recurrence**:
  * **Fast Loop (5–10 ms)**: Hardware safety + contact detection running on microcontroller/firmware where available; exposes heartbeat to the runtime.
  * **Mid Loop (50–100 ms)**: Policy inference + state estimation + actuator command emission (default loop above).
  * **Slow Loop (0.5–1 s)**: Map updates, diagnostics, and background health checks; bounded to <10% CPU.

### 5.4 HOPE Wave/Particle Recurrence
* **Wave (Scene/State)**: Aggregates multi-sensor context into a latent scene graph every mid-loop tick; maintains short-term buffers for trajectory history.
* **Particle (Action Planning)**: Samples/plans candidate actions from the Wave state, prunes by safety envelopes, and commits the selected action to the mid loop.
* **Integration**: Both phases log intermediate tensors to RLDS `observation`/`action` extras for debuggability and cloud replay.

### 5.5 Brain Profile Loader
* **Profiles**: Versioned bundles containing policy graphs (TFLite/ONNX), HAL configuration, safety parameters, and CMS loop settings. Profiles are described by a `profile.yaml` manifest with signatures and semantic versioning.
* **Loader Behavior**: Validates bundle signatures, ensures accelerator compatibility (Edge TPU vs CPU fallback), warms models, and atomically swaps active profiles. Rejects profiles that exceed loop latency budgets or HAL capability.

### 5.6 Edge Bundle & OTA Flow with Rollback
* **Package Format**: Immutable “edge bundles” (container image or signed tarball) carrying the brain profile, HOPE configs, and firmware blobs for microcontrollers.
* **Deployment Flow**: Download to a staging slot, verify signatures/checksums, run preflight health + latency checks, then activate with an atomic symlink swap. Maintain A/B slots for rollback.
* **Rollback**: Automatic rollback on watchdog triggers (missed heartbeats, control loop >100 ms for 3 consecutive ticks, or HAL fault) and manual rollback via XR/CLI with preserved logs for cloud diagnosis.

### 5.7 On-Device Learning Scope
* **Supported**: Short-horizon fine-tuning of value/policy heads (e.g., adapter layers) using recent RLDS buffers; supervised reward model calibration; latency-aware distillation for Edge TPU.
* **Not Supported**: Full foundation model retraining on-device. Large-batch updates are deferred to Continuon Cloud; the runtime only performs lightweight, bounded compute updates and logs gradients/metrics for cloud review.
* **Safety**: On-device updates are sandboxed; failed updates revert to the prior parameters and trigger a health report to the cloud.

### 5.8 Seed Model Promotion to Continuon Cloud
* **Seed Model**: Runtime ships with a seed TFLite policy suited for Pi 5 + Edge TPU. Promotion path: `seed.tflite` → local adapter fine-tunes → packaged as edge bundle → uploaded via lifecycle plan gates.
* **Continuon Cloud Integration**: Cloud validates provenance, replays RLDS episodes, and retrains at scale; successful cloud builds produce new brain profiles signed and pushed back to the OTA channel.
* **Operator Controls**: XR app surfaces promotion intent, shows diff of policy metadata, and requires explicit approval before cloud promotion or OTA activation.
