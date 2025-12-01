**Product Requirements Document (PRD): ContinuonXR**
**Repository Owner:** @ContinuonAI
**Repository Name:** `ContinuonXR`
**Version:** 1.0 (Initial Development)

***

## I. Strategic Foundations & Goals

### 1.1 Product Vision
ContinuonXR is the human-facing application that serves as the secure, spatial workstation, trainer interface, and the **primary data ingestion point** for the Continuon self-learning ecosystem. It overcomes the "static" nature of traditional models by continually supplying high-quality, human-generated **RLDS episodes** (gold data) to the **Continuon Cloud Factory**. All Continuon products and services (XR, Cloud, ContinuonBrain/OS runtime, web explorer, and orchestration) are built and shipped from this repository to keep architecture and contracts aligned.

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
*   **Communication:** Must establish a local **gRPC/WebRTC link** to the **ContinuonBrain/OS** running on the docked Pixel 10 for real-time state and command exchange. Must use HTTPS/WebSockets to communicate with **Continuon Cloud**.
*   **Edge Integration:** Must respect and use the stable **Robot API** exposed by ContinuonBrain/OS.

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
| **Phase 1 (MVP)** | **Lab Prototype** | Jetpack XR MVP with basic panels + functional teleop (Mode A) to a mock ContinuonBrain/OS instance. Single local service saving RLDS episodes. |

See [`docs/unified-roadmap.md`](docs/unified-roadmap.md) for the authoritative owners, dates, and current phase status that align these MVP targets with the Pi 5 lifecycle milestones.

### 4.3 Success Metrics (KPIs)

| Metric Category | Target KPI | Rationale |
| :--- | :--- | :--- |
| **Data Quality/Retention** | **95%** of all completed XR Trainer Mode (Mode A) sessions are logged as valid, clean RLDS episodes. | Ensures sufficient **gold data** is produced for imitation learning (high data retention rate). |
| **Performance** | **100 Hz** reliable BLE data streaming from Continuon Glove to the XR app. | Guarantees low latency and fidelity for dexterous training inputs. |
| **Integration** | 100% successful bidirectional communication (gRPC/WebRTC) between ContinuonXR and the ContinuonBrain/OS runtime module in this repo. | Verifies the core link in the data loop is robust. |
| **Engagement** | Daily/Weekly Active Users (DAU/WAU) of the XR Trainer Mode (Mode A). | Measures developer/operator engagement in creating new training data. |

### 4.4 Ingestion and Upload Safety (Offline-First)
- Default to offline/local RLDS logging across XR and ContinuonBrain/OS; uploads are manual/opt-in only and must follow the gating rules in [`continuon-lifecycle-plan.md`](continuon-lifecycle-plan.md).
- Before enabling uploads, apply the [Upload Readiness Checklist](docs/upload-readiness-checklist.md): capture explicit consent, curate/redact episodes, package zips with provenance manifests, and verify checksums/signatures over TLS before marking data as exported.
- Cloud-facing ingestion features in MVP and later phases must inherit these controls so provenance and operator intent remain aligned with the lifecycle plan.
