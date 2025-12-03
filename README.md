# Continuon AI: Self-Learning Robotics Ecosystem

**Transforming personal robots into continuously learning assistants through the "One Brain, Many Shells" architecture.**

## Build & CI Status

---

## Scoped Contributor Guidance

This repository relies on nested `AGENTS.md` files for area-specific rules (toolchains, testing expectations, and product
boundaries). See [`docs/AGENTS_ENFORCEMENT.md`](docs/AGENTS_ENFORCEMENT.md) for tips and automation ideas that help teams
consistently follow the right scope before editing or shipping changes.

## Project Status & Milestones

The roadmap below highlights where the Android XR app and data capture stack stand today, what is actively in-flight, and what is targeted next. Owners reflect day-to-day accountability for delivery.

| Area | Current Status | Owner | Target Date |
| --- | --- | --- | --- |
| XR App Shell & Navigation | Core shell, spatial panels, and input routing shipped in dogfood build; stability fixes in progress | XR Client Team | 2025-11-24 |
| Glove BLE Parser & Telemetry | BLE link stable with flex/force streaming; expanding reconnection safeguards | XR Client Team | 2025-11-24 |
| RLDS Logging on Device | Proto generation complete and logging enabled for head/hand poses; optimizing write batching | XR + Data Infra | 2025-11-24 |
| Data Capture Rig (Sensors + Video) | Depth + RGB capture validated; synchronized audio alignment underway | XR + Sensors | 2025-11-24 |
| Cloud Ingestion Hooks | Upload path wired to staging bucket; signed upload & provenance tagging next | Cloud Ingestion | 2025-11-24 |
| **Pi5 Robot Arm Integration** | **OAK-D + PCA9685 validated; Flutter UI + RLDS recorder ready for hardware** | **Edge Team** | **2025-12-01** |

### Pi5 Robot Arm Next Steps

**Hardware Auto-Detection:**
ContinuonBrain now includes intelligent hardware auto-detection for cameras, HATs, servo controllers, and accessories:
```bash
# Auto-detect all connected hardware
PYTHONPATH=$PWD python3 continuonbrain/sensors/hardware_detector.py

# System health check (automatic on wake from sleep)
PYTHONPATH=$PWD python3 continuonbrain/system_health.py --quick

# Wake robot with full services (discovery, API, modes)
PYTHONPATH=$PWD python3 continuonbrain/startup_manager.py
```

See [Hardware Detection](docs/hardware-detection.md), [System Health](docs/system-health.md), and [Robot Wake-Up](docs/robot-wakeup.md) for details.

**Without Hardware (Current Phase):**
1. ✅ Design validated with mock mode (OAK-D Lite + PCA9685 + SO-ARM101)
2. ✅ Hardware auto-detection for OAK-D, PCA9685, Hailo HAT+, IMUs
3. ✅ Wake-up orchestration with LAN discovery and mode management
4. ✅ Flutter UI with manual training, autonomous, and sleep learning modes
5. 🔄 Test Robot API communication flow (JSON-over-TCP)
6. 🔄 Prepare episode upload pipeline to cloud

**When Robot Arm Arrives:**
1. Connect SO-ARM101 servos to PCA9685
2. Run `PYTHONPATH=$PWD python3 continuonbrain/tests/integration_test.py --real-hardware` (auto-detects hardware)
3. Calibrate joint limits in `ArmConfig` per physical constraints
4. Record 16+ training episodes for first LoRA adapter

**When AI HAT+ Arrives:**
1. Stack Hailo-8L accelerator on Pi5 (auto-detects via PCIe)
2. Convert Gemma-based VLA to Hailo format
3. Update manifests to load from Hailo accelerator
4. Test inference latency with real depth frames

See `continuonbrain/README.md` and `continuonbrain/PI5_CAR_READINESS.md` for implementation details.

---

## Overview

Continuon is an end-to-end platform that overcomes the static nature of traditional AI models by creating a continuous learning loop where every human interaction becomes training data. The platform decouples robot intelligence from physical form factors, enabling a single AI brain to control diverse robotic platforms while continuously improving through real-world experience. This repository is the **sole home** for the full Continuon stack (XR apps, ContinuonBrain/OS runtime, cloud training factory, web explorer, and org tooling); all products and services are built and versioned here with clear module boundaries instead of across separate repos.

### Core Philosophy: "One Brain, Many Shells"

The central intelligence (the "Brain") is morphology-agnostic and can inhabit any robotic platform (the "Shells"). This separation enables:
- **Unified Learning**: All robots contribute to and benefit from a shared learning system
- **Rapid Deployment**: New robot platforms can be supported by implementing a standard Hardware Abstraction Layer
- **Continuous Improvement**: The brain evolves through multi-modal data from diverse deployment contexts

---

## Architectural Overview

### The HOPE Architecture: Hierarchical Optimizer, Perpetual Engine

Continuon's AI is built on the **Nested Learning (NL)** paradigm, treating the Vision-Language-Action (VLA) model as a system of nested, multi-level optimization problems that update at different frequencies to prevent catastrophic forgetting.

#### Continuum Memory System (CMS): Multi-Timescale Learning

The CMS structures learning into three loops:

| Loop | Timescale | Responsibility | Update Mechanism |
|------|-----------|----------------|------------------|
| **Fast Loop** | 50-100 ms | Low-level motor skills, reflexive safety, teleop mirroring | Online gradient steps, safety overrides |
| **Mid Loop** | 0.5-10 s | Skill sequencing, intent inference, short-horizon world modeling | Episodic fine-tuning, contextual bandits |
| **Slow Loop** | Minutes-Hours | High-level planning, semantic alignment, world model training | Corpus-scale training, generative pre-training |

All three loops run in both places: on-device/edge execution keeps latency low, while cloud retraining replays the same loops for regression and distillation. The Slow loop produces the "core" checkpoints that distill into Fast/Mid edge bundles, and on-device Fast/Mid traces flow back into the next cloud retrain to preserve reflexes and UI/session adaptations.

See [Model Lifecycle: HOPE/CMS Governance](docs/model_lifecycle.md) for how OTA packaging, Memory Plane persistence, and cloud replay map to these loops.

#### Sequence Core Beyond Transformers
HOPE/CMS is about multi-timescale memory, not a mandatory attention backbone. We implement the continuum memory with a particle + wave split: the local "particle" path (small attention windows, conv/MLP adapters) updates every step for exact positions and short-range dependencies, while the "wave" path uses SSMs and spectral operators (Mamba, Hyena/GFN, Griffin-style hybrids) to maintain compressed global state. Fast/Mid loops on edge (Pi 5 + Hailo; see `continuonbrain/README.md` and `apps/continuonxr/README.md`) update particle paths continuously and refresh compact SSM states per chunk/episode. The Slow loop in cloud (`continuonai/continuon-cloud/README.md`) trains longer-horizon SSM/spectral cores and ships OTA bundles that merge with the Memory Plane instead of overwriting it, keeping HOPE's nested optimization intact while scaling past attention-only Transformers.

#### VLA Stack: Unified Multi-Task Architecture

The robot's intelligence comprises five specialized heads sharing a common perception backbone:

1. **VisionCore**: World-aligned perception with 3D scene understanding
2. **LanguagePlanner**: Next-token generation, reasoning, and high-level planning
3. **World Model Head**: Predictive simulation and foresight (0.5-1s lookahead)
4. **SkillPolicies**: Manipulation and locomotion primitives
5. **SafetyHead**: Reflexive hazard detection and policy guardrails

---

## The Continuon Ecosystem: Monorepo Architecture

This single repository hosts every Continuon product with clear in-repo module boundaries instead of relying on other repos:

| Module | Path | Purpose | Status |
|--------|------|---------|--------|
| **ContinuonXR** | `apps/continuonxr/` | Spatial UI & data capture rig (Android XR, glove BLE parsing, RLDS logging) | Active |
| **ContinuonBrain/OS Runtime** | `continuonbrain/` + `continuonai/` | Robot OS/edge runtime scaffolding, HAL interfaces, OTA client & contracts | Active (consolidated here) |
| **ContinuonAI App + RLDS Portal** | `continuonai/` (+ `continuonai/continuon-cloud/`) | Flutter consumer app/robot controller with integrated WorldTape RLDS browser/annotation surfaces; Cloud ingestion/train specs | Active (worldtape consolidated here) |

### In-Repository Contracts

**RLDS Schema** (`docs/rlds-schema.md`): Versioned data contract for all robot experiences
**XR App Contract** (`docs/xr-app-spec.md`): Modes, runtime interfaces, and logging expectations for the Android XR client
**Robot API** (`proto/continuonbrain_link.proto`): gRPC/WebRTC interface between XR and robot runtime
**Edge Bundles** (`docs/bundle_manifest.md`): Signed, versioned AI model deployment packages

---

## Data Flow: From XR Capture to Deployed Intelligence

### Quickstart: Donkey Car + Flutter Companion → Continuon Cloud

Use this path to run the continuous learning loop on a Raspberry Pi 5-powered Donkey Car with an iPhone 16 Pro companion app (Cloud ingest/train/package staging lives under `continuonai/continuon-cloud/` for the Google Cloud backend):

1. **Edge bridge on Pi (ContinuonBrain/OS)**
   - Run the lightweight ContinuonBrain/OS bridge to expose the Robot API over gRPC/WebRTC.
   - Normalize Donkey Car steering/throttle into `action.command` and log drivetrain + sensor feedback into `observation.robot_state` per RLDS.
   - Keep camera/IMU/encoder timestamps aligned within **≤5 ms** so video and robot state stay synchronized for training.

2. **Flutter companion as XR shell**
   - Use the iPhone app for setup, quick teleop, and data review; emit RLDS episodes tagged with `xr_mode="trainer"` and `action.source="human_teleop_xr"`.
   - Capture substitute head/hand/gaze signals from phone sensors plus on-screen joystick commands, then buffer locally before forwarding episodes + media to Cloud over HTTPS/WebSocket using the `metadata.json` + `steps/*.jsonl` layout.

3. **Cloud ingestion + training loop**
   - Land uploads in RawLake → cleaning → standardized RLDS to preserve provenance and handle intermittent sensors.
   - Train the multi-head VLA stack (world model, policy, language, safety) on new episodes and export quantized TFLite heads into an edge bundle.

4. **Deployment back to Pi**
   - Deliver signed edge bundles via the ContinuonBrain/OS OTA module in this repo; authenticate, download, verify, and hot-swap models while keeping a local fallback policy.
   - Continue logging RLDS during autonomous follow-me runs to fuel the next training round.

5. **Operational checklist**
   - Verify sensor sync (≤5 ms), populate `episode_metadata` with environment IDs (e.g., `donkey-pi5`) and software versions, and tag modes correctly for follow-me demonstrations.
   - Use the in-repo module split intentionally: **ContinuonXR**/companion for capture, **Continuon Cloud** modules for training, **ContinuonBrain/OS** for deployment—all housed here.

### Ingestion safety & offline-first policy

- Default to **offline-first logging** for XR and Pi 5 targets. Cloud uploads are **manual/opt-in** and should only be enabled after following the [Upload Readiness Checklist](docs/upload-readiness-checklist.md).
- The Pi 5 lifecycle plan documents the gating steps (manual consent, provenance signing/checksums, TLS validation, post-upload verification) in [`continuon-lifecycle-plan.md`](continuon-lifecycle-plan.md); ingestion features must inherit those rules.
- When enabling uploads, stage curated RLDS zips locally, include provenance manifests, and perform checksum/signature verification before marking episodes as exported.

### 1. Data Capture (ContinuonXR & ContinuonBrain/OS)

ContinuonXR captures high-fidelity human demonstrations across multiple modes:

See the detailed on-device contract in [`docs/xr-app-spec.md`](docs/xr-app-spec.md) for required endpoints, mode tags, and RLDS logging expectations that keep capture aligned with cloud ingestion.

#### Mode A: XR Trainer (Direct Robot Control)
- **What**: Human teleoperates robot through XR headset
- **Data Logged**: 
  - XR head/hand poses (100 Hz)
  - Egocentric video + depth
  - Continuon Glove telemetry (flex, force, orientation)
  - Robot state feedback (joints, end-effector pose)
  - Synchronized audio
- **RLDS Tags**: `xr_mode="trainer"`, `control_role="human_teleop"`

#### Mode B: Spatial Workstation (Workflow Context)
- **What**: Developer uses XR as PC replacement while managing robots
- **Data Logged**:
  - UI context (active panels, file focus, dashboard state)
  - Gaze fixation points
  - Workflow actions (run tests, deploy, label runs)
  - Voice commands
- **RLDS Tags**: `xr_mode="workstation"`, `source="human_dev_xr"`

#### Mode C: Observer (Annotation & Supervision)
- **What**: User adds safety boundaries and quality labels
- **Data Logged**:
  - Polygon/mask annotations
  - Safety zone definitions
  - Success/failure labels
- **RLDS Tags**: `xr_mode="observer"`, annotations in `action.annotation`

#### Mode D: YouTube/Cloud TV (Internet Data)
- **What**: Curated internet videos normalized to RLDS
- **Data Logged**:
  - Video with synthetic pose estimation
  - ASR-derived audio transcripts
  - Vision-derived depth and affordances
- **RLDS Tags**: `xr_mode="youtube_tv"`, provenance metadata

### 2. Cloud Ingestion & Augmentation (Continuon-Cloud)

**Data Pipeline Flow:**

```
Raw Episodes → RawLake → Scizor/Golden Cleaning → Standardized RLDS
                                                         ↓
                                              NeRF Synthesis (multi-view)
                                              Tactile Hallucination Networks
                                                         ↓
                                              Augmented Training Corpus
```

**Key Capabilities:**
- **NeRF Scene Reconstruction**: Generate synthetic viewpoints to supervise state encoders
- **Visuo-Tactile Hallucination**: Predict missing tactile feedback when glove data absent
- **Quality Stratification**: Tag episodes by source confidence (XR gold data vs internet video)
- **Handling Missing Data**: Preserve `valid` flags and provenance for uncertainty-aware training

### 3. Multi-Task Training (Continuon-Cloud)

**Nested Optimization Training Loss:**

```
L_total = L_world_model + L_policy + L_language + L_safety
```

**Training Infrastructure:**
- Platform: Google Cloud (Vertex AI / GKE)
- Data: RLDS episodes from all sources
- Co-Training: VisionCore backbone updates require synchronized retraining of all dependent heads
- Output: Quantized TFLite models with INT8 precision

### 4. Edge Bundle Creation (Continuon-Cloud)

**Edge Bundle Contents:**
- **Model Files**: Quantized TFLite models for each VLA head
- **Manifest**: Version, compatibility constraints, preferred hardware backends
- **Gemma Weights**: Language model parameters
- **Digital Signature**: Cryptographic verification payload

**Manifest Example:**
```json
{
  "bundle_version": "2.1.0",
  "models": {
    "vision_core": "vision_core_v2.1.0_int8.tflite",
    "skill_policy": "skills_v2.1.0_int8.tflite"
  },
  "preferred_backends": ["nnapi", "xnnpack"],
  "compatibility": {
    "min_continuonos_version": "1.5.0"
  }
}
```

### 5. Secure Deployment (ContinuonBrain/OS OTA)

**Deployment Process:**

1. **Authentication**: Robot OTA client authenticates with Cloud
2. **Download**: Fetch signed Edge Bundle over HTTPS
3. **Verification**: 
   - Signature validation
   - Integrity checksums
   - Compatibility checks
4. **Hot-Swap**: 
   - Load new model while old model continues running
   - Atomic switch after functional verification
   - Maintain last-known-good model for instant rollback
5. **Telemetry**: Report deployment success/failure to Cloud

**Safety Guarantees:**
- No downtime during updates
- Automatic rollback on verification failure
- Coordinated updates for co-dependent models

### 6. Edge Runtime Execution (ContinuonBrain/OS)

**Platform-Agnostic Brain Runtime:**

```
src/core/           # Platform-neutral control loop, scheduler, TFLite loader
platform/android/   # NNAPI delegate for Pixel TPU acceleration
platform/linux_sbc/ # V4L2/GPIO for Raspberry Pi
```

**Hardware Abstraction Layer (HAL):**
- `SensorInterface`: Camera, depth, IMU abstractions
- `ActuatorInterface`: Motor/servo control
- `TimeInterface`: Monotonic clocks for synchronization
- `StorageInterface`: Episode logging and caching

**Dynamic Backend Selection:**
- Same TFLite model adapts to available hardware
- Pixel NPU via NNAPI
- ARM CPU via XNNPACK
- x86 desktop via CPU interpreter

**Real-Time Execution:**
- **Fast Loop**: On-device inference at 50-100 Hz for reactive control
- **Mid Loop**: World model imagination at 0.5-1 Hz for proactive planning
- **Slow Loop**: LLM reasoning invoked event-driven (5-60s or on-demand)

---

## Robot Platform Coverage

This repository ships both the XR-side tooling and the ContinuonBrain/OS runtime/HAL modules that power the robots. The table below tracks current coverage and obvious gaps so contributors know where to help.

| Robot Platform | HAL Interfaces Available | Current Coverage | Known Gaps |
|----------------|--------------------------|------------------|------------|
| **Hello Robot Stretch 3** | `ActuatorInterface` (arm + lift), `SensorInterface` (RGB-D), `SafetyHead` hooks | Teleop verified with mock backend; RLDS logging validated | Gripper force control missing; depth → point cloud conversion still stubbed; e-stop feedback not plumbed |
| **Unitree Go2** | `ActuatorInterface` (locomotion), `SensorInterface` (stereo, IMU), `TimeInterface` | Locomotion skill replay in sim; waypoint streaming | Manipulator/gripper HAL unimplemented; onboard perception passthrough to cloud missing |
| **Franka Emika Panda** | `ActuatorInterface` (arm), `SensorInterface` (RGB), `StorageInterface` | Offline teleop playback via mock brain | Real-time torque mode not exposed; tactile pads unsupported; workspace safety envelopes manual only |
| **ROS 2 Generic** | `continuonbrain_link.proto` bridge | Command/feedback bridge tested in bag replay | Sensor discovery limited to RGB; depth/point cloud topics ignored; no standard gripper mapping |

### Action List for Missing Integrations
- Implement **gripper drivers** for Stretch 3 (force + slip sensing) and Panda (torque-aware grasping) under ContinuonBrain HAL adapters in this repo; validate end-to-end through XR teleop.
- Add **depth/point cloud ingestion** for Stretch 3 and ROS 2 generic bridge, wiring conversions into RLDS logging and VisionCore calibration.
- Expose **perception passthrough** on Unitree Go2 (stereo + IMU) to the XR client so spatial UI mirrors robot state.
- Wire **safety systems** (e-stop feedback and workspace bounds) into `SafetyHead` for Stretch 3 and Panda, surfacing alerts in the XR HUD.
- Extend **manipulator interfaces** for Unitree Go2 when an arm is present, including gripper mapping and calibration flows.

---

## Building the Ecosystem: Phased Development Plan

The single source of truth for owners, dates, and KPIs across Phases 0–4 lives in [`docs/unified-roadmap.md`](docs/unified-roadmap.md). The roadmap links the PRD MVP scope, KPIs, and the Pi 5 lifecycle milestones; Phase 1 is the current focus.

### Phase 0: Contracts & Architecture ✓

**Deliverables:**
- [x] RLDS schema definition (`docs/rlds-schema.md`)
- [x] XR app specification (`docs/xr-app-spec.md`)
- [x] HOPE/CMS architecture (`docs/hope-cms-vla.md`)
- [x] Repository boundaries (`docs/monorepo-structure.md`)
- [x] Ecosystem alignment (`docs/ecosystem-alignment.md`)

### Phase 1: MVP Data Capture (Current)

**Goals:**
- [ ] Jetpack XR app with basic spatial UI
- [ ] Mode A teleop to mock ContinuonBrain/OS
- [ ] Continuon Glove BLE integration (100 Hz)
- [ ] Local RLDS episode writer with schema validation
- [ ] Manual upload to Cloud

**Key Metrics:**
- 95% of Mode A sessions → valid RLDS episodes
- 100 Hz reliable glove data streaming
- Bidirectional gRPC with ContinuonBrain/OS mock

### Phase 2: Cloud Integration

**Goals:**
- [ ] Automated RLDS upload with retry logic
- [ ] Scizor/Golden cleaning pipeline
- [ ] Basic VLA training loop (VisionCore + SkillPolicy)
- [ ] First edge bundle deployment

### Phase 3: Closed-Loop Learning

**Goals:**
- [ ] OTA updates from Cloud to ContinuonBrain/OS
- [ ] Hot-swap with rollback capability
- [ ] Telemetry feedback loop
- [ ] Multi-timescale training (Fast/Mid/Slow)

### Phase 4: Production Scale

**Goals:**
- [ ] Fleet management dashboard
- [ ] ContinuonAI RLDS/annotation tools (WorldTape portal in-app)
- [ ] Full VLA stack (all 5 heads)
- [ ] Continuous self-improvement metrics

---

## ContinuonXR Repository: Getting Started

This repository contains the **spatial UI and data capture rig** component of the Continuon ecosystem.

### Repository Structure

```
  ContinuonXR/
    apps/
      continuonxr/          # Android XR application (Kotlin + Jetpack XR)
    docs/                   # Architecture and contract documentation
      rlds-schema.md        # RLDS data contract
      hope-cms-vla.md       # HOPE architecture details
      ecosystem-alignment.md
      xr-app-spec.md
    proto/                  # Protobuf definitions
      rlds_episode.proto    # RLDS schema
      continuonbrain_link.proto  # Robot API
    continuonbrain/         # ContinuonBrain/OS scaffolding, Robot API server entrypoint, trainer modules
    continuonai/            # ContinuonAI Flutter app (web/iOS/Android/Linux) plus consolidated Cloud docs
    worldtapeai.com/        # redirect stub; WorldTape RLDS portal now lives inside continuonai/
  ```

Note: `continuonbrain/trainer/` contains an offline Pi/Jetson LoRA adapter-training scaffold (bounded jobs, RLDS-only inputs, safety-gated promotion) to stay aligned with ContinuonBrain/OS goals. The Robot API server lives in `continuonbrain/robot_api_server.py`; integrate deeper continuonos runtime/OTA paths in the dedicated `continuonos` repo.

Quick Pi 5 migration tips (Jetson → Pi):
- Copy RLDS episodes to `/opt/continuonos/brain/rlds/episodes`.
- Place your Gemma base model (non-quantized edge build if it fits) and point `base_model_path` in `pi5-donkey.json`.
- Use `continuonbrain/model/manifest.pi5.example.json` as a template for `flutter_gemma` to load base + LoRA.
- For smoke tests, you can duplicate a few episodes to meet `min_episodes`, then replace with real runs before training.
## Pi 5 Offline Brain Setup (Gemma + LoRA, no quant)

1) Clone on Pi  ```bash
cd /opt/continuonos/brain
git clone https://github.com/continuonai/ContinuonXR.git pi5-brain
cd pi5-brain
```

2) Create dirs & place models  ```bash
sudo mkdir -p /opt/continuonos/brain/model/{base_model,adapters/current,adapters/candidate,adapters/history}
sudo mkdir -p /opt/continuonos/brain/rlds/episodes
sudo mkdir -p /opt/continuonos/brain/train
```
- Put Gemma 3n base (non-quant edge build if it fits) at `/opt/continuonos/brain/model/base_model/gemma-3n.tflite`.
- If you have current adapters, place `/opt/continuonos/brain/model/adapters/current/lora_adapters.pt`.
- Stage RLDS episodes (JSON/JSONL/TFRecord) under `/opt/continuonos/brain/rlds/episodes/`.

3) Config paths  ```bash
cp continuonbrain/configs/pi5-donkey.json /opt/continuonos/brain/train/pi5-donkey.json
# edit base_model_path/rlds_dir/budgets as needed
```

4) Python deps (venv recommended)  ```bash
python3 -m venv venv
source venv/bin/activate
pip install torch  # choose the Pi build you use
```

5) Optional: runtime manifest for flutter_gemma  ```bash
cp continuonbrain/model/manifest.pi5.safety.example.json /opt/continuonos/brain/model/manifest.json
# edit base/adapter/safety paths
```

6) Run trainer (Gemma hooks + gating scaffold)  ```bash
python -m continuonbrain.trainer.examples.pi5_integration --config /opt/continuonos/brain/train/pi5-donkey.json
```
- Fails fast if `base_model_path` or `min_episodes` not met.
- Saves adapters to `adapters/candidate/`, promotes to `adapters/current/` on safety pass.

7) Wire gating & safety (replace stubs in the ContinuonBrain/OS runtime here)  - `gating_continuonos.py`: connect idle/battery/thermal/teleop to real signals; set thresholds (e.g., battery ≥40%, CPU ≤75C).  - `safety_head_stub.py`: clamp/log violations; swap in your safety head; log safety flags to RLDS.  - `gemma_hooks.py`: in-place LoRA for Gemma proj layers (`q_proj/k_proj/v_proj/o_proj`); plug in your real loader/loss.

8) Offline guarantee  - No internet for any loop; uploads are manual/opt-in (see `continuonbrain/trainer/CLOUD_EXPORT.md`).


### Prerequisites

- **Android Studio**: Koala or later
- **Android SDK**: Level 35
- **Gradle**: 8.7 (wrapper included)
- **Galaxy XR Device**: For production deployment
- **Continuon Glove v0**: For tactile data capture (optional for testing)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/continuonai/ContinuonXR.git
cd ContinuonXR

# Make gradlew executable
chmod +x gradlew

# Build the XR app
./gradlew :apps:continuonxr:assembleDebug

# Generate protobuf stubs
./gradlew :apps:continuonxr:generateDebugProto

# Run tests
./gradlew :apps:continuonxr:testDebugUnitTest
```

### Development Workflow

1. **Read the Contracts**: Start with `docs/rlds-schema.md` and `docs/xr-app-spec.md`
2. **Review Code Structure**: Explore `apps/continuonxr/src/main/java/com/continuonxr/app/`
3. **Understand Data Flow**: Read `docs/human-centric-data.md`
4. **Setup Environment**: Follow `docs/dev-setup.md`

### Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| Glove BLE | `app/glove/` | Parse Continuon Glove telemetry |
| XR Input | `app/xr/` | SceneCore pose/gaze integration |
| Teleop | `app/teleop/` | Map XR inputs → robot commands |
| RLDS Logging | `app/logging/` | Write/validate/upload episodes |
| Connectivity | `app/connectivity/` | gRPC/WebRTC to ContinuonBrain/OS runtime in this repo |

### Current Implementation Status

**✓ Completed:**
- RLDS schema and validation framework
- Glove BLE parser with MTU negotiation
- Mock ContinuonBrain/OS for testing
- Basic RLDS episode writer

**⚠ In Progress:**
- Jetpack XR/SceneCore integration (stubbed, gated by `ENABLE_XR_DEPS`)
- Live gRPC/WebRTC endpoints (mocked in dev)
- Audio capture pipeline
- Production upload with retry logic

**🔜 Planned:**
- Mode B workstation UI panels
- Mode C annotation tools
- Real-time episode preview
- Fleet configuration management

---

## RLDS Data Contract: The Universal Training Format

All data in the Continuon ecosystem flows through the **RLDS (Reinforcement Learning Dataset)** schema, a standardized format that unifies human demonstrations, robot telemetry, and internet data.

### Episode Structure

```
episode/
├── metadata.json           # Episode-level tags and config
└── steps/
    ├── 000000.jsonl       # Timestamped observations + actions
    ├── 000001.jsonl
    └── ...
```

### Critical Fields for HOPE/CMS

| RLDS Block | Fast Loop Use | Mid Loop Use | Slow Loop Use |
|------------|---------------|--------------|---------------|
| **Poses** (`xr_headset_pose`, `xr_hand_*_pose`) | Stabilize teleop, visual servoing | Calibration drift correction | Scene graph alignment |
| **Gaze** (`gaze.origin`, `direction`, `target_id`) | Attention gating for SafetyHead | Intent decoding | Saliency maps for VisionCore |
| **Vision** (`egocentric_video`, `egocentric_depth`) | Obstacle detection | Object affordance refresh | 3D world model training |
| **Audio** (`audio.uri`, `sample_rate_hz`) | Wake-word reflexes | Dialogue grounding | Multimodal alignment |
| **Glove** (`glove.flex`, `fsr`, `orientation_quat`) | Force reflexes, gripper mirroring | Slip detection | Haptic priors for manipulation |
| **Robot State** (joints, EE pose, velocities) | Safety envelope checks | State estimation fusion | Dynamics model fitting |

### Handling Missing Data

- All sensor blocks optional with `valid` flags
- Cloud can hallucinate missing modalities (tactile, audio)
- Provenance metadata preserved for uncertainty-aware training
- Quality tags enable stratified sampling during training

---

## CI/CD for AI Models: From Data to Deployment

### Model Lifecycle

```
Human Demos (XR) ─┬─→ RLDS Episodes ─→ Cloud Training ─→ Edge Bundle
Robot Telemetry   ─┤                          ↓
Internet Videos   ─┘                   Signed Package
                                               ↓
                                ContinuonBrain/OS OTA Client
                                              ↓
                              ┌──────────────────────┐
                              │ Signature Verify     │
                              │ Hot-Swap (no downtime)│
                              │ Rollback on Failure  │
                              └──────────────────────┘
```

### Bundle Signing & Verification

**Cloud (Continuon-Cloud):**
```python
# Package and sign edge bundle
bundle = create_edge_bundle(
    models={"vision_core": vision_model, "skills": skill_model},
    version="2.1.0",
    backends=["nnapi", "xnnpack"]
)
signed_bundle = sign_bundle(bundle, private_key)
upload_to_cdn(signed_bundle)
```

**Edge (ContinuonBrain/OS):**
```cpp
// OTA client downloads and verifies
EdgeBundle bundle = download_from_cloud(bundle_id);
if (!verify_signature(bundle, public_key)) {
    log_error("Signature verification failed");
    return ROLLBACK;
}
if (!check_compatibility(bundle)) {
    return REJECT_INCOMPATIBLE;
}
hot_swap_models(bundle);  // Atomic switch
```

### Coordinated Updates

When updating shared components like VisionCore:
1. **Cloud**: Co-train VisionCore + SkillPolicies + SafetyHead together
2. **Packaging**: Bundle all dependent models with matching version tags
3. **Deployment**: ContinuonBrain/OS verifies all models present before hot-swap
4. **Validation**: Run functional tests on new model ensemble
5. **Rollback**: If any test fails, instantly revert to previous bundle

---

## Security & Safety

### Data Privacy
- RLDS episodes encrypted at rest and in transit
- PII scrubbing in audio/video streams
- User consent for data upload (opt-in per session)

### Model Security
- All edge bundles cryptographically signed
- Public key pinning in ContinuonBrain/OS
- Version pinning prevents downgrade attacks

### Operational Safety
- SafetyHead provides reflex-level overrides (Fast Loop)
- Latency monitoring triggers safe-stop on degraded performance
- Anomaly detection in deployment telemetry

---

## Contributing

We welcome contributions across all modules in this monorepo. Please:

1. Read the relevant module's `CONTRIBUTING.md`
2. Follow the RLDS schema contract for data-related changes
3. Maintain platform-agnostic design in shared components
4. Add tests for new functionality
5. Update documentation to match code changes

### Key Principles

- **Minimal Changes**: Surgical modifications only
- **Contract Adherence**: Never break RLDS schema or API contracts without version bumps
- **Safety First**: All policy changes must pass SafetyHead regression tests
- **Reproducibility**: Document build/deployment steps thoroughly

---

## Documentation Index

**Architecture:**
- `docs/system-architecture.md` - Complete system architecture and training lifecycle (reconciled design)
- `docs/hope-cms-vla.md` - HOPE architecture and CMS timescales
- `docs/model_lifecycle.md` - Model lifecycle and Memory Plane persistence
- `docs/ecosystem-alignment.md` - In-repo module alignment guide
- `docs/architecture.md` - System overview

**Data Contracts:**
- `docs/rlds-schema.md` - Canonical RLDS schema
- `docs/human-centric-data.md` - Mode-specific data capture

**Implementation:**
- `docs/xr-app-spec.md` - XR application specification
- `docs/glove-ble.md` - Glove integration details
- `docs/dev-setup.md` - Development environment setup

**Project Management:**
- `PRD.md` - Product Requirements Document
- `docs/monorepo-structure.md` - Repository boundaries

---

## Learn More

- **Website**: [continuon.ai](https://continuon.ai) *(in development)*
- **RLDS Browser**: ContinuonAI app web build (WorldTape RLDS portal; see `continuonai/README.md`)
- **Research**: See `docs/hope-cms-vla.md` for technical details
- **Community**: Discord server *(coming soon)*

---

## License

[To be determined - placeholder]

---

**Continuon**: Building robots that learn from every interaction, one episode at a time.
## Pi 5 Offline Brain Setup (Gemma + LoRA, no quant)

1) Clone on Pi  
```bash
cd /opt/continuonos/brain
git clone https://github.com/continuonai/ContinuonXR.git pi5-brain
cd pi5-brain
```

2) Create dirs & place models  
```bash
sudo mkdir -p /opt/continuonos/brain/model/{base_model,adapters/current,adapters/candidate,adapters/history}
sudo mkdir -p /opt/continuonos/brain/rlds/episodes
sudo mkdir -p /opt/continuonos/brain/train
```
- Put Gemma 3n base (non-quant edge build if it fits) at `/opt/continuonos/brain/model/base_model/gemma-3n.tflite`.
- If you have current adapters, place `/opt/continuonos/brain/model/adapters/current/lora_adapters.pt`.
- Stage RLDS episodes (JSON/JSONL/TFRecord) under `/opt/continuonos/brain/rlds/episodes/`.

3) Config paths  
```bash
cp continuonbrain/configs/pi5-donkey.json /opt/continuonos/brain/train/pi5-donkey.json
# edit base_model_path/rlds_dir/budgets as needed
```

4) Python deps (venv recommended)  
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch  # choose the Pi build you use
```

5) Optional: runtime manifest for flutter_gemma  
```bash
cp continuonbrain/model/manifest.pi5.safety.example.json /opt/continuonos/brain/model/manifest.json
# edit base/adapter/safety paths
```

6) Run trainer (Gemma hooks + gating scaffold)  
```bash
python -m continuonbrain.trainer.examples.pi5_integration --config /opt/continuonos/brain/train/pi5-donkey.json
```
- Fails fast if `base_model_path` or `min_episodes` not met.
- Saves adapters to `adapters/candidate/`, promotes to `adapters/current/` on safety pass.

7) Wire gating & safety (replace stubs in the ContinuonBrain/OS runtime here)
- `gating_continuonos.py`: connect idle/battery/thermal/teleop to real signals; set thresholds (e.g., battery ≥40%, CPU ≤75C).  
- `safety_head_stub.py`: clamp/log violations; swap in your safety head; log safety flags to RLDS.  
- `gemma_hooks.py`: in-place LoRA for Gemma proj layers (`q_proj/k_proj/v_proj/o_proj`); plug in your real loader/loss.

8) Offline guarantee  
- No internet for any loop; uploads are manual/opt-in (see `continuonbrain/trainer/CLOUD_EXPORT.md`).
