# ContinuonXR System Architecture and Training Lifecycle

This document provides the reconciled design for the ContinuonXR multi-loop learning architecture, unifying three perspectives: the draft lifecycle document (fast/mid loops on-device, slow loop in cloud), the architecture plan (cloud-hosted RLDS with Vertex AI training and OTA updates), and the Donkey Car initiator plan (offline, on-robot data handling).

## Overview

The ContinuonXR project employs a multi-loop learning architecture that spans on-device edge processing and cloud-based training. In this design, all loops work in concert:

- **Immediate safety checks and learning** happen on the edge device (Raspberry Pi 5 + AI Hat)
- **The cloud** aggregates experience data via WorldTapeAI and performs heavy training (slow loop) with Vertex AI
- **New skill packs or model checkpoints** are delivered back to devices as over-the-air (OTA) updates
- **The device's Memory Plane** (persistent local learning state) is preserved across updates and merged with incoming global models at boot

This design maintains an "edge-first" ethos for safety and responsiveness, while leveraging cloud-scale learning for long-term improvement. We adhere to the nested learning principles from the HOPE/CMS research, layering fast, mid, and slow learning processes analogous to different frequency bands in a wave spectrum. Within the edge device diagram, the Pi-first safety boundary also aligns the "particle/wave" split to the two arms: particle = left arm (fast, reflexive ticks) and wave = right arm (slightly slower coordination), both anchored on-device.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ContinuonXR High-Level Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────┐          ┌─────────────────────────────────────┐ │
│   │  Edge Device        │          │       Cloud Infrastructure          │ │
│   │  (Pi 5 + AI Hat)    │          │                                     │ │
│   │                     │          │  ┌─────────────────────────────────┐│ │
│   │  ┌───────────────┐  │  RLDS    │  │     WorldTapeAI Portal          ││ │
│   │  │  Fast Loop    │  │  Upload  │  │  - Curated RLDS ingestion       ││ │
│   │  │  (Real-time)  │  │ ───────► │  │  - Opt-in data sharing          ││ │
│   │  └───────────────┘  │          │  │  - Episode aggregation          ││ │
│   │         │           │          │  └──────────────┬──────────────────┘│ │
│   │         ▼           │          │                 │                   │ │
│   │  ┌───────────────┐  │          │                 ▼                   │ │
│   │  │  Mid Loop     │  │          │  ┌─────────────────────────────────┐│ │
│   │  │  (Periodic)   │  │          │  │       Vertex AI Training        ││ │
│   │  └───────────────┘  │          │  │  - Slow loop consolidation      ││ │
│   │         │           │          │  │  - Global model training        ││ │
│   │         ▼           │          │  │  - Skill pack generation        ││ │
│   │  ┌───────────────┐  │  OTA     │  └──────────────┬──────────────────┘│ │
│   │  │ Memory Plane  │◄─│◄─────────│─────────────────┘                   │ │
│   │  │  (Persistent) │  │  Update  │       Model Registry & OTA Service  │ │
│   │  └───────────────┘  │          │                                     │ │
│   └─────────────────────┘          └─────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Edge-First Design: On-Device Fast and Mid Loops

### Edge Device (Pi 5 + AI Hat)

The Raspberry Pi 5 with an attached AI accelerator (Hat) serves as the on-site intelligence hub. It runs the current model locally for real-time inference and Fast-Loop learning.

#### Fast Loop

The fast loop executes 10–20 ms control ticks per arm (dual-arm, mirrored scheduling), driving real-time reflexes and safety validation before any actuator command is accepted.

| Aspect | Details |
|--------|---------|
| **Timescale** | 10–20 ms control ticks per arm (synchronous dual-arm cadence) |
| **Inputs** | Joint states, force/torque sensors, gripper state, Hailo vision features, SSM state (per arm) |
| **Outputs** | Torque/PWM commands and gripper micro-actions for both arms |
| **Responsibility** | Low-level motor skills, reflexive safety, teleop mirroring |
| **Update Mechanism** | Online gradient steps, safety overrides |
| **Example** | Per-tick validation of joint limits, grip force caps, and collision proxies while updating micro-actions from Hailo features |

The Pi enforces these 10–20 ms ticks locally so each arm receives immediate validation of joint/force envelopes and gripper micro-actions derived from the latest Hailo vision features and SSM state. This satisfies the edge-first requirement that critical safety checks and input validation happen on the device itself, keeping torque/PWM outputs bounded even during rapid teleop or novel stimuli.

#### Mid Loop

In addition to the fast loop, a Mid-Loop training and smoothing cycle runs on the device.

| Aspect | Details |
|--------|---------|
| **Timescale** | 100–500 ms |
| **Responsibility** | Per-arm stability smoothing plus bimanual coordination (coupled torques, timing sync) on Pi 5 + Hailo |
| **Update Mechanism** | Episodic fine-tuning, contextual bandits |
| **Example** | Periodic batch updates that smooth torque spikes per arm while adjusting coupled torques and timing synchronization across both arms |

Both fast and mid loops remain on-device as per the original lifecycle concept – they do not require cloud connectivity and thus can function offline. The Donkey Car initiator plan originally assumed all reinforcement learning data (episodes) would be handled offline on the robot; our reconciled design preserves the spirit of that by keeping the immediate learning (fast/mid loops) local while using the mid-loop cadence to coordinate dual-arm timing on the Pi 5 + Hailo stack.

### Safety & Real-Time Constraints

Because the Pi hosts the inference and fast loop learning, it can enforce real-time safety constraints:

- Before any control command is executed, it passes through on-device validation logic (limit checkers, rule-based failsafes)
- This guarantees that no cloud latency or dependency can compromise immediate safety
- The edge device's AI Hat accelerator (e.g., a Coral TPU or NPU) can run vision or sensor models with low latency
- Simultaneously perform quick learning updates on small batches if needed
- Only vetted, curated data is later shared to the cloud (opt-in)

---

## RLDS Data Ingestion via WorldTapeAI (Slow Loop Initiation)

When the edge device has accumulated a set of interesting experiences or training data (episodes of interaction, sensor logs, etc.), it enters the data ingestion phase for the slow loop. ContinuonXR keeps **manual/opt-in uploads as the default**; WorldTapeAI is only used once an operator explicitly enables the cloud path.

**Shared ingest strategy (aligned with the lifecycle plan)**
- Default posture: offline logging only. Upload daemons are disabled unless a user opts in.
- WorldTapeAI path: after opt-in, the device batches curated RLDS episodes, zips them with a manifest, and uploads via the configured ingest endpoint.
- Gating/curation: local filters remove sensitive frames, tag quality, and attach operator consent in the manifest; uploads are hashed/signed when supported for provenance.
- The exact enablement steps are in the [Upload Readiness Checklist](./upload-readiness-checklist.md), referenced by field teams and the lifecycle plan.

### Reinforcement Learning Dataset (RLDS) Format

The device packages its recorded episodes in a standardized RLDS format before upload:

- RLDS (Reinforcement Learning Datasets) is an ecosystem of tools and formats for logging sequential decision-making data in a lossless, shareable way
- Each episode consists of sequences of observations, actions, rewards, etc., recorded during operation
- Using a standard format ensures cloud training jobs can readily consume the data without custom preprocessing
- RLDS defines a consistent structure of episodes/steps including all details needed for replay

See [docs/rlds-schema.md](./rlds-schema.md) for the authoritative field definitions.

### Curated, Opt-In Sharing

Not all locally collected data will be sent to the cloud – only curated segments that are useful for global learning and meet privacy/quality criteria:

- The WorldTapeAI portal allows the robot (or user) to opt in to sharing certain data after passing the shared checklist.
- The device might automatically flag novel or challenging scenarios (edge cases) and ask for permission to upload.
- Sensitive information can be filtered out or anonymized prior to upload; rejected segments stay local.
- Each upload carries a manifest with operator/device identity plus checksums or signatures to preserve provenance.

### Offline vs cloud-connected decision table
| Mode | When to use | Prerequisites | Security & provenance |
|------|-------------|---------------|-----------------------|
| **Offline-only logging** | Default during bring-up, demos without connectivity, or sensitive runs | Sufficient local storage and rotation policy; upload services disabled | Keep RLDS local; enforce device-level auth for XR/teleop; no tokens provisioned |
| **Cloud-connected with WorldTapeAI ingest** | When operator opts in to share curated runs for slow-loop training | Network path to WorldTapeAI, valid ingest token, curated/trimmed episodes staged | Follow upload checklist: TLS, checksums/signature, manifest with operator/device identity, retain local copy until acknowledgment |

### Triggering Slow Loop Jobs

Once curated episodes are uploaded to WorldTapeAI's cloud repository:

1. They trigger the slow-loop training pipeline
2. WorldTapeAI aggregates data from all participating devices in the field
3. When enough new data or certain criteria are met (time-based schedule or threshold of new episodes), WorldTapeAI initiates a cloud training workflow
4. This corresponds to storing RLDS in a cloud database and invoking Google Cloud Vertex AI pipelines
5. The ingestion portal may do preprocessing (auto-labeling, validation, merging with existing datasets)

---

## Cloud Training Pipeline (Slow Loop Consolidation)

In the Slow Loop stage, the cloud performs intensive learning on the aggregated data using Vertex AI.

### Cloud Responsibilities

| Function | Description |
|----------|-------------|
| **RLDS Aggregation & Storage** | Episodes from WorldTapeAI stored centrally (Cloud Storage or database). Combines episodes from multiple edge devices for global perspective. |
| **Training Orchestration** | Vertex AI Pipelines coordinate training workflow. Jobs use TensorFlow + TF-Agents or PyTorch for RL or imitation learning. |
| **Hyperparameter Tuning & Evaluation** | Cloud compute enables sweeps and parallel training experiments. Validation on test sets before deployment. |
| **Global Skill Pack Output** | New model checkpoint or "global skill pack" with improved behaviors. Versioned and packaged for distribution. |
| **MLOps and CI/CD Integration** | Continuous training (CI/CT) as new data comes in. Continuous delivery (CD) pushes models to production. |

### Slow Loop Training Details

| Aspect | Details |
|--------|---------|
| **Timescale** | Minutes to Hours |
| **Responsibility** | High-level planning, semantic alignment, world model training |
| **Update Mechanism** | Corpus-scale training, generative pre-training |
| **Platform** | Google Cloud (Vertex AI / GKE) |
| **Data** | RLDS episodes from all sources |
| **Output** | Quantized TFLite models with INT8 precision |

The cloud slow-loop corresponds to what the draft lifecycle doc envisioned as the "Slow loop (cloud-based RLDS replay)." By using Vertex AI's managed infrastructure, we offload computationally expensive retraining from the Pi to scalable GPU/TPU resources.

---

## Over-the-Air Skill Updates and Model Integration

After a successful cloud training cycle, the newly trained global model (or skill pack) must be delivered back to all relevant edge devices through an Over-the-Air (OTA) update service.

### OTA Update Process

1. **Model Publishing**: Vertex AI pipeline registers new model in repository (Vertex Model Registry or Cloud Storage with version tag)
2. **Notification & Delivery**: Devices pull updates periodically or receive push notifications. Secure download with authentication and cryptographic signature verification
3. **Integration with Memory Plane**: The update contains new global model parameters but does not simply overwrite everything

### Update Integration Steps

```
1. Device saves current memory state (local fine-tuned weights or episodic memory)
   to persistent storage before applying update

2. Device loads new global model checkpoint into runtime

3. On boot, device merges Memory Plane with new model:
   - Memory Plane = extra parameters or knowledge base unique to device
   - Merging could involve:
     • Initializing layers with combination of old and new weights
     • Re-playing device's recent data on new model for quick fine-tuning
     • Using adapter modules that are preserved across updates
```

### Durability Across Updates

The system guards against catastrophic forgetting of local info when applying OTA updates:

- Model architecture designed to be modular, enabling layer-wise updates
- Global model update covers core layers ("long-term knowledge")
- Memory Plane corresponds to smaller portion (last-layer adapter or side memory) that remains untouched except for controlled merge
- Techniques from continual learning (elastic weight consolidation, partitioned networks) ensure device doesn't relearn everything

The OTA update is the deployment step of our continuous training loop. It closes the loop by taking the model from the model registry to production (edge). After deployment, the device runs the new model, monitors performance, collects new data, and eventually sends back the next round of RLDS to WorldTapeAI.

---

## Memory Plane and Persistent Learning on Device

The Memory Plane is a conceptual layer representing the device's persistent learning state – the union of fast- and mid-loop learnings accumulated over time.

### Memory Plane Implementation Options

| Implementation | Description |
|----------------|-------------|
| **Episodic Memory Buffer** | Cache of recent experiences (in RLDS or other format) used by mid-loop trainer. Persisted through updates for immediate fine-tuning after new model installation. |
| **Local Adaptation Weights** | Small neural network or calibration parameters trained on-device, stored separately from main model weights. Could be adapter layers updated by device but not overwritten by OTA. |
| **Meta-Data and Skill Usage Stats** | Logs of which skills/outputs were most used, typical environment conditions. Influences how new model is initialized or operates initially. |

### Integration at Boot

Prior to update, device model M_old consists of:
- θ_base^old (from last global update)
- θ_mem^old (local memory-trained parameters)

After OTA, we have θ_base^new from cloud. Merge function:

```
θ_combined = f(θ_base^new, θ_mem^old)
```

Function f could be:
- Concatenation (if different layers)
- Averaging (if appropriate)
- Knowledge distillation (new model fine-tuned briefly on data generated from old local model)

The key principle: old memory is not discarded. Over time, θ_mem might get relatively smaller as global model picks up more patterns, but it will always capture device-specific information.

---

## Multi-Loop Learning Continuum (Nested Learning Analogy)

ContinuonXR's architecture follows a nested learning loops approach, inspired by the HOPE/CMS Nested Learning framework. The model is seen as an integrated system of multi-level memories, each operating at its own timescale.

### Loop Definitions

| Loop | Frequency | Description | HOPE/CMS Analogy |
|------|-----------|-------------|------------------|
| **Fast Loop** | Highest (every step or few seconds) | Updates happen almost every step. On-device real-time adjustment (updating internal state or short-term synaptic weight based on latest input and outcome). | First MLP module that updates with every new sample. Analogous to synaptic consolidation occurring immediately during wakefulness. |
| **Mid Loop** | Medium (minutes, hours, event triggers) | Updates happen less frequently with larger batches. On-device retraining on recent data or periodic calibration. Intermediate consolidation stabilizing fast loop changes over longer horizon. | Second MLP module updated at configurable intervals (e.g., every N steps) rather than every step, where N is larger than the fast loop interval. |
| **Slow Loop** | Lowest (many episodes or hours) | Rare but high-impact updates (new global model). Implemented in cloud (Vertex AI) on aggregated experiences from many devices. | Outermost memory module, updated only occasionally but incorporating broad context. Analogous to brain's offline systems consolidation during sleep. |

### Wave Spectral Analogy

The wave spectral analogy comes from thinking of each loop as capturing a different frequency band of learning:

- **Fast loop** = high-frequency oscillations (quick, fine-grained tweaks)
- **Mid loop** = medium-frequency band (intermediate stabilization)
- **Slow loop** = low-frequency waves (gradual, sweeping changes)

By combining them, the system can respond to immediate fluctuations while steadily improving in a long-term sense - just as a composite signal can be reconstructed from its spectral components.

## Particle-Wave Blueprint for ContinuonOS (post-Transformer path)
- **Why:** Attention is demoted to a local specialist. Long-range structure and continuous-time memory come from SSMs and spectral mixers (Mamba/Selective SSMs, Hyena/GFN, Griffin/Hawk hybrids). This keeps HOPE/CMS intact while scaling beyond attention-only limits.
- **Tier 1: Particle path (Fast, on Pi + Hailo):** Tiny attention windows, local convs/MLPs, and small TFLite/ONNX heads for frame-by-frame reactivity. Runs on Hailo HEF for vision and on Pi CPU/NPU for policy heads; adapters live in `apps/continuonxr/` and `continuonbrain/`.
- **Tier 2: Wave path (Mid, on Pi):** A compact SSM cell (S4/Mamba-style) plus optional lightweight spectral mixer (short FFT over a small buffer with a learnable mask) maintains a continuous latent state between steps. Runs on Pi CPU, feeds the Memory Plane, and is updated per chunk/episode rather than every gradient step.
- **Tier 3: Global wave consolidator (Slow, cloud):** Larger Mamba/Hyena/Griffin models train on RLDS in `continuonai/continuon-cloud/` to learn long-range kernels and spectral filters. OTA bundles ship updated SSM kernels, spectral weights, adapters, and HEFs back to edge; merge with the Memory Plane instead of overwriting (see `docs/model_lifecycle.md`).
- **Immediate steps to implement on edge:**
  1) Add a 1-layer SSM cell beside the current particle head to carry hidden state across steps (NumPy/TF Lite custom op or lightweight Kotlin/ND arrays).
  2) Add a minimal spectral mixer: FFT over the last N hidden states (e.g., N=8), apply a learnable frequency mask, iFFT back, and feed the policy head.
  3) Fuse particle + wave features into the policy head (either exported together as TFLite or as a pre-head module in `continuonbrain/`).
  4) In cloud, train larger SSM/spectral models on RLDS and distill their kernels/filters/adapters for OTA delivery.
  5) OTA merge: apply new kernels/filters while preserving on-device Memory Plane state to keep HOPE/CMS multi-timescale learning intact.

---

## Roles and Responsibilities Breakdown

### Function/Phase by Component

| Function / Phase | Edge Device (Pi 5 + AI Hat) | WorldTapeAI Cloud Portal | Cloud (Vertex AI & OTA) |
|------------------|----------------------------|-------------------------|------------------------|
| **Real-Time Inference & Control** | Runs model locally for immediate perception and control. Ensures safety and I/O validation. Executes Fast Loop learning with minor online weight adjustments. | Not involved | Not involved |
| **On-Device Training (Mid Loop)** | Periodically retrains on recent data. Uses local compute (Pi + AI accelerator). Maintains Memory Plane of learned adjustments. | Not involved | Cloud provides initial model; mid-loop handled on device |
| **Data Collection & Curation** | Collects sensor data, observations, actions, rewards. Logs episodes. Filters and curates data. Packages in RLDS format and uploads (opt-in). | Receives RLDS episodes. Stores in dataset. Provides interface for user review/approval. Ensures anonymization and compliance. Queues data for training. | May store data in central repository. Prepares data for training (sharding, format compatibility). |
| **Triggering Training (Slow Loop)** | Signals or initiates upload when new data ready (or on schedule). May send notification that "training needed." | Detects sufficient new data or time threshold. Triggers Vertex AI pipeline. May provide API for this trigger. | Runs Vertex AI training job/pipeline. Loads aggregated RLDS data. Monitors training progress. |
| **Model Training & Validation** | Only lightweight training locally in fast/mid loops for interim performance. | Not involved in actual training computation. | Performs heavy model training on cloud infrastructure (GPUs/TPUs). Runs evaluation on validation datasets or simulations. Ensures new model meets performance and safety criteria. |
| **Model Artifact Generation** | Does not create final model artifacts for others; may save local checkpoints of mid-loop tuning (Memory Plane snapshots). | Not involved | Packages new global model (skill pack). Exports TensorFlow SavedModel, TensorRT engine, etc. Registers artifact in Model Registry. |
| **OTA Delivery** | Receives OTA updates. Downloads signed Edge Bundle over HTTPS. Verifies signature and compatibility. Hot-swaps models. Reports deployment success/failure. | May host download URLs or notification service. | Publishes model for devices to download. Manages OTA service for notification and delivery. |
| **Memory Plane Management** | Preserves Memory Plane during OTA updates. Merges Memory Plane with new global model at boot. | Not involved | May provide guidance on merge strategies via bundle metadata. |

---

## Related Documentation

- [Model Lifecycle: HOPE/CMS Governance](./model_lifecycle.md) - OTA delivery phases and Memory Plane persistence
- [RLDS Schema](./rlds-schema.md) - Data contract for all robot experiences
- [HOPE/CMS VLA](./hope-cms-vla.md) - Detailed signal routing to VLA heads
- [Architecture Overview](./architecture.md) - System module layout
- [Ecosystem Alignment](./ecosystem-alignment.md) - Integration with broader Continuon ecosystem
- [Donkey Car Lifecycle Plan](../continuon-lifecycle-plan.md) - Offline-first implementation details
