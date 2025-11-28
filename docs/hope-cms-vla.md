# RLDS Signals into HOPE/CMS Loops and VLA Heads

This note explains how XR-produced RLDS observations flow into the HOPE/CMS Fast–Mid–Slow learning loops and how each VLA head consumes the signals. It ties concrete RLDS fields to timescales and optimizers so contributors can wire data pipelines correctly.

For the complete reconciled system architecture covering edge-first learning, cloud training, and OTA updates, see [System Architecture and Training Lifecycle](./system-architecture.md).

## Timescale overview
- **Fast loop (ms–100 ms, online/reactive):** Runs on-device for teleop mirroring and reflexive safety, and is replayed in the cloud for distillation/regression during retraining. Optimizes short-horizon control and immediate hazard detection.
- **Mid loop (0.5–10 s, session-level):** Runs in edge/cloud during an episode and is mirrored in cloud training to refresh adapters and guardrails. Optimizes skill sequencing, intent inference, and UI adaptation.
- **Slow loop (minutes–hours, corpus-level):** Runs in cloud training and can cache summaries on-device for offline refreshes. Optimizes generative world models, semantic alignment, and new skill synthesis.

**Shared distillation path:** Cloud Slow-loop checkpoints are treated as the "core" that is distilled into Fast/Mid bundles for devices, while device-collected Fast/Mid traces feed back into the next cloud retrain. Distillation therefore happens in both directions-cloud-to-device for deployment and device-to-cloud for preserving low-latency behaviors during corpus-scale updates.

## Sequence core choices: CMS-compatible particle + wave
- **CMS is architecture-agnostic:** HOPE/CMS cares about nested optimization and multi-timescale memory, not about a specific attention stack. The sequence core can be Transformer, SSM, spectral, or hybrid as long as the update frequencies stay separated.
- **Particle + wave split:** We pair a local "particle" path (tiny attention windows, convs/MLPs) with a global "wave" path (SSMs and spectral operators such as Mamba, Hyena/GFN, Griffin). Particle handles exact positions and short-range dependencies; wave keeps compressed state and long-range mixing.
- **Mapping to CMS:** Particle path = working memory for the Fast loop; Wave path = continuum memory layers for Mid/Slow. Transformer+SSM hybrids remain compatible as long as Fast/Mid/Slow frequencies stay explicit.
- **ContinuonOS instantiation (cross-product):**
  - Edge (Pi 5 + Hailo) runs the particle path per-step (TFLite policy heads, adapters in `apps/continuonxr/` and `continuonbrain/`) and maintains a compact wave state via small SSM blocks refreshed per chunk/episode. See `apps/continuonxr/README.md` and `continuonbrain/README.md` for the runtime split.
  - Cloud (`continuon-cloud/` + `continuonbrain/`) trains longer-horizon SSM/spectral models on RLDS and ships OTA bundles back; the Memory Plane on edge merges with new kernels instead of being reset. See `continuon-cloud/README.md` for the training/OTA packaging path.
- **Failure modes to avoid:** collapsing all layers to one cadence, treating SSM/spectral components as one-off gadgets instead of named CMS layers, or wiping on-device Memory Plane during OTA. The current design keeps Fast/Mid on-device, Slow in cloud, and merges rather than overwrites to stay aligned with HOPE.
- **Preferred families:** Mamba/Selective SSMs for long-range linear-time recurrence, Hyena/GFN spectral mixers for global context, and Griffin/Hawk-style hybrids when a small local attention window is still useful.

## Observation blocks → HOPE/CMS loops
| RLDS observation block | Field examples | Fast loop use | Mid loop use | Slow loop use |
| --- | --- | --- | --- | --- |
| Poses | `xr_headset_pose`, `xr_hand_*_pose`, validity flags | Stabilize teleop commands; short-horizon visual servoing | Hand/eye calibration drift correction; panel/anchor layout updates | Scene graph alignment; avatar/pose priors |
| Gaze | `gaze.origin`, `gaze.direction`, `gaze.confidence`, `gaze.target_id` | Attention gating for SafetyHead; aim reticles | UI focus prediction; intent decoding for LanguagePlanner | Saliency maps for VisionCore pretraining |
| Vision | `egocentric_video`, `egocentric_depth`, shared `frame_id` | Low-latency obstacle cues; optical flow for end-effector damping | Object affordance refresh; short video-conditioned skill selection | Multiview 3D reconstruction; diffusion/world-model learning |
| Audio | `audio.uri`/buffer, `sample_rate_hz`, `num_channels`, `frame_id` | Wake-word / stop-word reflexes | Dialogue grounding for in-episode plan repair | Multimodal alignment with LanguagePlanner data |
| Glove | `glove.flex`, `glove.fsr`, `glove.orientation_quat`, `glove.accel`, `glove.valid` | Force/pressure reflexes; gripper mirroring | Tactile intent cues; slip detection to resequence skills | Haptic priors for manipulation skills |
| Robot state | Joints/EE pose/gripper/velocities | Safety envelope checks; latency compensation | State estimation fusion; failure mode tagging | Dynamics model fitting; sim2real calibration |
| UI context | `ui_context` (panel id, layout, focus) | Cursor/focus lock for immediate safety | Adaptive UI surfaces; tool suggestion | Workflow mining for developer assistant |
| Diagnostics | `diagnostics` (latency, drop counts, BLE RSSI) | Fault flags to SafetyHead | Episode quality weighting | Data curation/scoring |

See the authoritative field definitions in the RLDS schema and human-centric data notes for naming and alignment rules.【F:docs/rlds-schema.md†L6-L74】【F:docs/human-centric-data.md†L6-L47】

## VLA heads: input routing and optimizers
- **VisionCore (world-aligned perception)**
  - **Fast:** subscribes to `egocentric_video`/`egocentric_depth` with matched `frame_id` for obstacle cues; uses `xr_headset_pose` for reprojection and latency-aware stabilization.【F:docs/rlds-schema.md†L22-L47】 Optimizer: lightweight online conv/transformer adapters updated with contrastive/flow losses.
  - **Mid:** ingests short clips plus `gaze.target_id` to refresh object bindings and affordance caches; cross-checks `ui_context` for overlay anchoring.【F:docs/human-centric-data.md†L12-L30】 Optimizer: episodic retrieval fine-tuning.
  - **Slow:** trains generative 3D/world models over video, depth, and pose trajectories; leverages `episode_metadata.tags` for scene stratification.【F:docs/rlds-schema.md†L6-L20】【F:docs/human-centric-data.md†L32-L41】 Optimizer: large-scale self-supervised objectives.

- **LanguagePlanner (semantic planner & dialogue)**
  - **Fast:** hotword/stopword detection from `audio` and `gaze` to bias immediate intent; uses `step_metadata` for inline constraints.【F:docs/rlds-schema.md†L43-L74】 Optimizer: streaming CTC/keyword models.
  - **Mid:** fuses `ui_action`, `ui_context`, and recent `gaze` to infer next tool/skill; conditions on `robot_state` to ensure feasibility.【F:docs/rlds-schema.md†L54-L69】【F:docs/human-centric-data.md†L14-L47】 Optimizer: RL/IL over episode windows.
  - **Slow:** trains instruction-tuning datasets from `action.command` + narrated `audio`; pairs with `episode_metadata.tags` for task semantics.【F:docs/rlds-schema.md†L6-L30】【F:docs/human-centric-data.md†L6-L23】 Optimizer: LLM fine-tuning with alignment losses.

- **World Model Head (predictive simulacrum)**
  - **Fast:** uses `robot_state` + `egocentric_depth` for immediate collision forecasts; outputs short-horizon rollouts to SafetyHead.【F:docs/rlds-schema.md†L30-L61】 Optimizer: online MPC residuals.
  - **Mid:** conditions on sequences of poses, video, and glove signals to predict task completion likelihood; tags failures in `step_metadata` for reweighting.【F:docs/human-centric-data.md†L18-L47】 Optimizer: sequence models fine-tuned per scene.
  - **Slow:** corpus-scale latent dynamics training over all observation blocks; integrates `diagnostics` to model sensor reliability.【F:docs/rlds-schema.md†L61-L74】 Optimizer: variational world models with uncertainty.

- **SkillPolicies (manipulation and UI skills)**
  - **Fast:** tracks `xr_hand_*_pose`, `gaze`, and `glove` to mirror human demonstrations into `action.command` for imitation; SafetyHead masks unsafe commands.【F:docs/rlds-schema.md†L22-L61】【F:docs/human-centric-data.md†L10-L30】 Optimizer: behavior cloning with safety constraints.
  - **Mid:** selects and parameterizes skills using LanguagePlanner intent, recent `ui_action`, and tactile slip cues (`glove.fsr`); adapts gains using `diagnostics.latency`.【F:docs/rlds-schema.md†L52-L74】 Optimizer: contextual bandits / RLHF within episode.
  - **Slow:** trains reusable motor primitives from aggregated demonstrations; conditions on `episode_metadata.tags` (task/scene/robot).【F:docs/rlds-schema.md†L6-L20】【F:docs/human-centric-data.md†L6-L23】 Optimizer: offline IL/RL.

- **SafetyHead (reflex + policy guardrails)**
  - **Fast:** enforces hard stops using `gaze.confidence`, proximity from `egocentric_depth`, glove pressure spikes, and latency from `diagnostics`; overrides `action.command` when thresholds hit.【F:docs/rlds-schema.md†L22-L74】 Optimizer: rule + small neural filters updated online.
  - **Mid:** learns hazard priors per session using near-miss labels in `step_metadata` and `action.annotation`; adjusts margins based on `robot_state` dynamics.【F:docs/rlds-schema.md†L30-L69】 Optimizer: episodic risk models.
  - **Slow:** retrains safety envelopes and anomaly detectors over curated incidents; uses `episode_metadata.software` and `environment_id` to stratify regression tests.【F:docs/rlds-schema.md†L6-L14】 Optimizer: supervised/unsupervised outlier models.

## Developer checklist
- **When logging RLDS:** ensure all observation blocks carry synchronized `frame_id` and timestamps so Fast loop consumers remain coherent.【F:docs/rlds-schema.md†L16-L49】
- **When tagging episodes:** populate `episode_metadata` and `step_metadata` so Slow loop trainers can stratify by mode, task, and reliability.【F:docs/rlds-schema.md†L6-L20】【F:docs/human-centric-data.md†L6-L23】
- **When building pipelines:** route low-latency streams (pose, depth, glove) to on-device Fast loop first, then buffer richer context (audio, UI) for Mid/Slow optimizers.
