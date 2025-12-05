# Continuon Studio Implementation Blueprint

_This blueprint turns the Continuon Studio architectural specification into an implementation-ready outline for the on-device IDE that runs alongside the Continuon Brain runtime on PixelBrain hardware. It expands the roles, contracts, and UI ↔ backend bindings needed to ship the v0 experience described in the specification while keeping naming aligned with the rest of the repo._

## 1. Process Architecture on PixelBrain

Coordinate four cooperating services on-device. Keep HOPE/continuum control on performance cores/TPU and the Studio server on efficiency cores to preserve the 50 ms fast-loop deadline.

| Service | Language | Responsibilities | Interfaces |
| --- | --- | --- | --- |
| **HOPE Runtime Core** | C++/Rust with Python bindings | Runs HOPE recurrent world model (Wave/Particle/gating), maintains Continuum Memory System (CMS) fast/mid/slow loops, executes safety head. | `step(observation) -> {actions, w_t, p_t, g_t, safety_head}`; `cms_rollback(span_id)`; `apply_lora(update)`; `export_episode(buffer_id)` → RLDS. |
| **Thin Device Adaptor (TDA)** | Rust/C++ | Bridges Android XR HAL APIs (camera, audio, tactile, IMU, joints). Streams synchronized multimodal frames; executes low-latency action commands; maintains `RobotManifest` capabilities. | ObservationStream → HOPE; ActionStream ← HOPE/teleop; `capabilities` surfaced to Studio server. |
| **Continuon Studio Server (FastAPI + Uvicorn)** | Python | Serves SPA, terminates WebRTC signaling and WebSocket telemetry, exposes REST/WS control, enforces QoS/backpressure. | REST/WS as below; WebRTC setup; shared-memory or UDS bridge to HOPE runtime. |
| **Continuon Studio Frontend (SPA)** | TypeScript (Vite + React + Zustand/Redux + Monaco) | Renders Neuro-Symbolic Canvas, Sensory Homunculus, Continuum Timeline, Shadow Mode, Skill Composer, Safety Head visualizer; caches episodes/annotations. | WebRTC vision/audio, WebSocket telemetry/events, REST actions, data-channel teleop. |

**Process wiring**
- **TDA ↔ HOPE runtime:** In-process bindings or shared memory/ZeroMQ. HOPE subscribes to ObservationStream; publishes ActionStream + latent/safety snapshots.
- **HOPE runtime ↔ Studio server:** Shared-memory ring buffer or Unix domain sockets to expose `{w_t, p_t, g_t}` at ~20–50 Hz, SafetyHead diagnostics, training progress. Server republishes over WebSocket.
- **TDA ↔ Studio server:** WebRTC handles vision/audio; Python orchestrates signaling only. WebSocket/REST used for capabilities and fallbacks.

## 2. Core Interfaces and Data Contracts

### 2.1 Thin Device Adaptor (concrete shape)

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, TypedDict
import numpy as np

class ObservationFrame(TypedDict):
    timestamp_ns: int
    vision: np.ndarray           # H x W x 3, uint8
    depth: np.ndarray | None     # H x W, float32 or None
    audio: np.ndarray            # T x C, float32
    tactile: np.ndarray | None   # N_sensors, float32
    joints: np.ndarray           # N_joints, float32
    imu: np.ndarray              # 6- or 9-dof

class ThinDeviceAdaptor(ABC):
    @abstractmethod
    async def stream_observations(self) -> AsyncGenerator[ObservationFrame, None]:
        ...

    @abstractmethod
    async def execute_action(self, action_head: str, command: np.ndarray) -> None:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        ...  # feeds /api/v1/system/capabilities
```

### 2.2 Telemetry WebSocket (logical schema)

Use FlatBuffers on the wire; logical payload per message:

```json
{
  "timestamp_ns": 1733333333333,
  "mode": "AUTONOMOUS" | "SHADOW" | "PAUSED",
  "g_t": 0.72,
  "w_t_proj": [0.23, -0.11, 0.89],
  "p_t_energy": 1.83,
  "joint_state": [...],
  "tactile_summary": {"max": 0.9, "min": 0.0, "mean": 0.24},
  "safety": {"violation_imminent": false, "margin": 0.18}
}
```

Full latent tensors (w_t, p_t) stream on-demand or downsampled to avoid starving control loops.

### 2.3 RLDS Edge Episode

Metadata kept local for playback, Shadow Mode labeling, and later World Tape sync:

- `episode_id`, `robot_id`, `hardware_manifest_version`
- `start_time`, `end_time`
- `frames[]` → observation (vision/audio/tactile/pose), action, optional reward
- `info` → `mode` (AUTONOMOUS/SHADOW/REPLAY), `user_intervention` metadata, `golden` flag, `tags[]`, optional `loss_trace[]`

### 2.4 Event messages

Events ride a separate WebSocket `/streams/events` channel: `HELP_REQUEST`, `MODE_SWITCH`, `SOUND_CLASS`, `TRAIN_PROGRESS`, `SAFETY_ALERT`, etc., each carrying timestamp, label, confidence/metadata.

## 3. REST and WebSocket Surface (Studio Server)

All endpoints are served by FastAPI with mutual TLS + mandate gating for risky operations.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/v1/system/capabilities` | GET | Return `RobotManifest` (sensors, action heads, limits, CMS tiers exposed). |
| `/api/v1/system/health` | GET | Control-loop timing, TPU utilization, temperature, QoS/backpressure state. |
| `/api/v1/streams/telemetry` | WS | Streams telemetry frames (schema above). |
| `/api/v1/streams/events` | WS | Streams semantic events (audio classes, help requests, training progress, safety alerts). |
| `/api/v1/control/mode` | POST | Set mode: `AUTONOMOUS`/`SHADOW`/`PAUSED`. |
| `/api/v1/control/teleop` | POST | Convenience teleop (REST fallback); primary path is WebRTC data channel. |
| `/api/v1/brain/state/snapshot` | GET | On-demand `{w_t_proj, p_t_energy, g_t, recent_modes, cms_activity}` snapshot. |
| `/api/v1/brain/gate/clamp` | POST | Force Wave/Particle dominance (`mode: WAVE | PARTICLE | AUTO`, duration_ms). |
| `/api/v1/cms/rollback` | POST | Roll back CMS weights for selected span/tier (mandate required for long spans). |
| `/api/v1/episodes/start` | POST | Begin RLDS capture, returns `episode_id`. |
| `/api/v1/episodes/{id}/end` | POST | Close capture. |
| `/api/v1/episodes/{id}` | GET | Episode metadata. |
| `/api/v1/episodes/{id}/replay` | GET/WS | Time-travel replay for Timeline scrubbing. |
| `/api/v1/train/lora` | POST | Kick off on-device LoRA adapter training on `episode_ids`; returns `job_id`. |
| `/api/v1/train/jobs/{job_id}` | GET | Training status, loss curve, adapter location. |
| `/api/v1/skills/compile` | POST | Compile Python or blocks → skill bytecode + SafetyHead report. |
| `/api/v1/skills/{skill_id}/execute` | POST | Execute compiled skill with parameters after preflight. |

## 4. Frontend Component → Data Needs Map

| Feature | UI Components | Backend needs |
| --- | --- | --- |
| **Neuro-Symbolic Canvas** | `CameraFeedCanvas`, `LatentStateHorizon`, `CognitiveModeIndicator`, force/attention overlays | WebRTC video/depth, attention masks (~10–20 Hz), particle vector fields, `w_t_proj`, `g_t`, safety margin. |
| **Sensory Homunculus** | `TactileGripperView`, `AudioSemanticTimeline` | Tactile tensors + slip vectors, gripper pose; audio semantic events from Gemini Nano classifier on `/streams/events`. |
| **Continuum Timeline** | `Timeline` with Fast/Mid/Slow lanes | Indexed RLDS episode replay; manager decisions and semantic events; hook to `/cms/rollback` when a span is selected. |
| **Skill Composer** | `NaturalLanguageSkillPrompt`, `SkillCodeEditor` (Monaco), `BlocksView` | On-device LLM for prompt-to-Python, `/skills/compile` for SafetyHead check, safety annotations inline, `/skills/{id}/execute` for test runs. |

## 5. Shadow Mode: End-to-End Flow

1. **Help request:** HOPE uncertainty emits `HELP_REQUEST` on `/streams/events` with `w_t_proj` and context tags; Studio shows Android XR notification.
2. **Enter Shadow Mode:** Frontend POSTs `/control/mode {"mode":"SHADOW"}`; optionally clamp gate via `/brain/gate/clamp` toward Particle.
3. **Teleop:** VR controllers/joystick stream commands via WebRTC data channel; TDA executes; HOPE logs as teacher outputs.
4. **Golden episode:** On completion, user marks “Golden”; server sets `golden=true`, `mode=SHADOW`, `user_intervention` metadata in RLDS episode.
5. **On-device LoRA:** User POSTs `/train/lora {episode_ids, adapter_name}`; HOPE freezes base weights, trains adapter on TPU; progress events stream via `/streams/events` (`TRAIN_PROGRESS`).
6. **Verification:** Switch to `AUTONOMOUS`; execute skill via `/skills/{id}/execute`; Continuum Timeline shows before/after performance, safety margins, and CMS drift.

## 6. QoS, Safety, and Failure Modes

- **Control-loop starvation:** HOPE watchdog drops Studio data and can disconnect UI if fast-loop >48 ms; surface metrics in `/system/health` and UI banner.
- **Adaptive telemetry:** Downsample overlays first (particle density, latent frequency) before touching safety/control channels when under load.
- **CMS rollback guardrails:** Provide “dry run” simulation path and require mandates for large spans; warn when rollback would invalidate pending adapters.
- **Skill/safety conflicts:** SafetyHead Visualizer blocks execution when simulated collisions predicted; quarantine flagged skills until explicit override/mandate.
- **Streaming overload:** Prefer AV1/H.264 hardware encode; if throughput drops, shed video frames before tactile/latents; keep <100 ms photon-to-motion target.

## 7. Follow-on Docs to Produce

- **API contract doc (Studio ↔ robot):** Formal REST/WS schemas + FlatBuffers IDs; link to [`docs/rlds-schema.md`](./rlds-schema.md) for RLDS alignment.
- **TDA adapter guide:** Implementation recipes for Android XR + Pixel 10 baseline, quadruped RS485 servos, and lightweight Pi-class variants.
- **Studio UI wireframes:** Neuro-Symbolic Canvas, Sensory Homunculus, Continuum Timeline, Skill Composer, Safety Head visualizer mapped to data needs above.
- **Performance/telemetry budget:** Per 50 ms tick budget for TDA → HOPE → actions, allowed Studio overhead, LoRA concurrency limits, and QoS downsampling policy.
