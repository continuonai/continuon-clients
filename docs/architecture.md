# ContinuonXR Architecture (Draft)

This document outlines the early architecture that satisfies the PRD and Phase 0 deliverables. It is intentionally implementation-neutral while mapping the data loop between the XR app, ContinuonBrain/OS, and Continuon Cloud. For a HOPE/CMS-oriented view of OTA delivery and learning loops, see [Model Lifecycle: HOPE/CMS Governance](./model_lifecycle.md).

## High-level system
- **XR client (Android, Kotlin, Jetpack XR/Compose):** Runs on Galaxy XR hardware. Provides spatial workstation panels, teleop UI, and observer overlays. Emits RLDS episodes locally and forwards to Continuon Cloud.
- **ContinuonBrain/OS bridge (dock/phone):** Receives gRPC/WebRTC streams from XR client for robot state, commands, and telemetry. Exposes the stable Robot API.
- **Continuon Cloud:** Stores RLDS episodes, runs training/validation jobs, and manages OTA updates to ContinuonBrain/OS and XR client configs.
- **Companion apps:** Flutter clients for setup, quick teleop, and RLDS browsing; optional WebXR/Unity viewers for simulation and playback.

## Modules (proposed)
- **XR UI shell:** Scene composition, window/panel manager for IDE, terminals, dashboards, and teleop widgets.
- **Teleop & control adapters:** Translates XR pose/gesture/voice to normalized robot actions (EE velocity, joint deltas).
- **Observer & annotation tools:** Safety overlays, trajectory previews, polygon/mask annotation pipeline.
- **Input devices:** Headset pose, hand tracking, gaze ray, microphone audio, Continuon Glove BLE stream; input fusion and synchronization.
- **Data logging:** RLDS episode builder enforcing schema; local persistence; upload/forwarder to Cloud.
- **Connectivity:** gRPC/WebRTC client to ContinuonBrain/OS; HTTPS/WebSocket client to Cloud; BLE manager for glove.
- **Config & feature flags:** Mode selection (trainer/workstation/observer), environment selection (lab/mock), and rollout toggles.

## Data flow (Mode A: Trainer)
1. XR captures headset/hand poses, gaze ray, audio, and glove frames at ~100 Hz.
2. Teleop adapter maps inputs to normalized robot actions.
3. Bridge sends commands to ContinuonBrain/OS over gRPC/WebRTC; receives robot state feedback.
4. RLDS logger syncs observation (poses, gaze ray, audio, glove, video, robot state) with action and timestamps; stores locally.
5. Episodes are sealed with `episode_metadata` tags (`continuon.xr_mode`, `continuon.control_role`) and uploaded.

## Data flow (Mode B: Workstation)
1. Panels expose IDE/terminal/log dashboards.
2. UI actions (open file, run tests, rollout) are captured as RLDS step metadata with focus context.
3. Episodes tagged with `continuon.xr_mode = "workstation"` and `source = "human_dev_xr"`.

## Data flow (Mode C: Observer)
1. Live state and predictions rendered with safety overlays.
2. User adds annotations (polygons, thresholds, safe zones) emitted as `steps[*].action` records.

## MVP boundaries (Phase 1)
- Basic XR shell with limited panels.
- Teleop to mock ContinuonBrain/OS instance (local service).
- Local RLDS episode writer with deterministic schema validation.
- BLE glove ingestion with MTU negotiation and frame parsing.
- Manual upload/export flow to Cloud (full pipeline can be mocked).
