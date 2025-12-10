# Android XR App Scaffold

This folder sketches the Kotlin/Jetpack XR app described in `PRD.md`. Gradle wiring ships with Jetpack XR/SceneCore dependencies enabled.

## Modules
- `MainActivity.kt` — entry point, routes to mode-specific shells (trainer/workstation/observer).
- `config/` - configuration loading and flags (mode, connectivity, logging, glove).
- `connectivity/` - ContinuonBrain/OS bridge client for gRPC/WebRTC streams.
- `glove/` - BLE ingestion for Continuon Glove v0 (MTU negotiation, frame parser, diagnostics).
- `audio/` - audio capture stub for synchronized RLDS audio logging.
- `teleop/` - input fusion and command mapping for Mode A teleop; records RLDS steps.
- `logging/` - RLDS schema enforcement, validation, file sink, and upload hook stubs.
- `ui/` - UI context tracker for workstation mode logging.

## Build/test quickstart
- Requires Android Studio Koala or later (AGP 8.5) and a local Gradle install or wrapper.
- Build app: `./gradlew :apps:xr:assembleDebug`
- Run unit tests: `./gradlew :apps:xr:testDebugUnitTest`
- Generate proto stubs: `./gradlew :apps:xr:generateDebugProto`

## Next steps
1. Implement the Raspberry Pi 5 ContinuonBrain/OS bridge client, normalizing Donkey Car steering/throttle into `action.command` and logging drivetrain state into `observation.robot_state` with ≤5 ms timestamp alignment.
2. Add Flutter companion hooks for quick teleop and RLDS episode emission (tagged `xr_mode="trainer"` and `action.source="human_teleop_xr"`), including buffered uploads with `metadata.json` + `steps/*.jsonl` packaging.
3. Wire Compose XR/SceneCore inputs plus iPhone sensor proxies into `InputFusion` and `CommandMapper` so both XR and phone shells drive the same teleop surface.
4. Extend the RLDS writer with schema validation, provenance tags (environment IDs, software versions), and media blob handling for Cloud ingestion.
5. Expand unit tests for schema validation, BLE parsing, teleop command mapping, and OTA bundle handoff triggers once the edge bundle path is plumbed.

Conversation log: Pi5 startup/training optimization (2025-12-10) summarized at `../../docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).