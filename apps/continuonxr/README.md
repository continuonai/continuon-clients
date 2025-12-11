# Android XR App Scaffold

This folder sketches the Kotlin/Jetpack XR app described in `PRD.md`. Gradle wiring ships with Jetpack XR/SceneCore dependencies enabled.

## Status (2025-12-10)

- BLE parser + RLDS writer shipped; SceneCore/Compose XR rendering and the live bridge to the Continuon Brain runtime are in progress (see `docs/unified-roadmap.md` for phase targets).

## Modules
- `MainActivity.kt` — entry point, routes to mode-specific shells (trainer/workstation/observer).
- `config/` - configuration loading and flags (mode, connectivity, logging, glove).
- `connectivity/` - Continuon Brain runtime bridge client for gRPC/WebRTC streams.
- `glove/` - BLE ingestion for Continuon Glove v0 (MTU negotiation, frame parser, diagnostics).
- `audio/` - audio capture stub for synchronized RLDS audio logging.
- `teleop/` - input fusion and command mapping for Mode A teleop; records RLDS steps.
- `logging/` - RLDS schema enforcement, validation, file sink, and upload hook stubs.
- `ui/` - UI context tracker for workstation mode logging.

## Build/test quickstart
- Requires Android Studio Koala or later (AGP 8.5) and a local Gradle install or wrapper.
- Build app: `./gradlew :apps:continuonxr:assembleDebug`
- Run unit tests: `./gradlew :apps:continuonxr:testDebugUnitTest`
- Generate proto stubs: `./gradlew :apps:continuonxr:generateDebugProto`

## Next steps
1. Finish the Continuon Brain runtime bridge (gRPC/WebRTC) for Pi 5, normalizing Donkey Car steering/throttle into `action.command` and logging drivetrain state into `observation.robot_state` with ≤5 ms timestamp alignment.
2. Wire Compose XR/SceneCore inputs plus iPhone sensor proxies into `InputFusion` and `CommandMapper` so both XR and phone shells drive the same teleop surface.
3. Keep RLDS writing aligned with `docs/rlds-schema.md` (tags: `xr_mode`, `action.source`, provenance/env IDs) and buffer uploads in the `metadata.json` + `steps/*.jsonl` layout.
4. Expand unit tests for schema validation, BLE parsing, teleop command mapping, and OTA bundle handoff triggers once the edge bundle path is plumbed.

Conversation log: Pi5 startup/training optimization (2025-12-10) summarized at `../../docs/conversation-log.md` (headless Pi5 boot defaults, optional background trainer, tuned Pi5 training config, RLDS origin tagging).