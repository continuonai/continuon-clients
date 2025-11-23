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
1. Hook up Compose XR panels to live SceneCore input streams. Use `XrInputProvider`/`SceneCoreInputManager` (now streaming pose/gaze/audio) to drive the teleop shell.
2. Harden ContinuonBrain/OS connectivity (retries/backpressure) on both gRPC and WebRTC once a production endpoint is available.
3. Extend RLDS writer (file sink added) with schema validation and upload/export.
4. Wire real XR input (head/hand pose, gaze ray, voice/audio, UI events) into `InputFusion` and `CommandMapper`.
5. Expand unit tests for schema validation, BLE parsing, and teleop command mapping.
