# Android XR App Scaffold

This folder sketches the Kotlin/Jetpack XR app described in `PRD.md`. Gradle wiring is present; XR dependencies remain to be added.

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
1. Add Jetpack XR/SceneCore dependencies and hook up Compose scenes for XR panels. Use `XrInputProvider`/`SceneCoreInputManager` to feed pose/gaze/audio into teleop.
2. Implement ContinuonBrain/OS Robot API proto stubs from `proto/` and replace the mock stream in `ContinuonBrainClient`.
3. Extend RLDS writer (file sink added) with schema validation and upload/export.
4. Wire real XR input (head/hand pose, gaze ray, voice/audio, UI events) into `InputFusion` and `CommandMapper`.
5. Expand unit tests for schema validation, BLE parsing, and teleop command mapping.
