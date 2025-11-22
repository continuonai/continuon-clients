# Android XR App Scaffold

This folder sketches the Kotlin/Jetpack XR app described in `PRD.md`. Gradle wiring is present; XR dependencies remain to be added.

## Modules
- `MainActivity.kt` — entry point, routes to mode-specific shells (trainer/workstation/observer).
- `config/` — configuration loading and flags (mode, connectivity, logging, glove).
- `connectivity/` — PixelBrain/OS bridge client for gRPC/WebRTC streams.
- `glove/` — BLE ingestion for Continuon Glove v0 (MTU negotiation, frame parser, diagnostics).
- `teleop/` — input fusion and command mapping for Mode A teleop; records RLDS steps.
- `logging/` — RLDS schema enforcement and local persistence hooks.

## Build/test quickstart
- Requires Android Studio Koala or later (AGP 8.5) and a local Gradle install or wrapper.
- Build app: `./gradlew :apps:xr:assembleDebug`
- Run unit tests: `./gradlew :apps:xr:testDebugUnitTest`
- Generate proto stubs: `./gradlew :apps:xr:generateDebugProto`

## Next steps
1. Add Jetpack XR/SceneCore dependencies and hook up Compose scenes for XR panels.
2. Implement PixelBrain/OS Robot API proto stubs from `proto/` and a mock bridge.
3. Extend RLDS writer (file sink added) with schema validation and upload/export.
4. Wire real XR input (head/hand pose, voice) into `InputFusion` and `CommandMapper`.
5. Expand unit tests for schema validation, BLE parsing, and teleop command mapping.
