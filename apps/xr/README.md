# Android XR App Scaffold

This folder sketches the Kotlin/Jetpack XR app described in `PRD.md`. It is intentionally light-weight and does not include Gradle wiring yet.

## Modules
- `MainActivity.kt` — entry point, routes to mode-specific shells (trainer/workstation/observer).
- `config/` — configuration loading and flags (mode, connectivity, logging, glove).
- `connectivity/` — PixelBrain/OS bridge client for gRPC/WebRTC streams.
- `glove/` — BLE ingestion for Continuon Glove v0 (MTU negotiation, frame parser, diagnostics).
- `teleop/` — input fusion and command mapping for Mode A teleop; records RLDS steps.
- `logging/` — RLDS schema enforcement and local persistence hooks.

## Next steps
1. Add Gradle build files with Jetpack XR/Compose, coroutines, and protobuf dependencies.
2. Implement PixelBrain/OS Robot API proto stubs from `proto/`.
3. Flesh out RLDS writer to persist to disk and perform schema validation.
4. Wire real XR input (head/hand pose, voice) into `InputFusion` and `CommandMapper`.
5. Add unit tests for schema validation, glove parsing, and teleop command mapping.

