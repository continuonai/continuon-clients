# ContinuonXR

ContinuonXR is the human-facing XR application that powers Continuon's self-learning loop by turning every interaction into training data. This repository holds the documentation, schemas, and stubs for the Android XR app, companion clients, and data ingestion services described in `PRD.md`.

## Repository layout
- `docs/` - architecture notes, RLDS schema contract, and the XR app specification (Phase 0 deliverables).
- `apps/` - application stubs, starting with the Android XR app scaffold plus placeholders for mock services and Flutter companions.
- `proto/` - protobuf definitions for the Continuon XR <-> PixelBrain/OS link and RLDS ingestion.

## Getting started
1. Read `docs/rlds-schema.md` and `docs/xr-app-spec.md` to understand required contracts.
2. Review `apps/xr/README.md` for the proposed Android module breakdown and the Kotlin stubs in `apps/xr/src/main/java/`.
3. Extend the protobufs in `proto/` to match the Robot API exposed by PixelBrain/OS.

## Build (draft)
- Requires Android Studio Koala or later, Android SDK 35, and a local Gradle install or wrapper.
- Build XR app: `./gradlew :apps:xr:assembleDebug`
- Generate Kotlin proto stubs: `./gradlew :apps:xr:generateDebugProto`

## Next steps (tracking)
- Add Jetpack XR/SceneCore dependencies and Compose scenes for XR panels in `apps/xr`.
- Finalize PixelBrain/OS Robot API fields in `proto/` and implement gRPC/WebRTC client plus mock server.
- Glove BLE: wire MTU/notification flow in `GloveBleClient`; parser draft + test lives in `GloveFrameParser`.
- RLDS logging: file sink stub is in place; add schema validation and upload pipeline in `RldsEpisodeWriter`.
- Testing/tooling: expand coverage (schema validation, teleop mapping, BLE parsing) and add a Gradle wrapper for reproducible builds.

## Phase alignment
- **Phase 0 (contracts):** The documentation in `docs/` captures the RLDS schema and XR app spec.
- **Phase 1 (MVP):** Use `apps/xr/` to bootstrap the Jetpack XR MVP (Mode A teleop to a mock PixelBrain/OS, local RLDS writer).
