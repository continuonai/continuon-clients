# ContinuonXR

ContinuonXR is the human-facing XR application that powers Continuon's self-learning loop by turning every interaction into training data. This repository holds the documentation, schemas, and stubs for the Android XR app, companion clients, and data ingestion services described in `PRD.md`.

## Repository layout
- `docs/` - architecture notes, RLDS schema contract, and the XR app specification (Phase 0 deliverables).
- `apps/continuonxr/` - Android XR app scaffold and stubs.
- `proto/` - protobuf definitions for the Continuon XR <-> ContinuonBrain/OS link and RLDS ingestion.
- `docs/monorepo-structure.md` - how this repo fits into the decoupled `@continuonai` architecture ("One Brain, Many Shells").
- `continuonbrain/`, `continuon-cloud/`, `continuonai/`, `worldtapeai.com/` - placeholders noting separate repos in the decoupled architecture.

## Getting started
1. Read `docs/rlds-schema.md` and `docs/xr-app-spec.md` to understand required contracts.
2. Review `apps/continuonxr/README.md` for the proposed Android module breakdown and the Kotlin stubs in `apps/continuonxr/src/main/java/`.
3. Extend the protobufs in `proto/` to match the Robot API exposed by ContinuonBrain/OS.
4. See `docs/human-centric-data.md` for how modes map into RLDS and the “One Brain, Many Shells” data flow.
5. See `docs/dev-setup.md` for local build/test prerequisites.

## Build (draft)
- Requires Android Studio Koala or later, Android SDK 35. Gradle wrapper included (Gradle 8.7).
- Build XR app: `./gradlew :apps:continuonxr:assembleDebug`
- Generate Kotlin proto stubs: `./gradlew :apps:continuonxr:generateDebugProto`

## Next steps (tracking)
- Jetpack XR/SceneCore: add real dependencies and wire live pose/gaze/audio streams into `XrInputProvider`/teleop (currently stubbed and gated by ENABLE_XR_DEPS).
- ContinuonBrain link: wire live gRPC/WebRTC endpoints with TLS/auth and turn off mock in prod; current client/stubs are placeholders.
- Glove BLE: swap placeholder service/characteristic UUIDs for real firmware values; improve runtime permission UX and MTU/notification robustness.
- Audio/UI logging: implement microphone capture pipeline and workstation UI event logging to populate `observation.audio`/`ui_action`.
- RLDS logging/upload: add durable storage, retries, and production upload flow beyond current file sink + HTTP scaffold.
- Testing/tooling: expand coverage (schema validation, teleop mapping, BLE parsing/audio) and keep wrapper-based builds reproducible.

## Phase alignment
- **Phase 0 (contracts):** The documentation in `docs/` captures the RLDS schema and XR app spec.
- **Phase 1 (MVP):** Use `apps/continuonxr/` to bootstrap the Jetpack XR MVP (Mode A teleop to a mock ContinuonBrain/OS, local RLDS writer).
