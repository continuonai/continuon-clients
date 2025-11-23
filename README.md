# ContinuonXR

ContinuonXR is the human-facing XR application that powers Continuon's self-learning loop by turning every interaction into training data. This repository holds the documentation, schemas, and stubs for the Android XR app, companion clients, and data ingestion services described in `PRD.md`.

## Repository layout
- `docs/` - architecture notes, RLDS schema contract, and the XR app specification (Phase 0 deliverables).
- `apps/` - application stubs, starting with the Android XR app scaffold plus placeholders for mock services and Flutter companions.
- `proto/` - protobuf definitions for the Continuon XR <-> ContinuonBrain/OS link and RLDS ingestion.
- `docs/monorepo-structure.md` - how this repo fits into the decoupled `@continuonai` architecture (“One Brain, Many Shells”).
- `continuonbrain/`, `continuon-cloud/`, `continuonai/`, `worldtapeai.com/` - placeholders noting separate repos in the decoupled architecture.

## Getting started
1. Read `docs/rlds-schema.md` and `docs/xr-app-spec.md` to understand required contracts.
2. Review `apps/xr/README.md` for the proposed Android module breakdown and the Kotlin stubs in `apps/xr/src/main/java/`.
3. Extend the protobufs in `proto/` to match the Robot API exposed by ContinuonBrain/OS.
4. See `docs/human-centric-data.md` for how modes map into RLDS and the “One Brain, Many Shells” data flow.
5. See `docs/dev-setup.md` for local build/test prerequisites.

## Build (draft)
- Requires Android Studio Koala or later, Android SDK 35. Gradle wrapper included (Gradle 8.7).
- Build XR app: `./gradlew :apps:xr:assembleDebug`
- Generate Kotlin proto stubs: `./gradlew :apps:xr:generateDebugProto`

## Next steps (tracking)
- Jetpack XR/SceneCore: add dependencies and hook real pose/gaze/audio feeds into `XrInputProvider`/teleop.
- ContinuonBrain link: finalize proto, enable TLS/auth, and wire live gRPC/WebRTC endpoints (mock off by default).
- Glove BLE: replace placeholder UUIDs, handle runtime perms UX, and harden MTU/notification flow.
- Audio/UI: complete audio capture pipeline and workstation UI context logging into RLDS.
- RLDS logging/upload: keep validation, add durable storage + upload pipeline (HTTP uploader scaffold present).
- Testing/tooling: expand coverage (schema validation, teleop mapping, BLE parsing/audio) and keep builds reproducible via wrapper.

## Phase alignment
- **Phase 0 (contracts):** The documentation in `docs/` captures the RLDS schema and XR app spec.
- **Phase 1 (MVP):** Use `apps/xr/` to bootstrap the Jetpack XR MVP (Mode A teleop to a mock ContinuonBrain/OS, local RLDS writer).
