# Protobuf Definitions (Draft)

Draft proto files for the XR data loop:
- `continuonxr/rlds/v1/rlds_episode.proto` — RLDS episode contract for ContinuonXR logging and export.
- `continuonxr/continuonbrain/v1/continuonbrain_link.proto` — gRPC bridge between XR client and ContinuonBrain/OS.

Next steps:
1. Finalize Robot API fields and align `RobotState` with ContinuonBrain/OS contracts.
2. Generate Kotlin/JVM and TypeScript stubs for the XR app and mock ContinuonBrain server.
3. Add auth/versioning metadata to RPC requests once Cloud/API requirements are known.

## Codegen
- Kotlin/JVM (used by `apps/continuonxr`): `./gradlew generateProtoKotlin`
- TypeScript (mock server / tooling): `./gradlew generateProtoTypescript` (uses the checked-in `buf.gen.yaml`).
- Schema lint: `./gradlew validateProtoSchemas`

Schema highlights:
- Observation now supports gaze, audio, UI context, and glove validity flags, plus per-step metadata map for contextual tagging.
