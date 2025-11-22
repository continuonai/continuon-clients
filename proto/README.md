# Protobuf Definitions (Draft)

Draft proto files for the XR data loop:
- `rlds_episode.proto` — RLDS episode contract for ContinuonXR logging and export.
- `pixelbrain_link.proto` — gRPC bridge between XR client and PixelBrain/OS.

Next steps:
1. Finalize Robot API fields and align `RobotState` with PixelBrain/OS contracts.
2. Generate Kotlin/JVM and TypeScript stubs for the XR app and mock PixelBrain server.
3. Add auth/versioning metadata to RPC requests once Cloud/API requirements are known.

## Codegen
- Kotlin/JVM (used by `apps/xr`): `./gradlew :apps:xr:generateDebugProto`
- TypeScript (mock server / tooling) suggestion: use `buf` + `ts-proto` or `@grpc/grpc-js`. Example:
  - `buf generate --template buf.gen.yaml` (template not included yet).
  - Or `npx protoc --ts_out src --grpc_out=grpc_js --plugin=protoc-gen-grpc=$(npm bin)/grpc_tools_node_protoc_plugin proto/*.proto`
