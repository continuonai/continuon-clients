# Mock ContinuonBrain/OS Service (Stub)

Use this directory to host a lightweight mock of ContinuonBrain/OS for Phase 1 lab development. Goals:
- Replay robot state streams for XR client consumption.
- Accept normalized command vectors and log them for inspection.
- Optionally echo back latency/diagnostic metrics.

Suggested tooling: Kotlin/JVM or Node/TypeScript gRPC server generated from `proto/` definitions with canned trajectories for testing teleop.

## Quick start ideas
- Kotlin: Generate stubs via `./gradlew generateProtoKotlin` (uses shared `proto/` folder), then create a small Ktor/gRPC server that streams `StreamRobotStateResponse` messages.
- TypeScript: Run `npm install && npm run build` in this directory to generate stubs (via `ts-proto` + `buf`) and type-check the mock server helpers.
- Add a flag to jitter latency/drop rate to validate XR-side resilience.
