# Mock ContinuonBrain/OS Service (Stub)

Use this directory to host a lightweight mock of ContinuonBrain/OS for Phase 1 lab development. Goals:
- Replay robot state streams for XR client consumption.
- Accept normalized command vectors and log them for inspection.
- Optionally echo back latency/diagnostic metrics.

Suggested tooling: Kotlin/JVM or Node/TypeScript gRPC server generated from `proto/` definitions with canned trajectories for testing teleop.

## Quick start ideas
- Kotlin: Generate stubs via `./gradlew :apps:xr:generateDebugProto` (uses shared `proto/` folder), then create a small Ktor/gRPC server that streams `RobotStateEnvelope` messages.
- TypeScript: Generate via `buf` + `ts-proto` or `grpc-tools` and serve canned trajectories from a Node process.
- Add a flag to jitter latency/drop rate to validate XR-side resilience.
