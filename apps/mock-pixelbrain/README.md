# Mock PixelBrain/OS Service (Stub)

Use this directory to host a lightweight mock of PixelBrain/OS for Phase 1 lab development. Goals:
- Replay robot state streams for XR client consumption.
- Accept normalized command vectors and log them for inspection.
- Optionally echo back latency/diagnostic metrics.

Suggested tooling: Kotlin/JVM or Node/TypeScript gRPC server generated from `proto/` definitions with canned trajectories for testing teleop.

