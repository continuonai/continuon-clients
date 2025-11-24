# ContinuonBrain (scaffolding only)

ContinuonBrain/OS runtime lives in the separate `continuonos` repo (platform-agnostic core, HAL adapters, backends, configs). This folder only carries **scaffolding and contracts** used by ContinuonXR:
- `proto/continuonbrain_link.proto` (shared contract; mirror downstream).
- `trainer/` offline Pi/Jetson adapter-training scaffold (bounded, RLDS-only, safety-gated) to align with ContinuonBrain/OS goals.

No production runtime should live here; wire real implementations inside `continuonos` while keeping the interface consistent.
