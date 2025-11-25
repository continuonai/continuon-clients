# ContinuonBrain (scaffolding only)

ContinuonBrain/OS runtime lives in the separate `continuonos` repo (platform-agnostic core, HAL adapters, backends, configs). This folder only carries **scaffolding and contracts** used by ContinuonXR:
- `proto/continuonbrain_link.proto` (shared contract; mirror downstream).
- `trainer/` offline Pi/Jetson adapter-training scaffold (bounded, RLDS-only, safety-gated) to align with ContinuonBrain/OS goals. Synthetic RLDS samples for dry-runs sit under `continuonbrain/rlds/episodes/`. Sample manifest in `continuonbrain/model/manifest.pi5.example.json` shows how Pi 5 + `flutter_gemma` can load base + LoRA without extra quantization.

No production runtime should live here; wire real implementations inside `continuonos` while keeping the interface consistent.
