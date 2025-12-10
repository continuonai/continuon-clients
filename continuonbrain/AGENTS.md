# Agent Instructions (ContinuonBrain)

Scope: `continuonbrain/`.

- The Continuon Brain runtime and scaffolding are now co-located in this monorepo. Keep docs clear about what is production-ready versus staged scaffolding so downstream consumers can promote pieces confidently.
- Prefer small, dependency-light utilities. Avoid adding heavy ML packages beyond what the trainer stubs already expect; keep optional imports guarded so Pi/Jetson environments can still import modules.
- Maintain alignment with RLDS inputs/outputs: update comments/examples when changing schema expectations, and keep sample manifests/configs consistent with the trainer defaults.
- Use type hints and clear docstrings for trainer hooks/safety adapters. Keep safety gating logic explicit and easy to override from continuonos.
- Testing expectations:
  - For Python modules, run `python -m continuonbrain.trainer.local_lora_trainer --help` or an equivalent import check after altering trainer code.
  - For manifest/config changes, validate JSON syntax and ensure sample paths remain coherent with README references.
  Mention any skipped checks due to unavailable dependencies or hardware.

### Pi 5 HAT vision seed (reference names)
- Base model placeholder: `/opt/continuonos/brain/model/base_model/hat_vision_seed.pt`
- Current adapter target: `/opt/continuonos/brain/model/adapters/current/lora_hat_vision.pt`
- Candidate adapter target: `/opt/continuonos/brain/model/adapters/candidate/lora_hat_vision.pt`
- RLDS dir: `/opt/continuonos/brain/rlds/episodes/` (camera-only acceptable when PCA is down; OAK-D Lite provides RGB+depth)
- If on-device LLM/tools (Gemma 3n 2B + MCP/http), log tool traces into `step_metadata` for later cloud/JAX ingestion.
- Hailo: prefer Hailo for inference when available; placeholder HEF path `/opt/continuonos/brain/model/base_model/model.hef` with CPU fallback when absent.

Recent updates:
- JAX-first model selection is preferred (set `CONTINUON_PREFER_JAX=1`, default). Transformers/Gemma remains optional fallback; avoid heavy imports on startup when possible.
- Hailo export/runtime is still a placeholder; `.hef` presence is checked, but compilation/runtime requires the Hailo SDK. Document clearly when placeholders are used.
- Robot API server has been partially decomposed (chat/tasks extracted). Further splits (devices/routes) should continue to keep the server small and testable.
- New JAX inference utilities: CPU inference CLI (`jax_models/export/infer_cpu.py`) and hardware-aware inference router; keep imports guarded on constrained devices.
- Hailo notes: `inference_router` will load `.hef` and configure the device when `hailo_platform` is installed; VStream tensor I/O is still NotImplemented until wired with Hailo SDK and tensor specs. Export creates a placeholder `.hef` if Hailo tools are absent.
- UI templates: the router auto-materializes UI/control HTML into `server/templates/` from the existing providers; once generated, templates can be edited without touching `robot_api_server.py`.