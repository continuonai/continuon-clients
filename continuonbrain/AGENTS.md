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
