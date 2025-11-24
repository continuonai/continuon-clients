# Trainer scaffolding (offline, Pi-class targets)

This directory hosts **offline-first** scaffolding for local adapter training on Pi/Jetson-class hardware, aligning with ContinuonBrain/OS goals (no cloud dependency, bounded jobs, RLDS-only inputs).

## Key modules
- `local_lora_trainer.py`: core loop with budgets, candidate/current/history rotation, and promotion gate.
- `hooks_torch.py`: lazy Torch hook builder (expects caller-provided base model loader + LoRA injector + loss).
- `rlds_loader.py`: JSON/JSONL loader and TFRecord wrapper helper.
- `safety.py`: simple action-distance and bounds heuristics for shadow evaluation.
- `scheduler.py`: gating on idle/battery/thermals/teleop before running a job.
- `configs/pi5-donkey.json`: sample job config for Pi 5 Donkeycar.
- `examples/pi5_integration.py`: scaffold showing how to plug real model loader, LoRA injector, safety bounds, and gating hooks for Pi 5.

## Usage (examples)

Dry-run with stub hooks:
```bash
python -m continuonbrain.trainer.local_lora_trainer --config continuonbrain/configs/pi5-donkey.json --use-stub-hooks
```

Integrate real Torch model:
```python
from continuonbrain.trainer import build_torch_hooks, make_episode_loader, SafetyGateConfig, LocalTrainerJobConfig, maybe_run_local_training
from continuonbrain.trainer.safety import build_simple_action_guards

cfg = LocalTrainerJobConfig.from_json(Path("continuonbrain/configs/pi5-donkey.json"))
action_distance, violates = build_simple_action_guards(
    numeric_abs_limit=None,
    per_key_bounds={"steering": (-1.0, 1.0), "throttle": (-1.0, 1.0)},
    delta_weight=1.0,
)
hooks = build_torch_hooks(
    base_model_loader=load_base_model,
    lora_injector=inject_lora,
    loss_fn=behavior_cloning_loss,
    action_distance_fn=action_distance,
    safety_fn=violates,
)
episode_loader = make_episode_loader()  # or a tfrecord loader
result = maybe_run_local_training(cfg, hooks, SafetyGateConfig(), gating=None, episode_loader=episode_loader)
```

## Notes
- Keep real runtime code in `continuonos`; this scaffold is for planning/testing and aligns with the mock ContinuonBrain goals (bounded offline jobs, RLDS I/O, safety gate before swapping adapters).
- TensorFlow is optional; provide `tfrecord_iter`/`example_parser` if you prefer TFRecord episodes without importing TF globally.
- The mock ContinuonBrain service in `apps/mock-continuonbrain` remains for XR contract testing only; do not run trainer there. Use Pi/Jetson hardware for adapter updates.
