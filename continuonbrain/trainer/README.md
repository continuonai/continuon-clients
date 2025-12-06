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
- `CLOUD_EXPORT.md`: how to package RLDS and train in Colab/TPU, then bring adapters back.
- `GEMMA_PI5.md`: plan for running Gemma 3n + LoRA on Pi 5 with `flutter_gemma` (runtime) and sidecar trainer.
- `../model/manifest.pi5.example.json`: sample manifest for `flutter_gemma` to load base + adapter paths on Pi 5 (use non-quantized Gemma 3n if memory permits).
- `gemma_hooks.py`: in-place LoRA injector for Gemma models (torch) compatible with the trainer interface.
- `gating_continuonos.py`: helper to wire trainer gating to continuonos runtime signals (idle/battery/thermals/teleop).
- `safety_head_stub.py`: simple clamp/violation logger for steering/throttle; replace with your real safety head.
- `vla_adapter.py`: routes WaveCore `encode_for_policy()` outputs plus RLDS metadata into minimal VLA heads (SkillPolicies/LanguagePlanner) with switchable routing modes.
- `INTEGRATION_CHECKLIST.md`: step-by-step to drop the scaffold into continuonos (model loader, safety, gating, manifest, RLDS).
- `sidecar_runner.py`: RLDS-aware wrapper so a sidecar process can gate on idle/battery/thermals and only promote adapters after the safety gate passes.

## Usage (examples)

Dry-run with stub hooks:
```bash
python -m continuonbrain.trainer.local_lora_trainer --config continuonbrain/configs/pi5-donkey.json --use-stub-hooks
```

Integrate real Torch model:
```python
from continuonbrain.trainer import build_torch_hooks, make_episode_loader, SafetyGateConfig, LocalTrainerJobConfig, maybe_run_local_training
from continuonbrain.trainer.safety import build_simple_action_guards
from continuonbrain.trainer.vla_adapter import AdapterRoutingConfig, build_policy_request, forward_to_vla_head

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

# Route WaveCore embeddings + RLDS metadata into a VLA head with configurable routing
wavecore_output = wave_model.encode_for_policy(rlds_obs["egocentric_video"])  # user-supplied WaveCore call
sample = build_policy_request(
    wavecore_output=wavecore_output,
    rlds_observation=rlds_obs,  # RLDS observation dict with poses/gaze/audio/glove/diagnostics
    routing=AdapterRoutingConfig(mode="hybrid"),  # switch to "wave_only" or "attention_only" without changing inputs
)
action = forward_to_vla_head(language_planner_head, sample)
```

## Notes
- The runtime now lives in this repo; use this scaffold for planning/testing and clearly label what is placeholder versus production-ready (bounded offline jobs, RLDS I/O, safety gate before swapping adapters).
- TensorFlow is optional; provide `tfrecord_iter`/`example_parser` if you prefer TFRecord episodes without importing TF globally.
- Use the production Robot API server (`python -m continuonbrain.robot_api_server`) for end-to-end XR contract tests on hardware; do not run trainer inside the server process. Use Pi/Jetson hardware for adapter updates.
- `base_model_path` in configs (e.g., `/opt/continuonos/brain/model/base_policy.pt`) points to the frozen policy checkpoint used before attaching LoRA adapters.
- `make_batch_iterator` accepts a `drop_last` flag (default `True`) so you can emit the final partial batch on tiny smoke-test datasets instead of discarding it.
- The Pi integration example aborts if `base_model_path` or `rlds_dir` are missing/insufficient; ensure both exist on-device before running.
- Synthetic RLDS samples for dry-runs live under `continuonbrain/rlds/episodes/`; duplicate them into your Pi `rlds_dir` to meet `min_episodes` when testing.
- For Jetson-to-Pi migration, see `continuonbrain/rlds/episodes/CONVERT_JETSON_TO_PI5.md` for a short checklist (copy RLDS, set base model path, adjust manifest, run trainer).
- Keep `min_episodes` realistic; for smoke tests you can duplicate a tiny set, but swap in real data before any meaningful training.
- Use `build_gemma_lora_hooks` if you have a torch Gemma loader and want in-place LoRA without peft wrapping; use `build_pi5_gating` to connect to real idle/battery/thermals/teleop signals in continuonos.
- For runtime safety, use `safety_head_stub` as a placeholder to clamp actions and log violations; integrate your actual safety head in continuonos.
