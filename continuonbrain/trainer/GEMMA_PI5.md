# Pi 5 Brain Plan with Gemma 3n + LoRA (offline)

Goal: run a small brain on Raspberry Pi 5 using Gemma 3n for language/control logic with LoRA adapters, keeping everything offline and compatible with `flutter_gemma`.

## Architecture
- **Runtime (Flutter/Dart)**: `flutter_gemma` hosts Gemma 3n for inference in the app; loads base model + current LoRA adapters.
- **Sidecar trainer (Python/Rust)**: uses the local trainer scaffold to fine-tune adapters on RLDS; writes to `model/adapters/candidate` and promotes on safety pass.
- **Storage layout**:
  - `/opt/continuonos/brain/model/base_model/` — Gemma 3n weights (quantized).
  - `/opt/continuonos/brain/model/adapters/current/` — active LoRA adapters.
  - `/opt/continuonos/brain/model/adapters/candidate/` — newly trained adapters.
  - `/opt/continuonos/brain/rlds/episodes/` — local RLDS only.
  - `/opt/continuonos/brain/model/manifest.json` — manifest consumed by `flutter_gemma` (see `continuonbrain/model/manifest.pi5.example.json`).

## Build/Packaging Steps (Pi 5)
1) **Prepare Gemma 3n (no extra quantization if it fits)**:
   - Use the standard edge build that `flutter_gemma` can load on Pi 5 (8 GB).
   - Place at `/opt/continuonos/brain/model/base_model/gemma-3n.tflite` (or your supported format).
   - Set `base_model_path` in `continuonbrain/configs/pi5-donkey.json`.
2) **Adapters + manifest**:
   - Keep adapters in `/opt/continuonos/brain/model/adapters/current/lora_adapters.pt`.
   - Use `continuonbrain/model/manifest.pi5.example.json` as a template for `flutter_gemma` to load base + adapters.
3) **Sidecar trainer wiring** (Python/Rust):
   - Use `continuonbrain/trainer/examples/pi5_integration.py` with your real Gemma model loader + LoRA injector.
   - Point `rlds_dir` to `/opt/continuonos/brain/rlds/episodes`.
   - Run the trainer when idle/battery/temp allow; promotion gate handles candidate → current.
4) **Safety + gating** (Pi):
   - Implement `build_gates()` with real idle/battery/thermal/teleop hooks from continuonos.
   - Implement action bounds/deltas in `build_simple_action_guards` for your steering/throttle.

## Flutter integration (sketch)
- Add a Dart service that:
  - Reads a manifest (JSON) with `base_model_path` and `adapter_path`.
  - Calls `flutter_gemma` to load base + adapter.
  - On adapter promotion (file change or signal), hot-reloads the adapter.
- Expose a minimal IPC (e.g., Unix socket or gRPC) for the sidecar to notify the Flutter process after promotion.

## Training loop notes (sidecar)
- Replace placeholder model hooks with a Gemma 3n-compatible LoRA injector (Torch/PEFT or equivalent).
- For speed on Pi 5: keep rank low (e.g., 4–8), cap steps/wall-time as in config.
- Use RLDS JSON/TFRecord from `rlds_dir`; keep `min_episodes` conservative until you have real data.
- HOPE loops remain local/offline on Pi 5: Fast (control), Mid (bounded trainer), Slow (docked consolidation) all run without internet; uploads are opt-in only.
- Safety/other heads: keep a small safety head (bounds/violations) running alongside policy; log safety flags in RLDS `step_metadata`. If you add a world/planner head, ensure it runs locally or stays disabled until an edge build is available.

## Cloud lift
- When more data is ready, follow `CLOUD_EXPORT.md` to zip RLDS, train on TPU/Colab with a larger setup, then ship back adapters (or a merged edge bundle).
