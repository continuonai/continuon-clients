# Converting a Jetson Donkeycar setup to Raspberry Pi 5 (8 GB)

This repo now carries both scaffolding and the runtime; use these steps as a quick checklist to keep Pi and Jetson episode layouts in sync.

## Steps
1) **Reuse RLDS data**: copy your Jetson-generated RLDS episodes to the Pi:
   ```bash
   scp -r jetson:/opt/continuonos/brain/rlds/episodes pi:/opt/continuonos/brain/rlds/episodes
   ```
2) **Place base model**: put Gemma 3n base weights (non-quantized edge build if it fits) on the Pi:
   - Path: `/opt/continuonos/brain/model/base_model/gemma-3n.tflite` (or your supported format).
   - Update `continuonbrain/configs/pi5-donkey.json` `base_model_path` accordingly.
3) **Adapters layout**:
   ```
   /opt/continuonos/brain/model/adapters/current/      # active
   /opt/continuonos/brain/model/adapters/candidate/    # trainer writes here
   /opt/continuonos/brain/model/adapters/history/      # rotated
   ```
4) **Runtime manifest**: adapt `continuonbrain/model/manifest.pi5.example.json` for your paths so `flutter_gemma` can load base + LoRA adapters.
5) **Trainer config**: set `rlds_dir`, `base_model_path`, budgets in `pi5-donkey.json`. Ensure `min_episodes` is satisfied after copying data.
6) **Run trainer** on Pi when idle/battery/temp are OK:
   ```bash
   python -m continuonbrain.trainer.examples.pi5_integration --config /opt/continuonos/brain/train/pi5-donkey.json
   ```
7) **Safety gate & promotion**: trainer will promote to `adapters/current` only after passing the shadow eval; runtime should reload adapters or be signaled.

Notes:
- Keep everything offline; uploads are manual/opt-in (see CLOUD_EXPORT.md if needed).
- Avoid extra quantization unless required; Pi 5 8 GB can host a small Gemma edge build with LoRA if budgets are respected.
- If `min_episodes` is high, duplicate a small set for smoke tests, then replace with real data before meaningful training.
- Use `gemma_hooks.py` in the trainer to inject LoRA into Gemma proj layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and `gating_continuonos.py` to wire idle/battery/thermals/teleop from continuonos.
- Use `safety_head_stub.py` or your safety head in runtime to clamp actions; include a safety head path in your manifest (`manifest.pi5.safety.example.json`).
