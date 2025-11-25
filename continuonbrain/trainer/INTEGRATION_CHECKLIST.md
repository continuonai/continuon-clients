# Continuonos Integration Checklist (Pi 5, Gemma + LoRA + Safety)

Use this to drop the trainer scaffold into the real continuonos runtime.

## Model plumbing
- [ ] Implement a real Gemma loader (Torch or supported format) and point `base_model_path` in `pi5-donkey.json`.
- [ ] Replace the trainer placeholder loss with your behavior cloning / control loss.
- [ ] Use `build_gemma_lora_hooks` (or your PEFT equivalent) with target proj layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

## Safety
- [ ] Integrate your safety head or use `safety_head_stub.py` as a clamp until ready.
- [ ] Add safety flags/clamps to RLDS `step_metadata` (e.g., `safety_clamped=true`, `violations` map).
- [ ] Include a safety head path in the manifest if you have a model (`manifest.pi5.safety.example.json`).

## Gating (idle/battery/thermal/teleop)
- [ ] Wire `build_pi5_gating` to continuonos signals for idle, teleop active, battery, and CPU temp.
- [ ] Set sensible thresholds (battery >= 40%, CPU temp <= 75C) before running Mid/Slow loops.

## Runtime manifest
- [ ] Populate `/opt/continuonos/brain/model/manifest.json` using `manifest.pi5.example.json` (or safety variant) with correct base/adapter paths.
- [ ] Ensure the Flutter runtime (`flutter_gemma`) reloads adapters on promotion or via a small IPC signal.

## RLDS + data
- [ ] Stage RLDS episodes in `/opt/continuonos/brain/rlds/episodes` and meet `min_episodes`.
- [ ] Log safety metadata and loop tags (`fast_loop_tick`, `mid_loop_train_id`, `safety_violation`) for later training.

## Promotion/rollback
- [ ] Maintain `adapters/current`, `adapters/candidate`, `adapters/history`; keep a factory/last-good set.
- [ ] On promotion, signal the runtime to reload adapters; on failure, revert to last-good immediately.

## Offline guarantee
- [ ] Disable any network calls in all loops; uploads are manual/opt-in only (see `CLOUD_EXPORT.md`).
