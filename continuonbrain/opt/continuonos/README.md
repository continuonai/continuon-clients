# ContinuonOS Developer Kit Layout

This tree mirrors the `/opt/continuonos/` layout expected on the Pi 5 target so you can stage data and loops alongside the Continuon Brain runtime. The contents stay lightweight and pair with the existing runtime assets under `/opt/continuonos/brain/` (model manifests, RLDS outputs, and Hailo/Gemma placeholders).

## Directory map
- `00_data/`: Staging area for replayable RLDS snippets, calibration captures, and any Pi-specific manifests before they are promoted into `/opt/continuonos/brain/rlds/episodes/`.
- `01_vision_dreamer/`: Vision/HAT experiments that can call into `hailo_inference` or CPU/JAX fallbacks; use it to keep HEF exports and perception prompts scoped per experiment.
- `02_liquid_reflex/`: Control/policy sketches that delegate actuator commands to the runtime’s PCA9685/Robot API routes.
- `03_mamba_brain/`: Planning/eval scaffolding for the Mamba-style world model work; keep shard manifests consistent with `docs/wiki-rag-plan.md` and the librarian service defaults.
- `main_loop.py`: Entry point for driving the above modules on-device. The stub keeps imports light so it can run before optional dependencies (Hailo SDK, camera drivers) are installed.

## Syncing to a Pi 5 target
1. Prepare the Pi per `PI5_EDGE_BRAIN_INSTRUCTIONS.md` (I2C enabled, depth camera attached, and the runtime directories created under `/opt/continuonos/brain/{model,trainer,rlds}`).
2. From the repo root on your workstation, push this tree to the Pi:
   ```bash
   PI_HOST=pi@raspberrypi.local
   rsync -av --delete continuonbrain/opt/continuonos/ "$PI_HOST":/opt/continuonos/
   ```
   - The `--delete` flag keeps the Pi copy in lockstep with this repo.
   - If you already have runtime assets under `/opt/continuonos/brain/`, this sync will **not** replace them; it only updates the developer kit directories shown above.
3. On the Pi, verify the runtime placeholders are still present:
   - Hailo HEF placeholder: `/opt/continuonos/brain/model/base_model/model.hef` (CPU fallback when missing).
   - Adapter targets: `/opt/continuonos/brain/model/adapters/{current,candidate}/`.
   - RLDS output root: `/opt/continuonos/brain/rlds/episodes/`.

## How it fits with the Continuon Brain runtime
- The runtime remains the source of truth for hardware IO and inference. Developer kit scripts should import from `continuonbrain` modules (e.g., `hailo_inference.InferenceRouter`, `sensors.hardware_detector`, `robot_api_server`) instead of duplicating drivers.
- Keep experiment outputs pointed at the existing runtime paths above so training and the Flutter runtime can reuse them without extra copies.
- Hailo usage stays optional: the HEF path is checked when available, and CPU/JAX inference paths remain valid when the Hailo SDK is absent.
- When promoting adapters or RLDS episodes generated from this kit, move or symlink them into `/opt/continuonos/brain/model/adapters/current/` and `/opt/continuonos/brain/rlds/episodes/` to align with the Pi startup manager and health checks.

## Running the developer loop
On the Pi (or a dev machine with hardware mocks), run:
```bash
PYTHONPATH=/opt/continuonos/brain:$PYTHONPATH \
  python /opt/continuonos/main_loop.py --describe
```
Use additional flags to target specific experiment subfolders as you extend `main_loop.py`.
