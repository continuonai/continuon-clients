# Cloud/Colab Export Guide (RLDS → TPU/Cloud Training)

Goal: move locally captured RLDS (Pi/Jetson) to a cloud trainer (Colab TPU/VM), fine-tune with a larger model, then return adapters/edge bundle to the robot.

## On-device (Pi/Jetson)
1) Ensure episodes exist and meet quality gates:
   - Path: `/opt/continuonos/brain/rlds/episodes/`
   - Count: >= `min_episodes` in your trainer config
   - Include any synthetic files only for smoke tests; use real data for training.
2) Package episodes:
   ```bash
   cd /opt/continuonos/brain
   zip -r episodes_pi5_follow.zip rlds/episodes
   ```
3) Copy to your workstation or directly to cloud storage (scp/rsync/gsutil).

## Colab TPU/Cloud VM
1) In Colab, download the zip (Drive/S3/GCS) and unzip:
   ```python
   !mkdir -p /content/data
   !gdown --id YOUR_FILE_ID -O /content/data/episodes.zip  # or gsutil cp / curl
   !unzip -q /content/data/episodes.zip -d /content/data/rlds
   ```
2) Install deps:
   ```python
   !pip install torch torchvision transformers peft datasets
   # Add tensorflow if using TFRecord parsing.
   ```
3) Load RLDS and train (pseudo):
   ```python
   from pathlib import Path
   import json

   def load_json_episode(path: Path):
       payload = json.loads(path.read_text())
       for step in payload["steps"]:
           yield step["observation"], step["action"]

   data_dir = Path("/content/data/rlds/episodes")
   episodes = sorted(data_dir.glob("*.json"))

   # TODO: replace with your real model, tokenizer, and LoRA config
   for obs, action in load_json_episode(episodes[0]):
       pass
   ```
   Replace with your full training script (e.g., Hugging Face Trainer + PEFT/QLoRA) targeting TPU/Accelerator.
4) Save adapters/checkpoints:
   ```python
   out_dir = Path("/content/out/pi5_adapters")
   out_dir.mkdir(parents=True, exist_ok=True)
   # torch.save(lora_state_dict, out_dir / "lora_adapters.pt")
   ```
5) Package for edge:
   - Zip adapters or create your edge bundle manifest (outside this repo).
   - Download to your workstation/Pi.

## Return to robot
1) Place adapters on device:
   ```bash
   scp lora_adapters.pt pi:/opt/continuonos/brain/model/adapters/candidate/
   ```
2) Run the safety gate/promotion or restart brain to load `adapters/current`.

## Notes
- Keep credentials out of notebooks; use short-lived tokens.
- Use synthetic episodes only for smoke tests; real driving requires real data.
- Align episode parser with your on-robot schema (JSON/TFRecord). Update loaders accordingly.
