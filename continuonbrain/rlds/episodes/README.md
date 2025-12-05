# RLDS Episodes (local)

This folder can hold example or synthetic RLDS episodes for local trainer dry-runs. Keep real robot data outside the repo; these samples are non-sensitive scaffolds.

Files:
- `pi5_follow_synthetic.json`: Synthetic Pi 5 Donkeycar episode following a human around a house using a rolling-shutter RGB camera. Five steps with RGB frame URIs, robot state, steering/throttle actions, and human-distance tags.
- `follow_user_mr_v1/` and `follow_user_mr_v2/`: Sample mixed-reality pilot runs for a "follow_user" task, stored in RLDS-style layout with `metadata.json` plus small `steps/*.jsonl` blobs. V1 is marked golden, V2 is the newest capture and is autonomy-eligible.
- `stack_boxes_vr/`: VR copilot stacking dry-run scaffolded as `metadata.json` plus `steps/*.jsonl`.

Usage:
- Copy or duplicate synthetic files into your Pi’s `rlds_dir` (e.g., `/opt/continuonos/brain/rlds/episodes/`) to satisfy `min_episodes` during trainer testing. Replace with real data before meaningful training.
