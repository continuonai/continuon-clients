# RLDS Episodes (local)

This folder can hold example or synthetic RLDS episodes for local trainer dry-runs. Keep real robot data outside the repo; these samples are non-sensitive scaffolds.

Files:
- `pi5_follow_synthetic.json`: Synthetic Pi 5 Donkeycar episode following a human around a house using a rolling-shutter RGB camera. Five steps with RGB frame URIs, robot state, steering/throttle actions, and human-distance tags.

Usage:
- Copy or duplicate synthetic files into your Pi’s `rlds_dir` (e.g., `/opt/continuonos/brain/rlds/episodes/`) to satisfy `min_episodes` during trainer testing. Replace with real data before meaningful training.
