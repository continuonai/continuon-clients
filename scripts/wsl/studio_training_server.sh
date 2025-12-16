#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8081}"
RUNTIME_ROOT="${2:-/opt/continuonos/brain}"
CONFIG_DIR="${3:-/tmp/continuonbrain_demo}"
REPO_PATH="${4:-/mnt/c/Users/CraigM/source/repos/ContinuonXR}"
LOG_PATH="${5:-/tmp/studio_training_server.log}"

cd "$REPO_PATH"

if [[ -f "/root/.venvs/continuon_repo/bin/activate" ]]; then
  # Pinned CUDA JAX venv we set up earlier
  # shellcheck disable=SC1091
  source "/root/.venvs/continuon_repo/bin/activate"
elif [[ -f "$HOME/.venvs/continuon_repo/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.venvs/continuon_repo/bin/activate"
fi

mkdir -p "$RUNTIME_ROOT"/{rlds/episodes,trainer/logs,trainer/checkpoints/core_model_seed,model/adapters/{candidate,current}/core_model_seed}

pkill -f continuonbrain.studio_training_server >/dev/null 2>&1 || true

nohup python -m continuonbrain.studio_training_server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --runtime-root "$RUNTIME_ROOT" \
  --config-dir "$CONFIG_DIR" \
  >"$LOG_PATH" 2>&1 &

echo $! > /tmp/studio_training_server.pid

# Keep WSL alive while the server runs.
exec tail -f /dev/null


