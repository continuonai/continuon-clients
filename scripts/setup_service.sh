#!/usr/bin/env bash
set -euo pipefail

# Unified ContinuonBrain service setup script.
# Configures the environment and installs a systemd user service.

MODE="desktop"
REPO_DIR=""
CONFIG_DIR="${HOME}/.continuonbrain"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --repo) REPO_DIR="${2:-}"; shift 2 ;;
    --config-dir) CONFIG_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      cat << EOF
Usage: $0 [options]

Options:
  --mode desktop|rpi  Operating mode (default: desktop)
  --repo PATH         Path to repo (default: auto-detect)
  --config-dir PATH   Config dir (default: ~/.continuonbrain)
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Auto-detect repo
if [[ -z "${REPO_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
VENV="${REPO_DIR}/.venv"

echo "== ContinuonBrain Unified Setup (mode=${MODE}) =="

# 1. Environment Setup
if [[ ! -d "${VENV}" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --upgrade pip --quiet
pip install -r "${REPO_DIR}/continuonbrain/requirements.txt" --quiet

# 2. Systemd Setup
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_USER_DIR}"
SERVICE_FILE="${SYSTEMD_USER_DIR}/continuonbrain.service"

cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=ContinuonBrain Unified Service (mode=${MODE})
After=network-online.target

[Service]
Type=forking
Environment=CONFIG_DIR=${CONFIG_DIR}
Environment=DISPLAY=:0
Environment=XAUTHORITY=${HOME}/.Xauthority
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/scripts/start_services.sh start --mode ${MODE}
ExecStop=${REPO_DIR}/scripts/start_services.sh stop
PIDFile=${CONFIG_DIR}/runtime_pids/startup_manager.pid
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# 3. Linger & Enable
sudo loginctl enable-linger "${USER}" || true
systemctl --user daemon-reload
systemctl --user enable continuonbrain.service
systemctl --user restart continuonbrain.service

echo "✅ Unified setup complete!"
systemctl --user status continuonbrain.service --no-pager
