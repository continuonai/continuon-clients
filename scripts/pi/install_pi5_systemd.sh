#!/usr/bin/env bash
set -euo pipefail

# Install/enable ContinuonBrain systemd services on Pi.
#
# Usage:
#   ./scripts/pi/install_pi5_systemd.sh
#   ./scripts/pi/install_pi5_systemd.sh --repo /home/pi/ContinuonXR --user pi
#
# Installs:
# - continuonbrain-startup.service (boot runtime)
# - continuonbrain-wavecore.service (optional oneshot trainer)
#
# After install:
#   sudo systemctl status continuonbrain-startup.service
#   sudo journalctl -u continuonbrain-startup.service -f

REPO_DIR=""
RUN_USER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_DIR="${2:-}"; shift 2 ;;
    --user) RUN_USER="${2:-}"; shift 2 ;;
    -h|--help) sed -n '1,160p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${REPO_DIR}" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -z "${RUN_USER}" ]]; then
  RUN_USER="${SUDO_USER:-$USER}"
fi

echo "== Continuon Pi5 systemd install =="
echo "repo: ${REPO_DIR}"
echo "user: ${RUN_USER}"

if [[ ! -d "${REPO_DIR}/continuonbrain/systemd" ]]; then
  echo "ERROR: missing continuonbrain/systemd/ in repo" >&2
  exit 1
fi

sudo mkdir -p /etc/continuonbrain
if [[ ! -f /etc/continuonbrain/continuonbrain.env ]]; then
  sudo cp "${REPO_DIR}/continuonbrain/systemd/continuonbrain.env.example" /etc/continuonbrain/continuonbrain.env
fi

# Point env at this repo and its venv.
sudo sed -i \
  -e "s|^CONTINUON_REPO=.*|CONTINUON_REPO=${REPO_DIR}|g" \
  -e "s|^PYTHONPATH=.*|PYTHONPATH=${REPO_DIR}|g" \
  -e "s|^CONTINUON_PYTHON=.*|CONTINUON_PYTHON=${REPO_DIR}/.venv/bin/python3|g" \
  /etc/continuonbrain/continuonbrain.env

sudo cp "${REPO_DIR}/continuonbrain/systemd/continuonbrain-startup.service" /etc/systemd/system/continuonbrain-startup.service
sudo cp "${REPO_DIR}/continuonbrain/systemd/continuonbrain-wavecore.service" /etc/systemd/system/continuonbrain-wavecore.service

sudo systemctl daemon-reload
sudo systemctl enable --now continuonbrain-startup.service

echo
echo "✅ Enabled: continuonbrain-startup.service"
echo "Optional: run on-demand training:"
echo "  sudo systemctl start continuonbrain-wavecore.service"
echo "Logs:"
echo "  sudo journalctl -u continuonbrain-startup.service -f"


