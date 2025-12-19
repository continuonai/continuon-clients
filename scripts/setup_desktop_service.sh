#!/usr/bin/env bash
set -euo pipefail

# Setup ContinuonBrain desktop-level systemd user service
# Automatically detects repo path and user, and configures service to run start_desktop_services.sh

REPO_DIR=""
CONFIG_DIR="${HOME}/.continuonbrain"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_DIR="${2:-}"; shift 2 ;;
    --config-dir) CONFIG_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      cat << EOF
Setup ContinuonBrain desktop-level systemd user service

Usage: $0 [options]

Options:
  --repo PATH       Path to ContinuonXR repository (default: auto-detect)
  --config-dir PATH Config directory (default: ~/.continuonbrain)
  -h, --help        Show this help message

This script will:
  1. Create user systemd service directory
  2. Install and enable continuonbrain-desktop.service
  3. Enable systemd user linger (for auto-start without login)
  4. Start the service
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Auto-detect repo directory if not provided
if [[ -z "${REPO_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/start_desktop_services.sh" ]]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  else
    if [[ -f "scripts/start_desktop_services.sh" ]]; then
      REPO_DIR="$(pwd)"
    else
      echo "ERROR: Could not find ContinuonXR repository" >&2
      exit 1
    fi
  fi
fi

REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
USER="${USER:-$(whoami)}"

echo "== ContinuonBrain Desktop Service Setup =="
echo "Repository: ${REPO_DIR}"
echo "User: ${USER}"
echo ""

# Create systemd user directory
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_USER_DIR}"

# Create service file
SERVICE_FILE="${SYSTEMD_USER_DIR}/continuonbrain-desktop.service"
echo "Creating service file: ${SERVICE_FILE}"

cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=ContinuonBrain Desktop Services (Startup Manager + API + UI)
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
Environment=CONFIG_DIR=${CONFIG_DIR}
Environment=DISPLAY=:0
Environment=XAUTHORITY=${HOME}/.Xauthority
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/scripts/start_desktop_services.sh start
ExecStop=${REPO_DIR}/scripts/start_desktop_services.sh stop
PIDFile=${CONFIG_DIR}/runtime_pids/startup_manager.pid
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

echo "✅ Service file created"

# Enable systemd user linger
echo "Enabling systemd user linger..."
sudo loginctl enable-linger "${USER}" || {
  echo "⚠️  Warning: Could not enable linger (may need sudo)" >&2
}

# Reload systemd and enable service
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

echo "Enabling continuonbrain-desktop.service..."
systemctl --user enable continuonbrain-desktop.service

echo "Starting continuonbrain-desktop.service..."
systemctl --user start continuonbrain-desktop.service || {
  echo "⚠️  Warning: Service start failed. Check logs with:" >&2
  echo "   systemctl --user status continuonbrain-desktop.service" >&2
}

echo ""
echo "✅ Setup complete!"
echo "Useful commands:"
echo "  systemctl --user status continuonbrain-desktop.service"
echo "  journalctl --user -u continuonbrain-desktop.service -f"
