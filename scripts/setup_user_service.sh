#!/usr/bin/env bash
set -euo pipefail

# Setup ContinuonBrain user-level systemd service
# Automatically detects repo path and user, installs dependencies, and configures service
#
# Usage:
#   ./scripts/setup_user_service.sh
#   ./scripts/setup_user_service.sh --repo /path/to/ContinuonXR

REPO_DIR=""
CONFIG_DIR="${HOME}/.continuonbrain"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_DIR="${2:-}"; shift 2 ;;
    --config-dir) CONFIG_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      cat << EOF
Setup ContinuonBrain user-level systemd service

Usage: $0 [options]

Options:
  --repo PATH       Path to ContinuonXR repository (default: auto-detect)
  --config-dir PATH Config directory (default: ~/.continuonbrain)
  -h, --help        Show this help message

This script will:
  1. Install Python dependencies (including zeroconf)
  2. Create user systemd service directory
  3. Install and enable continuonbrain-startup.service
  4. Enable systemd user linger (for auto-start without login)
  5. Start the service

After setup:
  systemctl --user status continuonbrain-startup.service
  systemctl --user logs continuonbrain-startup.service -f
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Auto-detect repo directory if not provided
if [[ -z "${REPO_DIR}" ]]; then
  # Try to find repo from script location
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/../continuonbrain/startup_manager.py" ]]; then
    REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  else
    # Try current directory
    if [[ -f "continuonbrain/startup_manager.py" ]]; then
      REPO_DIR="$(pwd)"
    else
      echo "ERROR: Could not find ContinuonXR repository" >&2
      echo "Please run from repo root or use --repo PATH" >&2
      exit 1
    fi
  fi
fi

REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
USER="${USER:-$(whoami)}"
# Use standard .venv (not .venv-pi) for normal procedural Python environment
VENV="${REPO_DIR}/.venv"

echo "== ContinuonBrain User Service Setup =="
echo "Repository: ${REPO_DIR}"
echo "User: ${USER}"
echo "Config Dir: ${CONFIG_DIR}"
echo "Virtual Env: ${VENV} (standard .venv for procedural Python)"
echo ""

# Check if virtual environment exists (using standard .venv, not .venv-pi)
if [[ ! -d "${VENV}" ]]; then
  echo "Creating virtual environment at ${VENV}..."
  python3 -m venv "${VENV}"
fi

# Activate venv and install dependencies
echo "Installing Python dependencies..."
source "${VENV}/bin/activate"
pip install --upgrade pip --quiet
pip install -r "${REPO_DIR}/continuonbrain/requirements.txt" --quiet

# Ensure zeroconf is installed (should be in requirements, but double-check)
if ! python -c "import zeroconf" 2>/dev/null; then
  echo "Installing zeroconf..."
  pip install zeroconf --quiet
fi

echo "✅ Dependencies installed"
echo ""

# Create systemd user directory
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_USER_DIR}"

# Create service file from template
SERVICE_FILE="${SYSTEMD_USER_DIR}/continuonbrain-startup.service"
echo "Creating service file: ${SERVICE_FILE}"

cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=ContinuonBrain Robot startup (health checks + Robot API server)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=CONFIG_DIR=${CONFIG_DIR}
Environment=DISPLAY=:0
Environment=XAUTHORITY=${HOME}/.Xauthority
WorkingDirectory=${REPO_DIR}
ExecStart=${VENV}/bin/python3 -m continuonbrain.startup_manager --config-dir \${CONFIG_DIR}
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

echo "✅ Service file created"
echo ""

# Enable systemd user linger (allows services to run without user login)
echo "Enabling systemd user linger..."
sudo loginctl enable-linger "${USER}" || {
  echo "⚠️  Warning: Could not enable linger (may need sudo)" >&2
}

# Reload systemd and enable service
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

echo "Enabling continuonbrain-startup.service..."
systemctl --user enable continuonbrain-startup.service

echo "Starting continuonbrain-startup.service..."
systemctl --user start continuonbrain-startup.service || {
  echo "⚠️  Warning: Service start failed. Check logs with:" >&2
  echo "   systemctl --user status continuonbrain-startup.service" >&2
  exit 1
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "Service status:"
systemctl --user status continuonbrain-startup.service --no-pager -l || true
echo ""
echo "Useful commands:"
echo "  systemctl --user status continuonbrain-startup.service"
echo "  systemctl --user logs continuonbrain-startup.service -f"
echo "  systemctl --user restart continuonbrain-startup.service"
echo "  systemctl --user stop continuonbrain-startup.service"

