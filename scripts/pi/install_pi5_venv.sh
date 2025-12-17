#!/usr/bin/env bash
set -euo pipefail

# Pi 5 installer (repo-local venv + runtime dirs + minimal deps).
#
# Usage:
#   ./scripts/pi/install_pi5_venv.sh
#   ./scripts/pi/install_pi5_venv.sh --repo /home/pi/ContinuonXR --with-sam3 --with-opencv
#
# Notes:
# - Safe-by-default: does NOT install heavy ML deps unless requested.
# - Installs DepthAI from Luxonis extra index (recommended by Luxonis).
# - Attempts to install JAX CPU wheels (needed for WaveCore loops). If that fails,
#   the script prints next steps but still completes the base setup.

REPO_DIR=""
WITH_SAM3=0
WITH_OPENCV=0
WITH_JAX=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_DIR="${2:-}"; shift 2 ;;
    --with-sam3) WITH_SAM3=1; shift ;;
    --with-opencv) WITH_OPENCV=1; shift ;;
    --no-jax) WITH_JAX=0; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${REPO_DIR}" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

echo "== Continuon Pi5 install =="
echo "repo: ${REPO_DIR}"

if [[ ! -d "${REPO_DIR}/continuonbrain" ]]; then
  echo "ERROR: repo dir does not look like ContinuonXR (missing continuonbrain/)" >&2
  exit 1
fi

echo "== apt deps (sudo) =="
sudo apt-get update -y
sudo apt-get install -y \
  python3 python3-venv python3-pip python3-dev \
  git ca-certificates curl \
  build-essential pkg-config \
  swig liblgpio-dev \
  i2c-tools v4l-utils ffmpeg \
  libusb-1.0-0

echo "== runtime dirs (/opt) =="
sudo mkdir -p /opt/continuonos/brain/{model/base_model,model/adapters/{current,candidate},rlds/episodes,trainer/{logs,checkpoints/core_model_seed}}
sudo chown -R "${USER}:${USER}" /opt/continuonos/brain || true

echo "== venv (.venv) =="
cd "${REPO_DIR}"
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "== python deps (base, light) =="
python -m pip install \
  numpy psutil requests python-dotenv \
  smbus2 \
  adafruit-circuitpython-servokit \
  adafruit-circuitpython-pca9685 \
  adafruit-circuitpython-motor \
  lgpio

echo "== depthai (Luxonis index) =="
python -m pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-local/ depthai || {
  echo "WARN: depthai install failed. You may need a different OS build or updated pip. Continuing." >&2
}

if [[ "${WITH_OPENCV}" == "1" ]]; then
  echo "== opencv (headless) =="
  python -m pip install opencv-python-headless || {
    echo "WARN: opencv install failed; recorder can still write .npy frames. Continuing." >&2
  }
fi

if [[ "${WITH_JAX}" == "1" ]]; then
  echo "== jax (cpu) =="
  # Best-effort: CPU-only wheels. If unavailable for your OS, you'll need a platform-specific guide.
  python -m pip install "jax[cpu]" || {
    echo "WARN: jax[cpu] install failed. WaveCore loops won't run until JAX is installed." >&2
  }
fi

if [[ "${WITH_SAM3}" == "1" ]]; then
  echo "== SAM3 deps (heavy) =="
  echo "NOTE: SAM3 may be too heavy to run fast on Pi5. Treat it as offline enrichment."
  python -m pip install "torch>=2.2.0" "transformers>=4.40.0" "accelerate>=0.29.0" pillow || {
    echo "WARN: torch/transformers install failed on this platform. Continuing." >&2
  }
fi

echo
echo "✅ Done."
echo "- venv: ${REPO_DIR}/.venv"
echo "- episodes: /opt/continuonos/brain/rlds/episodes"
echo "- episodes: /opt/continuonos/brain/rlds/episodes"
echo "- next: run ./scripts/pi/install_pi5_systemd.sh to enable boot startup"

# Ensure startup scripts are executable
chmod +x "${REPO_DIR}/scripts/start_rpi.sh" "${REPO_DIR}/scripts/start_desktop_services.sh"


