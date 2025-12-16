#!/usr/bin/env bash
set -euo pipefail

# Periodically run the existing training report script and append to a log.
#
# Default output log:
#   /opt/continuonos/brain/logs/training_status_reports.log
#
# Controls:
#   CONFIG_DIR=/opt/continuonos/brain
#   TRAINING_REPORT_INTERVAL_S=1800

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "ERROR: python3 not found" >&2
  exit 1
fi

DEFAULT_CONFIG_DIR="/opt/continuonos/brain"
if [[ -d "${DEFAULT_CONFIG_DIR}" ]]; then
  CONFIG_DIR="${CONFIG_DIR:-${DEFAULT_CONFIG_DIR}}"
else
  CONFIG_DIR="${CONFIG_DIR:-${HOME}/.continuonbrain}"
fi

INTERVAL_S="${TRAINING_REPORT_INTERVAL_S:-1800}"
LOG_PATH="${TRAINING_REPORT_LOG_PATH:-${CONFIG_DIR}/logs/training_status_reports.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export CONFIG_DIR="${CONFIG_DIR}"

runner=()
if [[ ! -w "$(dirname "${LOG_PATH}")" ]]; then
  runner=(sudo -E)
fi

echo "Training report daemon:"
echo "  CONFIG_DIR=${CONFIG_DIR}"
echo "  interval_s=${INTERVAL_S}"
echo "  log=${LOG_PATH}"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo
    echo "===== training_status_report ${ts} ====="
    "${runner[@]}" "${PY}" -u "${REPO_ROOT}/continuonbrain/scripts/training_status_report.py" || true
  } >> "${LOG_PATH}" 2>&1
  sleep "${INTERVAL_S}"
done

