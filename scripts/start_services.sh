#!/usr/bin/env bash
set -euo pipefail

# Continuon Brain unified launcher.
# Supports both "desktop" (dev) and "rpi" (device) modes.
#
# Usage:
#   ./scripts/start_services.sh start [--mode desktop|rpi]
#   ./scripts/start_services.sh stop
#   ./scripts/start_services.sh restart [--mode desktop|rpi]
#   ./scripts/start_services.sh status
#   ./scripts/start_services.sh logs

MODE="desktop"
COMMAND=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|restart|status|logs) COMMAND="$1"; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${COMMAND}" ]]; then
  echo "Usage: $0 {start|stop|restart|status|logs} [--mode desktop|rpi]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || true)"
fi

# Configuration directory detection
DEFAULT_CONFIG_DIR="${REPO_ROOT}/brain_config"
if [[ -d "${DEFAULT_CONFIG_DIR}" ]]; then
  CONFIG_DIR="${CONFIG_DIR:-${DEFAULT_CONFIG_DIR}}"
elif [[ -d "/opt/continuonos/brain" ]]; then
  CONFIG_DIR="${CONFIG_DIR:-/opt/continuonos/brain}"
else
  CONFIG_DIR="${CONFIG_DIR:-${HOME}/.continuonbrain}"
fi

LOG_FILE="${CONFIG_DIR}/logs/startup_manager_console.log"
PIDFILE="${CONFIG_DIR}/runtime_pids/startup_manager.pid"
REPORT_PIDFILE="${CONFIG_DIR}/runtime_pids/training_report_daemon.pid"
WAVECORE_PIDFILE="${CONFIG_DIR}/runtime_pids/wavecore_trainer.pid"

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "$(dirname "${PIDFILE}")"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CONFIG_DIR="${CONFIG_DIR}"

# Mode-based defaults
if [[ "${MODE}" == "rpi" ]]; then
  export CONTINUON_FORCE_MOCK_HARDWARE="${CONTINUON_FORCE_MOCK_HARDWARE:-0}"
  export CONTINUON_FORCE_REAL_HARDWARE="${CONTINUON_FORCE_REAL_HARDWARE:-1}"
  export CONTINUON_SKIP_MOTION_HW="${CONTINUON_SKIP_MOTION_HW:-0}"
  export CONTINUON_HEADLESS="${CONTINUON_HEADLESS:-1}"
  export CONTINUON_UI_AUTOLAUNCH="${CONTINUON_UI_AUTOLAUNCH:-0}"
  export CONTINUON_ENABLE_BACKGROUND_TRAINER="${CONTINUON_ENABLE_BACKGROUND_TRAINER:-1}"
  export CONTINUON_LOG_CHAT_RLDS="${CONTINUON_LOG_CHAT_RLDS:-1}"
  export CONTINUON_USE_LITERT="${CONTINUON_USE_LITERT:-1}"
  export CONTINUON_ALLOW_TRANSFORMERS_CHAT="${CONTINUON_ALLOW_TRANSFORMERS_CHAT:-0}"
  export CONTINUON_RUN_WAVECORE="${CONTINUON_RUN_WAVECORE:-0}"
else
  export CONTINUON_FORCE_MOCK_HARDWARE="${CONTINUON_FORCE_MOCK_HARDWARE:-1}"
  export CONTINUON_FORCE_REAL_HARDWARE="${CONTINUON_FORCE_REAL_HARDWARE:-0}"
  export CONTINUON_SKIP_MOTION_HW="${CONTINUON_SKIP_MOTION_HW:-1}"
  export CONTINUON_HEADLESS="${CONTINUON_HEADLESS:-0}"
  export CONTINUON_UI_AUTOLAUNCH="${CONTINUON_UI_AUTOLAUNCH:-1}"
  export CONTINUON_ENABLE_BACKGROUND_TRAINER="${CONTINUON_ENABLE_BACKGROUND_TRAINER:-0}"
  export CONTINUON_LOG_CHAT_RLDS="${CONTINUON_LOG_CHAT_RLDS:-0}"
  export CONTINUON_USE_LITERT="${CONTINUON_USE_LITERT:-0}"
  export CONTINUON_ALLOW_TRANSFORMERS_CHAT="${CONTINUON_ALLOW_TRANSFORMERS_CHAT:-1}"
  export CONTINUON_RUN_WAVECORE="${CONTINUON_RUN_WAVECORE:-1}"
fi

export CONTINUON_START_TRAINING_REPORT_DAEMON="${CONTINUON_START_TRAINING_REPORT_DAEMON:-1}"

env_prefix() {
  cat <<EOF
env \
PYTHONPATH=${PYTHONPATH} \
CONFIG_DIR=${CONFIG_DIR} \
CONTINUON_FORCE_REAL_HARDWARE=${CONTINUON_FORCE_REAL_HARDWARE} \
CONTINUON_SKIP_MOTION_HW=${CONTINUON_SKIP_MOTION_HW} \
CONTINUON_HEADLESS=${CONTINUON_HEADLESS} \
CONTINUON_UI_AUTOLAUNCH=${CONTINUON_UI_AUTOLAUNCH} \
CONTINUON_FORCE_MOCK_HARDWARE=${CONTINUON_FORCE_MOCK_HARDWARE} \
CONTINUON_ENABLE_BACKGROUND_TRAINER=${CONTINUON_ENABLE_BACKGROUND_TRAINER} \
CONTINUON_LOG_CHAT_RLDS=${CONTINUON_LOG_CHAT_RLDS} \
CONTINUON_USE_LITERT=${CONTINUON_USE_LITERT} \
CONTINUON_ALLOW_TRANSFORMERS_CHAT=${CONTINUON_ALLOW_TRANSFORMERS_CHAT} \
CONTINUON_RUN_WAVECORE=${CONTINUON_RUN_WAVECORE} \
CONTINUON_CHAT_DEVICE=cpu
EOF
}

is_running() {
  [[ -f "${PIDFILE}" ]] && kill -0 $(cat "${PIDFILE}") 2>/dev/null
}

kill_stray_processes() {
  local pids
  pids="$(pgrep -f 'continuonbrain.startup_manager' || true)"
  [[ -n "${pids}" ]] && kill ${pids} 2>/dev/null || true
  pids="$(pgrep -f 'continuonbrain.api.server' || true)"
  [[ -n "${pids}" ]] && kill ${pids} 2>/dev/null || true
}

cmd_start() {
  if is_running; then
    echo "Already running (pid=$(cat "${PIDFILE}"))"
    return 0
  fi
  kill_stray_processes

  echo "Starting Continuon services (mode=${MODE})..."
  (
    exec $(env_prefix) "${PY}" -m continuonbrain.startup_manager --config-dir "${CONFIG_DIR}" --port 8081 \
      >> "${LOG_FILE}" 2>&1
  ) &
  echo $! > "${PIDFILE}"
  sleep 1

  if [[ "${CONTINUON_START_TRAINING_REPORT_DAEMON}" == "1" ]]; then
    ( $(env_prefix) "${REPO_ROOT}/scripts/training_report_daemon.sh" ) >> "${LOG_FILE}" 2>&1 &
    echo $! > "${REPORT_PIDFILE}"
  fi

  if [[ "${CONTINUON_RUN_WAVECORE}" == "1" ]]; then
    WAVECORE_CMD="from continuonbrain.services.wavecore_trainer import WavecoreTrainer; t=WavecoreTrainer(); t._run_sync({\"fast\":{}, \"mid\":{}, \"slow\":{}, \"compact_export\": True})"
    ( $(env_prefix) "${PY}" -c "${WAVECORE_CMD}" ) >> "${LOG_FILE}" 2>&1 &
    echo $! > "${WAVECORE_PIDFILE}"
  fi
  echo "Started."
}

cmd_stop() {
  if is_running; then
    local pid=$(cat "${PIDFILE}")
    kill "${pid}" 2>/dev/null || true
    rm -f "${PIDFILE}"
  fi
  for pf in "${REPORT_PIDFILE}" "${WAVECORE_PIDFILE}"; do
    if [[ -f "${pf}" ]]; then
      kill $(cat "${pf}") 2>/dev/null || true
      rm -f "${pf}"
    fi
  done
  kill_stray_processes
  echo "Stopped."
}

case "${COMMAND}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; sleep 1; cmd_start ;;
  status) is_running && echo "RUNNING" || echo "STOPPED" ;;
  logs) tail -n 200 -f "${LOG_FILE}" ;;
esac
