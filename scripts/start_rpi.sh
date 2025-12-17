#!/usr/bin/env bash
set -euo pipefail

# Continuon Brain runtime "RPi" launcher.
#
# Starts the same orchestration path as real device boot:
#   python -m continuonbrain.startup_manager
#
# Default behavior matches the "real hardware" request:
# - real-hardware ON (CONTINUON_FORCE_REAL_HARDWARE=1)
# - skip motion HW OFF (CONTINUON_SKIP_MOTION_HW=0)
#
# Usage:
#   ./scripts/start_rpi.sh start
#   ./scripts/start_rpi.sh stop
#   ./scripts/start_rpi.sh restart
#   ./scripts/start_rpi.sh status
#   ./scripts/start_rpi.sh logs
#
# Optional env overrides:
#   CONFIG_DIR=/opt/continuonos/brain
#   CONTINUON_FORCE_REAL_HARDWARE=1|0
#   CONTINUON_FORCE_MOCK_HARDWARE=1|0
#   CONTINUON_SKIP_MOTION_HW=1|0
#   CONTINUON_HEADLESS=1|0
#   CONTINUON_UI_AUTOLAUNCH=1|0
#   CONTINUON_ENABLE_BACKGROUND_TRAINER=1|0
#   CONTINUON_LOG_CHAT_RLDS=1|0

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

# Prefer a repo-local config dir for dev runs, but fallback to system paths.
DEFAULT_CONFIG_DIR="${REPO_ROOT}/brain_config"
if [[ -d "${DEFAULT_CONFIG_DIR}" ]]; then
  CONFIG_DIR="${CONFIG_DIR:-${DEFAULT_CONFIG_DIR}}"
elif [[ -d "/opt/continuonbrain/config" ]]; then
  CONFIG_DIR="${CONFIG_DIR:-/opt/continuonbrain/config}"
elif [[ -d "/opt/continuonos/brain" ]]; then
  CONFIG_DIR="${CONFIG_DIR:-/opt/continuonos/brain}"
else
  CONFIG_DIR="${CONFIG_DIR:-${HOME}/.continuonbrain}"
fi

LOG_DIR="${CONFIG_DIR}/logs"
LOG_FILE="${LOG_DIR}/startup_manager_console.log"

PIDFILE="${CONFIG_DIR}/runtime_pids/startup_manager.pid"
REPORT_PIDFILE="${CONFIG_DIR}/runtime_pids/training_report_daemon.pid"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PIDFILE}")"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export CONFIG_DIR="${CONFIG_DIR}"

# RPi default: use REAL hardware.
export CONTINUON_FORCE_MOCK_HARDWARE="${CONTINUON_FORCE_MOCK_HARDWARE:-0}"
export CONTINUON_FORCE_REAL_HARDWARE="${CONTINUON_FORCE_REAL_HARDWARE:-1}"
export CONTINUON_SKIP_MOTION_HW="${CONTINUON_SKIP_MOTION_HW:-0}"

# RPi is typically headless/kiosk.
export CONTINUON_HEADLESS="${CONTINUON_HEADLESS:-1}"
export CONTINUON_UI_AUTOLAUNCH="${CONTINUON_UI_AUTOLAUNCH:-0}"

# Self-training controls
export CONTINUON_ENABLE_BACKGROUND_TRAINER="${CONTINUON_ENABLE_BACKGROUND_TRAINER:-1}"
export CONTINUON_LOG_CHAT_RLDS="${CONTINUON_LOG_CHAT_RLDS:-1}"
export CONTINUON_APPLY_SELF_TRAINING_SETTINGS="${CONTINUON_APPLY_SELF_TRAINING_SETTINGS:-1}"
export CONTINUON_START_TRAINING_REPORT_DAEMON="${CONTINUON_START_TRAINING_REPORT_DAEMON:-1}"

# Use LiteRT on RPi for better performance.
export CONTINUON_USE_LITERT="${CONTINUON_USE_LITERT:-1}"
export CONTINUON_ALLOW_TRANSFORMERS_CHAT="${CONTINUON_ALLOW_TRANSFORMERS_CHAT:-0}"
export CONTINUON_PREFER_JAX="${CONTINUON_PREFER_JAX:-0}"
export CONTINUON_ENABLE_JAX_GEMMA_CHAT="${CONTINUON_ENABLE_JAX_GEMMA_CHAT:-0}"
export CONTINUON_SUBAGENT_REQUIRE_NON_MOCK="${CONTINUON_SUBAGENT_REQUIRE_NON_MOCK:-1}"
# Prevent heavy model downloads if not needed
export CONTINUON_ALLOW_MODEL_DOWNLOADS="${CONTINUON_ALLOW_MODEL_DOWNLOADS:-1}"

# Ensure modules are not disabled
export CONTINUON_DISABLE_CHAT_LEARN="0"
export CONTINUON_DISABLE_AUTONOMOUS_LEARNER="0"
export CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR="0"

needs_sudo() {
  [[ ! -w "${CONFIG_DIR}" ]]
}

sudo_env_prefix() {
  cat <<EOF
sudo env \
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
CONTINUON_PREFER_JAX=${CONTINUON_PREFER_JAX} \
CONTINUON_ENABLE_JAX_GEMMA_CHAT=${CONTINUON_ENABLE_JAX_GEMMA_CHAT} \
CONTINUON_SUBAGENT_REQUIRE_NON_MOCK=${CONTINUON_SUBAGENT_REQUIRE_NON_MOCK} \
CONTINUON_ALLOW_MODEL_DOWNLOADS=${CONTINUON_ALLOW_MODEL_DOWNLOADS} \
CONTINUON_DISABLE_CHAT_LEARN=${CONTINUON_DISABLE_CHAT_LEARN} \
CONTINUON_DISABLE_AUTONOMOUS_LEARNER=${CONTINUON_DISABLE_AUTONOMOUS_LEARNER} \
CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR=${CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR}
EOF
}

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
CONTINUON_PREFER_JAX=${CONTINUON_PREFER_JAX} \
CONTINUON_ENABLE_JAX_GEMMA_CHAT=${CONTINUON_ENABLE_JAX_GEMMA_CHAT} \
CONTINUON_SUBAGENT_REQUIRE_NON_MOCK=${CONTINUON_SUBAGENT_REQUIRE_NON_MOCK} \
CONTINUON_ALLOW_MODEL_DOWNLOADS=${CONTINUON_ALLOW_MODEL_DOWNLOADS} \
CONTINUON_DISABLE_CHAT_LEARN=${CONTINUON_DISABLE_CHAT_LEARN} \
CONTINUON_DISABLE_AUTONOMOUS_LEARNER=${CONTINUON_DISABLE_AUTONOMOUS_LEARNER} \
CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR=${CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR}
EOF
}

is_running() {
  if [[ -f "${PIDFILE}" ]]; then
    local pid
    pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

report_running() {
  if [[ -f "${REPORT_PIDFILE}" ]]; then
    local pid
    pid="$(cat "${REPORT_PIDFILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

kill_stray_processes() {
  local pids
  pids="$(pgrep -f 'continuonbrain.startup_manager' || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping stray startup_manager PID(s): ${pids}"
    if needs_sudo; then sudo kill ${pids} 2>/dev/null || true; else kill ${pids} 2>/dev/null || true; fi
  fi
  pids="$(pgrep -f 'continuonbrain.robot_api_server' || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping stray robot_api_server PID(s): ${pids}"
    if needs_sudo; then sudo kill ${pids} 2>/dev/null || true; else kill ${pids} 2>/dev/null || true; fi
  fi
}

cmd_start() {
  if is_running; then
    echo "Already running (pid=$(cat "${PIDFILE}"))"
    exit 0
  fi

  kill_stray_processes

  if [[ "${CONTINUON_APPLY_SELF_TRAINING_SETTINGS}" == "1" ]]; then
    "${REPO_ROOT}/scripts/enable_self_training.sh" || true
  fi

  echo "Repo: ${REPO_ROOT}"
  echo "Python: ${PY}"
  echo "CONFIG_DIR: ${CONFIG_DIR}"
  echo "Log: ${LOG_FILE}"

  if needs_sudo; then
    (
      $(sudo_env_prefix) "${PY}" -m continuonbrain.startup_manager --config-dir "${CONFIG_DIR}" --port 8081 \
        >> "${LOG_FILE}" 2>&1
    ) &
  else
    (
      $(env_prefix) "${PY}" -m continuonbrain.startup_manager --config-dir "${CONFIG_DIR}" --port 8081 \
        >> "${LOG_FILE}" 2>&1
    ) &
  fi

  local pid=$!
  echo "${pid}" > "${PIDFILE}"
  sleep 1

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Started (pid=${pid})."
    echo "UI: http://127.0.0.1:8080/ui"
    echo "Status: http://127.0.0.1:8080/status"
  else
    echo "Failed to start (pid exited). See logs: ${LOG_FILE}" >&2
    exit 1
  fi

  if [[ "${CONTINUON_START_TRAINING_REPORT_DAEMON}" == "1" ]]; then
    if ! report_running; then
       if needs_sudo; then
         ($(sudo_env_prefix) "${REPO_ROOT}/scripts/training_report_daemon.sh") >> "${LOG_FILE}" 2>&1 &
       else
         ($(env_prefix) "${REPO_ROOT}/scripts/training_report_daemon.sh") >> "${LOG_FILE}" 2>&1 &
       fi
       echo $! > "${REPORT_PIDFILE}"
       echo "Training report daemon started."
    fi
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "Not running."
    rm -f "${PIDFILE}" || true
    return 0
  fi
  local pid
  pid="$(cat "${PIDFILE}")"
  echo "Stopping pid=${pid} ..."
  if needs_sudo; then sudo kill "${pid}" 2>/dev/null || true; else kill "${pid}" 2>/dev/null || true; fi
  sleep 1
  rm -f "${PIDFILE}" || true
  echo "Stopped."

  if report_running; then
    local rpid
    rpid="$(cat "${REPORT_PIDFILE}")"
    echo "Stopping training report daemon pid=${rpid} ..."
    if needs_sudo; then sudo kill "${rpid}" 2>/dev/null || true; else kill "${rpid}" 2>/dev/null || true; fi
    rm -f "${REPORT_PIDFILE}" || true
  fi
  kill_stray_processes
}

cmd_status() {
  if is_running; then
    echo "RUNNING pid=$(cat "${PIDFILE}")"
  else
    echo "STOPPED"
  fi
}

cmd_logs() {
  echo "Tailing ${LOG_FILE} (Ctrl+C to exit)"
  if needs_sudo; then sudo tail -n 200 -f "${LOG_FILE}"; else tail -n 200 -f "${LOG_FILE}"; fi
}

cmd_restart() {
  cmd_stop || true
  cmd_start
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_restart ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
