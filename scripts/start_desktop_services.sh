#!/usr/bin/env bash
set -euo pipefail

# Continuon Brain runtime "desktop" launcher.
#
# Starts the same orchestration path as real device boot:
#   python -m continuonbrain.startup_manager
#
# Default behavior matches the "real hardware, no arms/servos" request:
# - real-hardware ON (CONTINUON_FORCE_REAL_HARDWARE=1)
# - skip motion HW ON (CONTINUON_SKIP_MOTION_HW=1)
#
# Usage:
#   ./scripts/start_desktop_services.sh start
#   ./scripts/start_desktop_services.sh stop
#   ./scripts/start_desktop_services.sh restart
#   ./scripts/start_desktop_services.sh status
#   ./scripts/start_desktop_services.sh logs
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
#   CONTINUON_APPLY_SELF_TRAINING_SETTINGS=1|0
#   CONTINUON_START_TRAINING_REPORT_DAEMON=1|0

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

LOG_DIR="${CONFIG_DIR}/logs"
LOG_FILE="${LOG_DIR}/startup_manager_console.log"

PIDFILE="${CONFIG_DIR}/runtime_pids/startup_manager.pid"
REPORT_PIDFILE="${CONFIG_DIR}/runtime_pids/training_report_daemon.pid"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${PIDFILE}")"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export CONFIG_DIR="${CONFIG_DIR}"

# Some Python builds are compiled with gcov; without these, they can spam stderr
# with "profiling: ... .gcda: Cannot open". Redirect coverage outputs to /tmp.
export GCOV_PREFIX="${GCOV_PREFIX:-/tmp/continuon_gcov}"
export GCOV_PREFIX_STRIP="${GCOV_PREFIX_STRIP:-3}"
mkdir -p "${GCOV_PREFIX}" 2>/dev/null || true
chmod -R a+rwx "${GCOV_PREFIX}" 2>/dev/null || true

# "Real hardware mode, but no arms/servos"
export CONTINUON_FORCE_REAL_HARDWARE="${CONTINUON_FORCE_REAL_HARDWARE:-1}"
export CONTINUON_SKIP_MOTION_HW="${CONTINUON_SKIP_MOTION_HW:-1}"

# Avoid surprise browser pops on desktop unless explicitly enabled.
export CONTINUON_HEADLESS="${CONTINUON_HEADLESS:-1}"
export CONTINUON_UI_AUTOLAUNCH="${CONTINUON_UI_AUTOLAUNCH:-0}"

export CONTINUON_FORCE_MOCK_HARDWARE="${CONTINUON_FORCE_MOCK_HARDWARE:-0}"

# Self-training controls (default ON for desktop convenience; override to 0 to disable)
export CONTINUON_ENABLE_BACKGROUND_TRAINER="${CONTINUON_ENABLE_BACKGROUND_TRAINER:-1}"
export CONTINUON_LOG_CHAT_RLDS="${CONTINUON_LOG_CHAT_RLDS:-1}"
export CONTINUON_APPLY_SELF_TRAINING_SETTINGS="${CONTINUON_APPLY_SELF_TRAINING_SETTINGS:-1}"
export CONTINUON_START_TRAINING_REPORT_DAEMON="${CONTINUON_START_TRAINING_REPORT_DAEMON:-1}"

# For this desktop "self training" launcher, force-enable the schedulers unless
# the operator explicitly edits the script (these env vars may be inherited as 1
# from other shells/services).
export CONTINUON_DISABLE_CHAT_LEARN="0"
export CONTINUON_DISABLE_AUTONOMOUS_LEARNER="0"
export CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR="0"

needs_sudo() {
  # Need sudo if we cannot write to CONFIG_DIR (common for /opt installs).
  [[ ! -w "${CONFIG_DIR}" ]]
}

sudo_env_prefix() {
  # Print a sudo+env prefix that forces critical vars through sudo.
  # We intentionally enumerate CONTINUON_* vars because sudo may drop -E env.
  cat <<EOF
sudo env \
PYTHONPATH=${PYTHONPATH} \
CONFIG_DIR=${CONFIG_DIR} \
GCOV_PREFIX=${GCOV_PREFIX} \
GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} \
CONTINUON_FORCE_REAL_HARDWARE=${CONTINUON_FORCE_REAL_HARDWARE} \
CONTINUON_SKIP_MOTION_HW=${CONTINUON_SKIP_MOTION_HW} \
CONTINUON_HEADLESS=${CONTINUON_HEADLESS} \
CONTINUON_UI_AUTOLAUNCH=${CONTINUON_UI_AUTOLAUNCH} \
CONTINUON_FORCE_MOCK_HARDWARE=${CONTINUON_FORCE_MOCK_HARDWARE} \
CONTINUON_ENABLE_BACKGROUND_TRAINER=${CONTINUON_ENABLE_BACKGROUND_TRAINER} \
CONTINUON_LOG_CHAT_RLDS=${CONTINUON_LOG_CHAT_RLDS} \
CONTINUON_DISABLE_CHAT_LEARN=${CONTINUON_DISABLE_CHAT_LEARN} \
CONTINUON_DISABLE_AUTONOMOUS_LEARNER=${CONTINUON_DISABLE_AUTONOMOUS_LEARNER} \
CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR=${CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR}
EOF
}

env_prefix() {
  # Same as sudo_env_prefix but without sudo.
  cat <<EOF
env \
PYTHONPATH=${PYTHONPATH} \
CONFIG_DIR=${CONFIG_DIR} \
GCOV_PREFIX=${GCOV_PREFIX} \
GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} \
CONTINUON_FORCE_REAL_HARDWARE=${CONTINUON_FORCE_REAL_HARDWARE} \
CONTINUON_SKIP_MOTION_HW=${CONTINUON_SKIP_MOTION_HW} \
CONTINUON_HEADLESS=${CONTINUON_HEADLESS} \
CONTINUON_UI_AUTOLAUNCH=${CONTINUON_UI_AUTOLAUNCH} \
CONTINUON_FORCE_MOCK_HARDWARE=${CONTINUON_FORCE_MOCK_HARDWARE} \
CONTINUON_ENABLE_BACKGROUND_TRAINER=${CONTINUON_ENABLE_BACKGROUND_TRAINER} \
CONTINUON_LOG_CHAT_RLDS=${CONTINUON_LOG_CHAT_RLDS} \
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
  # Best-effort cleanup for stale services (common when ports are left bound).
  # We intentionally target module names to avoid killing unrelated python.
  local pids
  pids="$(pgrep -f 'continuonbrain.startup_manager' || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping stray startup_manager PID(s): ${pids}"
    sudo kill ${pids} 2>/dev/null || true
  fi
  pids="$(pgrep -f 'continuonbrain.robot_api_server' || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping stray robot_api_server PID(s): ${pids}"
    sudo kill ${pids} 2>/dev/null || true
  fi
}

cmd_start() {
  if is_running; then
    echo "Already running (pid=$(cat "${PIDFILE}"))"
    exit 0
  fi

  # Ensure we own port 8080 and don't accidentally spawn duplicates.
  kill_stray_processes

  if [[ "${CONTINUON_APPLY_SELF_TRAINING_SETTINGS}" == "1" ]]; then
    "${REPO_ROOT}/scripts/enable_self_training.sh" || true
  fi

echo "Repo: ${REPO_ROOT}"
echo "Python: ${PY}"
echo "CONFIG_DIR: ${CONFIG_DIR}"
echo "Log: ${LOG_FILE}"
echo "Flags: FORCE_REAL=${CONTINUON_FORCE_REAL_HARDWARE} SKIP_MOTION=${CONTINUON_SKIP_MOTION_HW} HEADLESS=${CONTINUON_HEADLESS} UI_AUTOLAUNCH=${CONTINUON_UI_AUTOLAUNCH} FORCE_MOCK=${CONTINUON_FORCE_MOCK_HARDWARE}"
echo "Training: BACKGROUND_TRAINER=${CONTINUON_ENABLE_BACKGROUND_TRAINER} LOG_CHAT_RLDS=${CONTINUON_LOG_CHAT_RLDS} CHAT_LEARN_DISABLED=${CONTINUON_DISABLE_CHAT_LEARN} AUTO_LEARN_DISABLED=${CONTINUON_DISABLE_AUTONOMOUS_LEARNER} ORCH_DISABLED=${CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR}"

  # Start in background; write pidfile.
  # Note: startup_manager will spawn robot_api_server and exit non-zero if the child dies.
  if needs_sudo; then
    # Force env vars through sudo (sudo often drops -E env).
    (
      $(sudo_env_prefix) "${PY}" -m continuonbrain.startup_manager --config-dir "${CONFIG_DIR}" \
        >> "${LOG_FILE}" 2>&1
    ) &
  else
    (
      $(env_prefix) "${PY}" -m continuonbrain.startup_manager --config-dir "${CONFIG_DIR}" \
        >> "${LOG_FILE}" 2>&1
    ) &
  fi

  local pid=$!
  echo "${pid}" > "${PIDFILE}"

  # Give it a moment to bind ports.
  sleep 1

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Started (pid=${pid})."
    echo "UI: http://127.0.0.1:8080/ui"
    echo "Status: http://127.0.0.1:8080/status"
  else
    echo "Failed to start (pid exited). See logs: ${LOG_FILE}" >&2
    exit 1
  fi

  # Optional: run training report daemon (existing report script) in background.
  if [[ "${CONTINUON_START_TRAINING_REPORT_DAEMON}" == "1" ]]; then
    if report_running; then
      echo "Training report daemon already running (pid=$(cat "${REPORT_PIDFILE}"))"
    else
      if needs_sudo; then
        (
          $(sudo_env_prefix) "${REPO_ROOT}/scripts/training_report_daemon.sh"
        ) >> "${LOG_FILE}" 2>&1 &
      else
        $(env_prefix) "${REPO_ROOT}/scripts/training_report_daemon.sh" >> "${LOG_FILE}" 2>&1 &
      fi
      echo $! > "${REPORT_PIDFILE}"
      echo "Training report daemon started (pid=$(cat "${REPORT_PIDFILE}"))."
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
  if needs_sudo; then
    sudo kill "${pid}" 2>/dev/null || true
  else
    kill "${pid}" 2>/dev/null || true
  fi
  sleep 1
  rm -f "${PIDFILE}" || true
  echo "Stopped."

  if report_running; then
    local rpid
    rpid="$(cat "${REPORT_PIDFILE}")"
    echo "Stopping training report daemon pid=${rpid} ..."
    if needs_sudo; then
      sudo kill "${rpid}" 2>/dev/null || true
    else
      kill "${rpid}" 2>/dev/null || true
    fi
    rm -f "${REPORT_PIDFILE}" || true
  fi

  # Also stop any strays not tracked by pidfiles.
  kill_stray_processes
}

cmd_status() {
  if is_running; then
    local pid
    pid="$(cat "${PIDFILE}")"
    echo "RUNNING pid=${pid}"
  else
    echo "STOPPED"
  fi
  echo "CONFIG_DIR=${CONFIG_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  if report_running; then
    echo "TRAINING_REPORT_DAEMON=RUNNING pid=$(cat "${REPORT_PIDFILE}")"
  else
    echo "TRAINING_REPORT_DAEMON=STOPPED"
  fi
}

cmd_logs() {
  echo "Tailing ${LOG_FILE} (Ctrl+C to exit)"
  if needs_sudo; then
    sudo tail -n 200 -f "${LOG_FILE}"
  else
    tail -n 200 -f "${LOG_FILE}"
  fi
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

