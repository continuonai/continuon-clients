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
#   CONTINUON_RUN_WAVECORE=1|0

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

# Prefer a repo-local config dir for desktop/dev runs (keeps writes local, no sudo).
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
WAVECORE_PIDFILE="${CONFIG_DIR}/runtime_pids/wavecore_trainer.pid"

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

# Desktop default: use MOCK hardware to avoid requiring system audio/i2c deps.
# Override with CONTINUON_FORCE_REAL_HARDWARE=1 if you really want to probe devices.
export CONTINUON_FORCE_MOCK_HARDWARE="${CONTINUON_FORCE_MOCK_HARDWARE:-1}"
export CONTINUON_FORCE_REAL_HARDWARE="${CONTINUON_FORCE_REAL_HARDWARE:-0}"
export CONTINUON_SKIP_MOTION_HW="${CONTINUON_SKIP_MOTION_HW:-1}"

# Enable UI autolaunch for desktop convenience.
export CONTINUON_HEADLESS="${CONTINUON_HEADLESS:-0}"
export CONTINUON_UI_AUTOLAUNCH="${CONTINUON_UI_AUTOLAUNCH:-1}"

# (kept above; default is 1 for desktop)

# Self-training controls (default ON for desktop convenience; override to 0 to disable)
export CONTINUON_ENABLE_BACKGROUND_TRAINER="${CONTINUON_ENABLE_BACKGROUND_TRAINER:-1}"
export CONTINUON_LOG_CHAT_RLDS="${CONTINUON_LOG_CHAT_RLDS:-1}"
export CONTINUON_APPLY_SELF_TRAINING_SETTINGS="${CONTINUON_APPLY_SELF_TRAINING_SETTINGS:-1}"
export CONTINUON_START_TRAINING_REPORT_DAEMON="${CONTINUON_START_TRAINING_REPORT_DAEMON:-1}"
export CONTINUON_RUN_WAVECORE="${CONTINUON_RUN_WAVECORE:-1}"

# Prefer non-mock LLM backends on desktop (offline-first; uses local HF cache).
# - LiteRT is disabled because it can silently fall back to a mock implementation without mediapipe-genai.
# - Transformers chat is allowed even when headless, so we can avoid mock text.
export CONTINUON_USE_LITERT="${CONTINUON_USE_LITERT:-0}"
export CONTINUON_ALLOW_TRANSFORMERS_CHAT="${CONTINUON_ALLOW_TRANSFORMERS_CHAT:-1}"
export CONTINUON_PREFER_JAX="${CONTINUON_PREFER_JAX:-0}"
export CONTINUON_ENABLE_JAX_GEMMA_CHAT="${CONTINUON_ENABLE_JAX_GEMMA_CHAT:-0}"
export CONTINUON_SUBAGENT_REQUIRE_NON_MOCK="${CONTINUON_SUBAGENT_REQUIRE_NON_MOCK:-1}"
# Desktop convenience: allow downloading missing HF models (for non-mock subagents).
export CONTINUON_ALLOW_MODEL_DOWNLOADS="${CONTINUON_ALLOW_MODEL_DOWNLOADS:-1}"

# Ensure HOPE's background learning + chat-learn + autonomy orchestrator threads run.
# (Runtime also reads settings.json; this just prevents env-based disable.)
export CONTINUON_START_TRAINING_REPORT_DAEMON="${CONTINUON_START_TRAINING_REPORT_DAEMON:-1}"
export CONTINUON_DISABLE_CHAT_LEARN="0"
export CONTINUON_DISABLE_AUTONOMOUS_LEARNER="0"
export CONTINUON_DISABLE_AUTONOMY_ORCHESTRATOR="0"

# (Legacy note) We used to hard-force these to 0. Now we default them to 0 above but allow overrides.

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

wavecore_running() {
  if [[ -f "${WAVECORE_PIDFILE}" ]]; then
    local pid
    pid="$(cat "${WAVECORE_PIDFILE}" 2>/dev/null || true)"
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

  # Start Wavecore background trainer (seed model loops)
  if [[ "${CONTINUON_RUN_WAVECORE}" == "1" ]]; then
    if wavecore_running; then
      echo "Wavecore trainer already running (pid=$(cat "${WAVECORE_PIDFILE}"))"
    else
      echo "Starting Wavecore trainer..."
      # Use same logic/command as continuonbrain-wavecore.service
      # We just run it in background with nohup/subshell.
      # Note: This is an infinite(ish) loop or long running process depending on implementation, 
      # but systemd service says 'oneshot' running '_run_sync' which implies it runs ONCE then exits? 
      # Checking service file: Type=oneshot. 
      # So it runs one set of loops then finishes. 
      # If we want it to run periodically, we'd need a loop or timer. 
      # However, for desktop startup, running it once on boot is the requested parity.
      
      WAVECORE_CMD="from continuonbrain.services.wavecore_trainer import WavecoreTrainer; t=WavecoreTrainer(); t._run_sync({'fast':{}, 'mid':{}, 'slow':{}, 'compact_export': True})"
      
      if needs_sudo; then
        (
          $(sudo_env_prefix) "${PY}" -c "${WAVECORE_CMD}"
        ) >> "${LOG_FILE}" 2>&1 &
      else
        $(env_prefix) "${PY}" -c "${WAVECORE_CMD}" >> "${LOG_FILE}" 2>&1 &
      fi
      local wpid=$!
      echo "${wpid}" > "${WAVECORE_PIDFILE}"
      echo "Wavecore trainer started (pid=${wpid})."
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

  if wavecore_running; then
    local wpid
    wpid="$(cat "${WAVECORE_PIDFILE}")"
    echo "Stopping Wavecore trainer pid=${wpid} ..."
    if needs_sudo; then
      sudo kill "${wpid}" 2>/dev/null || true
    else
      kill "${wpid}" 2>/dev/null || true
    fi
    rm -f "${WAVECORE_PIDFILE}" || true
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
  if wavecore_running; then
    echo "WAVECORE_TRAINER=RUNNING pid=$(cat "${WAVECORE_PIDFILE}")"
  else
    echo "WAVECORE_TRAINER=STOPPED"
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

