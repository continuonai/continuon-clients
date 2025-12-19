#!/usr/bin/env bash
set -euo pipefail

# Enable self-training across Agent Manager / subagents using existing runtime scripts.
#
# What this does:
# - Updates CONFIG_DIR/settings.json (via SettingsStore) to:
#   - chat.log_rlds = true (explicit opt-in for chat→RLDS)
#   - training.enable_sidecar_trainer = true
#   - agent_manager.chat_learn.enabled = true (scheduled)
#   - agent_manager.chat_learn.modes = ["idle","autonomous"]
#   - agent_manager.chat_learn.delegate_model_hint = "google/gemma-370m" (lightweight subagent)
#   - agent_manager.autonomy_orchestrator.enabled = true
#
# It does NOT upload anything to cloud; it stays local/offline-first.

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

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export CONFIG_DIR="${CONFIG_DIR}"

echo "Enabling self-training settings in CONFIG_DIR=${CONFIG_DIR}"

runner=()
if [[ ! -w "${CONFIG_DIR}" ]]; then
  runner=(sudo -E)
fi

"${runner[@]}" "${PY}" - <<'PY'
from pathlib import Path
from continuonbrain.settings_manager import SettingsStore

cfg_dir = Path(__import__("os").environ.get("CONFIG_DIR") or "/opt/continuonos/brain")
store = SettingsStore(cfg_dir)
settings = store.load()

settings.setdefault("chat", {})["log_rlds"] = True
settings.setdefault("training", {})["enable_sidecar_trainer"] = True

agent_mgr = settings.setdefault("agent_manager", {})
# Ensure autonomous/background learning remains enabled.
agent_mgr["enable_autonomous_learning"] = bool(agent_mgr.get("enable_autonomous_learning", True))
chat_learn = agent_mgr.setdefault("chat_learn", {})
chat_learn["enabled"] = True
chat_learn["modes"] = ["idle", "autonomous"]
# Force desired schedule defaults (overwrite any old values).
chat_learn["interval_s"] = 300
chat_learn["turns_per_cycle"] = 12
# Main agent is HOPE v1. Subagent teaching is provided via consult:<model>.
chat_learn["model_hint"] = "hope-v1"
# Prefer a subagent model that matches the default local-cache model id (offline-first).
# Prefer a smaller, non-gated subagent model so we avoid mock outputs.
# If not cached, it can be downloaded when CONTINUON_ALLOW_MODEL_DOWNLOADS=1.
chat_learn["delegate_model_hint"] = "consult:google/gemma-3-270m-it"
chat_learn["topic"] = "curiosity-driven HOPE v1 self-improvement: CMS, safety, planning, tool-use, and robot skills"

orch = agent_mgr.setdefault("autonomy_orchestrator", {})
orch["enabled"] = True
orch["modes"] = ["autonomous"]

validated = store.save(settings)
print("Wrote", store.path)
print("chat.log_rlds =", validated["chat"]["log_rlds"])
print("training.enable_sidecar_trainer =", validated["training"]["enable_sidecar_trainer"])
print("agent_manager.chat_learn =", validated["agent_manager"]["chat_learn"])
print("agent_manager.autonomy_orchestrator.enabled =", validated["agent_manager"]["autonomy_orchestrator"]["enabled"])
PY

echo "Done."

