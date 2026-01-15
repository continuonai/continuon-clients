#!/bin/bash
# SessionStart Hook - Initialize Brain B state for this session
#
# This hook runs when a Claude Code session starts.
# It initializes Brain B's state and loads any existing guardrails.

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
BRAIN_B_DIR="${PROJECT_DIR}/brain_b_data"
STATE_FILE="${BRAIN_B_DIR}/session_state.json"

# Create Brain B data directory if needed
mkdir -p "${BRAIN_B_DIR}/sessions"
mkdir -p "${BRAIN_B_DIR}/training_buffer"

# Read session info from stdin
SESSION_INFO=$(cat)
SESSION_ID=$(echo "$SESSION_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','unknown'))" 2>/dev/null || echo "unknown")

# Initialize session state
cat > "${STATE_FILE}" << EOF
{
  "session_id": "${SESSION_ID}",
  "started_at": "$(date -Iseconds)",
  "iteration": 0,
  "actions_recorded": 0,
  "guardrails_triggered": 0,
  "status": "active"
}
EOF

# Load guardrails count
GUARDRAILS_FILE="${BRAIN_B_DIR}/guardrails.json"
if [ -f "$GUARDRAILS_FILE" ]; then
  GUARDRAILS_COUNT=$(python3 -c "import json; print(len(json.load(open('${GUARDRAILS_FILE}')).get('guardrails',[])))" 2>/dev/null || echo "0")
else
  GUARDRAILS_COUNT="0"
fi

# Persist environment for other hooks
if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo "export BRAIN_B_SESSION_ID='${SESSION_ID}'" >> "$CLAUDE_ENV_FILE"
  echo "export BRAIN_B_STATE_FILE='${STATE_FILE}'" >> "$CLAUDE_ENV_FILE"
  echo "export BRAIN_B_DATA_DIR='${BRAIN_B_DIR}'" >> "$CLAUDE_ENV_FILE"
fi

# Output context for Claude
cat << EOF

Brain B initialized for session ${SESSION_ID}.
- Data directory: ${BRAIN_B_DIR}
- Guardrails loaded: ${GUARDRAILS_COUNT}
- Training buffer ready

Robot actions will be validated through Brain B before execution.
Session data will be exported as RLDS episodes for Brain A training.

EOF

exit 0
