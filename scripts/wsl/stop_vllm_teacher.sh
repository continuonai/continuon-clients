#!/usr/bin/env bash
set -euo pipefail

PID_DIR="/tmp/vllm_teacher"

for name in chat embed; do
  pid_file="$PID_DIR/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file" || true)"
    if [[ -n "${pid:-}" ]]; then
      echo "Stopping $name pid=$pid"
      kill "$pid" 2>/dev/null || true
    fi
  fi
done

## Also stop any orphaned vLLM api_server processes, but NEVER kill the current shell.
me="$$"
parent="${PPID:-0}"
for pid in $(pgrep -f "vllm.entrypoints.openai.api_server" || true); do
  if [[ "$pid" != "$me" && "$pid" != "$parent" ]]; then
    echo "Stopping orphan vLLM pid=$pid"
    kill "$pid" 2>/dev/null || true
  fi
done

echo "Remaining vLLM processes:"
pgrep -af "vllm.entrypoints.openai.api_server" || true

