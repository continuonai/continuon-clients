#!/usr/bin/env bash
set -euo pipefail

CHAT_PORT="${CHAT_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"
TIMEOUT_S="${TIMEOUT_S:-180}"

deadline=$(( $(date +%s) + TIMEOUT_S ))

wait_one () {
  local port="$1"
  local name="$2"
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/tmp/vllm_teacher/models_${name}.json 2>/dev/null; then
      echo "${name}_ready port=${port}"
      return 0
    fi
    if [[ $(date +%s) -ge $deadline ]]; then
      echo "timeout waiting for ${name} port=${port}"
      return 1
    fi
    sleep 2
  done
}

mkdir -p /tmp/vllm_teacher
wait_one "$CHAT_PORT" "chat"
wait_one "$EMBED_PORT" "embed"

echo "---chat models---"
head -c 240 /tmp/vllm_teacher/models_chat.json || true
echo
echo "---embed models---"
head -c 240 /tmp/vllm_teacher/models_embed.json || true
echo

