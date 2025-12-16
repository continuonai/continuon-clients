#!/usr/bin/env bash
set -euo pipefail

# Start two vLLM OpenAI-compatible servers:
#  - Chat/planner teacher (text LLM) on port 8000
#  - Embeddings teacher on port 8001
#
# This is the Option B path: WSL2 + CUDA + vLLM.
#
# NOTE: On a 4GB laptop GPU, prefer VERY small models. Recommended defaults:
#  - CHAT_MODEL: Qwen/Qwen2.5-0.5B-Instruct (text-only; used for planner/tool traces)
#  - EMBED_MODEL: BAAI/bge-small-en-v1.5
#
# A true VLM (vision-language) teacher typically requires more VRAM than 4GB.
# If you run a VLM, set CHAT_MODEL to a compatible vision model and ensure it fits.

ROOT_DIR="${ROOT_DIR:-/mnt/c/Users/CraigM/source/repos/ContinuonXR}"
VENV="${VENV:-/root/.venvs/continuon_repo}"

CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"

CHAT_PORT="${CHAT_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"

cd "$ROOT_DIR"
source "$VENV/bin/activate"

mkdir -p /tmp/vllm_teacher

echo "Starting vLLM chat server: model=$CHAT_MODEL port=$CHAT_PORT"
nohup python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$CHAT_PORT" \
  --model "$CHAT_MODEL" \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.92 \
  >/tmp/vllm_teacher/chat.log 2>&1 &
echo $! >/tmp/vllm_teacher/chat.pid

echo "Starting vLLM embeddings server: model=$EMBED_MODEL port=$EMBED_PORT"
nohup python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$EMBED_PORT" \
  --model "$EMBED_MODEL" \
  --task embedding \
  --gpu-memory-utilization 0.65 \
  >/tmp/vllm_teacher/embed.log 2>&1 &
echo $! >/tmp/vllm_teacher/embed.pid

sleep 1
echo "PIDs: chat=$(cat /tmp/vllm_teacher/chat.pid) embed=$(cat /tmp/vllm_teacher/embed.pid)"
echo "Logs: /tmp/vllm_teacher/chat.log /tmp/vllm_teacher/embed.log"

