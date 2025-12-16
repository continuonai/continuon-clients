#!/usr/bin/env bash
set -euo pipefail

# Install Option-B dependencies into the repo's WSL venv.
# This is intentionally explicit so we don't accidentally add heavy deps to the Windows host.

ROOT_DIR="${ROOT_DIR:-/mnt/c/Users/CraigM/source/repos/ContinuonXR}"
VENV="${VENV:-/root/.venvs/continuon_repo}"

cd "$ROOT_DIR"
source "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel

# CUDA torch (adjust cu version if your WSL CUDA runtime differs)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# vLLM OpenAI-compatible server + HF tokenizer stack
python -m pip install vllm==0.6.6 transformers==4.45.2 sentencepiece==0.2.0

python - <<PY
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

