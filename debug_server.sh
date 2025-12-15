#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CONFIG_DIR=/home/craigm26/.continuonbrain

# NOTE: Keep secrets out of repo scripts.

# Load secrets from .htoken if present
if [ -f ".htoken" ]; then
    export HUGGINGFACE_TOKEN=$(cat .htoken | tr -d '\n')
    # echo "Loaded HUGGINGFACE_TOKEN from .htoken"
fi

# Set these in your shell environment (or a secure systemd env/credential) before running:
#   export GEMINI_API_KEY="..."
#   export HUGGINGFACE_TOKEN="..."

echo "Starting Server Manual Debug (Unbuffered)..." > debug_server.log
./.venv/bin/python3 -u -m continuonbrain.api.server --port 8080 --real-hardware >> debug_server.log 2>&1
