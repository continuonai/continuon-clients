#!/bin/bash
export PYTHONPATH=/home/craigm26/ContinuonXR:/home/craigm26/.local/lib/python3.13/site-packages
export CONFIG_DIR=/home/craigm26/.continuonbrain
export GEMINI_API_KEY=AIzaSyDhXpcXCwhNP6Xl8HGNcAMAWi6TBdBw20A
export HUGGINGFACE_TOKEN=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

echo "Starting Server Manual Debug (Unbuffered)..." > debug_server.log
/home/craigm26/ContinuonXR/.venv/bin/python3 -u -m continuonbrain.api.server --port 8080 --real-hardware >> debug_server.log 2>&1
