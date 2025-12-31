# Continuon Brain runtime: Colab quickstart

This runbook shows how to bring up the **Continuon Brain runtime** in a fresh Google Colab session for API experiments (chat, context graph, decision traces) without extra hardware. It favors lightweight dependencies so the notebook starts quickly.

## 1) One-cell bootstrap (minimal CPU setup)

Paste the following cell into Colab. It installs system libs, pulls the repo, and launches the Brain API on port `8080` in mock-hardware mode with a local config directory under `/content/.continuonbrain`.

```bash
%%bash
set -euxo pipefail

# System deps for OpenCV/sentence-transformers
apt-get update -qq
apt-get install -y -qq libgl1 libglib2.0-0

# Python deps (CPU JAX; remove sentence-transformers to skip embeddings)
pip install --quiet "jax[cpu]" flax chex optax numpy scipy sentence-transformers fastapi uvicorn[standard] pyyaml requests

# Repo checkout
cd /content
if [ ! -d ContinuonXR ]; then
  git clone --depth 1 https://github.com/<your-org>/ContinuonXR.git
fi
cd ContinuonXR
export PYTHONPATH=$PWD
export CONTINUON_CONFIG_DIR=/content/.continuonbrain

# Launch server (mock hardware; keep this cell running)
python -m continuonbrain.api.server \
  --config-dir "${CONTINUON_CONFIG_DIR}" \
  --port 8080 \
  --mock-hardware
```

Keep this cell running to serve requests. Open a new Colab cell to interact.

## 2) Verify the runtime is up

In a new cell:

```bash
%%bash
curl -s http://localhost:8080/api/status | head
curl -s "http://localhost:8080/api/runtime/context" | head
```

If JAX is available, the Brain will auto-prime the JAX world-model adapter; otherwise it falls back to a stub and continues to serve chat + context graph APIs.

## 3) Chat and context graph calls

```bash
# Basic chat
curl -s -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello from colab"}'

# Context graph (scoped subgraph)
curl -s "http://localhost:8080/api/context/graph?depth=2&limit=25"

# Decision traces (policy/tool edges)
curl -s "http://localhost:8080/api/context/graph/decisions?depth=2&limit=25"
```

## 4) Optional: seed a tiny RLDS episode and ingest

```bash
%%bash
cd /content/ContinuonXR
export PYTHONPATH=$PWD
export CONTINUON_CONFIG_DIR=/content/.continuonbrain

# Create a small RLDS transcript from a mock chat
python - <<'PY'
from continuonbrain.services.brain_service import BrainService

svc = BrainService(config_dir=CONTINUON_CONFIG_DIR, prefer_real_hardware=False, auto_detect=False)
svc._save_rlds_episode(
    conversation=[
        {"role": "user", "content": "test goal"},
        {"role": "assistant", "content": "ack from mock brain"},
    ]
)
PY

# Inspect the ingested context graph
curl -s "http://localhost:8080/api/context/graph?depth=2&limit=25"
```

## 5) Tips for Colab quirks

- The API process must stay in its own running cell. Use a second cell for curls/requests.
- If installs are too slow, drop `sentence-transformers`; semantic search will degrade gracefully.
- To expose the port externally, use a Colab tunnel (e.g., `cloudflared` or `lt`) in a separate cell; keep API traffic private by default.
- Logs, context graph SQLite DB, and RLDS snippets live under `${CONTINUON_CONFIG_DIR}`. Delete the directory to reset state between runs.
