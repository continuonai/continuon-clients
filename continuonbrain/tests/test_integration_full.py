import pytest
import requests
import time
import subprocess
import os
import sys
from pathlib import Path

# repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

@pytest.fixture(scope="module")
def system_stack():
    """Starts the full system stack for testing."""
    test_port = 8086
    kernel_port = 5007
    config_dir = REPO_ROOT / "tmp" / "test_integration"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Start Kernel
    kernel_proc = subprocess.Popen(
        [sys.executable, "-m", "continuonbrain.kernel.safety_kernel", "--port", str(kernel_port)],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    )
    
    # 2. Start Server
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "continuonbrain.api.server", "--port", str(test_port), "--config-dir", str(config_dir), "--mock-hardware"],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "CONTINUON_ALLOW_MOCK_AUTH": "1"}
    )
    
    time.sleep(5)
    yield f"http://localhost:{test_port}"
    
    server_proc.terminate()
    kernel_proc.terminate()

def test_rbac_creator_access(system_stack):
    headers = {"Authorization": "Bearer MOCK_creator_admin@continuon.ai"}
    resp = requests.get(f"{system_stack}/api/ping", headers=headers)
    assert resp.status_code == 200

def test_safety_clipping(system_stack):
    headers = {"Authorization": "Bearer MOCK_creator_admin@continuon.ai"}
    resp = requests.post(
        f"{system_stack}/api/robot/joints", 
        headers=headers, 
        json={"joint_index": 0, "value": 10.0}
    )
    assert resp.status_code == 200
    assert resp.json()["clipping"] == True

def test_context_graph_populated(system_stack):
    resp = requests.get(f"{system_stack}/api/context/graph")
    assert resp.status_code == 200
    assert "nodes" in resp.json()
