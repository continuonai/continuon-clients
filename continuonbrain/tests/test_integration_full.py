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


def test_decision_traces_logged(system_stack):
    headers = {"Authorization": "Bearer MOCK_creator_admin@continuon.ai"}
    for _ in range(10):
        try:
            ready = requests.get(f"{system_stack}/api/ping", timeout=1)
            if ready.status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        pytest.skip("Server not reachable for decision trace logging")

    plan_resp = requests.post(
        f"{system_stack}/api/context/decision/plan",
        json={
            "plan_text": "Pick and place with gripper",
            "tools": ["gripper"],
            "session_id": "itest",
            "actor": "hope_llm",
            "provenance": {"source": "integration_test"},
        },
    )
    assert plan_resp.status_code == 200
    plan_node = plan_resp.json().get("plan_node")
    assert plan_node

    safety_resp = requests.post(
        f"{system_stack}/api/robot/joints",
        headers=headers,
        json={"joint_index": 0, "value": 10.0, "session_id": "itest"},
    )
    assert safety_resp.status_code == 200

    human_resp = requests.post(
        f"{system_stack}/api/context/decision/feedback",
        json={
            "session_id": "itest",
            "action_ref": plan_node,
            "approved": False,
            "user_id": "integration_tester",
            "notes": "Manual rejection for test coverage",
        },
    )
    assert human_resp.status_code == 200

    graph_resp = requests.get(f"{system_stack}/api/context/graph/decisions?limit=25&depth=3")
    assert graph_resp.status_code == 200
    payload = graph_resp.json()
    edges = payload.get("edges", [])
    assert any(e["type"] == "tool_use" and e["source"] == plan_node for e in edges)
    assert any(e["type"] == "policy" and e["provenance"].get("source") == "safety_kernel" for e in edges)
    assert any(e["provenance"].get("source") == "human" for e in edges)
