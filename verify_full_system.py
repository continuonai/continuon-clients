import os
import sys
import time
import json
import subprocess
import requests
import shutil
import signal
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class SystemHarness:
    """Manages the lifecycle of the Brain Server and Safety Kernel for verification."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.config_dir = test_dir / "config"
        self.log_dir = test_dir / "logs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.server_port = 8085
        self.kernel_port = 5006
        self.socket_path = str(self.config_dir / "safety.sock")
        
        self.server_proc: Optional[subprocess.Popen] = None
        self.kernel_proc: Optional[subprocess.Popen] = None
        
        # Paths
        self.python_exec = sys.executable
        
    def start(self):
        print(f"--- Starting System Harness in {self.test_dir} ---")
        
        # 1. Start Safety Kernel
        print(f"Starting Safety Kernel on port {self.kernel_port}...")
        kernel_env = {{**os.environ, "PYTHONPATH": str(REPO_ROOT)}}
        self.kernel_proc = subprocess.Popen(
            [self.python_exec, "-m", "continuonbrain.kernel.safety_kernel", 
             "--port", str(self.kernel_port), 
             "--socket", self.socket_path],
            env=kernel_env,
            stdout=(self.log_dir / "kernel.log").open("w"),
            stderr=subprocess.STDOUT
        )
        
        # 2. Start Brain Server
        print(f"Starting Brain Server on port {self.server_port}...")
        server_env = {
            **os.environ, 
            "PYTHONPATH": str(REPO_ROOT),
            "CONTINUON_ALLOW_MOCK_AUTH": "1",
            "CONTINUON_NO_UI_LAUNCH": "1"
        }
        self.server_proc = subprocess.Popen(
            [self.python_exec, "-m", "continuonbrain.api.server", 
             "--port", str(self.server_port), 
             "--config-dir", str(self.config_dir),
             "--mock-hardware"],
            env=server_env,
            stdout=(self.log_dir / "server.log").open("w"),
            stderr=subprocess.STDOUT
        )
        
        # Wait for startup
        print("Waiting for services to initialize...")
        time.sleep(5)
        print("--- Systems Ready ---")

    def stop(self):
        print("--- Stopping System Harness ---")
        if self.server_proc:
            self.server_proc.terminate()
            self.server_proc.wait()
        if self.kernel_proc:
            self.kernel_proc.terminate()
            self.kernel_proc.wait()
        print("--- Systems Offline ---")

    def get_api_url(self, path: str) -> str:
        return f"http://localhost:{self.server_port}{path}"

def run_tests(harness: SystemHarness) -> Dict[str, Any]:
    results = {"subsystems": {}, "overall_pass": True}
    
    def log_result(name: str, passed: bool, details: str):
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {details}")
        results["subsystems"][name] = {"passed": passed, "details": details}
        if not passed:
            results["overall_pass"] = False

    # --- 1. RBAC & AUTH ---
    print("\nVerifying RBAC...")
    try:
        # Unauthorized (No header)
        resp = requests.post(harness.get_api_url("/api/mode/autonomous"))
        log_result("RBAC_Unauthorized", resp.status_code == 401, "No header check")
        
        # Forbidden (Consumer trying Admin)
        headers = {"Authorization": "Bearer MOCK_consumer_test@user.com"}
        resp = requests.post(harness.get_api_url("/api/admin/promote_candidate"), headers=headers, json={{}})
        log_result("RBAC_Forbidden", resp.status_code == 403, "Consumer accessing admin")
        
        # Authorized (Creator)
        headers = {"Authorization": "Bearer MOCK_creator_admin@continuon.ai"}
        resp = requests.get(harness.get_api_url("/api/ping"), headers=headers)
        log_result("RBAC_Authorized", resp.status_code == 200, "Creator ping")
    except Exception as e:
        log_result("RBAC_Exception", False, str(e))

    # --- 2. SAFETY KERNEL ---
    print("\nVerifying Safety Kernel...")
    try:
        headers = {"Authorization": "Bearer MOCK_creator_admin@continuon.ai"}
        # Valid command
        resp = requests.post(harness.get_api_url("/api/robot/joints"), headers=headers, json={"joint_index": 0, "value": 0.5})
        log_result("Safety_Valid", resp.status_code == 200 and resp.json().get("success"), "Normal joint move")
        
        # Out of bounds command
        resp = requests.post(harness.get_api_url("/api/robot/joints"), headers=headers, json={"joint_index": 0, "value": 5.0})
        # Check if it was clipped (success=True but value adjusted)
        data = resp.json()
        log_result("Safety_Clipping", data.get("success") and data.get("clipping"), "OOB value clipped")
    except Exception as e:
        log_result("Safety_Exception", False, str(e))

    # --- 3. CONTEXT GRAPHS ---
    print("\nVerifying Context Graphs...")
    try:
        # Trigger an ingestion (ping introspection often does this or we can trigger it)
        resp = requests.get(harness.get_api_url("/api/status/introspection"))
        time.sleep(1)
        # Check graph
        resp = requests.get(harness.get_api_url("/api/context/graph"))
        data = resp.json()
        log_result("Graph_Query", "nodes" in data and len(data["nodes"]) > 0, f"Found {len(data.get('nodes', []))} nodes")
    except Exception as e:
        log_result("Graph_Exception", False, str(e))

    return results

if __name__ == "__main__":
    test_root = REPO_ROOT / "tmp" / "system_verify"
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True)
    
    harness = SystemHarness(test_root)
    try:
        harness.start()
        report = run_tests(harness)
        
        # Save Report
        with open(test_root / "system_audit_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*40)
        print(f"VERIFICATION {'SUCCESS' if report['overall_pass'] else 'FAILED'}")
        print("="*40)
        
        if not report["overall_pass"]:
            sys.exit(1)
            
    finally:
        harness.stop()