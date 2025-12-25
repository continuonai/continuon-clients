
import requests
import json
import time
import subprocess
import os
import sys

def run_rbac_tests():
    # Set environment variables for mock auth
    os.environ["CONTINUON_ALLOW_MOCK_AUTH"] = "1"
    os.environ["CONTINUON_HEADLESS"] = "1" # Avoid launching browser
    os.environ["CONTINUON_NO_UI_LAUNCH"] = "1"
    
    port = 8084
    base_url = f"http://localhost:{port}"
    
    # Start the server
    print(f"Starting server on port {port}...")
    # Use -u for unbuffered output
    server_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "continuonbrain.api.server", "--port", str(port), "--config-dir", "/tmp/rbac_test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to be ready
    max_retries = 10
    connected = False
    for i in range(max_retries):
        try:
            resp = requests.get(f"{base_url}/api/ping", timeout=2)
            if resp.status_code == 200:
                print("Server is up!")
                connected = True
                break
        except Exception:
            print("Waiting for server...")
            time.sleep(2)
    
    if not connected:
        print("❌ Failed to start server")
        server_process.terminate()
        return

    try:
        # 1. Test Unauthenticated Access to Protected Endpoint
        print("\n--- Test 1: Unauthenticated Access to Protected Endpoint ---")
        # /api/robot/drive requires CONSUMER
        resp = requests.post(f"{base_url}/api/robot/drive", json={"steering": 0, "throttle": 0})
        print(f"POST /api/robot/drive (No Auth) -> {resp.status_code}")
        if resp.status_code == 401:
            print("✅ Success: Unauthenticated access blocked")
        else:
            print(f"❌ Failed: Unauthenticated access should be 401 (got {resp.status_code})")

        # 2. Test Protected Endpoint (Consumer Role)
        print("\n--- Test 2: Protected Endpoint (Consumer) ---")
        headers = {"Authorization": "Bearer MOCK_consumer_user@example.com"}
        resp = requests.post(f"{base_url}/api/robot/drive", json={"steering": 0, "throttle": 0}, headers=headers)
        print(f"POST /api/robot/drive (CONSUMER) -> {resp.status_code}")
        if resp.status_code == 200:
            print("✅ Success: Consumer can access drive endpoint")
        else:
            print(f"❌ Failed: {resp.text}")

        # 3. Test Creator Endpoint with Consumer Role
        print("\n--- Test 3: Creator Endpoint with Consumer Role ---")
        resp = requests.post(f"{base_url}/api/v1/models/activate", json={"model_id": "test"}, headers=headers)
        print(f"POST /api/v1/models/activate (CONSUMER) -> {resp.status_code}")
        if resp.status_code == 403:
            print("✅ Success: Consumer blocked from Creator endpoint")
        else:
            print(f"❌ Failed: Consumer should be blocked (got {resp.status_code})")

        # 4. Test Creator Endpoint with Creator Role
        print("\n--- Test 4: Creator Endpoint with Creator Role ---")
        # Email craigm26@gmail.com is hardcoded as CREATOR in auth.py
        headers = {"Authorization": "Bearer MOCK_creator_craigm26@gmail.com"}
        resp = requests.post(f"{base_url}/api/v1/models/activate", json={"model_id": "mock"}, headers=headers)
        print(f"POST /api/v1/models/activate (CREATOR) -> {resp.status_code}")
        if resp.status_code == 200:
            print("✅ Success: Creator can access model activation")
        else:
            print(f"❌ Failed: {resp.text}")

        # 5. Test Learning Endpoint (Now Protected)
        print("\n--- Test 5: Learning Endpoint (Verify if Protected) ---")
        resp = requests.get(f"{base_url}/api/training/cloud_readiness")
        print(f"GET /api/training/cloud_readiness (No Auth) -> {resp.status_code}")
        if resp.status_code == 401:
            print("✅ Success: Learning endpoint now protected")
        else:
            print(f"❌ Failed: Learning endpoint should be protected (got {resp.status_code})")

        # 6. Test Learning Endpoint with Developer Role
        print("\n--- Test 6: Learning Endpoint with Developer Role ---")
        headers = {"Authorization": "Bearer MOCK_developer_dev@example.com"}
        resp = requests.get(f"{base_url}/api/training/cloud_readiness", headers=headers)
        print(f"GET /api/training/cloud_readiness (DEVELOPER) -> {resp.status_code}")
        if resp.status_code == 200:
            print("✅ Success: Developer can access learning readiness")
        else:
            print(f"❌ Failed: {resp.text}")

finally:
    print("\nStopping server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

if __name__ == "__main__":
    run_rbac_tests()
