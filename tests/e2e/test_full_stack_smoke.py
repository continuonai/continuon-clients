"""
End-to-End Smoke Test for ContinuonXR Full Stack.
Verifies Brain Startup, API Reachability, and Web UI Content.
"""
import pytest
import requests
import subprocess
import time
import socket
import os
import signal
from pathlib import Path


def is_port_open(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="module")
def brain_server():
    """
    Fixture to launch the ContinuonBrain startup_manager as a background process.
    Uses a temporary config directory to avoid messing with local settings.
    """
    config_dir = Path("/tmp/continuon_e2e_test")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old state
    state_file = config_dir / ".startup_state"
    if state_file.exists():
        state_file.unlink()

    # Launch brain
    cmd = [
        "python", "-m", "continuonbrain.startup_manager",
        "--config-dir", str(config_dir),
        "--port", "8080",
        "--robot-name", "E2ETestBot"
    ]
    
    env = os.environ.copy()
    env["CONTINUON_HEADLESS"] = "1"
    env["CONTINUON_MOCK_HARDWARE"] = "1"
    env["CONTINUON_UI_AUTOLAUNCH"] = "0"
    
    print(f"\n🚀 Launching Brain Server: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )

    # Wait for port 8080 to open (max 30 seconds)
    timeout = 30
    start_time = time.time()
    opened = False
    while time.time() - start_time < timeout:
        if is_port_open(8080):
            opened = True
            break
        if process.poll() is not None:
            # Process died
            stdout, _ = process.communicate()
            pytest.fail(f"Brain server died unexpectedly during startup:\n{stdout}")
        time.sleep(1)
    
    if not opened:
        process.terminate()
        pytest.fail("Brain server failed to open port 8080 within timeout")

    yield process

    # Teardown
    print("\n🛑 Tearing down Brain Server...")
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.wait(timeout=5)


def test_layer_1_network_reachability(brain_server):
    """
    Layer 1: Verify API and Web endpoints return 200 OK.
    """
    base_url = "http://127.0.0.1:8080"
    
    # Check root API
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200
    
    # Check UI endpoint
    response = requests.get(f"{base_url}/ui")
    assert response.status_code == 200
    
    # Check Health endpoint (if exists)
    response = requests.get(f"{base_url}/health")
    # Some endpoints might return 404 if not implemented, but we expect 200 for core ones
    assert response.status_code in [200, 404]


def test_layer_2_content_inspection(brain_server):
    """
    Layer 2: Verify UI content contains expected artifacts (Flutter Web App).
    """
    base_url = "http://127.0.0.1:8080"
    
    # Inspect UI page
    response = requests.get(f"{base_url}/ui")
    html = response.text.lower()
    
    # Verify it's a Flutter Web app or contains Continuon branding
    # Based on the project, the UI is served from studio_server or similar
    assert "continuon" in html
    assert "flutter" in html or "main.dart.js" in html or "canvaskit" in html
    
    # Verify scripts are present
    assert "<script" in html


def test_layer_3_headless_ui_smoke(brain_server):
    """
    Layer 3: Verify UI renders correctly in a headless browser.
    Requires playwright: pip install playwright pytest-playwright && playwright install
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed. Skipping Layer 3 UI test.")

    base_url = "http://127.0.0.1:8080/ui"
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to UI
        print(f"🌐 Navigating to {base_url}...")
        page.goto(base_url, wait_until="networkidle")
        
        # Check for title or key element
        # Flutter usually takes a moment to bootstrap
        page.wait_for_selector("body", timeout=10000)
        
        # Wait for either the Flutter canvas or a specific text
        # If it's the "Brain Studio" or "Continuon", verify it
        title = page.title()
        print(f"📄 Page Title: {title}")
        
        # Check if the page contains 'Continuon' (case insensitive)
        content = page.content().lower()
        assert "continuon" in content or "brain" in content
        
        browser.close()
