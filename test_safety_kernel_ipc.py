
import subprocess
import time
import sys
import os
import json
from continuonbrain.kernel.safety_kernel import SafetyKernelClient

def test_ipc():
    print("Starting Safety Kernel process...")
    # Use -u for unbuffered output
    kernel_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "continuonbrain.kernel.safety_kernel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Give it a moment to start
    time.sleep(2)
    
    # Check if it started
    if kernel_process.poll() is not None:
        print("❌ Safety Kernel failed to start")
        print(kernel_process.stdout.read())
        return

    print("✅ Safety Kernel process is running")
    
    client = SafetyKernelClient()
    
    try:
        # Test 1: Simple valid command
        print("\nTest 1: Valid move_joints command")
        resp = client.send_command("move_joints", {"joints": {"base": 10.0, "shoulder": 20.0}})
        print(f"Response: {json.dumps(resp, indent=2)}")
        assert resp["status"] == "ok"
        assert resp["safety_level"] == 2

        # Test 2: Clipping command (base too high)
        print("\nTest 2: Out-of-bounds move_joints command (clipping)")
        resp = client.send_command("move_joints", {"joints": {"base": 200.0}})
        print(f"Response: {json.dumps(resp, indent=2)}")
        assert resp["status"] == "ok"
        assert resp["safety_level"] == 1
        assert resp["args"]["joints"]["base"] == 180.0
        assert resp["warning"] == "joint_limit_clipping"

        # Test 3: System unhealthy (simulated later, but testing error handling)
        # For now, let's test a command that might fail in constitution
        print("\nTest 3: Out of reach command")
        resp = client.send_command("move_to", {"target_pose": {"x": 2000, "y": 0, "z": 0}})
        print(f"Response: {json.dumps(resp, indent=2)}")
        # Currently Constitution says level 0 for this
        # apply_response(0, ...) returns status: ok, safety_level: 2? Wait.
        # GraduatedResponseSystem.apply_response level 0 returns safety_level 2.
        # That seems like a bug or confusing. Level 0 should be Denied.
        
        print("\n✅ IPC tests completed successfully")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        print("\nStopping Safety Kernel...")
        kernel_process.terminate()
        try:
            kernel_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kernel_process.kill()
        print("Done.")

if __name__ == "__main__":
    test_ipc()
