import os
import sys
import socket
import json
import time
import unittest
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuonbrain.kernel.safety_kernel import SafetyKernel

class SafetyKernelClient:
    """Helper client for testing SafetyKernel."""
    def __init__(self, host='localhost', port=5005, socket_path='/tmp/continuon_safety.sock'):
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self.use_unix_socket = sys.platform != 'win32'
        self.sock = None

    def connect(self):
        if self.use_unix_socket:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))

    def send_command(self, command, args=None):
        payload = {"command": command, "args": args or {}}
        self.sock.sendall(json.dumps(payload).encode('utf-8'))
        response = self.sock.recv(4096)
        return json.loads(response.decode('utf-8'))

    def close(self):
        if self.sock:
            self.sock.close()

class TestSafetyKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start kernel in a background thread
        cls.kernel = SafetyKernel(port=5006, socket_path='/tmp/test_safety.sock')
        cls.kernel.start()
        time.sleep(0.5) # Wait for it to start

    @classmethod
    def tearDownClass(cls):
        cls.kernel.stop()

    def test_basic_command(self):
        client = SafetyKernelClient(port=5006, socket_path='/tmp/test_safety.sock')
        client.connect()
        try:
            response = client.send_command("move_arm", {"x": 10, "y": 20})
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["command"], "move_arm")
            self.assertEqual(response["args"]["x"], 10)
        finally:
            client.close()

    def test_velocity_clipping(self):
        client = SafetyKernelClient(port=5006, socket_path='/tmp/test_safety.sock')
        client.connect()
        try:
            # max_velocity is 1.0 in skeleton
            response = client.send_command("drive", {"velocity": 2.5})
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["args"]["velocity"], 1.0)
            
            response = client.send_command("drive", {"velocity": -5.0})
            self.assertEqual(response["args"]["velocity"], -1.0)
        finally:
            client.close()

if __name__ == "__main__":
    unittest.main()
