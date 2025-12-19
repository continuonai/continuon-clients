import os
import sys
import socket
import json
import logging
import signal
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from continuonbrain.kernel.constitution import Constitution
from continuonbrain.kernel.graduated_response import GraduatedResponseSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SafetyKernel: %(message)s'
)
logger = logging.getLogger("SafetyKernel")

class SafetyKernel:
    """
    Ring 0 Safety Kernel.
    Deterministic gatekeeper for all actuation commands.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 5005, socket_path: str = '/tmp/continuon_safety.sock', config: Optional[Dict[str, Any]] = None):
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self.running = False
        self.server_socket = None
        
        # Use Unix Domain Socket on Linux, TCP on others (Windows)
        self.use_unix_socket = sys.platform != 'win32'
        
        # Actuation limits (The Constitution)
        self.constitution = Constitution(config)
        self.response_system = GraduatedResponseSystem(self)
        self.system_safe = True

    def start(self):
        """Start the safety kernel listener."""
        self.running = True
        
        if self.use_unix_socket:
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            os.chmod(self.socket_path, 0o666) # Ensure accessibility
            logger.info(f"Listening on Unix socket: {self.socket_path}")
        else:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            logger.info(f"Listening on TCP: {self.host}:{self.port} (Windows fallback)")

        self.server_socket.listen(5)
        
        # Start the accept loop in a separate thread
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()
        
        # Start background health monitoring
        self.health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self.health_thread.start()
        
        logger.info("✅ Safety Kernel active and ready.")

    def stop(self):
        """Stop the safety kernel."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.use_unix_socket and os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        logger.info("🛑 Safety Kernel shut down.")

    def _health_loop(self):
        """Monitor system vitals and trigger hard halts if needed."""
        while self.running:
            # Placeholder for real sensor reading
            metrics = {"voltage": 12.1, "cpu_temp": 45.0} 
            self.system_safe = self.constitution.check_system_health(metrics)
            
            if not self.system_safe:
                logger.error("🚨 CRITICAL SYSTEM HALT TRIGGERED BY CONSTITUTION")
            
            time.sleep(1.0)

    def _accept_loop(self):
        while self.running:
            try:
                client_sock, addr = self.server_socket.accept()
                logger.info(f"Accepted connection from {addr}")
                client_thread = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
                client_thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
                break

    def _handle_client(self, client_sock):
        try:
            while self.running:
                data = client_sock.recv(4096)
                if not data:
                    break
                
                try:
                    payload = json.loads(data.decode('utf-8'))
                    command = payload.get('command')
                    args = payload.get('args', {})
                    
                    if not self.system_safe:
                         response = self.response_system.apply_response(2, command, args, "system_not_safe")
                    else:
                        # VALIDATION (Ring 0 Logic)
                        level, safe_args, reason = self.constitution.validate_actuation(command, args)
                        
                        response = self.response_system.apply_response(level, command, safe_args, reason)
                        response["timestamp"] = time.time()
                    
                    client_sock.sendall(json.dumps(response).encode('utf-8'))
                    
                except json.JSONDecodeError:
                    logger.warning("Received malformed JSON")
                    client_sock.sendall(json.dumps({"error": "malformed_json"}).encode('utf-8'))
        finally:
            client_sock.close()

class SafetyKernelClient:
    """Helper client for interacting with the SafetyKernel IPC."""
    def __init__(self, host='localhost', port=5005, socket_path='/tmp/continuon_safety.sock'):
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self.use_unix_socket = sys.platform != 'win32'

    def send_command(self, command: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to the Safety Kernel and wait for validated response."""
        try:
            if self.use_unix_socket:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self.socket_path)
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
            
            payload = {"command": command, "args": args or {}}
            sock.sendall(json.dumps(payload).encode('utf-8'))
            
            response = sock.recv(4096)
            sock.close()
            return json.loads(response.decode('utf-8'))
        except Exception as e:
            logger.error(f"Safety Kernel communication failed: {e}")
            return {"status": "error", "reason": "kernel_offline"}

def handle_signals(kernel):
    def signal_handler(sig, frame):
        logger.info("Interrupt received...")
        kernel.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    kernel = SafetyKernel()
    handle_signals(kernel)
    kernel.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        kernel.stop()
