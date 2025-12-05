"""
ContinuonBrain Main API Server.
Modular replacement for the legacy robot_api_server.py.
"""
import sys
import os
import asyncio
import json
import logging
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.brain_service import BrainService
from continuonbrain.agent_identity import AgentIdentity
from continuonbrain.api.routes import ui_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainServer")

# Global Service Instance
brain_service: BrainService = None
identity_service: AgentIdentity = None

class BrainRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Brain API."""

    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/ui" or self.path == "/ui/":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_home_html().encode("utf-8"))
            
            elif self.path == "/ui/status":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_status_html().encode("utf-8"))
                
            elif self.path == "/ui/dashboard":
                # Fallback to legacy dashboard (we need to port the HTML from robot_api_server)
                # For now returning a placeholder
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Legacy Dashboard Placeholder</h1>")

            elif self.path == "/api/status/introspection":
                # Introspection endpoint for Brain Status page
                identity_service.self_report() # Updates internal state
                data = identity_service.identity
                self.send_json(data)

            elif self.path == "/api/status":
                # Legacy robot status
                # TODO: Implement full status serialization
                self.send_json({"status": "ok", "mode": "idle"})
                
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.send_error(500)

    def do_POST(self):
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len).decode('utf-8')
            
            if self.path == "/api/chat":
                data = json.loads(body)
                msg = data.get("message", "")
                
                response = brain_service.ChatWithGemma(msg, [])
                self.send_json({"response": response})
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"POST error: {e}")
            self.send_error(500)

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence default logging
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True

def main():
    global brain_service, identity_service
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/tmp/continuonbrain_demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--real-hardware", action="store_true", help="Prefer real hardware")
    parser.add_argument("--mock-hardware", action="store_true", help="Force mock hardware")
    args = parser.parse_args()
    
    prefer_real = args.real_hardware and not args.mock_hardware
    
    print(f"🧠 Starting ContinuonBrain Server on port {args.port}...")
    
    # Initialize Services
    identity_service = AgentIdentity(config_dir=args.config_dir)
    # Run identity check early to determine shell
    identity_service.self_report() 
    shell_type = identity_service.identity.get("shell", {}).get("type", "Unknown")
    
    # If Desktop Station, force mock hardware for robot components?
    # Or let BrainService handle it using "One Brain Many Shells" logic?
    # For now, we pass preferences.
    
    brain_service = BrainService(
        config_dir=args.config_dir,
        prefer_real_hardware=prefer_real,
        auto_detect=True
    )
    
    # Async Init (hack for sync constructor)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(brain_service.initialize())
    
    server = ThreadedHTTPServer(("0.0.0.0", args.port), BrainRequestHandler)
    print(f"🚀 Server listening on http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    
    brain_service.shutdown()
    server.server_close()
    print("Server stopped.")

if __name__ == "__main__":
    main()
