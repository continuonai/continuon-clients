"""
Minimal status endpoint for selected model/backend.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from typing import Dict, Any


def start_status_server(selected_model: Dict[str, Any], port: int = 8090) -> HTTPServer:
    """
    Start a lightweight HTTP server returning model/backend status.
    """

    class StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/status":
                self.send_response(404)
                self.end_headers()
                return
            payload = json.dumps({"selected_model": selected_model}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A003
            return  # silence default logging

    server = HTTPServer(("0.0.0.0", port), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"ℹ️  Status endpoint started on http://0.0.0.0:{port}/status")
    return server

