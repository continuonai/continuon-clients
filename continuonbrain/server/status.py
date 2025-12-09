"""
Minimal status endpoint for selected model/backend and battery status.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from typing import Dict, Any, Optional


def _get_battery_status() -> Optional[Dict[str, Any]]:
    """Get battery status if available."""
    try:
        from continuonbrain.sensors.battery_monitor import BatteryMonitor
        monitor = BatteryMonitor()
        return monitor.get_diagnostics()
    except Exception:
        return None


def start_status_server(selected_model: Dict[str, Any], port: int = 8090) -> HTTPServer:
    """
    Start a lightweight HTTP server returning model/backend status and battery info.
    """

    class StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/status":
                self.send_response(404)
                self.end_headers()
                return
            
            # Build status response
            status_data: Dict[str, Any] = {"selected_model": selected_model}
            
            # Add battery status if available
            battery_status = _get_battery_status()
            if battery_status:
                status_data["battery"] = battery_status
            
            payload = json.dumps(status_data, indent=2).encode("utf-8")
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

