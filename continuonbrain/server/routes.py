"""
Simple JSON/HTTP server extracted from robot_api_server.
"""

import asyncio
import json
from urllib.parse import parse_qs
from pathlib import Path
from typing import Dict, Any

from continuonbrain.settings_manager import SettingsStore, SettingsValidationError
from continuonbrain.server.tasks import TaskLibraryEntry, TaskSummary
from pathlib import Path


class SimpleJSONServer:
    """
    HTTP/JSON server for robot control and web UI.
    Supports both HTTP endpoints and raw JSON protocol.
    """

    # Chat configuration
    CHAT_HISTORY_LIMIT = 50  # Maximum number of chat messages to persist

    def __init__(self, service, ui_provider=None, control_provider=None):
        self.service = service
        self.ui_provider = ui_provider
        self.control_provider = control_provider
        self.server = None

    async def handle_http_request(self, request_line: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle HTTP request and return HTML/JSON response."""
        # Parse request line
        parts = request_line.split()
        method = parts[0] if len(parts) > 0 else "GET"
        full_path = parts[1] if len(parts) > 1 else "/"

        # Strip query string for routing
        path = full_path.split('?')[0]
        query_params = parse_qs(full_path.split('?', 1)[1]) if '?' in full_path else {}

        print(f"[HTTP] {method} {path}")

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if not line or line == b'\r\n' or line == b'\n':
                break
            header_line = line.decode().strip()
            if ':' in header_line:
                key, value = header_line.split(':', 1)
                headers[key.strip().lower()] = value.strip()

        # Route the request
        response = await self._route(path, method, query_params, headers, reader)

        writer.write(response.encode('utf-8') if isinstance(response, str) else response)
        await writer.drain()
        writer.close()

    async def _route(self, path: str, method: str, query_params: Dict[str, Any], headers: Dict[str, Any], reader: asyncio.StreamReader) -> str:
        if path == "/" or path == "/ui":
            response_body = self.get_web_ui_html()
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/control":
            await self.service.SetRobotMode("manual_control")
            response_body = self.get_control_interface_html()
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status, indent=2)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/loops":
            status = await self.service.GetLoopHealth()
            response_body = json.dumps(status)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/gates":
            gates = await self.service.GetGates()
            response_body = json.dumps(gates)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/settings" and method == "GET":
            store = SettingsStore(Path(self.service.config_dir))
            settings = store.load()
            response_body = json.dumps({"success": True, "settings": settings})
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path in {"/api/tasks", "/api/tasks/"}:
            include_ineligible = query_params.get("include_ineligible", ["false"])[0].lower() == "true"
            result = await self.service.ListTasks(include_ineligible=include_ineligible)
            response_body = json.dumps(result)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path.startswith("/api/tasks/summary/"):
            task_id = path.split("/")[-1]
            result = await self.service.GetTaskSummary(task_id)
            response_body = json.dumps(result)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/tasks/select" and method == "POST":
            content_length = int(headers.get('content-length', 0))
            body = await reader.read(content_length) if content_length > 0 else b''
            payload = json.loads(body.decode()) if body else {}
            result = await self.service.SelectTask(payload.get("task_id", ""), reason=payload.get("reason"))
            response_body = json.dumps(result)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/settings" and method == "POST":
            content_length = int(headers.get('content-length', 0))
            body = await reader.read(content_length) if content_length > 0 else b''
            payload = json.loads(body.decode()) if body else {}
            store = SettingsStore(Path(self.service.config_dir))
            try:
                validated = store.save(payload)
                response_body = json.dumps(
                    {"success": True, "settings": validated, "message": "Settings saved"}
                )
                status_line = "HTTP/1.1 200 OK"
            except SettingsValidationError as exc:
                response_body = json.dumps({"success": False, "message": str(exc)})
                status_line = "HTTP/1.1 400 Bad Request"

            return f"{status_line}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"

        # Fallback
        return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming client connections."""
        try:
            # Read the first line of the HTTP request
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return
            first_line_str = request_line.decode().strip()
            await self.handle_http_request(first_line_str, reader, writer)
        except Exception as exc:
            print(f"Error handling client: {exc}")
            writer.close()

    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the server."""
        self.server = await asyncio.start_server(
            self.handle_client,
            host=host,
            port=port,
        )
        addr = self.server.sockets[0].getsockname()
        print("📱 Web UI: http://{0}:{1}/ui".format(addr[0] if addr[0] != '0.0.0.0' else 'localhost', addr[1]))
        print("🔌 API Endpoint: http://{0}:{1}/status".format(addr[0] if addr[0] != '0.0.0.0' else 'localhost', addr[1]))

        async with self.server:
            await self.server.serve_forever()

    def get_web_ui_html(self) -> str:
        tmpl_path = Path(__file__).parent / "templates" / "ui.html"
        if tmpl_path.exists():
            return tmpl_path.read_text(encoding="utf-8")
        if self.ui_provider:
            content = self.ui_provider()
            tmpl_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl_path.write_text(content, encoding="utf-8")
            return content
        return "<html><body><h1>Robot UI</h1></body></html>"

    def get_control_interface_html(self) -> str:
        tmpl_path = Path(__file__).parent / "templates" / "control.html"
        if tmpl_path.exists():
            return tmpl_path.read_text(encoding="utf-8")
        if self.control_provider:
            content = self.control_provider()
            tmpl_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl_path.write_text(content, encoding="utf-8")
            return content
        return "<html><body><h1>Control Interface</h1></body></html>"

