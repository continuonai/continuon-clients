"""
Simple JSON/HTTP server extracted from robot_api_server.
"""

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

import jinja2

from continuonbrain.settings_manager import SettingsStore, SettingsValidationError
from continuonbrain.server.tasks import TaskLibraryEntry, TaskSummary


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
        
        # Initialize Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def render_template(self, template_name: str, context: Dict[str, Any] = None) -> str:
        """Render a Jinja2 template."""
        if context is None:
            context = {}
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
        except jinja2.TemplateNotFound:
            return f"<html><body><h1>Template {template_name} not found</h1></body></html>"
        except Exception as e:
            return f"<html><body><h1>Error rendering template: {e}</h1></body></html>"

    async def handle_http_request(self, request_line: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle HTTP request and return HTML/JSON/SSE response."""
        parts = request_line.split()
        method = parts[0] if len(parts) > 0 else "GET"
        full_path = parts[1] if len(parts) > 1 else "/"

        path = full_path.split("?")[0]
        query_params = parse_qs(full_path.split("?", 1)[1]) if "?" in full_path else {}

        print(f"[HTTP] {method} {path}")

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n" or line == b"\n":
                break
            header_line = line.decode().strip()
            if ":" in header_line:
                key, value = header_line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        # SSE endpoint
        if path == "/api/events":
            await self._handle_sse(writer)
            return

        # Static assets
        if path.startswith("/static/"):
            await self._serve_static(path, writer)
            return

        response = await self._route(path, method, query_params, headers, reader, writer)

        if response is not None and not writer.is_closing():
            writer.write(response.encode("utf-8") if isinstance(response, str) else response)
            await writer.drain()
            writer.close()

    async def _route(
        self,
        path: str,
        method: str,
        query_params: Dict[str, Any],
        headers: Dict[str, Any],
        reader: asyncio.StreamReader,
        writer: Optional[asyncio.StreamWriter],
    ) -> Any:
        # Page Routes
        if path == "/" or path == "/ui":
            response_body = self.render_template("home.html", {"active_page": "home"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/safety":
            response_body = self.render_template("safety.html", {"active_page": "safety"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/tasks":
            response_body = self.render_template("tasks.html", {"active_page": "tasks"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/skills":
            response_body = self.render_template("skills.html", {"active_page": "skills"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/research":
            response_body = self.render_template("research.html", {"active_page": "research"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/api_explorer":
            response_body = self.render_template("api_explorer.html", {"active_page": "api_explorer"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
            
        elif path == "/wiring":
            response_body = self.get_wiring_html()
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/settings":
            response_body = self.get_settings_html()
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
        elif path == "/api/wiring":
            response_body = json.dumps(self._get_wiring_stats())
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/loops":
            status = await self.service.GetLoopHealth()
            response_body = json.dumps(status)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/gates":
            gates = await self.service.GetGates()
            response_body = json.dumps(gates)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/status":
            status_path = Path("/opt/continuonos/brain/trainer/status.json")
            if status_path.exists():
                response_body = status_path.read_text()
            else:
                response_body = json.dumps({"status": "unknown", "message": "training status file not found"})
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/logs":
            log_dir = Path("/opt/continuonos/brain/trainer/logs")
            logs = sorted(log_dir.glob("trainer_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            payload = [{"path": str(p), "mtime": p.stat().st_mtime} for p in logs]
            response_body = json.dumps(payload)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/run" and method == "POST":
            # Trigger a background training run (non-blocking). Service must implement RunTraining().
            try:
                await self.service.RunTraining()
                response_body = json.dumps({"status": "started"})
                return f"HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/manual" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunManualTraining(payload or {})
                response_body = json.dumps({"status": "completed", "result": result})
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/wavecore_loops" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunWavecoreLoops(payload or {})
                response_body = json.dumps({"status": "completed", "result": result})
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/hope_eval" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunHopeEval(payload or {})
                response_body = json.dumps({"status": "completed", "result": result})
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/hope_eval_facts" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunFactsEval(payload or {})
                response_body = json.dumps({"status": "completed", "result": result})
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
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
        elif path in {"/api/skills", "/api/skills/"}:
            include_ineligible = query_params.get("include_ineligible", ["false"])[0].lower() == "true"
            result = await self.service.ListSkills(include_ineligible=include_ineligible)
            response_body = json.dumps(result)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path.startswith("/api/skills/summary/"):
            skill_id = path.split("/")[-1]
            result = await self.service.GetSkillSummary(skill_id)
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

        # Legacy/expected control endpoints restored for UI compatibility
        elif path.startswith("/api/mode/"):
            mode_name = path.split("/")[-1]
            result = await self.service.SetRobotMode(mode_name)
            return self._json_response(result)

        elif path == "/api/command" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            result = await self.service.SendCommand(payload or {})
            return self._json_response(result)

        elif path == "/api/drive" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            steering = payload.get("steering") if isinstance(payload, dict) else None
            throttle = payload.get("throttle") if isinstance(payload, dict) else None
            result = await self.service.Drive(steering, throttle)
            return self._json_response(result)

        elif path == "/api/safety/hold" and method == "POST":
            result = await self.service.TriggerSafetyHold()
            return self._json_response(result)

        elif path == "/api/safety/reset" and method == "POST":
            result = await self.service.ResetSafetyGates()
            return self._json_response(result)

        elif path == "/api/chat" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            message = ""
            history = []
            if isinstance(payload, dict):
                message = payload.get("message", "") or payload.get("msg", "")
                history = payload.get("history", []) or []
            result = await self.service.ChatWithGemma(message, history)
            return self._json_response(result)

        elif path == "/api/camera/frame":
            frame = await self.service.GetCameraFrameJPEG()
            if not frame:
                return "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
            header = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n"
                "Cache-Control: no-cache\r\n\r\n"
            )
            return header.encode("utf-8") + frame

        elif path == "/api/camera/stream":
            await self._serve_mjpeg(writer)
            return b""

        # Fallback
        return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"

    async def _serve_static(self, path: str, writer: asyncio.StreamWriter) -> None:
        """Serve static assets from server/static."""
        static_root = Path(__file__).parent / "static"
        relative = path[len("/static/") :]
        file_path = (static_root / relative).resolve()
        if not str(file_path).startswith(str(static_root.resolve())) or not file_path.exists():
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            return

        content = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"
        header = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Cache-Control: max-age=120\r\n"
            f"Content-Length: {len(content)}\r\n\r\n"
        )
        writer.write(header.encode("utf-8") + content)
        await writer.drain()
        writer.close()

    async def _handle_sse(self, writer: asyncio.StreamWriter) -> None:
        """Server-Sent Events stream for status/loop/task updates."""
        try:
            writer.write(
                (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: keep-alive\r\n\r\n"
                ).encode("utf-8")
            )
            await writer.drain()

            while True:
                if writer.is_closing():
                    break

                try:
                    status = await self.service.GetRobotStatus()
                    loops = await self.service.GetLoopHealth()
                    tasks = await self.service.ListTasks(include_ineligible=True)
                    skills = await self.service.ListSkills(include_ineligible=True)

                    payload = {"status": status, "loops": loops, "tasks": tasks, "skills": skills}
                    writer.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                    await writer.drain()
                except Exception as exc:  # noqa: BLE001
                    print(f"SSE error: {exc}")
                    break

                await asyncio.sleep(1.0)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _serve_mjpeg(self, writer: Optional[asyncio.StreamWriter]) -> None:
        """Serve a simple MJPEG stream for camera preview."""
        if writer is None:
            return

        boundary = "frame"
        try:
            writer.write(
                (
                    "HTTP/1.1 200 OK\r\n"
                    f"Content-Type: multipart/x-mixed-replace; boundary={boundary}\r\n"
                    "Cache-Control: no-cache\r\n\r\n"
                ).encode("utf-8")
            )
            await writer.drain()

            while not writer.is_closing():
                frame = await self.service.GetCameraFrameJPEG()
                if frame:
                    part_header = (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n"
                    ).encode("utf-8")
                    writer.write(part_header + frame + b"\r\n")
                    await writer.drain()
                await asyncio.sleep(0.2)
        except Exception as exc:  # noqa: BLE001
            print(f"MJPEG stream error: {exc}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_json_body(self, reader: asyncio.StreamReader, headers: Dict[str, Any]) -> Any:
        content_length = int(headers.get("content-length", 0))
        body = await reader.read(content_length) if content_length > 0 else b""
        if not body:
            return {}
        try:
            return json.loads(body.decode())
        except Exception:
            return {}

    def _get_wiring_stats(self) -> Dict[str, Any]:
        ep_dir = Path("/opt/continuonos/brain/rlds/episodes")
        compact_summary = Path("/opt/continuonos/brain/rlds/compact/compact_summary.json")

        def _count(prefix: str) -> int:
            return sum(1 for p in ep_dir.glob(f"{prefix}*.json") if p.is_file())

        episodes = [p for p in ep_dir.glob("*.json") if p.is_file()]
        summary: Dict[str, Any] = {
            "episodes_total": len(episodes),
            "hope_eval_episodes": _count("hope_eval_"),
            "facts_eval_episodes": _count("facts_eval_"),
            "hardware_mode": "unknown",
        }
        if compact_summary.exists():
            try:
                summary["compact"] = json.loads(compact_summary.read_text())
            except Exception:
                summary["compact"] = {"error": "failed to read compact summary"}
        return summary

    def _json_response(self, payload: Any, status_code: int = 200) -> bytes:
        body = json.dumps(payload)
        status_text = "OK" if status_code < 400 else "Error"
        return (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
            f"{body}"
        ).encode("utf-8")

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
        print("Web UI: http://{0}:{1}/ui".format(addr[0] if addr[0] != '0.0.0.0' else 'localhost', addr[1]))
        print("API Endpoint: http://{0}:{1}/status".format(addr[0] if addr[0] != '0.0.0.0' else 'localhost', addr[1]))

        async with self.server:
            await self.server.serve_forever()

    def get_web_ui_html(self) -> str:
        # Legacy fallback
        tmpl_path = Path(__file__).parent / "templates" / "ui.html"
        if tmpl_path.exists():
            return tmpl_path.read_text(encoding="utf-8")
        if self.ui_provider:
            content = self.ui_provider()
            tmpl_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl_path.write_text(content, encoding="utf-8")
            return content
        return "<html><body><h1>Robot UI</h1></body></html>"

    def get_wiring_html(self) -> str:
        tmpl_path = Path(__file__).parent / "templates" / "wiring.html"
        if tmpl_path.exists():
            return tmpl_path.read_text(encoding="utf-8")
        return "<html><body><h1>Wiring view unavailable.</h1></body></html>"

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

    def get_settings_html(self) -> str:
        tmpl_path = Path(__file__).parent / "templates" / "settings.html"
        if tmpl_path.exists():
            return tmpl_path.read_text(encoding="utf-8")
        return "<html><body><h1>Settings</h1></body></html>"
