"""
Simple JSON/HTTP server extracted from robot_api_server.
"""

import asyncio
import json
import mimetypes
import os
import re
import socket
import shutil
import time
import zipfile
import hashlib
import random
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import jinja2

from continuonbrain.settings_manager import SettingsStore, SettingsValidationError
from continuonbrain.server.tasks import TaskLibraryEntry, TaskSummary
from continuonbrain.server.websocket_handler import WebSocketHandler


class SimpleJSONServer:
    """
    HTTP/JSON server for robot control and web UI.
    Supports both HTTP endpoints and raw JSON protocol.
    """

    # Chat configuration
    CHAT_HISTORY_LIMIT = 50  # Maximum number of chat messages to persist

    def __init__(self, service, ui_provider=None, control_provider=None, state_aggregator=None):
        self.service = service
        self.ui_provider = ui_provider
        self.control_provider = control_provider
        self.server = None
        self.state_aggregator = state_aggregator
        
        # Real-time event queues
        self.cognitive_event_queue = asyncio.Queue(maxsize=100)
        
        # Wire aggregator to queue if provided
        if self.state_aggregator:
            self.state_aggregator.set_event_queue(self.cognitive_event_queue)
        
        # Initialize Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        # Initialize WebSocket handler for real-time bi-directional communication
        self.websocket_handler = WebSocketHandler(service=service)

        # SSE client connections for fallback
        self._sse_clients: list = []

    def _base_dir(self) -> Path:
        """
        Resolve the runtime base directory for offline-first storage.

        Packaged installs run with a configurable config-dir (e.g. /opt/continuonbrain/config).
        Older code used the legacy /opt/continuonos/brain path. Prefer the service's config_dir
        when present to keep UI + learning + RLDS writes consistent in packaged mode.
        """
        try:
            cfg = getattr(self.service, "config_dir", None)
            if cfg:
                return Path(str(cfg))
        except Exception:
            pass
        return Path("/opt/continuonos/brain")

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

    def _best_lan_ip(self) -> str:
        """
        Best-effort LAN IP selection for "device discoverability".

        Motivation: when the UI is opened *on-device* (e.g. `http://localhost:8081/ui`),
        QR codes and deep links must advertise a reachable LAN address (e.g. `192.168.x.y`)
        so phones / Continuon AI can connect.
        """

        # 1) Prefer the system's default route interface (no packets are sent).
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect(("8.8.8.8", 80))
                ip = sock.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
            finally:
                sock.close()
        except Exception:
            pass

        # 2) Fallback to hostname resolution (often works on Pi LANs).
        try:
            _, _, ips = socket.gethostbyname_ex(socket.gethostname())
            for ip in ips or []:
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass

        return "127.0.0.1"

    def _infer_advertise_base_url(
        self,
        *,
        requested_base_url: str = "",
        headers: Dict[str, Any],
        writer: Optional[asyncio.StreamWriter],
    ) -> str:
        """
        Infer a base URL that other devices on the LAN can reach.

        We accept a client-provided `requested_base_url`, but if it looks like localhost
        we replace it with a best-effort LAN IP.
        """

        requested_base_url = str(requested_base_url or "").strip()
        scheme = "http"
        host = ""
        port: Optional[int] = None

        # Prefer explicit client-provided base_url scheme/port if present.
        if requested_base_url:
            try:
                parsed = urlparse(requested_base_url)
                if parsed.scheme:
                    scheme = parsed.scheme
                if parsed.hostname:
                    host = parsed.hostname
                if parsed.port:
                    port = int(parsed.port)
            except Exception:
                pass

        # Use reverse-proxy / direct host headers if they exist (and aren't localhost).
        try:
            hdr_host = (headers.get("x-forwarded-host") or headers.get("host") or "")
            hdr_host = str(hdr_host).split(",", 1)[0].strip()
            if hdr_host:
                # If host header includes port, keep it; otherwise keep host only.
                if ":" in hdr_host and not hdr_host.endswith("]"):
                    host_part, port_part = hdr_host.rsplit(":", 1)
                    if host_part:
                        host = host_part.strip("[]")
                    if port is None:
                        try:
                            port = int(port_part)
                        except Exception:
                            pass
                else:
                    host = hdr_host.strip("[]")
        except Exception:
            pass

        # Port fallback: take the local listening port.
        if port is None:
            try:
                if writer is not None:
                    sockname = writer.get_extra_info("sockname")
                    if isinstance(sockname, (tuple, list)) and len(sockname) >= 2:
                        port = int(sockname[1])
            except Exception:
                port = None
        if port is None:
            port = 8081

        # Scheme can be overridden by proxy headers.
        try:
            xf_proto = str(headers.get("x-forwarded-proto") or "").split(",", 1)[0].strip().lower()
            if xf_proto in ("http", "https"):
                scheme = xf_proto
        except Exception:
            pass

        # If host is missing or localhost-ish, switch to best LAN IP.
        host_lower = (host or "").strip().lower()
        if host_lower in ("", "localhost", "127.0.0.1", "0.0.0.0", "::1", "::"):
            host = self._best_lan_ip()

        return f"{scheme}://{host}:{port}"

    async def handle_http_request(self, request_line: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle HTTP request and return HTML/JSON/SSE response."""
        pass
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

        # WebSocket upgrade for real-time bi-directional communication
        if path in {"/ws", "/ws/events"} and WebSocketHandler.is_websocket_upgrade(headers):
            await self.websocket_handler.handle_upgrade(headers, reader, writer)
            return

        # SSE endpoint (fallback for clients that don't support WebSocket)
        if path == "/api/events":
            await self._handle_sse(writer, headers)
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
        elif path == "/training_proof":
            response_body = self.render_template("training_proof.html", {"active_page": "training_proof"})
            response_bytes = response_body.encode("utf-8")
            return (
                f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode(
                    "utf-8"
                )
                + response_bytes
            )
        elif path == "/api_explorer":
            response_body = self.render_template("api_explorer.html", {"active_page": "api_explorer"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
            
        elif path == "/training":
            response_body = self.render_template("training.html", {"active_page": "training"})
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes

        elif path == "/ui/hope":
            response_body = self.render_template("hope.html", {"active_page": "hope", "hope_section": "training"})
            response_bytes = response_body.encode("utf-8")
            return (
                f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode(
                    "utf-8"
                )
                + response_bytes
            )

        elif path.startswith("/ui/hope/"):
            # Consolidated HOPE monitor: keep legacy URLs but serve the unified template.
            section = (path.split("/")[-1] or "training").strip().lower()
            # Back-compat: some docs reference /ui/hope/map.
            if section == "map":
                section = "dynamics"
            if section not in ("training", "stability", "memory", "dynamics", "performance"):
                section = "training"
            response_body = self.render_template("hope.html", {"active_page": "hope", "hope_section": section})
            response_bytes = response_body.encode("utf-8")
            return (
                f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode(
                    "utf-8"
                )
                + response_bytes
            )
            
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
        elif path == "/pair":
            token = (query_params.get("token") or "")

            # Get base URL for display
            base_url = self._infer_advertise_base_url(headers=headers, writer=writer)
            parsed_url = base_url.replace("http://", "").replace("https://", "")
            host = parsed_url.split("/")[0]  # Remove any path
            is_secure = base_url.startswith("https://")
            is_tunnel = ".ngrok" in base_url or ".trycloudflare" in base_url or ".loca.lt" in base_url

            # Load robot name from settings
            robot_name = "ContinuonBot"
            try:
                settings = SettingsStore(Path(self.service.config_dir)).load()
                identity = settings.get("identity", {}) or {}
                robot_name = identity.get("creator_display_name") or identity.get("robot_name") or robot_name
            except Exception:
                pass

            # Get RCAN RURI if available
            ruri = "rcan://unknown"
            try:
                rcan_info = self.service.rcan.get_discovery_info()
                ruri = rcan_info.get("ruri", ruri)
            except Exception:
                pass

            # Get capabilities
            capabilities = ["arm", "vision", "training", "autonomous", "pairing"]

            response_body = self.render_template("pair_qr.html", {
                "active_page": "",
                "token": token,
                "robot_name": robot_name,
                "ruri": ruri,
                "host": host,
                "secure": is_secure,
                "capabilities": capabilities,
                "is_tunnel": is_tunnel,
            })
            response_bytes = response_body.encode('utf-8')
            return f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(response_bytes)}\r\n\r\n".encode('utf-8') + response_bytes
        elif path == "/status":
            status = await self.service.GetRobotStatus()
            response_body = json.dumps(status, indent=2)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/status":
            status = await self.service.GetRobotStatus()
            # Add discovery endpoint URL for progressive enhancement
            if isinstance(status, dict):
                base_url = self._infer_advertise_base_url(headers=headers, writer=writer)
                status["discovery_url"] = f"{base_url}/api/discovery/info"
            response_body = json.dumps(status)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/realtime" and method == "GET":
            # Real-time connection info endpoint
            base_url = self._infer_advertise_base_url(headers=headers, writer=writer)
            ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
            payload = {
                "status": "ok",
                "websocket": {
                    "available": True,
                    "url": f"{ws_url}/ws/events",
                    "channels": ["status", "training", "cognitive", "chat", "loops", "camera"],
                    "protocol": "continuonbrain-v1",
                },
                "sse": {
                    "available": True,
                    "url": f"{base_url}/api/events",
                    "channels": ["status", "training", "cognitive", "chat", "loops"],
                },
                "connections": {
                    "websocket": self.websocket_handler.get_connection_count(),
                    "sse": len(self._sse_clients),
                },
            }
            response_body = json.dumps(payload)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path in {"/api/discovery", "/api/discovery/info"} and method == "GET":
            base_url = self._infer_advertise_base_url(headers=headers, writer=writer)
            session = None
            try:
                session = self.service.pairing.get_pending()
            except Exception:
                session = None
            
            # Load device identification
            device_id = None
            try:
                device_id_path = Path(self.service.config_dir) / "device_id.json"
                if device_id_path.exists():
                    data = json.loads(device_id_path.read_text())
                    device_id = data.get("device_id")
            except Exception:
                pass
            
            # Load robot name from settings
            robot_name = "ContinuonBot"
            try:
                settings = SettingsStore(Path(self.service.config_dir)).load()
                identity = settings.get("identity", {}) or {}
                robot_name = identity.get("creator_display_name") or identity.get("robot_name") or robot_name
            except Exception:
                pass
            
            # Get pairing status with enhanced info
            pairing_info: Dict[str, Any] = {"pending": False}
            if session:
                now = int(time.time())
                expires_unix_s = getattr(session, "expires_unix_s", None) or (now + 300)
                expires_in_seconds = max(0, expires_unix_s - now)
                # Only expose confirm_code if session is valid and not expired
                confirm_code = None
                if expires_in_seconds > 0:
                    confirm_code = getattr(session, "confirm_code", None)
                pairing_info = {
                    "pending": True,
                    "expires_unix_s": expires_unix_s,
                    "expires_in_seconds": expires_in_seconds,
                    "url": getattr(session, "url", None),
                    "pairing_url": getattr(session, "url", None),  # Alias for clarity
                    "confirm_code": confirm_code,
                }
            
            # Get RCAN info if available
            rcan_info = {}
            try:
                rcan_info = self.service.rcan.get_discovery_info()
            except Exception:
                rcan_info = {}
            
            payload: Dict[str, Any] = {
                "status": "ok",
                "product": "continuon_brain_runtime",
                "device_id": device_id,
                "robot_name": robot_name,
                "version": "0.1.0",
                "capabilities": [
                    "arm_control",
                    "depth_vision",
                    "training_mode",
                    "autonomous_mode",
                    "pairing",
                    "discovery",
                    "rcan",  # RCAN protocol support
                ],
                "base_url": base_url,
                "discovery": {"kind": "lan_http", "via": "continuonbrain/server/routes.py"},
                "rcan": rcan_info,  # RCAN protocol info
                "endpoints": {
                    "status": f"{base_url}/api/status",
                    "mobile_summary": f"{base_url}/api/mobile/summary",
                    "pair_landing": f"{base_url}/pair",
                    "pair_start": f"{base_url}/api/ownership/pair/start",
                    "pair_confirm": f"{base_url}/api/ownership/pair/confirm",
                    "pair_qr_png": f"{base_url}/api/ownership/pair/qr",
                    "discovery": f"{base_url}/api/discovery/info",
                    # RCAN endpoints
                    "rcan_discover": f"{base_url}/rcan/v1/discover",
                    "rcan_status": f"{base_url}/rcan/v1/status",
                    "rcan_claim": f"{base_url}/rcan/v1/auth/claim",
                    "rcan_release": f"{base_url}/rcan/v1/auth/release",
                    "rcan_command": f"{base_url}/rcan/v1/command",
                },
                "pairing": pairing_info,
            }
            return self._json_response(payload)
        elif path == "/api/mobile/summary":
            status = await self.service.GetRobotStatus()
            loops = await self.service.GetLoopHealth()
            tasks = await self.service.ListTasks(include_ineligible=False)

            status_block = status.get("status") if isinstance(status, dict) else {}
            battery = (status_block or {}).get("battery", {}) if isinstance(status_block, dict) else {}
            task_list = tasks.get("tasks") if isinstance(tasks, dict) else None

            payload = {
                "status": status,
                "loops": loops,
                "tasks": {
                    "success": tasks.get("success") if isinstance(tasks, dict) else None,
                    "tasks": (task_list or [])[:5],
                    "selected_task_id": tasks.get("selected_task_id") if isinstance(tasks, dict) else None,
                },
                "mobile": {
                    "mode": (status_block or {}).get("mode") or (status_block or {}).get("robot_mode"),
                    "battery_percent": battery.get("percent"),
                    "selected_task_id": tasks.get("selected_task_id") if isinstance(tasks, dict) else None,
                },
            }

            response_body = json.dumps(payload)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/wiring":
            response_body = json.dumps(self._get_wiring_stats())
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"

        elif path == "/api/system/events":
            # Lightweight UI feed: recent system lifecycle events (jsonl).
            limit_raw = (query_params.get("limit", ["60"]) or ["60"])[0]
            try:
                limit = max(1, min(500, int(limit_raw)))
            except Exception:
                limit = 60
            events_path = self._base_dir() / "logs" / "system_events.jsonl"
            items: list[dict[str, Any]] = []
            if events_path.exists():
                try:
                    lines = events_path.read_text(errors="replace").splitlines()
                    for ln in lines[-limit:]:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            payload = json.loads(ln)
                            if isinstance(payload, dict):
                                items.append(payload)
                        except Exception:
                            items.append({"raw": ln})
                except Exception as exc:  # noqa: BLE001
                    items = [{"error": str(exc)}]
            response_body = json.dumps({"status": "ok", "path": str(events_path), "items": items})
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
            status_path = self._base_dir() / "trainer" / "status.json"
            if status_path.exists():
                response_body = status_path.read_text()
            else:
                response_body = json.dumps({"status": "unknown", "message": "training status file not found"})
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/cloud_readiness":
            payload = self._build_cloud_readiness()
            return self._json_response(payload)
        elif path == "/api/training/export_zip" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = self._build_cloud_export_zip(payload or {})
                return self._json_response({"status": "ok", **result})
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"status": "error", "message": str(exc)}, status_code=500)
        elif path == "/api/training/exports":
            exports_dir = self._base_dir() / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            zips = sorted(exports_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]
            items = []
            for p in zips:
                try:
                    items.append(
                        {
                            "name": p.name,
                            "path": str(p),
                            "size_bytes": p.stat().st_size,
                            "mtime": p.stat().st_mtime,
                            "download_url": f"/api/training/exports/download/{p.name}",
                        }
                    )
                except Exception:
                    continue
            return self._json_response({"status": "ok", "exports_dir": str(exports_dir), "items": items})
        elif path.startswith("/api/training/exports/download/") and method == "GET":
            name = path.split("/")[-1]
            return self._download_export_zip(name)
        elif path == "/api/training/install_bundle" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = self._install_model_bundle(payload or {})
                return self._json_response({"status": "ok", **result})
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"status": "error", "message": str(exc)}, status_code=500)
        elif path == "/api/training/metrics":
            payload = self._read_training_metrics(query_params)
            return self._json_response(payload)
        elif path == "/api/training/eval_summary":
            payload = self._read_eval_summary(query_params)
            return self._json_response(payload)
        elif path == "/api/training/data_quality":
            payload = self._read_data_quality(query_params)
            return self._json_response(payload)
        elif path == "/api/runtime/control_loop" and method == "GET":
            try:
                limit_raw = (query_params.get("limit", ["180"]) or ["180"])[0]
                payload = await self.service.GetControlLoopMetrics({"limit": limit_raw})
                return self._json_response(payload)
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"status": "error", "message": str(exc)}, status_code=500)
        elif path == "/api/training/tool_dataset_summary":
            payload = self._read_tool_dataset_summary(query_params)
            return self._json_response(payload)
        elif path == "/api/training/logs":
            log_dir = self._base_dir() / "trainer" / "logs"
            logs = sorted(log_dir.glob("trainer_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            payload = [{"path": str(p), "mtime": p.stat().st_mtime} for p in logs]
            response_body = json.dumps(payload)
            return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/logs/tail":
            log_dir = (self._base_dir() / "trainer" / "logs").resolve()
            raw_path = (query_params.get("path") or [""])[0]
            raw_lines = (query_params.get("lines") or ["200"])[0]
            try:
                line_count = max(1, min(2000, int(raw_lines)))
            except Exception:
                line_count = 200

            if not raw_path:
                return self._json_response({"status": "error", "message": "path is required"}, status_code=400)

            requested_path = Path(raw_path)
            if not requested_path.is_absolute():
                requested_path = (log_dir / requested_path).resolve()
            try:
                requested_path.relative_to(log_dir)
            except Exception:
                return self._json_response({"status": "error", "message": "requested log is outside trainer/logs"}, status_code=400)

            if not requested_path.exists() or not requested_path.is_file():
                return self._json_response({"status": "error", "message": "log file not found"}, status_code=404)

            try:
                tail_lines = self._tail_file(requested_path, line_count)
                payload = {
                    "status": "ok",
                    "path": str(requested_path),
                    "lines": line_count,
                    "content": "".join(tail_lines),
                }
                return self._json_response(payload)
            except Exception as exc:
                return self._json_response({"status": "error", "message": f"Failed to read log file: {exc}"}, status_code=500)
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
                # Wavecore results may include dataclasses/config objects; serialize defensively.
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/hope_eval" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunHopeEval(payload or {})
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/hope_eval_facts" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunFactsEval(payload or {})
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/tool_router_train" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunToolRouterTrain(payload or {})
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/tool_router_predict" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.ToolRouterPredict(payload or {})
                response_body = json.dumps(result, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/tool_router_eval" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunToolRouterEval(payload or {})
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/chat_learn" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunChatLearn(payload or {})
                response_body = json.dumps({"status": "completed", "result": result}, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/curriculum/lessons" and method == "GET":
            result = await self.service.ListLessons()
            return self._json_response(result)

        elif path == "/api/curriculum/run" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            lesson_id = payload.get("lesson_id")
            if not lesson_id:
                return self._json_response({"status": "error", "message": "lesson_id is required"}, status_code=400)
            # Run in background to not block SSE
            asyncio.create_task(self.service.RunCurriculumLesson(lesson_id))
            return self._json_response({"status": "started", "lesson_id": lesson_id})

        elif path == "/api/imagination/start" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunSymbolicSearch(payload or {})
                response_body = json.dumps(result, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
        elif path == "/api/training/architecture_status" and method == "GET":
            try:
                result = await self.service.GetArchitectureStatus()
                return self._json_response(result)
            except Exception as exc:
                return self._json_response({"status": "error", "message": str(exc)}, status_code=500)
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

        elif path == "/api/training/control/mode" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            mode = payload.get("mode")
            result = await self.service.SetRobotMode(mode)
            return self._json_response(result)

        elif path == "/api/training/jax/start" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            try:
                result = await self.service.RunJaxTraining(payload or {})
                response_body = json.dumps(result, default=str)
                return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            except Exception as exc:
                response_body = json.dumps({"status": "error", "message": str(exc)})
                return f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"

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

        elif path == "/api/admin/factory_reset" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"success": False, "message": "invalid payload"}, status_code=400)

            profile = str(payload.get("profile") or "factory").strip()
            confirm = str(payload.get("confirm") or "").strip()
            token = payload.get("token") or headers.get("x-continuon-admin-token")
            dry_run = bool(payload.get("dry_run", False))

            # Gate on mode and allow_motion
            try:
                status_res = await self.service.GetRobotStatus()
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"success": False, "message": f"status unavailable: {exc}"}, status_code=503)

            status_block = (status_res or {}).get("status") if isinstance(status_res, dict) else {}
            mode = (status_block or {}).get("mode")
            allow_motion = (status_block or {}).get("allow_motion")
            if mode not in ("idle", "emergency_stop") or allow_motion not in (False, None):
                return self._json_response(
                    {
                        "success": False,
                        "message": "reset blocked: requires mode=idle or emergency_stop and allow_motion=false",
                        "mode": mode,
                        "allow_motion": allow_motion,
                    },
                    status_code=409,
                )

            from continuonbrain.services.reset_manager import ResetManager, ResetProfile, ResetRequest

            manager = ResetManager()
            try:
                prof = ResetProfile(profile)
            except Exception:
                return self._json_response({"success": False, "message": f"unknown profile: {profile}"}, status_code=400)

            expected_confirm = manager.CONFIRM_FACTORY if prof == ResetProfile.FACTORY else manager.CONFIRM_MEMORIES
            if confirm != expected_confirm:
                return self._json_response(
                    {"success": False, "message": "confirmation required", "confirm_expected": expected_confirm},
                    status_code=400,
                )

            runtime_root = Path("/opt/continuonos/brain")
            config_dir = Path(getattr(self.service, "config_dir", "")) if getattr(self.service, "config_dir", None) else None
            if not manager.authorize(provided_token=token, runtime_root=runtime_root, config_dir=config_dir):
                return self._json_response(
                    {
                        "success": False,
                        "message": "not authorized (set CONTINUON_ADMIN_TOKEN or CONTINUON_ALLOW_UNSAFE_RESET=1 for dev)",
                    },
                    status_code=403,
                )

            req = ResetRequest(profile=prof, dry_run=dry_run, config_dir=config_dir, runtime_root=runtime_root)
            res = manager.run(req)
            return self._json_response({"success": res.success, "result": res.__dict__}, status_code=200 if res.success else 500)

        elif path == "/api/admin/promote_candidate" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"success": False, "message": "invalid payload"}, status_code=400)

            token = payload.get("token") or headers.get("x-continuon-admin-token")
            dry_run = bool(payload.get("dry_run", False))

            # Gate on mode and allow_motion
            try:
                status_res = await self.service.GetRobotStatus()
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"success": False, "message": f"status unavailable: {exc}"}, status_code=503)

            status_block = (status_res or {}).get("status") if isinstance(status_res, dict) else {}
            mode = (status_block or {}).get("mode")
            allow_motion = (status_block or {}).get("allow_motion")
            if mode not in ("idle", "emergency_stop") or allow_motion not in (False, None):
                return self._json_response(
                    {
                        "success": False,
                        "message": "promotion blocked: requires mode=idle or emergency_stop and allow_motion=false",
                        "mode": mode,
                        "allow_motion": allow_motion,
                    },
                    status_code=409,
                )

            from continuonbrain.services.promotion_manager import PromotionManager

            runtime_root = Path("/opt/continuonos/brain")
            config_dir = Path(getattr(self.service, "config_dir", "")) if getattr(self.service, "config_dir", None) else None
            mgr = PromotionManager()
            if not mgr.authorize(provided_token=token, runtime_root=runtime_root, config_dir=config_dir):
                return self._json_response(
                    {
                        "success": False,
                        "message": "not authorized (set CONTINUON_ADMIN_TOKEN or CONTINUON_ALLOW_UNSAFE_RESET=1 for dev)",
                    },
                    status_code=403,
                )

            res = mgr.promote(runtime_root=runtime_root, dry_run=dry_run)
            return self._json_response({"success": res.success, "result": res.__dict__}, status_code=200 if res.success else 500)

        elif path == "/api/chat" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            message = ""
            history = []
            session_id = None
            model_hint = None
            delegate_model_hint = None
            attach_camera_frame = False
            vision_requested = False
            if isinstance(payload, dict):
                message = payload.get("message", "") or payload.get("msg", "")
                history = payload.get("history", []) or []
                session_id = payload.get("session_id")
                model_hint = payload.get("model_hint")
                delegate_model_hint = payload.get("delegate_model_hint")
                attach_camera_frame = bool(payload.get("attach_camera_frame") or payload.get("use_vision"))
                # Allow requesting vision without forcing camera capture (e.g., client-provided image).
                vision_requested = bool(payload.get("vision_requested", attach_camera_frame))
            image_jpeg = None
            image_source = None
            # Prefer a client-provided image (base64) when present.
            if isinstance(payload, dict) and payload.get("image_jpeg"):
                try:
                    import base64

                    raw = payload.get("image_jpeg")
                    if isinstance(raw, str):
                        image_jpeg = base64.b64decode(raw.encode("utf-8"), validate=False)
                    elif isinstance(raw, (bytes, bytearray)):
                        image_jpeg = bytes(raw)
                    image_source = payload.get("image_source") or "client"
                except Exception:
                    image_jpeg = None
                    image_source = None

            if image_jpeg is None and attach_camera_frame:
                try:
                    image_jpeg = await self.service.GetCameraFrameJPEG()
                except Exception:
                    image_jpeg = None
            result = await self.service.ChatWithGemma(
                message,
                history,
                session_id=session_id,
                model_hint=model_hint,
                delegate_model_hint=delegate_model_hint,
                image_jpeg=image_jpeg,
                image_source=image_source or ("camera" if attach_camera_frame else payload.get("image_source") if isinstance(payload, dict) else None),
                vision_requested=vision_requested,
            )
            return self._json_response(result)

        elif path == "/api/audio/tts" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            result = await self.service.SpeakText(payload or {})
            return self._json_response(result)

        elif path == "/api/audio/record" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            result = await self.service.RecordMicrophone(payload or {})
            return self._json_response(result)

        elif path == "/api/audio/devices" and method == "GET":
            result = await self.service.ListAudioDevices()
            return self._json_response(result)

        elif path == "/api/ownership/status" and method == "GET":
            result = await self.service.GetOwnershipStatus()
            return self._json_response(result)

        elif path == "/api/ownership/pair/start" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                payload = {}
            # Ensure the QR advertises a LAN-reachable base_url even when the UI is opened
            # as `http://localhost:8081/ui` on the robot itself.
            payload["base_url"] = self._infer_advertise_base_url(
                requested_base_url=str(payload.get("base_url") or ""),
                headers=headers,
                writer=writer,
            )
            result = await self.service.StartPairing(payload or {})
            return self._json_response(result)

        elif path == "/api/ownership/pair/confirm" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            result = await self.service.ConfirmPairing(payload or {})
            return self._json_response(result)

        elif path == "/api/ownership/pair/qr" and method == "GET":
            session = None
            try:
                session = self.service.pairing.get_pending()
            except Exception:
                session = None
            if not session:
                return self._json_response({"status": "error", "message": "No pending pairing session"}, status_code=404)
            try:
                proc = subprocess.run(
                    ["qrencode", "-o", "-", "-t", "PNG", "-s", "6", session.url],
                    capture_output=True,
                    timeout=3,
                    check=True,
                )
                png = proc.stdout or b""
                if not png:
                    raise RuntimeError("qrencode produced empty output")
                header = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: image/png\r\n"
                    f"Content-Length: {len(png)}\r\n"
                    "Cache-Control: no-cache\r\n\r\n"
                )
                return header.encode("utf-8") + png
            except Exception as exc:  # noqa: BLE001
                return self._json_response({"status": "error", "message": str(exc)}, status_code=500)

        # =====================================================================
        # RCAN Protocol Endpoints (Robot Communication & Addressing Network)
        # See: docs/rcan-protocol.md, docs/rcan-technical-spec.md
        # =====================================================================
        
        elif path == "/rcan/v1/discover" and method == "POST":
            # Handle discovery request
            payload = await self._read_json_body(reader, headers)
            from continuonbrain.services.rcan_service import RCANMessage
            msg = RCANMessage.from_dict(payload or {})
            response = self.service.rcan.handle_discover(msg)
            return self._json_response(response.to_dict())
        
        elif path == "/rcan/v1/status" and method == "GET":
            # Get RCAN service status
            status = self.service.rcan.get_status()
            # Add discovery info
            status["discovery"] = self.service.rcan.get_discovery_info()
            return self._json_response(status)
        
        elif path == "/rcan/v1/auth/claim" and method == "POST":
            # Claim control of the robot
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"error": "Invalid payload"}, status_code=400)
            
            from continuonbrain.services.rcan_service import RCANMessage, UserRole
            msg = RCANMessage.from_dict({
                "source_ruri": payload.get("source_ruri", ""),
                "target_ruri": self.service.rcan.identity.ruri,
            })
            user_id = payload.get("user_id", "")
            role_str = payload.get("role", "guest").lower()
            role = UserRole[role_str.upper()] if hasattr(UserRole, role_str.upper()) else UserRole.GUEST
            
            response = self.service.rcan.handle_claim(msg, user_id, role)
            return self._json_response(response.to_dict())
        
        elif path == "/rcan/v1/auth/release" and method == "DELETE":
            # Release control
            payload = await self._read_json_body(reader, headers)
            session_id = (payload or {}).get("session_id", "")
            response = self.service.rcan.handle_release(session_id)
            return self._json_response(response.to_dict())
        
        elif path == "/rcan/v1/command" and method == "POST":
            # Send command via RCAN
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"error": "Invalid payload"}, status_code=400)
            
            session_id = payload.get("session_id", "")
            from continuonbrain.services.rcan_service import RCANMessage
            msg = RCANMessage.from_dict(payload.get("message", {}))
            
            response = self.service.rcan.handle_command(msg, session_id)
            return self._json_response(response.to_dict())
        
        elif path == "/rcan/v1/handoff" and method == "POST":
            # Transfer control between users
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"error": "Invalid payload"}, status_code=400)

            from_session_id = payload.get("session_id", "")
            to_user_id = payload.get("target_user_id", "")
            to_role_str = payload.get("target_role", "USER").upper()
            reason = payload.get("reason", "")

            if not from_session_id:
                return self._json_response({"error": "session_id required"}, status_code=400)
            if not to_user_id:
                return self._json_response({"error": "target_user_id required"}, status_code=400)

            # Parse role string to enum
            from continuonbrain.services.rcan_service import UserRole
            try:
                to_role = UserRole[to_role_str]
            except KeyError:
                valid_roles = [r.name for r in UserRole if r != UserRole.UNKNOWN]
                return self._json_response({
                    "error": f"Invalid target_role: {to_role_str}",
                    "valid_roles": valid_roles
                }, status_code=400)

            # Execute handoff
            success, result = self.service.rcan.handle_handoff(
                from_session_id=from_session_id,
                to_user_id=to_user_id,
                to_role=to_role,
                reason=reason,
            )

            if not success:
                return self._json_response(result, status_code=403)

            return self._json_response(result)

        elif path == "/api/planning/arm_search" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                payload = {}
            result = await self.service.PlanArmSearch(payload)
            return self._json_response(result)

        elif path == "/api/planning/arm_execute_delta" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                payload = {}
            result = await self.service.ExecuteArmDelta(payload)
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

        # =====================================================================
        # Feedback API - User feedback on robot responses
        # =====================================================================

        elif path == "/api/feedback" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"error": "Invalid payload"}, status_code=400)

            conversation_id = payload.get("conversation_id")
            if not conversation_id:
                return self._json_response({"error": "conversation_id is required"}, status_code=400)

            is_validated = payload.get("is_validated", False)
            correction = payload.get("correction")
            rating = payload.get("rating")
            tags = payload.get("tags")

            try:
                from continuonbrain.core.feedback_store import SQLiteFeedbackStore
                store = SQLiteFeedbackStore(str(self._base_dir() / "feedback.db"))
                store.initialize_db()
                store.add_feedback(
                    conversation_id=conversation_id,
                    is_validated=is_validated,
                    correction=correction,
                    rating=rating,
                    tags=tags,
                )
                return self._json_response({
                    "status": "ok",
                    "conversation_id": conversation_id,
                    "message": "Feedback recorded",
                })
            except ValueError as e:
                return self._json_response({"error": str(e)}, status_code=400)
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/feedback/summary" and method == "GET":
            try:
                from continuonbrain.core.feedback_store import SQLiteFeedbackStore
                store = SQLiteFeedbackStore(str(self._base_dir() / "feedback.db"))
                store.initialize_db()
                summary = store.get_summary()
                return self._json_response({"status": "ok", **summary})
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/feedback/list" and method == "GET":
            limit = int((query_params.get("limit", ["50"]) or ["50"])[0])
            validated_only = (query_params.get("validated_only", ["false"]) or ["false"])[0].lower() == "true"
            try:
                from continuonbrain.core.feedback_store import SQLiteFeedbackStore
                store = SQLiteFeedbackStore(str(self._base_dir() / "feedback.db"))
                store.initialize_db()
                items = store.list_recent(limit=limit, validated_only=validated_only)
                return self._json_response({"status": "ok", "count": len(items), "items": items})
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path.startswith("/api/feedback/") and method == "GET":
            # GET /api/feedback/{conversation_id}
            conv_id = path.split("/")[-1]
            if not conv_id or conv_id == "feedback":
                return self._json_response({"error": "conversation_id required"}, status_code=400)
            try:
                from continuonbrain.core.feedback_store import SQLiteFeedbackStore
                store = SQLiteFeedbackStore(str(self._base_dir() / "feedback.db"))
                store.initialize_db()
                feedback = store.get_feedback(conv_id)
                if not feedback:
                    return self._json_response({"error": "Feedback not found"}, status_code=404)
                return self._json_response({"status": "ok", "feedback": feedback})
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        # =====================================================================
        # Context Graph API - Knowledge graph queries
        # =====================================================================

        elif path == "/api/graph/nodes" and method == "GET":
            types_raw = query_params.get("types", [""])
            types = [t for t in types_raw[0].split(",") if t] if types_raw[0] else None
            tags_raw = query_params.get("tags", [""])
            tags = [t for t in tags_raw[0].split(",") if t] if tags_raw[0] else None
            limit = int((query_params.get("limit", ["100"]) or ["100"])[0])
            try:
                nodes = self.service.context_store.list_nodes(types=types, limit=limit, tags=tags)
                return self._json_response({
                    "status": "ok",
                    "count": len(nodes),
                    "nodes": [n.__dict__ if hasattr(n, "__dict__") else n for n in nodes],
                })
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path.startswith("/api/graph/nodes/") and method == "GET":
            node_id = path.replace("/api/graph/nodes/", "")
            if not node_id:
                return self._json_response({"error": "node_id required"}, status_code=400)
            try:
                node = self.service.context_store.get_node(node_id)
                if not node:
                    return self._json_response({"error": "Node not found"}, status_code=404)
                return self._json_response({
                    "status": "ok",
                    "node": node.__dict__ if hasattr(node, "__dict__") else node,
                })
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/graph/edges" and method == "GET":
            types_raw = query_params.get("types", [""])
            types = [t for t in types_raw[0].split(",") if t] if types_raw[0] else None
            source_id = (query_params.get("source_id", [""])[0]) or None
            target_id = (query_params.get("target_id", [""])[0]) or None
            min_conf = float((query_params.get("min_confidence", ["0.0"]) or ["0.0"])[0])
            limit = int((query_params.get("limit", ["100"]) or ["100"])[0])
            try:
                edges = self.service.context_store.list_edges(
                    source_ids=[source_id] if source_id else None,
                    target_ids=[target_id] if target_id else None,
                    limit=limit,
                    min_confidence=min_conf,
                    types=types,
                )
                return self._json_response({
                    "status": "ok",
                    "count": len(edges),
                    "edges": [e.__dict__ if hasattr(e, "__dict__") else e for e in edges],
                })
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/graph/subgraph" and method == "GET":
            seeds_raw = query_params.get("seeds", [""])
            seeds = [s for s in seeds_raw[0].split(",") if s] if seeds_raw[0] else []
            session_id = (query_params.get("session_id", [""])[0]) or None
            depth = int((query_params.get("depth", ["2"]) or ["2"])[0])
            min_conf = float((query_params.get("min_confidence", ["0.0"]) or ["0.0"])[0])
            limit = int((query_params.get("limit", ["50"]) or ["50"])[0])
            try:
                result = self.service.get_context_subgraph(
                    session_id=session_id,
                    tags=None,
                    depth=depth,
                    limit=limit,
                    min_confidence=min_conf,
                )
                return self._json_response({"status": "ok", **result})
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/graph/decisions" and method == "GET":
            depth = int((query_params.get("depth", ["2"]) or ["2"])[0])
            limit = int((query_params.get("limit", ["20"]) or ["20"])[0])
            min_conf = float((query_params.get("min_confidence", ["0.0"]) or ["0.0"])[0])
            try:
                result = self.service.get_decision_trace_subgraph(
                    depth=depth,
                    limit=limit,
                    min_confidence=min_conf,
                )
                return self._json_response({"status": "ok", **result})
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/graph/search" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                return self._json_response({"error": "Invalid payload"}, status_code=400)
            query_text = payload.get("query", "")
            limit = payload.get("limit", 10)
            try:
                # Generate embedding for query
                embedding = None
                if hasattr(self.service, "gemma_chat") and self.service.gemma_chat:
                    # Try to get embedding from chat service
                    try:
                        from continuonbrain.services.embedding_gemma import GemmaEmbedding
                        embedder = GemmaEmbedding()
                        embedding = embedder.embed(query_text)
                    except Exception:
                        embedding = None

                if embedding:
                    nodes = self.service.context_store.get_nearest_nodes(embedding, limit=limit)
                    return self._json_response({
                        "status": "ok",
                        "query": query_text,
                        "count": len(nodes),
                        "nodes": [n.__dict__ if hasattr(n, "__dict__") else n for n in nodes],
                    })
                else:
                    return self._json_response({
                        "status": "ok",
                        "query": query_text,
                        "count": 0,
                        "nodes": [],
                        "message": "Semantic search unavailable (no embedding model)",
                    })
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        # =====================================================================
        # Vision API - Detection, segmentation, depth
        # =====================================================================

        elif path == "/api/vision/detect" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                payload = {}
            conf_threshold = payload.get("conf_threshold", 0.25)
            backend = payload.get("backend", "auto")

            try:
                # Get frame from payload or camera
                frame = None
                if payload.get("image_jpeg"):
                    import base64
                    import numpy as np
                    import cv2
                    img_bytes = base64.b64decode(payload["image_jpeg"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Capture from camera
                    if hasattr(self.service, "vision_core") and self.service.vision_core:
                        frame = self.service.vision_core._capture_rgb()

                if frame is None:
                    return self._json_response({"error": "No frame available"}, status_code=503)

                # Run detection
                if hasattr(self.service, "vision_core") and self.service.vision_core:
                    scene = self.service.vision_core.perceive(
                        rgb_frame=frame,
                        run_detection=True,
                        run_segmentation=False,
                    )
                    detections = [
                        {
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "bbox": list(obj.bbox) if hasattr(obj, "bbox") else None,
                            "depth_mm": obj.depth_mm if hasattr(obj, "depth_mm") else None,
                        }
                        for obj in (scene.objects if scene else [])
                    ]
                    return self._json_response({
                        "status": "ok",
                        "detections": detections,
                        "count": len(detections),
                        "inference_time_ms": scene.inference_time_ms if hasattr(scene, "inference_time_ms") else None,
                        "backend": backend,
                    })
                else:
                    return self._json_response({"error": "Vision core not available"}, status_code=503)
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/vision/segment" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                payload = {}
            prompt = payload.get("prompt")
            points = payload.get("points")
            box = payload.get("box")

            try:
                # Get frame from payload or camera
                frame = None
                if payload.get("image_jpeg"):
                    import base64
                    import numpy as np
                    import cv2
                    img_bytes = base64.b64decode(payload["image_jpeg"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    if hasattr(self.service, "vision_core") and self.service.vision_core:
                        frame = self.service.vision_core._capture_rgb()

                if frame is None:
                    return self._json_response({"error": "No frame available"}, status_code=503)

                # Run segmentation
                if hasattr(self.service, "vision_core") and self.service.vision_core:
                    scene = self.service.vision_core.perceive(
                        rgb_frame=frame,
                        run_detection=False,
                        run_segmentation=True,
                        segmentation_prompt=prompt,
                    )
                    segments = [
                        {
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "bbox": list(obj.bbox) if hasattr(obj, "bbox") else None,
                            "mask_shape": list(obj.mask.shape) if hasattr(obj, "mask") and obj.mask is not None else None,
                        }
                        for obj in (scene.objects if scene else [])
                    ]
                    return self._json_response({
                        "status": "ok",
                        "segments": segments,
                        "count": len(segments),
                        "prompt": prompt,
                    })
                else:
                    return self._json_response({"error": "Vision core not available"}, status_code=503)
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/vision/perceive" and method == "POST":
            payload = await self._read_json_body(reader, headers)
            if not isinstance(payload, dict):
                payload = {}
            run_detection = payload.get("run_detection", True)
            run_segmentation = payload.get("run_segmentation", False)
            segmentation_prompt = payload.get("segmentation_prompt")

            try:
                frame = None
                if payload.get("image_jpeg"):
                    import base64
                    import numpy as np
                    import cv2
                    img_bytes = base64.b64decode(payload["image_jpeg"])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if hasattr(self.service, "vision_core") and self.service.vision_core:
                    scene = self.service.vision_core.perceive(
                        rgb_frame=frame,
                        run_detection=run_detection,
                        run_segmentation=run_segmentation,
                        segmentation_prompt=segmentation_prompt,
                    )
                    return self._json_response({
                        "status": "ok",
                        "object_count": scene.object_count if scene else 0,
                        "nearest_object_mm": scene.nearest_object_mm if scene else None,
                        "objects": [
                            {
                                "label": obj.label,
                                "confidence": obj.confidence,
                                "bbox": list(obj.bbox) if hasattr(obj, "bbox") else None,
                                "depth_mm": obj.depth_mm if hasattr(obj, "depth_mm") else None,
                            }
                            for obj in (scene.objects if scene else [])
                        ],
                    })
                else:
                    return self._json_response({"error": "Vision core not available"}, status_code=503)
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/vision/depth" and method == "GET":
            try:
                if hasattr(self.service, "vision_core") and self.service.vision_core:
                    frame_data = self.service.vision_core._capture_rgbd()
                    if frame_data and "depth" in frame_data:
                        import numpy as np
                        depth = frame_data["depth"]
                        return self._json_response({
                            "status": "ok",
                            "shape": list(depth.shape) if isinstance(depth, np.ndarray) else None,
                            "min_mm": float(np.min(depth)) if isinstance(depth, np.ndarray) else None,
                            "max_mm": float(np.max(depth)) if isinstance(depth, np.ndarray) else None,
                            "mean_mm": float(np.mean(depth)) if isinstance(depth, np.ndarray) else None,
                        })
                    else:
                        return self._json_response({"error": "No depth data available"}, status_code=503)
                else:
                    return self._json_response({"error": "Vision core not available"}, status_code=503)
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/vision/capabilities" and method == "GET":
            try:
                if hasattr(self.service, "vision_core") and self.service.vision_core:
                    caps = self.service.vision_core.get_capabilities()
                    return self._json_response({"status": "ok", "capabilities": caps})
                else:
                    return self._json_response({
                        "status": "ok",
                        "capabilities": {
                            "detection": False,
                            "segmentation": False,
                            "depth": False,
                            "message": "Vision core not initialized",
                        }
                    })
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        elif path == "/api/vision/stats" and method == "GET":
            try:
                if hasattr(self.service, "vision_core") and self.service.vision_core:
                    stats = self.service.vision_core.get_pipeline_stats()
                    return self._json_response({"status": "ok", "stats": stats})
                else:
                    return self._json_response({"status": "ok", "stats": {}})
            except Exception as e:
                return self._json_response({"error": str(e)}, status_code=500)

        # Fallback
        return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"

    def _build_cloud_readiness(self) -> Dict[str, Any]:
        """
        Lightweight, file-based readiness report for "cloud TPU v1 training" handoff.
        Intentionally offline-first: no uploads are performed here.
        """
        base_dir = self._base_dir()
        rlds_dir = base_dir / "rlds" / "episodes"
        tfrecord_dir = base_dir / "rlds" / "tfrecord"
        seed_export_dir = base_dir / "model" / "adapters" / "candidate" / "core_model_seed"
        seed_manifest = seed_export_dir / "model_manifest.json"
        ckpt_dir = base_dir / "trainer" / "checkpoints" / "core_model_seed"
        trainer_status = base_dir / "trainer" / "status.json"
        proof = base_dir / "proof_of_learning.json"

        def _count_json(prefix: Optional[str] = None) -> int:
            if not rlds_dir.exists():
                return 0
            if prefix:
                return sum(1 for p in rlds_dir.glob(f"{prefix}*.json") if p.is_file())
            return sum(1 for p in rlds_dir.glob("*.json") if p.is_file())

        episodes_total = _count_json()
        hope_eval = _count_json("hope_eval_")
        facts_eval = _count_json("facts_eval_")

        latest_episode = None
        if rlds_dir.exists():
            eps = sorted([p for p in rlds_dir.glob("*.json") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            if eps:
                latest_episode = {"path": str(eps[0]), "mtime": eps[0].stat().st_mtime}

        tfrecord_files = []
        if tfrecord_dir.exists():
            tfrecord_files = [p for p in tfrecord_dir.rglob("*") if p.is_file()]

        ckpts = []
        if ckpt_dir.exists():
            ckpts = [p for p in ckpt_dir.rglob("*") if p.is_file()]
        ckpts_sorted = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
        latest_ckpt = None
        if ckpts_sorted:
            latest_ckpt = {"path": str(ckpts_sorted[0]), "mtime": ckpts_sorted[0].stat().st_mtime, "size_bytes": ckpts_sorted[0].stat().st_size}

        ready = True
        gates = []

        def gate(name: str, ok: bool, detail: str) -> None:
            nonlocal ready
            gates.append({"name": name, "ok": ok, "detail": detail})
            if not ok:
                ready = False

        gate("episodes_present", episodes_total > 0, f"{episodes_total} episode(s) in {rlds_dir}")
        gate("seed_manifest_present", seed_manifest.exists(), f"manifest at {seed_manifest}" if seed_manifest.exists() else f"missing {seed_manifest}")
        gate("seed_checkpoints_present", len(ckpts) > 0, f"{len(ckpts)} file(s) in {ckpt_dir}" if ckpts else f"missing checkpoints in {ckpt_dir}")

        # Optional-but-helpful evidence signals
        optional = {
            "tfrecord_dir": {"path": str(tfrecord_dir), "exists": tfrecord_dir.exists(), "file_count": len(tfrecord_files)},
            "trainer_status": {"path": str(trainer_status), "exists": trainer_status.exists(), "mtime": trainer_status.stat().st_mtime if trainer_status.exists() else None},
            "proof_of_learning": {"path": str(proof), "exists": proof.exists(), "mtime": proof.stat().st_mtime if proof.exists() else None},
        }

        # High-signal, copyable command suggestions (not executed by the server).
        commands = {
            "zip_episodes": f"cd {base_dir} && zip -r episodes.zip rlds/episodes",
            "tfrecord_convert": f"python -m continuonbrain.jax_models.data.tfrecord_converter --input-dir {base_dir}/rlds/episodes --output-dir {base_dir}/rlds/tfrecord --compress",
            "cloud_tpu_train_template": "python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000",
        }

        return {
            "status": "ok",
            "ready_for_cloud_handoff": ready,
            "gates": gates,
            "rlds": {
                "dir": str(rlds_dir),
                "episodes_total": episodes_total,
                "hope_eval_episodes": hope_eval,
                "facts_eval_episodes": facts_eval,
                "latest_episode": latest_episode,
            },
            "seed": {
                "export_dir": str(seed_export_dir),
                "manifest_path": str(seed_manifest),
                "manifest_exists": seed_manifest.exists(),
                "checkpoint_dir": str(ckpt_dir),
                "checkpoint_file_count": len(ckpts),
                "latest_checkpoint": latest_ckpt,
            },
            "optional": optional,
            "commands": commands,
            "distribution": {
                "options": [
                    {
                        "id": "manual_zip",
                        "label": "Manual zip (download/upload yourself)",
                        "notes": "Build an export zip on-device, copy it to cloud/workstation, then paste the returned bundle URL or path into Install.",
                    },
                    {
                        "id": "signed_edge_bundle",
                        "label": "Signed OTA edge bundle (edge_manifest.json)",
                        "notes": "Preferred for production OTA: signature/checksum gating happens in Continuon AI app + device verifier.",
                    },
                    {
                        "id": "vertex_edge",
                        "label": "Google Vertex AI + Edge distribution (transport)",
                        "notes": "Use Vertex/GCS as the distribution channel; generate a signed URL to a .zip, then install it here (auto-detects bundle type).",
                    },
                ],
                "vertex_templates": {
                    "upload_to_gcs": "gcloud storage cp /opt/continuonos/brain/exports/<EXPORT_ZIP>.zip gs://<BUCKET>/<PREFIX>/<EXPORT_ZIP>.zip",
                    "sign_url_gcloud": "gcloud storage sign-url --duration=1h --private-key-file=<SERVICE_ACCOUNT_KEY.json> gs://<BUCKET>/<PREFIX>/<EXPORT_ZIP>.zip",
                    "sign_url_gsutil_legacy": "gsutil signurl -d 1h <SERVICE_ACCOUNT_KEY.json> gs://<BUCKET>/<PREFIX>/<EXPORT_ZIP>.zip",
                    "vertex_model_registry_hint": "Optional: register the trained bundle metadata in Vertex AI Model Registry for tracking; distribution to devices still uses signed URLs or your OTA pipeline.",
                },
            },
        }

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build_cloud_export_zip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        exports_dir = self._base_dir() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = payload.get("name") or f"cloud_handoff_{ts}.zip"
        # Sanitize: keep it simple and safe.
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        if not name.endswith(".zip"):
            name += ".zip"
        out_path = (exports_dir / name).resolve()
        if not str(out_path).startswith(str(exports_dir.resolve())):
            raise ValueError("Invalid export name")

        include = payload.get("include") if isinstance(payload.get("include"), dict) else {}
        include_episodes = bool(include.get("episodes", True))
        include_tfrecord = bool(include.get("tfrecord", False))
        include_seed = bool(include.get("seed_export", True))
        include_checkpoints = bool(include.get("checkpoints", True))
        include_status = bool(include.get("trainer_status", True))
        episode_limit = payload.get("episode_limit")
        try:
            episode_limit = int(episode_limit) if episode_limit is not None else None
        except Exception:
            episode_limit = None

        roots = []
        if include_episodes:
            roots.append(("rlds/episodes", self._base_dir() / "rlds" / "episodes"))
        if include_tfrecord:
            roots.append(("rlds/tfrecord", self._base_dir() / "rlds" / "tfrecord"))
        if include_seed:
            roots.append(("model/adapters/candidate/core_model_seed", self._base_dir() / "model" / "adapters" / "candidate" / "core_model_seed"))
        if include_checkpoints:
            roots.append(("trainer/checkpoints/core_model_seed", self._base_dir() / "trainer" / "checkpoints" / "core_model_seed"))
        if include_status:
            roots.append(("trainer/status.json", self._base_dir() / "trainer" / "status.json"))

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Always include a small metadata record for provenance.
            meta = {
                "created_at_unix_s": int(time.time()),
                "created_at": ts,
                "includes": {
                    "episodes": include_episodes,
                    "tfrecord": include_tfrecord,
                    "seed_export": include_seed,
                    "checkpoints": include_checkpoints,
                    "trainer_status": include_status,
                },
            }
            zf.writestr("handoff_manifest.json", json.dumps(meta, indent=2))

            for arc_root, src in roots:
                if not src.exists():
                    continue
                if src.is_file():
                    zf.write(src, arcname=arc_root)
                    continue

                files = [p for p in src.rglob("*") if p.is_file()]
                if arc_root == "rlds/episodes" and episode_limit and episode_limit > 0:
                    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:episode_limit]
                for f in files:
                    rel = f.relative_to(src)
                    zf.write(f, arcname=str(Path(arc_root) / rel))

        sha256 = self._sha256_file(out_path)
        return {
            "exports_dir": str(exports_dir),
            "zip_name": out_path.name,
            "zip_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256,
            "download_url": f"/api/training/exports/download/{out_path.name}",
        }

    def _download_export_zip(self, name: str) -> bytes:
        exports_dir = (self._base_dir() / "exports").resolve()
        exports_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        if safe != name or not safe.endswith(".zip"):
            return b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
        path = (exports_dir / safe).resolve()
        if not str(path).startswith(str(exports_dir)) or not path.exists() or not path.is_file():
            return b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        content = path.read_bytes()
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/zip\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Content-Length: {len(content)}\r\n"
            f"Content-Disposition: attachment; filename=\"{safe}\"\r\n\r\n"
        )
        return header.encode("utf-8") + content

    def _tail_file(self, path: Path, line_count: int) -> list[str]:
        """Return the last ``line_count`` lines from ``path`` safely and efficiently."""

        if line_count <= 0:
            return []

        # Efficiently read the last `line_count` lines from the file
        # Read in binary mode to avoid issues with encoding and newlines
        lines = []
        buffer = b""
        chunk_size = 4096
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                pos = file_size
                while pos > 0 and len(lines) <= line_count:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    data = f.read(read_size)
                    buffer = data + buffer
                    # Split lines
                    lines = buffer.splitlines()
            # Return the last `line_count` lines, decoded
            return [line.decode(errors="replace") for line in lines[-line_count:]]
        except Exception:
            # Fallback to original method if any error occurs
            buffer: deque[str] = deque(maxlen=line_count)
            with path.open("r", errors="replace") as handle:
                for line in handle:
                    buffer.append(line)
            return list(buffer)

    def _read_training_metrics(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read lightweight training metrics for UI visualization (sparklines).

        Sources:
        - /opt/continuonos/brain/trainer/logs/wavecore_{fast,mid,slow}_metrics.json
        """
        log_dir = self._base_dir() / "trainer" / "logs"
        limit_raw = (query_params.get("limit", ["120"]) or ["120"])[0]
        try:
            limit = max(10, min(2000, int(limit_raw)))
        except Exception:
            limit = 120

        def read_series(path: Path, *, y_key: str = "loss") -> Dict[str, Any]:
            if not path.exists():
                return {"path": str(path), "exists": False, "points": []}
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, list):
                    return {"path": str(path), "exists": True, "points": []}
                pts = []
                for item in data[-limit:]:
                    if not isinstance(item, dict):
                        continue
                    step = item.get("step")
                    y = item.get(y_key)
                    try:
                        pts.append({"step": int(step), y_key: float(y)})
                    except Exception:
                        continue
                return {"path": str(path), "exists": True, "points": pts, "mtime": path.stat().st_mtime}
            except Exception:
                return {"path": str(path), "exists": True, "points": []}

        return {
            "status": "ok",
            "limit": limit,
            "wavecore": {
                "fast": read_series(log_dir / "wavecore_fast_metrics.json", y_key="loss"),
                "mid": read_series(log_dir / "wavecore_mid_metrics.json", y_key="loss"),
                "slow": read_series(log_dir / "wavecore_slow_metrics.json", y_key="loss"),
            },
            "tool_router": {
                "loss": read_series(log_dir / "tool_router_metrics.json", y_key="loss"),
                "acc": read_series(log_dir / "tool_router_metrics.json", y_key="acc"),
            },
            "tool_router_eval": {
                "top1": read_series(log_dir / "tool_router_eval_metrics.json", y_key="top1"),
                "top5": read_series(log_dir / "tool_router_eval_metrics.json", y_key="top5"),
            },
        }

    def _read_eval_summary(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize recent eval RLDS episodes so the UI can render "intelligence" indicators.

        Until a formal grader exists, we use defensible heuristics:
        - success_rate: fraction of steps whose answer is non-empty and not an [error:...] stub
        - fallback_rate: fraction of steps with used_fallback=true
        - tier_coverage: distribution of obs.tier
        """
        rlds_dir = self._base_dir() / "rlds" / "episodes"
        limit_raw = (query_params.get("limit", ["6"]) or ["6"])[0]
        try:
            limit = max(1, min(30, int(limit_raw)))
        except Exception:
            limit = 6

        def summarize_prefix(prefix: str) -> Dict[str, Any]:
            if not rlds_dir.exists():
                return {"prefix": prefix, "episodes": [], "latest": None}
            files = sorted(rlds_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            episodes = []
            for p in files:
                try:
                    payload = json.loads(p.read_text())
                except Exception:
                    continue
                steps = payload.get("steps", [])
                if not isinstance(steps, list):
                    steps = []
                total = 0
                ok = 0
                fallback = 0
                tiers: Dict[str, int] = {}
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    total += 1
                    obs = s.get("obs") or s.get("observation") or {}
                    action = s.get("action") or {}
                    tier = (obs.get("tier") if isinstance(obs, dict) else None) or "unknown"
                    tiers[str(tier)] = tiers.get(str(tier), 0) + 1
                    ans = action.get("answer") if isinstance(action, dict) else None
                    used_fb = bool(action.get("used_fallback")) if isinstance(action, dict) else False
                    if used_fb:
                        fallback += 1
                    ans_s = str(ans or "")
                    if ans_s and not ans_s.startswith("[error:"):
                        ok += 1
                episodes.append(
                    {
                        "path": str(p),
                        "mtime": p.stat().st_mtime,
                        "steps": total,
                        "ok_steps": ok,
                        "fallback_steps": fallback,
                        "success_rate": (ok / total) if total else None,
                        "fallback_rate": (fallback / total) if total else None,
                        "tiers": tiers,
                    }
                )
            latest = episodes[0] if episodes else None
            return {"prefix": prefix, "episodes": episodes, "latest": latest}

        return {
            "status": "ok",
            "rlds_dir": str(rlds_dir),
            "limit": limit,
            "hope_eval": summarize_prefix("hope_eval"),
            "facts_eval": summarize_prefix("facts_eval"),
            "compare_eval": summarize_prefix("compare_eval"),
            "wiki_learn": summarize_prefix("wiki_learn"),
        }

    def _read_data_quality(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick RLDS JSON "learnability" stats for UI:
        - Are actions present and non-zero?
        - Are observations present with numeric content?
        This helps explain flat loss curves (e.g., constant zeros).
        """
        rlds_dir = self._base_dir() / "rlds" / "episodes"
        limit_raw = (query_params.get("limit", ["30"]) or ["30"])[0]
        step_cap_raw = (query_params.get("step_cap", ["2500"]) or ["2500"])[0]
        try:
            limit = max(1, min(200, int(limit_raw)))
        except Exception:
            limit = 30
        try:
            step_cap = max(200, min(20000, int(step_cap_raw)))
        except Exception:
            step_cap = 2500

        if not rlds_dir.exists():
            return {"status": "ok", "rlds_dir": str(rlds_dir), "episodes_scanned": 0, "steps_scanned": 0}

        files = sorted(rlds_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

        steps_scanned = 0
        action_present = 0
        action_nonzero = 0
        action_abs_sum = 0.0
        action_abs_sum_sq = 0.0
        action_len_total = 0

        obs_present = 0
        obs_numeric_fields = 0
        obs_numeric_scalars = 0

        # Track most common top-level observation keys (coverage)
        obs_key_counts: Dict[str, int] = {}
        episode_kind_counts: Dict[str, int] = {}

        def _flatten_numeric(x: Any) -> list[float]:
            out: list[float] = []
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                out.append(float(x))
                return out
            if isinstance(x, (list, tuple)):
                for v in x:
                    out.extend(_flatten_numeric(v))
            if isinstance(x, dict):
                for v in x.values():
                    out.extend(_flatten_numeric(v))
            return out

        def _action_vec(step: Dict[str, Any]) -> Optional[list[float]]:
            action = step.get("action")
            if not isinstance(action, dict):
                return None
            cmd = action.get("command")
            if isinstance(cmd, (list, tuple)) and cmd:
                try:
                    return [float(v) for v in cmd]
                except Exception:
                    return None
            # Some episodes might use nested action fields; treat as absent for now.
            return None

        for p in files:
            if steps_scanned >= step_cap:
                break
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            steps = payload.get("steps", [])
            if not isinstance(steps, list):
                continue
            # Episode kind heuristic
            kind = "unknown"
            base = p.name
            if base.startswith("hope_eval_") or base.startswith("hope_eval_followup_"):
                kind = "hope_eval"
            elif base.startswith("facts_eval_"):
                kind = "facts_eval"
            elif base.startswith("compare_eval_"):
                kind = "compare_eval"
            elif base.startswith("test_"):
                kind = "test"
            episode_kind_counts[kind] = episode_kind_counts.get(kind, 0) + 1
            for s in steps:
                if steps_scanned >= step_cap:
                    break
                if not isinstance(s, dict):
                    continue
                steps_scanned += 1

                # Action stats
                vec = _action_vec(s)
                if vec is not None:
                    action_present += 1
                    action_len_total += len(vec)
                    abs_sum = sum(abs(v) for v in vec)
                    action_abs_sum += abs_sum
                    action_abs_sum_sq += abs_sum * abs_sum
                    if abs_sum > 1e-9:
                        action_nonzero += 1

                # Observation stats
                obs = s.get("observation")
                if isinstance(obs, dict):
                    obs_present += 1
                    for k in obs.keys():
                        obs_key_counts[k] = obs_key_counts.get(k, 0) + 1
                    # Count numeric payload density (rough)
                    nums = _flatten_numeric(obs)
                    if nums:
                        obs_numeric_fields += 1
                        obs_numeric_scalars += len(nums)

        action_present_rate = (action_present / steps_scanned) if steps_scanned else None
        action_nonzero_rate = (action_nonzero / action_present) if action_present else None
        action_mean_abs_sum = (action_abs_sum / action_present) if action_present else None
        action_std_abs_sum = None
        if action_present and action_mean_abs_sum is not None:
            mean = action_mean_abs_sum
            var = (action_abs_sum_sq / action_present) - (mean * mean)
            if var < 0:
                var = 0.0
            action_std_abs_sum = var ** 0.5

        obs_present_rate = (obs_present / steps_scanned) if steps_scanned else None
        obs_numeric_rate = (obs_numeric_fields / obs_present) if obs_present else None
        obs_avg_numeric_scalars = (obs_numeric_scalars / obs_numeric_fields) if obs_numeric_fields else None

        top_obs_keys = sorted(obs_key_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]

        warnings = []
        if steps_scanned and (action_present == 0):
            warnings.append("No action.command vectors found in sampled steps. WaveCore training will see zero actions (flat loss) unless you filter to trainable episodes.")
        if action_nonzero_rate is not None and action_nonzero_rate < 0.1:
            warnings.append("Most action vectors are near-zero; training signal may be degenerate.")
        if obs_present == 0:
            warnings.append("No observation dicts found in sampled steps. WaveCore training will see empty observations (flat loss) unless you filter to trainable episodes.")
        if obs_numeric_rate is not None and obs_numeric_rate < 0.2:
            warnings.append("Most observations contain little/no numeric content; obs vector may be empty/zero-padded.")

        return {
            "status": "ok",
            "rlds_dir": str(rlds_dir),
            "episodes_scanned": len(files),
            "steps_scanned": steps_scanned,
            "episode_kinds": episode_kind_counts,
            "action": {
                "present_rate": action_present_rate,
                "nonzero_rate": action_nonzero_rate,
                "avg_len": (action_len_total / action_present) if action_present else None,
                "mean_abs_sum": action_mean_abs_sum,
                "std_abs_sum": action_std_abs_sum,
            },
            "observation": {
                "present_rate": obs_present_rate,
                "numeric_rate": obs_numeric_rate,
                "avg_numeric_scalars": obs_avg_numeric_scalars,
                "top_keys": [{"key": k, "count": c} for k, c in top_obs_keys],
            },
            "warnings": warnings,
        }

    def _read_tool_dataset_summary(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize imported tool-use RLDS episodes (e.g., toolchat_hf_toolbench, toolchat_hf_glaive, toolchat_hf).

        Returns counts + tool-call rate + top tool names to help visualize tool-use coverage in the UI.
        """
        base_dir = self._base_dir() / "rlds" / "episodes"
        limit_raw = (query_params.get("limit", ["2000"]) or ["2000"])[0]
        try:
            limit = max(50, min(200000, int(limit_raw)))
        except Exception:
            limit = 2000

        subdirs = []
        if base_dir.exists():
            for p in sorted(base_dir.glob("toolchat_hf*")):
                if p.is_dir():
                    subdirs.append(p)

        def summarize_dir(d: Path) -> Dict[str, Any]:
            files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            episodes = 0
            steps_total = 0
            tool_call_steps = 0
            dataset_ids: Dict[str, int] = {}
            tool_names: Dict[str, int] = {}

            for fp in files:
                try:
                    payload = json.loads(fp.read_text())
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                episodes += 1
                meta = payload.get("metadata") or {}
                dsid = meta.get("dataset_id")
                if isinstance(dsid, str) and dsid:
                    dataset_ids[dsid] = dataset_ids.get(dsid, 0) + 1
                steps = payload.get("steps", [])
                if not isinstance(steps, list):
                    continue
                steps_total += len(steps)
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    action = s.get("action") or {}
                    if not isinstance(action, dict):
                        continue
                    if action.get("type") == "tool_call":
                        tool_call_steps += 1
                        nm = action.get("name")
                        if isinstance(nm, str) and nm:
                            tool_names[nm] = tool_names.get(nm, 0) + 1

            top_tools = sorted(tool_names.items(), key=lambda kv: kv[1], reverse=True)[:20]
            tool_call_rate = (tool_call_steps / steps_total) if steps_total else None
            avg_steps = (steps_total / episodes) if episodes else None
            top_dataset_ids = sorted(dataset_ids.items(), key=lambda kv: kv[1], reverse=True)[:8]

            return {
                "dir": str(d),
                "episodes": episodes,
                "steps_total": steps_total,
                "avg_steps_per_episode": avg_steps,
                "tool_call_steps": tool_call_steps,
                "tool_call_rate": tool_call_rate,
                "top_tools": [{"name": n, "count": c} for n, c in top_tools],
                "dataset_ids": [{"id": i, "episodes": c} for i, c in top_dataset_ids],
            }

        return {
            "status": "ok",
            "base_dir": str(base_dir),
            "limit": limit,
            "sources": [summarize_dir(d) for d in subdirs],
        }

    def _install_model_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Install a returned artifact from "distribution" (manual URL/path).

        Supported kinds:
        - jax_seed_manifest: expects a zip/folder containing model_manifest.json + checkpoint; installs to candidate/core_model_seed
        - edge_bundle: expects a zip/folder containing edge_manifest.json; unpacks into model/bundles/<version or timestamp>
        """
        kind = payload.get("kind") or "jax_seed_manifest"
        source_url = payload.get("source_url")
        source_path = payload.get("source_path")
        if not source_url and not source_path:
            raise ValueError("Provide source_url or source_path")

        incoming_root = self._base_dir() / "model" / "_incoming"
        incoming_root.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        staging = incoming_root / f"incoming_{ts}"
        staging.mkdir(parents=True, exist_ok=True)

        # Fetch/copy into staging as either a file (zip) or a directory.
        local_in = None
        if source_path:
            p = Path(str(source_path)).expanduser()
            if not p.is_absolute():
                raise ValueError("source_path must be absolute")
            if not p.exists():
                raise FileNotFoundError(str(p))
            if p.is_dir():
                local_in = staging / "payload_dir"
                shutil.copytree(p, local_in)
            else:
                local_in = staging / p.name
                shutil.copy2(p, local_in)
        else:
            # URL download (best-effort; keep minimal deps)
            import urllib.request

            dest = staging / "download.bin"
            req = urllib.request.Request(str(source_url), headers={"User-Agent": "continuonbrain-ui/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                with dest.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            local_in = dest

        extracted = staging / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)

        if local_in.is_dir():
            extracted = local_in
        else:
            # If it's a zip, extract; else treat as error.
            if local_in.suffix.lower() != ".zip":
                raise ValueError("Expected a .zip file (or a directory) for install")
            with zipfile.ZipFile(local_in, "r") as zf:
                zf.extractall(extracted)

        # Vertex AI Edge is treated as a *distribution transport*; we auto-detect payload type.
        if kind in {"vertex_edge", "auto"}:
            if list(extracted.rglob("edge_manifest.json")):
                kind = "edge_bundle"
            elif list(extracted.rglob("model_manifest.json")):
                kind = "jax_seed_manifest"
            else:
                raise ValueError("Unable to auto-detect bundle type (expected edge_manifest.json or model_manifest.json)")

        if kind == "jax_seed_manifest":
            manifest_candidates = list(extracted.rglob("model_manifest.json"))
            if not manifest_candidates:
                raise ValueError("model_manifest.json not found in bundle")
            manifest_path = manifest_candidates[0]
            manifest = json.loads(manifest_path.read_text())

            target_dir = self._base_dir() / "model" / "adapters" / "candidate" / "core_model_seed"
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            backup_dir = None
            if target_dir.exists():
                backup_dir = target_dir.parent / f"{target_dir.name}_backup_{ts}"
                shutil.move(str(target_dir), str(backup_dir))
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy entire extracted bundle contents (flatten to target_dir root).
            # Keep it simple: copy files adjacent to manifest and its tree.
            src_root = manifest_path.parent
            for p in src_root.rglob("*"):
                if p.is_dir():
                    continue
                rel = p.relative_to(src_root)
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

            return {
                "kind": kind,
                "installed_to": str(target_dir),
                "backup_dir": str(backup_dir) if backup_dir else None,
                "manifest_path": str(target_dir / "model_manifest.json"),
                "notes": "Installed as candidate JAX seed; model selector should now detect it. (If this came from Vertex AI, it was treated as transport-only.)",
            }

        if kind == "edge_bundle":
            edge_candidates = list(extracted.rglob("edge_manifest.json"))
            if not edge_candidates:
                raise ValueError("edge_manifest.json not found in bundle")
            edge_path = edge_candidates[0]
            edge = json.loads(edge_path.read_text())
            version = str(edge.get("version") or ts)
            version_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", version)
            bundles_dir = self._base_dir() / "model" / "bundles"
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_dir = bundles_dir / version_safe
            if bundle_dir.exists():
                bundle_dir = bundles_dir / f"{version_safe}_{ts}"
            shutil.copytree(edge_path.parent, bundle_dir)
            return {
                "kind": kind,
                "installed_to": str(bundle_dir),
                "edge_manifest": str(bundle_dir / "edge_manifest.json"),
                "notes": "Bundle staged; apply/switch is handled by the runtime/app OTA flow (signature/checksum gating).",
            }

        raise ValueError(f"Unknown install kind: {kind}")

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
                    # Check for pending chat events
                    # We loop quickly to drain queue if multiple messages arrived
                    while True:
                        try:
                            if not self.service.chat_event_queue.empty():
                                chat_evt = self.service.chat_event_queue.get_nowait()
                                writer.write(f"data: {json.dumps({'chat': chat_evt})}\n\n".encode("utf-8"))
                                await writer.drain()
                            else:
                                break
                        except Exception:
                            break

                    # Check for pending cognitive events
                    while not self.cognitive_event_queue.empty():
                        try:
                            cog_evt = self.cognitive_event_queue.get_nowait()
                            writer.write(f"data: {json.dumps({'cognitive': cog_evt})}\n\n".encode("utf-8"))
                            await writer.drain()
                        except asyncio.QueueEmpty:
                            break

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
        base_dir = self._base_dir()
        ep_dir = base_dir / "rlds" / "episodes"
        compact_summary = base_dir / "rlds" / "compact" / "compact_summary.json"

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
        def _default(o: Any):  # noqa: ANN401
            # Make API responses robust to numpy/jax scalars (e.g. float32) and other
            # common non-JSON-native types.
            try:
                import numpy as np  # type: ignore

                if isinstance(o, np.generic):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
            except Exception:
                pass

            # Generic fallbacks (covers some torch/jax types as well).
            try:
                if hasattr(o, "tolist") and callable(getattr(o, "tolist")):
                    return o.tolist()
            except Exception:
                pass
            try:
                if hasattr(o, "item") and callable(getattr(o, "item")):
                    return o.item()
            except Exception:
                pass
            try:
                if isinstance(o, Path):
                    return str(o)
            except Exception:
                pass
            try:
                if isinstance(o, set):
                    return list(o)
            except Exception:
                pass
            return str(o)

        body = json.dumps(payload, default=_default)
        status_text = "OK" if status_code < 400 else "Error"
        return (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
            f"{body}"
        ).encode("utf-8")

    async def _handle_sse(self, writer: asyncio.StreamWriter, headers: Dict[str, Any] = None):
        """
        Handle Server-Sent Events (SSE) connection.

        Provides real-time event streaming for clients that don't support WebSocket.
        Events include: status updates, training progress, cognitive events, chat messages.
        """
        # Send SSE headers
        sse_headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(sse_headers.encode())
        await writer.drain()

        # Add to SSE clients list
        self._sse_clients.append(writer)

        # Send initial connected event
        await self._send_sse_event(writer, "connected", {
            "message": "SSE connection established",
            "channels": ["status", "training", "cognitive", "chat", "loops"],
            "websocket_available": True,
            "websocket_url": "/ws/events",
        })

        try:
            last_status_time = 0
            last_loop_time = 0
            while not writer.is_closing():
                now = time.time()

                # Send status update every 2 seconds
                if now - last_status_time >= 2:
                    try:
                        status = await self.service.GetRobotStatus()
                        await self._send_sse_event(writer, "status", status)
                        last_status_time = now
                    except Exception as e:
                        await self._send_sse_event(writer, "error", {"message": str(e)})

                # Send loop metrics every 1 second
                if now - last_loop_time >= 1:
                    try:
                        loops = await self.service.GetLoopHealth()
                        await self._send_sse_event(writer, "loops", loops)
                        last_loop_time = now
                    except Exception:
                        pass

                # Check cognitive event queue
                try:
                    event = self.cognitive_event_queue.get_nowait()
                    await self._send_sse_event(writer, "cognitive", event)
                except asyncio.QueueEmpty:
                    pass

                # Check chat event queue if available
                if hasattr(self.service, 'chat_event_queue'):
                    try:
                        chat_event = self.service.chat_event_queue.get_nowait()
                        await self._send_sse_event(writer, "chat", chat_event)
                    except asyncio.QueueEmpty:
                        pass

                # Send heartbeat every 15 seconds
                if now % 15 < 0.1:
                    await self._send_sse_event(writer, "heartbeat", {"timestamp": now})

                await asyncio.sleep(0.1)  # 10 Hz poll rate

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"SSE error: {e}")
        finally:
            if writer in self._sse_clients:
                self._sse_clients.remove(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _send_sse_event(self, writer: asyncio.StreamWriter, event_type: str, data: Any):
        """Send a single SSE event."""
        try:
            payload = json.dumps(data, default=str)
            message = f"event: {event_type}\ndata: {payload}\n\n"
            writer.write(message.encode())
            await writer.drain()
        except Exception:
            pass

    async def broadcast_sse_event(self, event_type: str, data: Any):
        """Broadcast an SSE event to all connected SSE clients."""
        dead_clients = []
        for writer in self._sse_clients:
            try:
                if writer.is_closing():
                    dead_clients.append(writer)
                else:
                    await self._send_sse_event(writer, event_type, data)
            except Exception:
                dead_clients.append(writer)

        for client in dead_clients:
            if client in self._sse_clients:
                self._sse_clients.remove(client)

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

    async def start(self, host: str = "0.0.0.0", port: int = 8081):
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
