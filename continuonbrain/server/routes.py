"""
Simple JSON/HTTP server extracted from robot_api_server.
"""

import asyncio
import json
import mimetypes
import os
import re
import shutil
import time
import zipfile
import hashlib
import random
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
            exports_dir = Path("/opt/continuonos/brain/exports")
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
        elif path == "/api/training/tool_dataset_summary":
            payload = self._read_tool_dataset_summary(query_params)
            return self._json_response(payload)
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
                # Wavecore results may include dataclasses/config objects; serialize defensively.
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
            session_id = None
            if isinstance(payload, dict):
                message = payload.get("message", "") or payload.get("msg", "")
                history = payload.get("history", []) or []
                session_id = payload.get("session_id")
            result = await self.service.ChatWithGemma(message, history, session_id=session_id)
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

        # Fallback
        return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"

    def _build_cloud_readiness(self) -> Dict[str, Any]:
        """
        Lightweight, file-based readiness report for "cloud TPU v1 training" handoff.
        Intentionally offline-first: no uploads are performed here.
        """
        rlds_dir = Path("/opt/continuonos/brain/rlds/episodes")
        tfrecord_dir = Path("/opt/continuonos/brain/rlds/tfrecord")
        seed_export_dir = Path("/opt/continuonos/brain/model/adapters/candidate/core_model_seed")
        seed_manifest = seed_export_dir / "model_manifest.json"
        ckpt_dir = Path("/opt/continuonos/brain/trainer/checkpoints/core_model_seed")
        trainer_status = Path("/opt/continuonos/brain/trainer/status.json")
        proof = Path("/opt/continuonos/brain/proof_of_learning.json")

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
            "zip_episodes": "cd /opt/continuonos/brain && zip -r episodes.zip rlds/episodes",
            "tfrecord_convert": "python -m continuonbrain.jax_models.data.tfrecord_converter --input-dir /opt/continuonos/brain/rlds/episodes --output-dir /opt/continuonos/brain/rlds/tfrecord --compress",
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
        exports_dir = Path("/opt/continuonos/brain/exports")
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
            roots.append(("rlds/episodes", Path("/opt/continuonos/brain/rlds/episodes")))
        if include_tfrecord:
            roots.append(("rlds/tfrecord", Path("/opt/continuonos/brain/rlds/tfrecord")))
        if include_seed:
            roots.append(("model/adapters/candidate/core_model_seed", Path("/opt/continuonos/brain/model/adapters/candidate/core_model_seed")))
        if include_checkpoints:
            roots.append(("trainer/checkpoints/core_model_seed", Path("/opt/continuonos/brain/trainer/checkpoints/core_model_seed")))
        if include_status:
            roots.append(("trainer/status.json", Path("/opt/continuonos/brain/trainer/status.json")))

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
        exports_dir = Path("/opt/continuonos/brain/exports").resolve()
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

    def _read_training_metrics(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read lightweight training metrics for UI visualization (sparklines).

        Sources:
        - /opt/continuonos/brain/trainer/logs/wavecore_{fast,mid,slow}_metrics.json
        """
        log_dir = Path("/opt/continuonos/brain/trainer/logs")
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
        }

    def _read_eval_summary(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize recent eval RLDS episodes so the UI can render "intelligence" indicators.

        Until a formal grader exists, we use defensible heuristics:
        - success_rate: fraction of steps whose answer is non-empty and not an [error:...] stub
        - fallback_rate: fraction of steps with used_fallback=true
        - tier_coverage: distribution of obs.tier
        """
        rlds_dir = Path("/opt/continuonos/brain/rlds/episodes")
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
        rlds_dir = Path("/opt/continuonos/brain/rlds/episodes")
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
        base_dir = Path("/opt/continuonos/brain/rlds/episodes")
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

        incoming_root = Path("/opt/continuonos/brain/model/_incoming")
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

            target_dir = Path("/opt/continuonos/brain/model/adapters/candidate/core_model_seed")
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
            bundles_dir = Path("/opt/continuonos/brain/model/bundles")
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
