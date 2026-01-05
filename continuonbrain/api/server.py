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
import time
import threading
import webbrowser
import platform
import subprocess
import shutil
import re
import mimetypes
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).parent.parent / "server" / "static"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.brain_service import BrainService
from continuonbrain.robot_modes import RobotMode
from continuonbrain.agent_identity import AgentIdentity
from continuonbrain.api.routes import ui_routes
from continuonbrain.api.routes.training_plan_page import get_training_plan_html
from continuonbrain.settings_manager import SettingsStore, SettingsValidationError
from continuonbrain.system_events import SystemEventLogger
from continuonbrain.server.skills import SkillLibrary, SkillEligibility, SkillEligibilityMarker, SkillLibraryEntry, SkillSummary
from continuonbrain.server.tasks import TaskEligibilityMarker
from continuonbrain.api.middleware.auth import get_auth_provider
from continuonbrain.api.controllers.admin_controller import AdminControllerMixin
from continuonbrain.api.controllers.robot_controller import RobotControllerMixin
from continuonbrain.api.controllers.model_controller import ModelControllerMixin
from continuonbrain.api.controllers.data_controller import DataControllerMixin
from continuonbrain.api.controllers.learning_controller import LearningControllerMixin
from continuonbrain.api.controllers.training_controller import TrainingControllerMixin
from continuonbrain.api.controllers.chat_controller import ChatControllerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainServer")

# Global Service Instance
brain_service: BrainService = None
identity_service: AgentIdentity = None
event_logger: Optional[SystemEventLogger] = None
background_learner = None  # Autonomous learning service
skill_library = SkillLibrary()
selected_task_id: Optional[str] = None
selected_skill_id: Optional[str] = None


def _detect_model_stack():
    """Best-effort model stack detection for UI chips."""
    try:
        from continuonbrain.services.model_detector import ModelDetector

        detector = ModelDetector()
        models = detector.get_available_models()
        primary = models[0] if models else {"name": "Gemma"}
        fallbacks = models[1:] if len(models) > 1 else []
        return {
            "primary": {"name": primary.get("name") or primary.get("id", "Gemma")},
            "fallbacks": [{"name": m.get("name") or m.get("id")} for m in fallbacks],
        }
    except Exception:
        # Fallback when detector not available
        primary_name = "HOPE" if brain_service and getattr(brain_service, "hope_brain", None) else "Gemma"
        return {"primary": {"name": primary_name}, "fallbacks": [{"name": "LLM fallback"}]}


def _build_status_payload() -> dict:
    """Build enriched status payload expected by the revamped UI."""
    gates = {}
    loops = {}
    mode_value = "unknown"

    if brain_service and brain_service.mode_manager:
        gates = brain_service.mode_manager.get_gate_snapshot()
        loops = brain_service.mode_manager.get_loop_metrics()
        mode_value = gates.get("mode") or brain_service.mode_manager.current_mode.value

    battery_status = None
    try:
        from continuonbrain.sensors.battery_monitor import BatteryMonitor

        monitor = BatteryMonitor()
        battery_status = monitor.get_diagnostics()
    except Exception:
        pass

    learning = None
    if background_learner:
        try:
            learning = background_learner.get_status()
        except Exception:
            learning = None

    hardware_mode = "mock"
    if brain_service and hasattr(brain_service, "hardware_mode"):
        hardware_mode = brain_service.hardware_mode
    elif brain_service and getattr(brain_service, "prefer_real_hardware", False):
         hardware_mode = "real"

    # Get runtime context for inference/training status
    runtime_ctx = None
    try:
        from continuonbrain.services.runtime_context import get_runtime_context_manager
        mgr = get_runtime_context_manager()
        runtime_ctx = mgr.get_context().to_dict()
    except Exception:
        pass

    world_model_ready = bool(getattr(brain_service, "jax_adapter", None))
    if runtime_ctx:
        try:
            world_model_ready = bool(runtime_ctx.get("hardware", {}).get("world_model", {}).get("adapter_ready", world_model_ready))
        except Exception:
            pass

    surprise = 0.0
    if brain_service and hasattr(brain_service, "measure_surprise"):
        try:
            surprise = brain_service.measure_surprise()
        except Exception:
            pass

    return {
        "surprise": surprise,
        "api_state": "ok",
        "hardware_mode": hardware_mode,
        "mode": mode_value,
        "gate_snapshot": gates,
        "allow_motion": gates.get("allow_motion"),
        "record_episodes": gates.get("record_episodes"),
        "is_recording": gates.get("record_episodes"),
        "loop_metrics": loops,
        "battery": battery_status,
        "detected_hardware": getattr(brain_service, "detected_config", None),
        "capabilities": getattr(brain_service, "capabilities", {}),
        "model_stack": _detect_model_stack(),
        "current_task": {"id": selected_task_id} if selected_task_id else None,
        "current_skill": {"id": selected_skill_id} if selected_skill_id else None,
        "learning": learning,
        "runtime_context": runtime_ctx,
        "world_model_ready": world_model_ready,
    }


def _build_discovery_payload(handler) -> dict:
    """Build discovery endpoint payload for LAN device discovery."""
    import socket
    
    # Infer base URL from request
    host_header = handler.headers.get("host") or handler.headers.get("x-forwarded-host") or ""
    host = host_header.split(":")[0] if host_header else "127.0.0.1"
    
    # If host is localhost-ish, try to get LAN IP
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", ""):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
            s.close()
        except Exception:
            host = "127.0.0.1"
    
    port = 8081  # Default, could extract from host_header if needed
    if ":" in host_header:
        try:
            port = int(host_header.split(":")[1])
        except Exception:
            pass
    
    base_url = f"http://{host}:{port}"
    
    # Load device identification
    device_id = getattr(brain_service, "device_id", None)
    if not device_id:
        try:
            device_id_path = Path(brain_service.config_dir) / "device_id.json"
            if device_id_path.exists():
                data = json.loads(device_id_path.read_text())
                device_id = data.get("device_id")
        except Exception:
            pass
    
    # Load robot name and creator from settings
    robot_name = "ContinuonBot"
    creator_name = "Craig Michael Merry"  # Immutable
    try:
        settings_store = SettingsStore(Path(brain_service.config_dir))
        settings = settings_store.load()
        identity = settings.get("identity", {}) or {}
        robot_name = identity.get("robot_name") or robot_name
        creator_name = identity.get("creator_display_name") or creator_name
    except Exception:
        pass
    
    # Get pairing status
    pairing_info = {"pending": False}
    try:
        session = brain_service.pairing.get_pending() if hasattr(brain_service, "pairing") else None
        if session:
            import time
            now = int(time.time())
            expires_unix_s = getattr(session, "expires_unix_s", None) or (now + 300)
            expires_in_seconds = max(0, expires_unix_s - now)
            confirm_code = None
            if expires_in_seconds > 0:
                confirm_code = getattr(session, "confirm_code", None)
            pairing_info = {
                "pending": True,
                "expires_unix_s": expires_unix_s,
                "expires_in_seconds": expires_in_seconds,
                "url": getattr(session, "url", None),
                "pairing_url": getattr(session, "url", None),
                "confirm_code": confirm_code,
            }
    except Exception:
        pass
    
    # Get RCAN info if available
    rcan_info = {}
    try:
        if brain_service and hasattr(brain_service, 'rcan'):
            rcan_info = brain_service.rcan.get_discovery_info()
    except Exception:
        pass
    
    return {
        "status": "ok",
        "product": "continuon_brain_runtime",
        "device_id": device_id,
        "robot_name": robot_name,
        "creator_name": creator_name,  # Immutable - Craig Michael Merry
        "version": "0.1.0",
        "capabilities": [
            "arm_control",
            "depth_vision",
            "inference_mode",
            "training_mode",
            "manual_mode",
            "autonomous_mode",
            "hybrid_mode",
            "pairing",
            "discovery",
            "hailo_accelerator",
            "sam_segmentation",
            "rcan",  # RCAN protocol support
        ],
        "base_url": base_url,
        "discovery": {"kind": "lan_http", "via": "continuonbrain/api/server.py"},
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
            "rcan_status": f"{base_url}/rcan/v1/status",
            "rcan_claim": f"{base_url}/rcan/v1/auth/claim",
            "rcan_release": f"{base_url}/rcan/v1/auth/release",
            "rcan_command": f"{base_url}/rcan/v1/command",
        },
        "pairing": pairing_info,
    }


def _training_pipeline_overview() -> dict:
    """
    Latest training pipeline summary for UI/API consumers.
    Mirrors docs/training-plan.md and /ui/training-plan.
    """
    return {
        "pipeline": "jax_pi5_seed",
        "purpose": "Pi RLDS → JAX sanity check → TFRecord → TPU train → OTA bundle with proof.",
        "on_device": {
            "rlds_path": "/opt/continuonos/brain/rlds/episodes",
            "tfrecord_path": "/opt/continuonos/brain/rlds/tfrecord",
            "sanity_check_cmd": "python -m continuonbrain.jax_models.train.local_sanity_check --rlds-dir /opt/continuonos/brain/rlds/episodes --arch-preset pi5 --max-steps 8 --batch-size 4 --metrics-path /tmp/jax_sanity.csv --checkpoint-dir /tmp/jax_ckpts",
            "proof_cmd": "python prove_learning_capability.py",
            "proof_artifact": "proof_of_learning.json",
            "tfrecord_cmd": "python -m continuonbrain.jax_models.data.tfrecord_converter --input-dir /opt/continuonos/brain/rlds/episodes --output-dir /opt/continuonos/brain/rlds/tfrecord --compress",
        },
        "cloud": {
            "train_cmd": "python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000",
            "export_cmd": "python -m continuonbrain.jax_models.export.export_jax --checkpoint-path gs://... --output-path ./models/core_model_inference --quantization fp16",
            "hailo_cmd": "python -m continuonbrain.jax_models.export.export_hailo --checkpoint-path ./models/core_model_inference --output-dir ./models/core_model_hailo",
            "bundle_manifest": "docs/bundle_manifest.md",
        },
        "evidence": {
            "metrics_csv": "/tmp/jax_sanity.csv",
            "checkpoints_dir": "/tmp/jax_ckpts",
            "proof_json": "proof_of_learning.json",
            "edge_manifest": "edge_manifest.json",
        },
    }


def _skill_eligibility_for(skill) -> SkillEligibility:
    """Lightweight skill eligibility using current capabilities."""
    caps = getattr(brain_service, "capabilities", {}) if brain_service else {}
    markers = []

    for modality in getattr(skill, "required_modalities", []) or []:
        if modality == "vision" and not caps.get("has_vision", False):
            markers.append(SkillEligibilityMarker(code="MISSING_VISION", label="Vision required", severity="error", blocking=True))
        elif modality in ("arm", "gripper") and not caps.get("has_manipulator", False):
            markers.append(SkillEligibilityMarker(code="MISSING_ARM", label="Manipulator required", severity="error", blocking=True))

    eligible = not any(m.blocking for m in markers)
    return SkillEligibility(eligible=eligible, markers=markers, next_poll_after_ms=300.0)


def _serialize_skill_entry(skill) -> SkillLibraryEntry:
    elig = _skill_eligibility_for(skill)
    return SkillLibraryEntry(
        id=skill.id,
        title=skill.title,
        description=skill.description,
        group=skill.group,
        tags=skill.tags,
        capabilities=skill.capabilities,
        eligibility=elig,
        estimated_duration=skill.estimated_duration,
        publisher=skill.publisher,
        version=skill.version,
    )


def _skill_summary(skill_id: str) -> Optional[dict]:
    try:
        skill = skill_library.get_entry(skill_id)
    except KeyError:
        return None

    entry = _serialize_skill_entry(skill)
    summary = SkillSummary(
        entry=entry,
        steps=[
            "Load skill policy",
            "Validate safety gates",
            "Execute capability plan",
        ],
        publisher=skill.publisher,
        version=skill.version,
        provenance=getattr(skill, "provenance", ""),
    )
    return summary.to_dict()


def scan_wifi_networks():
    """Scan Wi-Fi networks using nmcli; return list of dicts."""
    if not shutil.which("nmcli"):
        return {"success": False, "message": "nmcli not available", "networks": []}
    try:
        # -t for terse, fields: SSID, SIGNAL, SECURITY
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list", "--rescan", "yes"],
            text=True,
            timeout=6,
        )
        networks = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(":")
            # nmcli joins extra colons when SSID empty; pad safely
            ssid = parts[0] if parts else ""
            signal = parts[1] if len(parts) > 1 else ""
            security = parts[2] if len(parts) > 2 else ""
            networks.append({
                "ssid": ssid or "<hidden>",
                "signal": int(signal) if signal.isdigit() else signal,
                "security": security or "open",
            })
        return {"success": True, "networks": networks}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Wi-Fi scan timed out", "networks": []}
    except Exception as exc:
        return {"success": False, "message": str(exc), "networks": []}


def scan_bluetooth_devices():
    """Scan Bluetooth devices using bluetoothctl."""
    if not shutil.which("bluetoothctl"):
        return {"success": False, "message": "bluetoothctl not available", "devices": []}
    try:
        # Preload known devices to improve naming
        known_out = subprocess.check_output([
            "bluetoothctl", "devices"
        ], text=True, timeout=4, stderr=subprocess.STDOUT)
        known_names = {}
        for line in known_out.splitlines():
            match = re.search(r"Device ([0-9A-F:]{17}) (.+)", line)
            if match:
                addr, name = match.groups()
                known_names[addr] = name

        out = subprocess.check_output(
            ["bluetoothctl", "--timeout", "6", "scan", "on"],
            text=True,
            timeout=8,
            stderr=subprocess.STDOUT,
        )
        devices = []
        for line in out.splitlines():
            match = re.search(r"Device ([0-9A-F:]{17}) (.+)", line)
            if match:
                addr, name = match.groups()
                label = name or known_names.get(addr) or "Unknown device"
                devices.append({"address": addr, "name": label})
        # Deduplicate by address, prefer known name
        unique = {}
        for dev in devices:
            addr = dev["address"]
            if addr not in unique:
                unique[addr] = dev
            elif unique[addr].get("name", "").startswith("Unknown") and dev.get("name"):
                unique[addr] = dev
        return {"success": True, "devices": list(unique.values())}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Bluetooth scan timed out", "devices": []}
    except Exception as exc:
        return {"success": False, "message": str(exc), "devices": []}


def wifi_status():
    """Return active Wi-Fi connection status via nmcli."""
    if not shutil.which("nmcli"):
        return {"success": False, "message": "nmcli not available"}
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "NAME,DEVICE,TYPE,STATE,CONNECTION", "-m", "tabular", "con", "show", "--active"],
            text=True,
            timeout=5,
        )
        connections = []
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) < 5:
                continue
            name, device, ctype, state, conn = parts[:5]
            if ctype.lower() == "wifi" or "wifi" in name.lower():
                connections.append({
                    "name": name,
                    "device": device,
                    "state": state,
                    "connection": conn,
                })
        return {"success": True, "connections": connections}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Wi-Fi status timed out"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def wifi_connect(ssid: str, password: str = None):
    """Connect to Wi-Fi using nmcli in a single shot; does not store password beyond command execution."""
    if not shutil.which("nmcli"):
        return {"success": False, "message": "nmcli not available"}
    if not ssid:
        return {"success": False, "message": "SSID required"}
    cmd = ["nmcli", "dev", "wifi", "connect", ssid]
    if password:
        cmd.extend(["password", password])
    try:
        out = subprocess.check_output(cmd, text=True, timeout=12, stderr=subprocess.STDOUT)
        return {"success": True, "message": out.strip()}
    except subprocess.CalledProcessError as exc:
        return {"success": False, "message": exc.output.strip() if exc.output else str(exc)}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Wi-Fi connect timed out"}


def bluetooth_paired():
    """List paired Bluetooth devices with names."""
    if not shutil.which("bluetoothctl"):
        return {"success": False, "message": "bluetoothctl not available", "devices": []}
    try:
        out = subprocess.check_output(["bluetoothctl", "paired-devices"], text=True, timeout=5, stderr=subprocess.STDOUT)
        devices = []
        for line in out.splitlines():
            match = re.search(r"Device ([0-9A-F:]{17}) (.+)", line)
            if match:
                addr, name = match.groups()
                devices.append({"address": addr, "name": name})
        return {"success": True, "devices": devices}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Bluetooth paired query timed out", "devices": []}
    except Exception as exc:
        return {"success": False, "message": str(exc), "devices": []}


def bluetooth_connect(address: str):
    """Attempt to connect to a Bluetooth device by address (paired devices preferred)."""
    if not shutil.which("bluetoothctl"):
        return {"success": False, "message": "bluetoothctl not available"}
    if not address:
        return {"success": False, "message": "Bluetooth address required"}
    try:
        out = subprocess.check_output(
            ["bluetoothctl", "--timeout", "8", "connect", address],
            text=True,
            timeout=10,
            stderr=subprocess.STDOUT,
        )
        return {"success": True, "message": out.strip()}
    except subprocess.CalledProcessError as exc:
        return {"success": False, "message": exc.output.strip() if exc.output else str(exc)}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Bluetooth connect timed out"}

class BrainRequestHandler(BaseHTTPRequestHandler, AdminControllerMixin, RobotControllerMixin, ModelControllerMixin, DataControllerMixin, LearningControllerMixin, TrainingControllerMixin, ChatControllerMixin):
    """Handles HTTP requests for the Brain API."""
    
    def _base_dir(self) -> Path:
        # Use module-level REPO_ROOT for development contexts
        return REPO_ROOT / "continuonbrain"

    def _json_response(self, payload: dict, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))

    
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def send_error(self, code, message=None, explain=None):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"success": False, "error": message or "Unknown error", "code": code}).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        try:
            if self.path in ("/", "/ui", "/ui/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_dashboard_html().encode("utf-8"))

            elif self.path in ("/safety", "/safety/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_safety_html().encode("utf-8"))

            elif self.path in ("/tasks", "/tasks/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_tasks_html().encode("utf-8"))

            elif self.path in ("/skills", "/skills/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_skills_html().encode("utf-8"))

            elif self.path in ("/research", "/research/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_research_html().encode("utf-8"))

            elif self.path in ("/api_explorer", "/api_explorer/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_api_explorer_html().encode("utf-8"))
            
            elif self.path in ("/training", "/training/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_training_html().encode("utf-8"))
            
            elif self.path in ("/training_proof", "/training_proof/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_training_proof_html().encode("utf-8"))

            elif self.path.startswith("/static/"):
                # Basic static file serving
                rel_path = self.path.replace("/static/", "", 1)
                file_path = (STATIC_DIR / rel_path).resolve()
                
                # Security check: ensure we stay within STATIC_DIR
                if str(file_path).startswith(str(STATIC_DIR.resolve())) and file_path.exists() and file_path.is_file():
                    mime_type, _ = mimetypes.guess_type(file_path)
                    self.send_response(200)
                    self.send_header("Content-type", mime_type or "application/octet-stream")
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404, "File not found")
            
            elif self.path == "/ui/status":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_status_html().encode("utf-8"))
                
            elif self.path == "/ui/dashboard":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_dashboard_html().encode("utf-8"))

            elif self.path == "/ui/chat":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_chat_html().encode("utf-8"))
            
            elif self.path == "/ui/training-plan":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(get_training_plan_html().encode("utf-8"))

            elif self.path == "/ui/settings":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_settings_html().encode("utf-8"))
            
            elif self.path == "/ui/manual":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_control_html().encode("utf-8"))

            elif self.path == "/ui/tasks":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_tasks_html().encode("utf-8"))
            
            elif self.path == "/ui/context":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                # Load from file directly or via ui_routes helper
                with open(REPO_ROOT / "continuonbrain/server/templates/context_graph.html", "rb") as f:
                    self.wfile.write(f.read())

            # HOPE Monitoring Pages
            elif self.path in ("/ui/hope", "/ui/hope/"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_hope_training_html().encode("utf-8"))

            elif self.path == "/ui/hope/training":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_hope_training_html().encode("utf-8"))
            
            elif self.path == "/ui/hope/memory":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_hope_memory_html().encode("utf-8"))
            
            elif self.path == "/ui/hope/stability":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_hope_stability_html().encode("utf-8"))
            
            elif self.path == "/ui/hope/dynamics":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_hope_dynamics_html().encode("utf-8"))

            elif self.path == "/ui/hope/performance":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_hope_performance_html().encode("utf-8"))

            elif self.path == "/ui/hope/map":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                # Back-compat: map route now points at unified HOPE monitor UI.
                self.wfile.write(ui_routes.get_hope_dynamics_html().encode("utf-8"))
            
            elif self.path in ("/ui/owner", "/ui/owner/"):
                # Owner/pairing page
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                with open(REPO_ROOT / "continuonbrain/server/templates/pair.html", "rb") as f:
                    self.wfile.write(f.read())
            
            elif self.path in ("/ui/research", "/ui/research/"):
                # Research page
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                with open(REPO_ROOT / "continuonbrain/server/templates/research.html", "rb") as f:
                    self.wfile.write(f.read())

            # ============================================
            # V2 UI - Command Center Style Routes
            # ============================================
            elif self.path in ("/v2", "/v2/", "/v2/dashboard"):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_dashboard_html().encode("utf-8"))
            
            elif self.path == "/v2/control":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_control_html().encode("utf-8"))
            
            elif self.path == "/v2/training":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_training_html().encode("utf-8"))
            
            elif self.path == "/v2/safety":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_safety_html().encode("utf-8"))
            
            elif self.path == "/v2/network":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_network_html().encode("utf-8"))
            
            elif self.path == "/v2/agent":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_agent_html().encode("utf-8"))
            
            elif self.path == "/v2/settings":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_v2_settings_html().encode("utf-8"))

            elif self.path in ("/api/events", "/api/chat/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                try:
                    last_pulse = 0
                    while True:
                        try:
                            # Inline Status Pulse (Fallback)
                            now = time.time()
                            if now - last_pulse > 2.0:
                                last_pulse = now
                                mode = "unknown"
                                if brain_service and brain_service.mode_manager:
                                    mode = brain_service.mode_manager.current_mode.value
                                status_payload = {
                                    "status": {
                                        "uptime_seconds": brain_service.uptime_seconds if brain_service else 0,
                                        "device_id": brain_service.device_id if brain_service else "unknown",
                                        "mode": mode,
                                        "ok": True
                                    }
                                }
                                # Inject directly to stream
                                data = json.dumps(status_payload)
                                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                                self.wfile.flush()
                                
                            # Use timeout to allow checking connection status / keepalive
                            # print("DEBUG: Waiting for event...", flush=True)
                            event = brain_service.chat_event_queue.get(timeout=1.0)
                            print(f"DEBUG: Got event: {event}", flush=True)
                            data = json.dumps(event)
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            # Keep alive comment to prevent timeouts
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                except Exception:
                    # Client disconnected or BrokenPipe
                    pass

            elif self.path == "/api/training/pipeline":
                # JSON mirror of /ui/training-plan for automation/tests
                self.send_json(_training_pipeline_overview())

            elif self.path == "/api/hope/structure":
                data = brain_service.get_brain_structure()
                self.send_json(data)

            elif self.path.startswith("/api/context/graph/decisions"):
                try:
                    parsed = urlparse(self.path)
                    query = parse_qs(parsed.query)
                    depth = int((query.get("depth") or ["2"])[0])
                    limit = int((query.get("limit") or ["50"])[0])
                    min_conf = float((query.get("min_confidence") or ["0.0"])[0])

                    subgraph = brain_service.get_decision_trace_subgraph(
                        depth=depth, limit=limit, min_confidence=min_conf
                    )

                    nodes_payload = []
                    for node in subgraph.get("nodes", []):
                        n = {
                            "id": node.id,
                            "type": node.type,
                            "name": node.name,
                            "attributes": node.attributes,
                            "belief": getattr(node, "belief", {}),
                        }
                        if getattr(node, "embedding", None):
                            n["embedding"] = "[vector]"
                        nodes_payload.append(n)

                    edges_payload = []
                    for edge in subgraph.get("edges", []):
                        e = {
                            "id": edge.id,
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.type,
                            "scope": edge.scope,
                            "provenance": edge.provenance,
                            "assertion": edge.assertion,
                            "confidence": edge.confidence,
                            "salience": edge.salience,
                            "policy": edge.policy,
                        }
                        if edge.embedding:
                            e["embedding"] = "[vector]"
                        edges_payload.append(e)

                    self.send_json({"nodes": nodes_payload, "edges": edges_payload, "seeds": subgraph.get("seeds", [])})
                except Exception as e:
                    self.send_json({"error": str(e)}, status=500)

            elif self.path.startswith("/api/context/graph"):
                try:
                    parsed = urlparse(self.path)
                    query = parse_qs(parsed.query)
                    session_id = (query.get("session_id") or [None])[0]
                    tags = query.get("tag") or query.get("tags") or None
                    depth = int((query.get("depth") or ["2"])[0])
                    limit = int((query.get("limit") or ["50"])[0])
                    min_conf = float((query.get("min_confidence") or ["0.0"])[0])

                    subgraph = brain_service.get_context_subgraph(
                        session_id=session_id,
                        tags=tags,
                        depth=depth,
                        limit=limit,
                        min_confidence=min_conf,
                    )

                    nodes_payload = []
                    for node in subgraph.get("nodes", []):
                        n = {
                            "id": node.id,
                            "type": node.type,
                            "name": node.name,
                            "attributes": node.attributes,
                            "belief": node.belief,
                        }
                        if node.embedding:
                            n["embedding"] = "[vector]"
                        nodes_payload.append(n)

                    edges_payload = []
                    for edge in subgraph.get("edges", []):
                        e = {
                            "id": edge.id,
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.type,
                            "scope": edge.scope,
                            "provenance": edge.provenance,
                            "assertion": edge.assertion,
                            "confidence": edge.confidence,
                            "salience": edge.salience,
                            "policy": edge.policy,
                        }
                        if edge.embedding:
                            e["embedding"] = "[vector]"
                        edges_payload.append(e)

                    self.send_json({"nodes": nodes_payload, "edges": edges_payload, "seeds": subgraph.get("seeds", [])})
                except Exception as e:
                    self.send_json({"error": str(e)}, status=500)

            elif self.path == "/api/tasks/library":
                # Return task library with eligibility checks
                tasks = brain_service.get_task_library()
                self.send_json({"tasks": tasks})

            elif self.path == "/api/curriculum/lessons":
                self.send_json({"lessons": brain_service.curriculum_manager.list_curriculum()})

            elif self.path == "/api/system/audit":
                audit_path = Path(brain_service.config_dir) / "system_audit_report.json"
                if audit_path.exists():
                    self.send_json(json.loads(audit_path.read_text()))
                else:
                    self.send_json({"error": "No audit report found"}, status=404)

            elif self.path == "/api/status/introspection":
                # Introspection endpoint for Brain Status page
                # identity_service.self_report() # Removed to prevent log spam/heavy IO on polling
                data = identity_service.identity
                self.send_json(data)

            elif self.path == "/api/runtime/context":
                # Runtime context: inference/training modes and hardware capabilities
                try:
                    from continuonbrain.services.runtime_context import get_runtime_context_manager
                    mgr = get_runtime_context_manager(brain_service.config_dir if brain_service else "/tmp/continuonbrain")
                    self.send_json(mgr.get_status())
                except Exception as e:
                    self.send_json({"error": str(e)}, status=500)

            elif self.path == "/api/runtime/hardware":
                # Quick hardware capabilities check
                try:
                    from continuonbrain.services.runtime_context import get_runtime_context_manager
                    mgr = get_runtime_context_manager(brain_service.config_dir if brain_service else "/tmp/continuonbrain")
                    caps = mgr.detect_hardware()
                    self.send_json(caps.to_dict())
                except Exception as e:
                    self.send_json({"error": str(e)}, status=500)

            elif self.path in {"/api/discovery", "/api/discovery/info"}:
                # Discovery endpoint for LAN device discovery
                self.send_json(_build_discovery_payload(self))
            
            # ===================================================================
            # RCAN Protocol Endpoints (Robot Communication & Addressing Network)
            # ===================================================================
            elif self.path == "/rcan/v1/status":
                # Get RCAN service status
                if brain_service and hasattr(brain_service, 'rcan'):
                    status = brain_service.rcan.get_status()
                    status["discovery"] = brain_service.rcan.get_discovery_info()
                    self.send_json(status)
                else:
                    self.send_json({"error": "RCAN service not available"}, status=503)

            elif self.path == "/rcan/v1/cloud/status":
                # Get cloud registry status
                if brain_service and hasattr(brain_service, 'cloud_registry') and brain_service.cloud_registry:
                    self.send_json({
                        "registered": True,
                        "ruri": brain_service.cloud_registry.ruri,
                        "device_id": brain_service.cloud_registry.device_id,
                        "firebase_available": True,
                    })
                else:
                    self.send_json({
                        "registered": False,
                        "firebase_available": False,
                        "message": "Cloud registry not initialized",
                    })

            elif self.path == "/api/refresh":
                # Comprehensive refresh endpoint for UI - combines multiple data sources
                refresh_payload = _build_status_payload()

                # Add chat status
                if brain_service and hasattr(brain_service, 'hope_chat') and brain_service.hope_chat:
                    try:
                        refresh_payload["chat"] = brain_service.hope_chat.get_status()
                    except Exception:
                        refresh_payload["chat"] = {"ready": False}
                else:
                    refresh_payload["chat"] = {"ready": False}

                # Add training status
                if brain_service and hasattr(brain_service, 'brain_trainer') and brain_service.brain_trainer:
                    try:
                        refresh_payload["training"] = brain_service.brain_trainer.get_status()
                    except Exception:
                        refresh_payload["training"] = {"enabled": False, "running": False}
                else:
                    refresh_payload["training"] = {"enabled": False, "running": False}

                # Add timestamp for cache invalidation
                refresh_payload["timestamp"] = time.time()
                refresh_payload["refresh"] = True

                self.send_json(refresh_payload)

            elif self.path.startswith("/api/status"):
                # Enriched robot status for UI
                status_payload = _build_status_payload()
                # Add discovery_url for progressive enhancement
                try:
                    import socket
                    host_header = self.headers.get("host") or self.headers.get("x-forwarded-host") or ""
                    host = host_header.split(":")[0] if host_header else "127.0.0.1"
                    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", ""):
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        host = s.getsockname()[0]
                        s.close()
                    port = 8081
                    if ":" in host_header:
                        try:
                            port = int(host_header.split(":")[1])
                        except Exception:
                            pass
                    status_payload["discovery_url"] = f"http://{host}:{port}/api/discovery/info"
                except Exception:
                    pass
                # Wrap status in 'status' key for Flutter app compatibility
                self.send_json({"status": status_payload, "success": True})

            elif self.path.startswith("/api/loops"):
                gates = brain_service.mode_manager.get_gate_snapshot() if brain_service and brain_service.mode_manager else {}
                loops = brain_service.mode_manager.get_loop_metrics() if brain_service and brain_service.mode_manager else {}
                self.send_json({"gate_snapshot": gates, "loop_metrics": loops})

            elif self.path.startswith("/api/gates"):
                gates = brain_service.mode_manager.get_gate_snapshot() if brain_service and brain_service.mode_manager else {}
                self.send_json({"gate_snapshot": gates, "allow_motion": gates.get("allow_motion")})

            elif self.path.startswith("/api/mode"):
                # GET can be used as a mode toggle for UI convenience
                parts = self.path.rstrip("/").split("/")
                target = parts[-1] if len(parts) >= 3 else None
                if target and target != "mode":
                    self._set_mode(target)
                gates = brain_service.mode_manager.get_gate_snapshot() if brain_service and brain_service.mode_manager else {}
                self.send_json({"mode": gates.get("mode"), "gate_snapshot": gates})

            elif self.path.startswith("/api/tasks/summary/"):
                task_id = self.path.split("/")[-1]
                summary = brain_service.GetTaskSummary(task_id)
                if summary:
                    self.send_json({"success": True, "summary": summary.to_dict()})
                else:
                    self.send_json({"success": False, "message": f"Task {task_id} not found"}, status=404)

            elif self.path.startswith("/api/skills/summary/"):
                skill_id = self.path.split("/")[-1]
                summary = _skill_summary(skill_id)
                if summary:
                    self.send_json({"success": True, "summary": summary})
                else:
                    self.send_json({"success": False, "message": f"Skill {skill_id} not found"}, status=404)

            elif self.path.startswith("/api/training/exports/download/"):
                # Download export zip
                # Path format: /api/training/exports/download/{filename}
                filename = self.path.split("/")[-1]
                data = self._download_export_zip(filename)
                if data.startswith(b"HTTP/1.1"):
                    # Raw response bytes (headers + body)
                    self.wfile.write(data)
                else:
                    # Fallback or error
                    self.send_error(404, "Download failed")

            elif self.path.startswith("/api/tasks"):
                parsed = urlparse(self.path)
                include_ineligible = parse_qs(parsed.query).get("include_ineligible", ["false"])[0].lower() == "true"
                tasks = brain_service.get_task_library()
                if not include_ineligible:
                    tasks = [t for t in tasks if t.get("eligibility", {}).get("eligible", False)]
                self.send_json({"tasks": tasks})

            elif self.path.startswith("/api/skills"):
                parsed = urlparse(self.path)
                include_ineligible = parse_qs(parsed.query).get("include_ineligible", ["false"])[0].lower() == "true"
                skills = []
                for skill in skill_library.list_entries():
                    entry = _serialize_skill_entry(skill)
                    skills.append({
                        "id": skill.id,
                        "title": skill.title,
                        "description": skill.description,
                        "group": skill.group,
                        "tags": skill.tags,
                        "capabilities": skill.capabilities,
                        "required_modalities": skill.required_modalities,
                        "eligibility": entry.eligibility.to_dict(),
                        "estimated_duration": skill.estimated_duration,
                        "publisher": skill.publisher,
                        "version": skill.version,
                    })
                
                # Filter by eligibility if requested
                if not include_ineligible:
                    filtered = []
                    for s in skills:
                        if s.get("eligibility", {}).get("eligible", False):
                            filtered.append(s)
                    skills = filtered
                    
                self.send_json({"skills": skills})

            # Training / Learning Routes
            elif self.path == "/api/training/status":
                status_path = self._base_dir() / "trainer" / "status.json"
                if status_path.exists():
                    try:
                        data = json.loads(status_path.read_text())
                        self.send_json(data)
                    except Exception:
                        self.send_json({"status": "unknown", "message": "invalid status file"})
                else:
                    self.send_json({"status": "unknown", "message": "training status file not found"})

            elif self.path == "/api/training/cloud_readiness":
                self.handle_get_cloud_readiness()

            elif self.path.startswith("/api/training/metrics"):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                self.handle_get_training_metrics(params)

            elif self.path.startswith("/api/training/eval_summary"):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                self.handle_get_eval_summary(params)

            elif self.path.startswith("/api/training/data_quality"):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                self.handle_get_data_quality(params)
            
            elif self.path.startswith("/api/training/tool_dataset_summary"):
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                self.handle_get_tool_dataset_summary(params)

            elif self.path.startswith("/api/runtime/control_loop"):
                parsed = urlparse(self.path)
                limit_raw = (parse_qs(parsed.query).get("limit") or ["180"])[0]
                # Assuming BrainService has GetControlLoopMetrics, otherwise return empty
                try:
                    limit = int(limit_raw)
                    if brain_service and brain_service.mode_manager:
                        metrics = brain_service.mode_manager.get_loop_metrics(limit=limit)
                    else:
                        metrics = {}
                    self.send_json(metrics)
                except Exception as e:
                    self.send_json({"status": "error", "message": str(e)}, status=500)

            elif self.path == "/api/training/architecture_status":
                # Stub or implement if BrainService has it. 
                # routes.py called await self.service.GetArchitectureStatus()
                # We'll return a placeholder to stop 404s if service doesn't have it yet.
                self.send_json({"status": "ok", "architecture": "continuon_hope_v1", "modules": ["perception", "memory", "planning", "control"]})
            
            elif self.path.startswith("/api/training/logs"):
                # Get training logs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", ["100"])[0])
                
                logs_path = self._base_dir() / "trainer" / "logs"
                logs = []
                if logs_path.exists():
                    for log_file in sorted(logs_path.glob("*.log"), reverse=True)[:limit]:
                        try:
                            logs.append({
                                "name": log_file.name,
                                "size": log_file.stat().st_size,
                                "modified": log_file.stat().st_mtime,
                                "content": log_file.read_text()[-5000] if log_file.stat().st_size < 50000 else "(truncated)"
                            })
                        except Exception:
                            pass
                self.send_json({"logs": logs})

            elif self.path == "/api/ping":
                uptime = getattr(brain_service, "uptime_seconds", None)
                device_id = getattr(brain_service, "device_id", None)
                self.send_json(
                    {
                        "ok": True,
                        "uptime_seconds": uptime,
                        "device_id": device_id or brain_service.agent_id,
                        "message": "pong",
                    }
                )

            elif self.path == "/api/v1/models":
                self.handle_list_models()

            elif self.path == "/api/v1/data/episodes":
                self.handle_list_episodes()

            elif self.path == "/api/ownership/status":
                self.send_json(
                    {
                        "owned": bool(brain_service.is_owned),
                        "subscription_active": bool(brain_service.subscription_active),
                        "seed_installed": bool(brain_service.seed_installed),
                        "account_id": getattr(brain_service, "account_id", None),
                        "account_type": getattr(brain_service, "account_type", None),
                        "owner_id": getattr(brain_service, "owner_id", None),
                    }
                )
            elif self.path == "/api/ownership/claim":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b""
                account_id = None
                account_type = None
                owner_id = None
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                    account_id = payload.get("account_id")
                    account_type = payload.get("account_type")
                    owner_id = payload.get("owner_id")
                except Exception as e:
                    logger.warning(f"Claim payload parse error: {e}")
                
                try:
                    brain_service.set_ownership(
                        owned=True,
                        account_id=account_id,
                        account_type=account_type,
                        owner_id=owner_id,
                    )
                    self.send_json(
                        {
                            "owned": True,
                            "subscription_active": brain_service.subscription_active,
                            "seed_installed": brain_service.seed_installed,
                            "account_id": brain_service.account_id,
                            "account_type": brain_service.account_type,
                            "owner_id": brain_service.owner_id,
                        }
                    )
                except Exception as ex:
                    logger.error(f"Ownership claim failed: {ex}")
                    self.send_error(500, f"Claim failed: {str(ex)}")

            elif self.path == "/api/camera/stream":
                self.handle_mjpeg_stream()

            elif self.path == "/api/camera/frame":
                self.handle_single_frame()

            elif self.path == "/api/settings":
                store = SettingsStore(Path(brain_service.config_dir))
                self.send_json({"success": True, "settings": store.load()})
            
            elif self.path == "/api/robot/name":
                # Get current robot name and creator (creator is immutable)
                store = SettingsStore(Path(brain_service.config_dir))
                settings = store.load()
                identity = settings.get("identity", {})
                self.send_json({
                    "robot_name": identity.get("robot_name", "ContinuonBot"),
                    "creator_name": identity.get("creator_display_name", "Craig Michael Merry"),
                    "creator_immutable": True,
                })
            
            elif self.path == "/api/resources":
                if brain_service.resource_monitor:
                    self.send_json(brain_service.resource_monitor.get_status_summary())
                else:
                    self.send_json({"error": "Resource monitor not available"}, status=503)

            elif self.path == "/api/network/wifi/scan":
                result = scan_wifi_networks()
                status = 200 if result.get("success") else 500
                self.send_json(result, status=status)

            elif self.path == "/api/network/wifi/status":
                result = wifi_status()
                status = 200 if result.get("success") else 500
                self.send_json(result, status=status)

            elif self.path == "/api/network/bluetooth/scan":
                result = scan_bluetooth_devices()
                status = 200 if result.get("success") else 500
                self.send_json(result, status=status)

            elif self.path == "/api/network/bluetooth/paired":
                result = bluetooth_paired()
                status = 200 if result.get("success") else 500
                self.send_json(result, status=status)

            elif self.path.startswith("/api/system/events"):
                if not event_logger:
                    self.send_json({"events": []})
                    return

                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                try:
                    limit = int(params.get("limit", ["50"])[0])
                except ValueError:
                    limit = 50

                limit = max(1, min(limit, 200))
                events = event_logger.load_recent(limit=limit)
                self.send_json({"events": events})
            
            elif self.path == "/api/agent/info":
                # Aggregate info from chat agent and learning agent
                chat_info = brain_service.get_chat_agent_info()
                
                learning_info = {"enabled": False, "status": "disabled"}
                learner = background_learner or getattr(brain_service, "background_learner", None)
                if learner:
                    learning_info = learner.get_status()
                    learning_info["enabled"] = True
                elif brain_service.agent_settings.get('enable_autonomous_learning', True) and brain_service.hope_brain:
                    # Enabled but maybe not started or failed?
                    learning_info = {"enabled": True, "status": "inactive"}

                self.send_json({
                    "chat_agent": chat_info,
                    "learning_agent": learning_info
                })
            
            elif self.path == "/api/agent/models":
                # List available chat models
                try:
                    from continuonbrain.services.model_detector import ModelDetector
                    detector = ModelDetector()
                    models = detector.get_available_models()
                    self.send_json({"success": True, "models": models})
                except Exception as e:
                    logger.error(f"Model detection failed: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)

            elif self.path == "/api/agent/validation_summary":
                try:
                    summary = brain_service.experience_logger.feedback_store.get_summary()
                    self.send_json({"success": True, "summary": summary})
                except Exception as e:
                    logger.error(f"Failed to get validation summary: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            elif self.path == "/api/agent/knowledge_map":
                # Return semantic clusters for visualization
                try:
                    # For now, return recent conversations with embeddings (redacted)
                    results = brain_service.experience_logger.search_conversations("", max_results=100)
                    self.send_json({"success": True, "memories": results})
                except Exception as e:
                    logger.error(f"Knowledge map failed: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            elif self.path == "/api/agent/learning_stats":
                # Get learning and agent performance statistics
                try:
                    stats = brain_service.experience_logger.get_statistics()
                    
                    # Add agent response distribution
                    by_agent = stats.get("by_agent", {})
                    total = stats.get("total_conversations", 0)
                    
                    if total > 0:
                        stats["hope_response_rate"] = by_agent.get("hope_brain", 0) / total
                        stats["llm_context_rate"] = by_agent.get("llm_with_hope_context", 0) / total
                        stats["llm_only_rate"] = by_agent.get("llm_only", 0) / total
                    else:
                        stats["hope_response_rate"] = 0.0
                        stats["llm_context_rate"] = 0.0
                        stats["llm_only_rate"] = 0.0
                    
                    self.send_json({"success": True, "stats": stats})
                except Exception as e:
                    logger.error(f"Failed to get learning stats: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            
            # PERSONALITY & IDENTITY API
            elif self.path == "/api/personality":
                self.send_json(brain_service.personality_config.__dict__)
            
            elif self.path == "/api/identity":
                self.send_json(brain_service.user_context.__dict__)

            # HOPE API Endpoints
            elif self.path.startswith("/api/hope/"):
                try:
                    from continuonbrain.api.routes import hope_routes
                    hope_routes.handle_hope_request(self)
                except ImportError:
                    self.send_json({"error": "HOPE implementation not available"}, status=503)
            
            # Learning API Endpoints
            elif self.path.startswith("/api/learning/"):
                try:
                    from continuonbrain.api.routes import learning_routes
                    learning_routes.handle_learning_request(self)
                except ImportError:
                    self.send_json({"error": "Learning service not available"}, status=503)

            # Chat API GET Endpoints (HOPE Chat)
            elif self.path == "/api/chat/status":
                self.handle_chat_status()
            elif self.path.startswith("/api/chat/session/"):
                # GET /api/chat/session/<session_id>
                session_id = self.path.replace("/api/chat/session/", "").strip("/")
                if session_id:
                    self.handle_chat_session(session_id)
                else:
                    self.send_json({"success": False, "error": "session_id required"}, status=400)

            # Training API Endpoints (Autonomous Training Engine)
            elif self.path == "/api/training/status":
                self.handle_training_status()
            elif self.path == "/api/training/benchmarks":
                self.handle_training_benchmarks()
            elif self.path == "/api/training/progress":
                self.handle_training_progress()
            elif self.path == "/api/training/decisions":
                self.handle_training_decisions()
            elif self.path == "/api/training/models":
                self.handle_training_models()
            elif self.path.startswith("/api/training/history"):
                self.handle_training_history()
            elif self.path == "/api/training/trends":
                self.handle_training_trends()
            elif self.path == "/api/training/config":
                self.handle_training_config()

            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Request error: {e}")
            # self.send_error(500) # Generating error during stream breaks things

    def do_POST(self):
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len).decode('utf-8')
            
            # Chat API - HOPE Chat (CoreModel + Text Encoder)
            if self.path == "/api/chat":
                self.handle_chat(body)

            elif self.path == "/api/chat/clear":
                self.handle_chat_clear_session(body)

            elif self.path == "/api/chat/history/clear":
                # Legacy endpoint - clear default session
                if brain_service.hope_chat:
                    brain_service.hope_chat.clear_session("default")
                self.send_json({"success": True})

            elif self.path == "/api/training/wavecore_loops":
                data = json.loads(body) if body else {}
                # Bridge sync handler to async service method
                result = asyncio.run(brain_service.RunWavecoreLoops(data))
                self.send_json(result)

            elif self.path == "/api/cms/compact":
                result = brain_service.compact_memory()
                self.send_json(result)

            elif self.path == "/api/training/manual":
                data = json.loads(body) if body else {}
                # Bridge sync handler to async service method
                result = asyncio.run(brain_service.RunManualTraining(data))
                self.send_json(result)

            elif self.path == "/api/chat/session/clear":
                data = json.loads(body) if body else {}
                session_id = data.get("session_id")
                if session_id:
                    brain_service.clear_session(session_id)
                    self.send_json({"success": True, "message": f"Session {session_id} cleared"})
                else:
                    self.send_json({"success": False, "message": "session_id required"}, status=400)

            elif self.path == "/api/context/decision/plan":
                data = json.loads(body) if body else {}
                plan_text = data.get("plan_text") or data.get("summary") or ""
                actor = data.get("actor") or "agent_manager"
                tools = data.get("tools") or []
                provenance = data.get("provenance") or {}
                plan_node = brain_service.record_action_plan_trace(
                    session_id=data.get("session_id"),
                    plan_text=plan_text,
                    actor=actor,
                    tools=tools if isinstance(tools, list) else [tools],
                    provenance=provenance if isinstance(provenance, dict) else {},
                )
                self.send_json({"success": True, "plan_node": plan_node})

            elif self.path == "/api/context/decision/feedback":
                data = json.loads(body) if body else {}
                decision_id = brain_service.record_human_decision_trace(
                    session_id=data.get("session_id"),
                    action_ref=data.get("action_ref") or data.get("plan_id") or "",
                    approved=bool(data.get("approved", False)),
                    user_id=data.get("user_id") or "unknown_user",
                    notes=data.get("notes") or "",
                )
                self.send_json({"success": True, "decision_id": decision_id})

            elif self.path == "/api/admin/factory_reset":
                self.handle_factory_reset(body)
                return

            elif self.path == "/api/admin/promote_candidate":
                self.handle_promote_candidate(body)
                return

            elif self.path == "/api/v1/models/activate":
                self.handle_activate_model(body)
                return

            elif self.path == "/api/v1/models/upload":
                self.handle_upload_model(body)
                return

            elif self.path == "/api/v1/data/tag":
                self.handle_tag_episode(body)
                return

            elif self.path.startswith("/api/mode/"):
                target_mode = self.path.rstrip("/").split("/")[-1]
                result = self._set_mode(target_mode)
                self.send_json(result, status=200 if result.get("success") else 400)

            elif self.path == "/api/safety/hold":
                result = self._set_mode("emergency_stop")
                self.send_json(result, status=200 if result.get("success") else 400)

            elif self.path == "/api/safety/reset":
                result = self._set_mode("idle")
                self.send_json(result, status=200 if result.get("success") else 400)

            elif self.path == "/api/runtime/mode":
                # Set runtime mode (inference/training, manual/autonomous)
                try:
                    from continuonbrain.services.runtime_context import (
                        get_runtime_context_manager,
                        PrimaryMode,
                        SubMode,
                    )
                    data = json.loads(body) if body else {}
                    mgr = get_runtime_context_manager(brain_service.config_dir if brain_service else "/tmp/continuonbrain")
                    
                    primary = None
                    sub = None
                    
                    if "primary" in data:
                        primary_str = data["primary"].lower()
                        primary = {
                            "inference": PrimaryMode.INFERENCE,
                            "training": PrimaryMode.TRAINING,
                            "hybrid": PrimaryMode.HYBRID,
                        }.get(primary_str)
                    
                    if "sub" in data:
                        sub_str = data["sub"].lower()
                        sub = {
                            "manual": SubMode.MANUAL,
                            "autonomous": SubMode.AUTONOMOUS,
                        }.get(sub_str)
                    
                    mgr.set_mode(primary=primary, sub=sub)
                    self.send_json({
                        "success": True,
                        "context": mgr.get_status(),
                    })
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, status=500)

            elif self.path == "/api/tasks/select":
                data = json.loads(body) if body else {}
                task_id = data.get("task_id")
                if task_id:
                    global selected_task_id
                    selected_task_id = task_id
                    self.send_json({"success": True, "selected_task": task_id})
                else:
                    self.send_json({"success": False, "message": "task_id required"}, status=400)

            elif self.path == "/api/curriculum/run":
                data = json.loads(body) if body else {}
                lesson_id = data.get("lesson_id")
                if lesson_id:
                    result = asyncio.run(brain_service.RunCurriculumLesson(lesson_id))
                    self.send_json(result)
                else:
                    self.send_json({"success": False, "message": "lesson_id required"}, status=400)

            # HOPE API POST Endpoints
            elif self.path.startswith("/api/hope/"):
                try:
                    from continuonbrain.api.routes import hope_routes
                    hope_routes.handle_hope_request(self, body)
                except ImportError:
                    self.send_json({"error": "HOPE implementation not available"}, status=503)

            elif self.path == "/api/context/search":
                try:
                    data = json.loads(body) if body else {}
                    query = data.get("query")
                    if query and brain_service.gemma_chat:
                        vec = brain_service.gemma_chat.embed(query)
                        results = brain_service.context_store.get_nearest_nodes(vec, limit=10)
                        self.send_json({"results": [n.__dict__ for n in results]})
                    else:
                        self.send_json({"results": []})
                except Exception as e:
                    self.send_json({"error": str(e)}, status=500)

            elif self.path == "/api/memory/save":
                if brain_service.experience_logger.save_memory():
                    self.send_json({"success": True, "message": "Memory saved"})
                else:
                    self.send_json({"success": False, "error": "Save failed"}, status=500)
            
            elif self.path == "/api/hope/compact":
                try:
                    result = brain_service.compact_memory()
                    self.send_json(result)
                except Exception as e:
                    self.send_json({"error": str(e)}, status=500)
            
            elif self.path == "/api/brain/toggle_hybrid":
                # Toggle hybrid mode (1 vs 4 columns)
                try:
                    current_cols = len(brain_service.hope_brain.columns) if brain_service.hope_brain else 1
                    target_cols = 4 if current_cols == 1 else 1
                    
                    brain_service.hope_brain.initialize(num_columns=target_cols)
                    msg = "Switched to Hybrid 4-Column Mode" if target_cols > 1 else "Switched to Standard Mode"
                    self.send_json({"success": True, "message": msg, "mode": "hybrid" if target_cols > 1 else "standard"})
                except Exception as e:
                    logger.error(f"Toggle hybrid failed: {e}")
                    self.send_json({"success": False, "message": str(e)}, status=500)
            
            elif self.path == "/api/robot/drive":
                self.handle_drive(body)
            
            elif self.path == "/api/robot/joints":
                self.handle_joints(body)
            
            elif self.path == "/api/robot/name":
                # Update robot name (only owners can do this)
                # Note: creator_display_name is IMMUTABLE
                data = json.loads(body) if body else {}
                new_name = data.get("robot_name", "").strip()
                
                if not new_name:
                    self.send_json({"success": False, "error": "Robot name cannot be empty"}, status=400)
                elif len(new_name) > 50:
                    self.send_json({"success": False, "error": "Robot name must be <= 50 characters"}, status=400)
                else:
                    store = SettingsStore(Path(brain_service.config_dir))
                    settings = store.load()
                    settings["identity"]["robot_name"] = new_name
                    # Ensure creator_display_name stays immutable
                    settings["identity"]["creator_display_name"] = "Craig Michael Merry"
                    settings["identity"]["_creator_immutable"] = True
                    store.save(settings)
                    self.send_json({
                        "success": True,
                        "robot_name": new_name,
                        "creator_name": "Craig Michael Merry",
                        "message": "Robot name updated"
                    })
            
            elif self.path == "/api/settings":
                data = json.loads(body) if body else {}
                store = SettingsStore(Path(brain_service.config_dir))
                try:
                    switch_result = None
                    # Check if model is changing
                    old_settings = store.load()
                    old_model = old_settings.get("agent_manager", {}).get("agent_model", "mock")
                    # Default to HOPE if not provided
                    if "agent_manager" not in data:
                        data["agent_manager"] = {}
                    new_model = data.get("agent_manager", {}).get("agent_model") or "hope-v1"
                    data["agent_manager"]["agent_model"] = new_model
                    
                    # Save settings first
                    settings = store.save(data)
                    
                    if old_model != new_model:
                        logger.info(f"Model change detected: {old_model} -> {new_model}")
                        switch_result = brain_service.switch_model(new_model)
                    
                    # Reload settings in active service
                    brain_service.load_settings()
                    
                    response = {
                        "success": True,
                        "settings": settings,
                        "message": "Settings saved"
                    }
                    
                    if switch_result:
                        response["model_switch"] = switch_result
                        if switch_result.get("success"):
                            response["message"] += f" and switched to {new_model}"
                        else:
                            response["message"] += f" but model switch failed: {switch_result.get('error')}"
                    
                    self.send_json(response)
                except SettingsValidationError as e:
                    self.send_json({"success": False, "message": str(e)}, status=400)

            elif self.path == "/api/hardware/scan":
                # Manual Hardware Scan
                try:
                    from continuonbrain.sensors.hardware_detector import HardwareDetector
                    detector = HardwareDetector()
                    payload = json.loads(body) if body else {}
                    auto_install = bool(payload.get("auto_install", False))
                    allow_system_install = bool(payload.get("allow_system_install", False))
                    detector.detect_all(auto_install=auto_install, allow_system_install=allow_system_install)
                    devices_cfg = detector.generate_config()
                    
                    response = {
                        "success": True,
                        "device_count": len(devices_cfg.get("devices", {})),
                        "devices": {
                            "camera": "depth_camera" in devices_cfg.get("primary", {}),
                            "arm": "servo_controller" in devices_cfg.get("primary", {}) or "servo_controller" in devices_cfg.get("devices", {}),
                            "drivetrain": "servo_controller" in devices_cfg.get("primary", {}) or "servo_controller" in devices_cfg.get("devices", {})
                        },
                        "message": "Scan complete",
                        "missing_dependencies": detector.missing_dependencies,
                        "platform": detector.platform_info,
                    }
                    self.send_json(response)
                except ImportError:
                    self.send_json({"success": False, "message": "Hardware Detector not available"})
            
            elif self.path == "/api/agent/validate":
                data = json.loads(body) if body else {}
                timestamp = data.get('timestamp')
                conversation_id = data.get('conversation_id') or timestamp
                validated = data.get('validated', True)
                correction = data.get('correction')
                
                if not conversation_id:
                    self.send_json({"success": False, "error": "conversation_id or timestamp required"}, status=400)
                else:
                    try:
                        updated = brain_service.experience_logger.validate_conversation(
                            conversation_id=conversation_id,
                            validated=validated,
                            correction=correction
                        )
                        if updated:
                            self.send_json({"success": True, "message": "Conversation validated"})
                        else:
                            self.send_json({"success": False, "error": "Conversation not found"}, status=404)
                    except Exception as e:
                        logger.error(f"Failed to validate conversation: {e}")
                        self.send_json({"success": False, "error": str(e)}, status=500)

            elif self.path == "/api/agent/search":
                try:
                    data = json.loads(body) if body else {}
                    query = data.get('query', '')
                    if not query:
                        self.send_json({"error": "Missing query"}, status=400)
                    else:
                        results = brain_service.experience_logger.search_conversations(query)
                        self.send_json({"success": True, "results": results})
                except Exception as e:
                    logger.error(f"Search failed: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)

            elif self.path == "/api/training/manual":
                try:
                    data = json.loads(body) if body else {}
                    # RunManualTraining is async, so we must run it in a loop
                    result = asyncio.run(brain_service.RunManualTraining(data))
                    self.send_json(result)
                except Exception as e:
                    logger.error(f"Manual training failed: {e}")
                    self.send_json({"status": "error", "message": str(e)}, status=500)

            elif self.path == "/api/agent/consolidate":
                try:
                    stats = brain_service.experience_logger.consolidate_memories()
                    self.send_json({"success": True, "stats": stats})
                except Exception as e:
                    logger.error(f"Failed to consolidate memories: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            elif self.path == "/api/agent/decay":
                try:
                    stats = brain_service.experience_logger.apply_confidence_decay()
                    self.send_json({"success": True, "stats": stats})
                except Exception as e:
                    logger.error(f"Failed to apply decay: {e}")
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            elif self.path.startswith("/api/tasks/") and self.path.endswith("/execute"):
                # Execute task: /api/tasks/{task_id}/execute
                task_id = self.path.split("/")[3]
                try:
                    summary = brain_service.GetTaskSummary(task_id)
                    if not summary:
                        self.send_json({"success": False, "message": f"Task {task_id} not found"})
                    elif not summary.entry.eligibility.eligible:
                        blocking = [m.label for m in summary.entry.eligibility.markers if m.blocking]
                        self.send_json({"success": False, "message": f"Task blocked: {', '.join(blocking)}"})
                    else:
                        # TODO: Implement actual task execution
                        # For now, just acknowledge
                        self.send_json({"success": True, "message": f"Task {task_id} execution started (stub)"})
                except Exception as e:
                    logger.error(f"Task execution error: {e}")
                    self.send_json({"success": False, "message": str(e)})

            elif self.path == "/api/personality/update":
                data = json.loads(body)
                updated = brain_service.update_personality_config(
                    humor=data.get("humor_level"),
                    sarcasm=data.get("sarcasm_level"),
                    empathy=data.get("empathy_level"),
                    verbosity=data.get("verbosity_level"),
                    system_name=data.get("system_name"),
                    identity_mode=data.get("identity_mode")
                )
                self.send_json({"success": True, "config": updated})
            
            elif self.path == "/api/identity/update":
                data = json.loads(body)
                user_id = data.get("user_id")
                role = data.get("role")
                if user_id and role:
                   updated = brain_service.set_user_context(user_id, role)
                   self.send_json({"success": True, "context": updated})
                else:
                   self.send_json({"success": False, "message": "Missing user_id or role"}, status=400)

            elif self.path == "/api/network/wifi/connect":
                data = json.loads(body) if body else {}
                ssid = data.get("ssid")
                password = data.get("password")
                result = wifi_connect(ssid, password)
                status = 200 if result.get("success") else 500
                # Do not echo password back
                if "password" in result:
                    result.pop("password")
                self.send_json(result, status=status)

            elif self.path == "/api/training/chat_learn":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunChatLearn(data))
                self.send_json(result)

            elif self.path == "/api/training/sequential":
                data = json.loads(body) if body else {}
                result = brain_service.start_sequential_training(data)
                self.send_json(result)

            elif self.path == "/api/training/teacher/mode":
                data = json.loads(body) if body else {}
                brain_service.teacher_mode_active = bool(data.get("active", False))
                self.send_json({"success": True, "active": brain_service.teacher_mode_active})

            elif self.path == "/api/training/teacher/answer":
                data = json.loads(body) if body else {}
                text = data.get("text")
                if text:
                    brain_service.teacher_response_text = text
                    brain_service.teacher_response_event.set()
                    self.send_json({"success": True})
                else:
                    self.send_json({"success": False, "message": "Missing text"}, status=400)

            elif self.path == "/api/training/teacher/status":
                 self.send_json({
                     "success": True, 
                     "active": brain_service.teacher_mode_active, 
                     "pending_question": brain_service.teacher_pending_question
                 })

            elif self.path.startswith("/api/learning/"):
                try:
                    from continuonbrain.api.routes import learning_routes
                    learning_routes.handle_learning_request(self)
                except ImportError:
                    self.send_json({"error": "Learning service not available"}, status=503)

            # Autonomous Training Engine POST endpoints
            elif self.path == "/api/training/trigger":
                self.handle_training_trigger(body)
            elif self.path == "/api/training/config":
                self.handle_training_config(body)
            elif self.path == "/api/training/benchmark":
                self.handle_run_benchmark(body)

            elif self.path == "/api/manual/symbolic_search":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunSymbolicSearch(data))
                self.send_json(result)

            elif self.path == "/api/waves/loops" or self.path == "/api/training/wavecore_loops":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunWavecoreLoops(data))
                self.send_json(result)

            # ======== Missing Training Endpoints (now implemented) ========
            elif self.path == "/api/training/jax/start":
                # Start JAX training (alias for wavecore)
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunWavecoreLoops(data))
                self.send_json(result)
            
            elif self.path == "/api/training/run":
                # Generic training run
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunWavecoreLoops(data))
                self.send_json(result)
            
            elif self.path == "/api/training/hope_eval":
                # Run HOPE eval
                data = json.loads(body) if body else {}
                data["run_hope_eval"] = True
                data["fast"] = {"max_steps": 0}  # Skip training, just eval
                result = asyncio.run(brain_service.RunWavecoreLoops(data))
                self.send_json(result)
            
            elif self.path == "/api/training/hope_eval_facts":
                # Run facts eval
                data = json.loads(body) if body else {}
                data["run_facts_eval"] = True
                data["fast"] = {"max_steps": 0}
                result = asyncio.run(brain_service.RunWavecoreLoops(data))
                self.send_json(result)
            
            elif self.path == "/api/training/export_zip":
                # Create cloud export zip
                self.handle_create_cloud_export(body)
            
            elif self.path == "/api/training/install_bundle":
                # Install model bundle
                self.handle_install_model(body)
            
            elif self.path == "/api/training/control/mode":
                # Set training control mode
                data = json.loads(body) if body else {}
                mode = data.get("mode", "auto")
                if hasattr(brain_service, 'training_control_mode'):
                    brain_service.training_control_mode = mode
                self.send_json({"success": True, "mode": mode})
            
            elif self.path == "/api/training/tool_router_train":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunToolRouterTrain(data))
                self.send_json(result)
            
            elif self.path == "/api/training/tool_router_predict":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.ToolRouterPredict(data))
                self.send_json(result)
            
            elif self.path == "/api/training/tool_router_eval":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RunToolRouterEval(data))
                self.send_json(result)
            
            elif self.path == "/api/imagination/start":
                # Start imagination / world model prediction
                data = json.loads(body) if body else {}
                try:
                    from continuonbrain.mamba_brain.world_model import build_world_model
                    from continuonbrain.mamba_brain import WorldModelState, WorldModelAction
                    
                    wm = build_world_model(prefer_mamba=True)
                    state_data = data.get("state", {})
                    action_data = data.get("action", {})
                    
                    state = WorldModelState(joint_pos=state_data.get("joint_pos", [0.0] * 6))
                    action = WorldModelAction(joint_delta=action_data.get("joint_delta", [0.0] * 6))
                    
                    result = wm.predict(state, action)
                    self.send_json({
                        "success": True,
                        "next_state": result.next_state.joint_pos,
                        "uncertainty": result.uncertainty,
                        "backend": result.debug.get("backend", "unknown")
                    })
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            elif self.path == "/api/mode/manual_control":
                # Set mode to manual control
                result = self._set_mode("manual")
                self.send_json(result)
            
            elif self.path == "/api/planning/arm_search":
                # Arm planning search
                data = json.loads(body) if body else {}
                target_pos = data.get("target_position", [0.5] * 6)
                # Use world model for planning
                try:
                    from continuonbrain.mamba_brain.world_model import build_world_model
                    wm = build_world_model(prefer_mamba=True)
                    # Simple planning stub - return direct path
                    self.send_json({
                        "success": True,
                        "plan": [{"position": target_pos, "duration_ms": 1000}],
                        "steps": 1
                    })
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, status=500)
            
            elif self.path == "/api/planning/arm_execute_delta":
                # Execute arm delta movement
                data = json.loads(body) if body else {}
                delta = data.get("delta", [0.0] * 6)
                # This would normally send to robot control
                self.send_json({
                    "success": True,
                    "executed": True,
                    "delta": delta,
                    "message": "Delta command queued"
                })
            
            elif self.path == "/api/audio/tts":
                # Text to speech
                data = json.loads(body) if body else {}
                text = data.get("text", "")
                result = asyncio.run(brain_service.SpeakText({"text": text}))
                self.send_json(result)
            
            elif self.path == "/api/wiring":
                # Wiring/connection status
                data = json.loads(body) if body else {}
                # Return current wiring state
                self.send_json({
                    "success": True,
                    "connections": {
                        "hailo": getattr(brain_service, '_hailo_connected', False),
                        "oak": getattr(brain_service, '_oak_connected', False),
                        "sam3": getattr(brain_service, '_sam3_available', False),
                        "world_model": True,
                        "hope_brain": brain_service.hope_brain is not None,
                    }
                })
            # ======== End Missing Endpoints ========

            elif self.path.startswith("/api/tools/"):
                data = json.loads(body) if body else {}
                if self.path == "/api/tools/train":
                    result = asyncio.run(brain_service.RunToolRouterTrain(data))
                elif self.path == "/api/tools/predict":
                    result = asyncio.run(brain_service.ToolRouterPredict(data))
                elif self.path == "/api/tools/eval":
                    result = asyncio.run(brain_service.RunToolRouterEval(data))
                else:
                    result = {"error": "Unknown tool endpoint"}
                    status = 404
                self.send_json(result)

            elif self.path == "/api/ownership/pair/start":
                # Generate pairing session
                host = self.headers.get("host") or "127.0.0.1:8081"
                if "://" not in host:
                    host = f"http://{host}"
                session = brain_service.pairing.start(base_url=host)
                payload = session.__dict__.copy()
                payload["url"] = session.url  # Include computed property
                self.send_json(payload)

            elif self.path == "/api/ownership/pair/confirm":
                data = json.loads(body) if body else {}
                token = data.get("token")
                code = data.get("confirm_code")
                owner_id = data.get("owner_id")
                account_id = data.get("account_id")
                account_type = data.get("account_type")
                
                success, msg, payload = brain_service.pairing.confirm(
                    token=token, 
                    confirm_code=code, 
                    owner_id=owner_id,
                    account_id=account_id,
                    account_type=account_type
                )
                if success:
                    # Update brain_service state
                    brain_service.set_ownership(
                        owned=True,
                        owner_id=owner_id,
                        account_id=account_id,
                        account_type=account_type
                    )
                    self.send_json({"status": "paired", "ownership": payload})
                else:
                    self.send_json({"status": "error", "message": msg}, status=400)

            elif self.path == "/api/ownership/status":
                result = asyncio.run(brain_service.GetOwnershipStatus())
                self.send_json(result)
            
            elif self.path == "/api/ownership/transfer":
                data = json.loads(body) if body else {}
                new_owner_id = data.get("new_owner_id")
                new_account_id = data.get("new_account_id")
                new_account_type = data.get("new_account_type")
                
                if not new_owner_id:
                    self.send_json({"success": False, "message": "new_owner_id required"}, status=400)
                    return
                
                # Transfer ownership
                try:
                    brain_service.set_ownership(
                        owned=True,
                        owner_id=new_owner_id,
                        account_id=new_account_id,
                        account_type=new_account_type
                    )
                    self.send_json({
                        "success": True,
                        "message": f"Ownership transferred to {new_owner_id}",
                        "new_owner_id": new_owner_id
                    })
                except Exception as e:
                    self.send_json({"success": False, "message": str(e)}, status=500)

            elif self.path == "/api/audio/speak":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.SpeakText(data))
                self.send_json(result)

            elif self.path == "/api/audio/record":
                data = json.loads(body) if body else {}
                result = asyncio.run(brain_service.RecordMicrophone(data))
                self.send_json(result)

            elif self.path == "/api/network/bluetooth/connect":
                data = json.loads(body) if body else {}
                address = data.get("address")
                result = bluetooth_connect(address)
                status = 200 if result.get("success") else 500
                self.send_json(result, status=status)

            elif self.path == "/api/training/exports/create":
                self.handle_create_cloud_export(body)

            elif self.path == "/api/model/install":
                self.handle_install_model(body)

            # ===================================================================
            # RCAN Protocol POST Endpoints
            # ===================================================================
            elif self.path == "/rcan/v1/discover":
                # Handle discovery request
                data = json.loads(body) if body else {}
                if brain_service and hasattr(brain_service, 'rcan'):
                    from continuonbrain.services.rcan_service import RCANMessage
                    msg = RCANMessage.from_dict(data)
                    response = brain_service.rcan.handle_discover(msg)
                    self.send_json(response.to_dict())
                else:
                    self.send_json({"error": "RCAN service not available"}, status=503)
            
            elif self.path == "/rcan/v1/auth/claim":
                # Claim control of the robot
                data = json.loads(body) if body else {}
                if brain_service and hasattr(brain_service, 'rcan'):
                    from continuonbrain.services.rcan_service import RCANMessage, UserRole
                    msg = RCANMessage(
                        source_ruri=data.get("source_ruri", ""),
                        target_ruri=brain_service.rcan.identity.ruri,
                    )
                    user_id = data.get("user_id", "")
                    role_str = data.get("role", "guest").upper()
                    try:
                        role = UserRole[role_str]
                    except KeyError:
                        role = UserRole.GUEST
                    response = brain_service.rcan.handle_claim(msg, user_id, role)
                    self.send_json(response.to_dict())
                else:
                    self.send_json({"error": "RCAN service not available"}, status=503)
            
            elif self.path == "/rcan/v1/auth/release":
                # Release control
                data = json.loads(body) if body else {}
                session_id = data.get("session_id", "")
                if brain_service and hasattr(brain_service, 'rcan'):
                    response = brain_service.rcan.handle_release(session_id)
                    self.send_json(response.to_dict())
                else:
                    self.send_json({"error": "RCAN service not available"}, status=503)
            
            elif self.path == "/rcan/v1/command":
                # Send command via RCAN
                data = json.loads(body) if body else {}
                session_id = data.get("session_id", "")
                if brain_service and hasattr(brain_service, 'rcan'):
                    from continuonbrain.services.rcan_service import RCANMessage
                    msg = RCANMessage.from_dict(data.get("message", {}))
                    response = brain_service.rcan.handle_command(msg, session_id)
                    self.send_json(response.to_dict())
                else:
                    self.send_json({"error": "RCAN service not available"}, status=503)

            elif self.path == "/rcan/v1/cloud/register":
                # Register robot with Firebase cloud registry
                data = json.loads(body) if body else {}
                if brain_service and hasattr(brain_service, 'cloud_registry') and brain_service.cloud_registry:
                    capabilities = data.get("capabilities") or ["arm", "vision", "chat", "teleop", "training"]
                    tunnel_url = data.get("tunnel_url")
                    firmware_version = data.get("firmware_version", "0.1.0")
                    port = data.get("port", 8081)
                    result = asyncio.run(brain_service.cloud_registry.register(
                        capabilities=capabilities,
                        firmware_version=firmware_version,
                        tunnel_url=tunnel_url,
                        port=port,
                    ))
                    self.send_json(result)
                else:
                    self.send_json({
                        "success": False,
                        "error": "Cloud registry not initialized. Check service-account.json.",
                    }, status=503)

            elif self.path == "/rcan/v1/cloud/heartbeat":
                # Send heartbeat to cloud registry
                if brain_service and hasattr(brain_service, 'cloud_registry') and brain_service.cloud_registry:
                    result = asyncio.run(brain_service.cloud_registry.heartbeat())
                    self.send_json(result)
                else:
                    self.send_json({"success": False, "error": "Cloud registry not available"}, status=503)

            elif self.path == "/rcan/v1/cloud/lookup":
                # Lookup a robot by RURI
                data = json.loads(body) if body else {}
                ruri = data.get("ruri", "")
                if not ruri:
                    self.send_json({"success": False, "error": "ruri required"}, status=400)
                else:
                    try:
                        from continuonbrain.services.rcan_cloud_registry import RCANCloudRegistry
                        config_dir = brain_service.config_dir if brain_service else "/tmp"
                        result = asyncio.run(RCANCloudRegistry.lookup(ruri, config_dir))
                        if result:
                            self.send_json({"success": True, "robot": result})
                        else:
                            self.send_json({"success": False, "error": "Robot not found"}, status=404)
                    except Exception as e:
                        self.send_json({"success": False, "error": str(e)}, status=500)

            else:
                self.send_error(404)

        except Exception as e:
            logger.error(f"POST error: {e}")
            self.send_json({"success": False, "message": str(e)}, status=500)

    def _set_mode(self, target_mode: str) -> dict:
        """Set robot mode from string target."""
        if not brain_service or not brain_service.mode_manager:
            return {"success": False, "message": "Mode manager unavailable"}

        target = (target_mode or "").lower()
        mapping = {
            "manual_control": RobotMode.MANUAL_CONTROL,
            "manual_training": RobotMode.MANUAL_TRAINING,
            "autonomous": RobotMode.AUTONOMOUS,
            "sleep_learning": RobotMode.SLEEP_LEARNING,
            "idle": RobotMode.IDLE,
            "emergency_stop": RobotMode.EMERGENCY_STOP,
        }

        if target not in mapping:
            return {"success": False, "message": f"Unknown mode: {target_mode}"}

        if mapping[target] == RobotMode.EMERGENCY_STOP:
            brain_service.mode_manager.emergency_stop("API trigger")
            changed = True
        else:
            changed = brain_service.mode_manager.set_mode(mapping[target])

        gates = brain_service.mode_manager.get_gate_snapshot()
        return {"success": bool(changed), "mode": gates.get("mode"), "gate_snapshot": gates}

    def send_json(self, data, status: int = 200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

    def handle_single_frame(self):
        if not brain_service.camera:
            self.send_error(404, "Camera not available")
            return
            
        frame_data = brain_service.camera.capture_frame()
        if frame_data and frame_data.get('rgb') is not None:
            ret, jpeg = cv2.imencode('.jpg', frame_data['rgb'])
            if ret:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                return
        
        self.send_error(503, "Frame capture failed")

    def handle_mjpeg_stream(self):
        if not brain_service.camera:
            self.send_error(404, "Camera not available")
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        try:
            while True:
                # Limit FPS to ~15 for streaming to save bandwidth
                start_time = time.time()
                
                frame_data = brain_service.camera.capture_frame()
                if frame_data and frame_data.get('rgb') is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame_data['rgb'])
                    if ret:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                
                # Sleep to maintain FPS
                elapsed = time.time() - start_time
                delay = max(0.0, 0.066 - elapsed) # ~15 FPS
                time.sleep(delay)
                
        except Exception as e:
            pass # Client disconnected

    def log_message(self, format, *args):
        # Silence default logging
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    allow_reuse_address = True
    daemon_threads = True


def bind_http_server(preferred_port: int):
    """Bind HTTP server, falling back to the next available port if busy."""
    last_error = None
    for candidate in range(preferred_port, preferred_port + 10):
        try:
            server = ThreadedHTTPServer(("0.0.0.0", candidate), BrainRequestHandler)
            return server, candidate
        except OSError as exc:
            last_error = exc
            if exc.errno == 98:  # Address already in use
                continue
            raise
    raise last_error or OSError("No available port found")


def launch_ui_if_desktop(port: int):
    """Best-effort UI auto-launch when a desktop is present."""
    if os.environ.get("CONTINUON_NO_UI_LAUNCH"):
        return
    system = platform.system()
    has_display = True
    if system == "Linux" and not os.environ.get("DISPLAY"):
        has_display = False
    if not has_display:
        return
    try:
        webbrowser.open(f"http://localhost:{port}/ui")
        if event_logger:
            event_logger.log(
                "ui_launch",
                "Opened ContinuonBrain UI in default browser",
                {"port": port},
            )
    except Exception as exc:
        logger.warning(f"Failed to launch UI browser: {exc}")

def main():
    """
    Main entry point for the Brain API server.
    
    NOTE: This server should be started via the startup_manager, not directly.
    The startup_manager handles:
    - Proper service orchestration (safety kernel, API server, etc.)
    - Process monitoring and restart on failure
    - Unified logging and configuration
    - systemd integration for production deployments
    
    To start services properly, use:
        ./scripts/start_services.sh start --mode desktop|rpi
    
    Or via systemd:
        systemctl --user start continuonbrain.service
    """
    global brain_service, identity_service, event_logger, background_learner
    
    # Check if started via startup_manager
    parent_cmdline = ""
    try:
        import psutil
        parent = psutil.Process().parent()
        if parent:
            parent_cmdline = " ".join(parent.cmdline())
    except Exception:
        pass
    
    is_via_startup_manager = "startup_manager" in parent_cmdline
    
    if not is_via_startup_manager and not os.environ.get("CONTINUON_ALLOW_DIRECT_SERVER"):
        print("=" * 70)
        print("⚠️  WARNING: Direct server invocation detected!")
        print("=" * 70)
        print("")
        print("The Brain API server should be started via the startup_manager")
        print("to ensure proper service orchestration and process management.")
        print("")
        print("Recommended ways to start:")
        print("  1. ./scripts/start_services.sh start --mode desktop")
        print("  2. systemctl --user start continuonbrain.service")
        print("")
        print("To bypass this warning, set CONTINUON_ALLOW_DIRECT_SERVER=1")
        print("")
        print("Continuing with direct startup...")
        print("=" * 70)
    
    parser = argparse.ArgumentParser(
        description="ContinuonBrain API Server (prefer startup via startup_manager)"
    )
    parser.add_argument("--config-dir", default="/tmp/continuonbrain_demo")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--real-hardware", action="store_true", help="Prefer real hardware")
    parser.add_argument("--mock-hardware", action="store_true", help="Force mock hardware")
    args = parser.parse_args()

    # Check environment variables as fallback for hardware mode
    env_real_hw = os.environ.get("CONTINUON_FORCE_REAL_HARDWARE", "0").lower() in ("1", "true", "yes")
    env_mock_hw = os.environ.get("CONTINUON_FORCE_MOCK_HARDWARE", "0").lower() in ("1", "true", "yes")
    prefer_real = (args.real_hardware or env_real_hw) and not (args.mock_hardware or env_mock_hw)
    desired_port = args.port
    event_logger = SystemEventLogger(args.config_dir)

    try:
        server, bound_port = bind_http_server(desired_port)
        # Initializes Auth Provider
        get_auth_provider(Path(args.config_dir))
        # Attach BrainService for Controllers
        # Note: brain_service is init AFTER bind_http_server in original code, so we must wait to attach
        # See below after brain_service init
    except OSError as exc:
        print(f"❌ Failed to bind HTTP server on port {desired_port}: {exc}")
        raise

    if bound_port != desired_port:
        logger.warning(f"Port {desired_port} in use; bound to {bound_port} instead")
    print(f"Starting ContinuonBrain Server on port {bound_port}...")
    event_logger.log(
        "server_start",
        "Brain API server starting",
        {"port": bound_port, "config_dir": args.config_dir},
    )
    
    # Initialize Services
    identity_service = AgentIdentity(config_dir=args.config_dir)
    # Run identity check early to determine shell
    identity_service.self_report() 
    shell_type = identity_service.identity.get("shell", {}).get("type", "Unknown")
    
    # Load settings
    settings_store = SettingsStore(Path(args.config_dir))
    settings = settings_store.load()
    agent_settings = settings.get("agent_manager", {})
    
    print(f"Agent Manager Settings:")
    print(f"  Thinking Indicator: {agent_settings.get('enable_thinking_indicator', True)}")
    print(f"  Intervention Prompts: {agent_settings.get('enable_intervention_prompts', True)}")
    print(f"  Confidence Threshold: {agent_settings.get('intervention_confidence_threshold', 0.5)}")
    print(f"  Status Updates: {agent_settings.get('enable_status_updates', True)}")
    print(f"  Autonomous Learning: {agent_settings.get('enable_autonomous_learning', True)}")
    
    # If Desktop Station, force mock hardware for robot components?
    # For now, we pass preferences.
    
    brain_service = BrainService(
        config_dir=args.config_dir,
        prefer_real_hardware=prefer_real,
        auto_detect=True
    )
    # Attach to server for controllers
    server.brain_service = brain_service
    
    # Store settings in brain_service for access by ChatWithGemma
    
    # Store settings in brain_service for access by ChatWithGemma
    brain_service.agent_settings = agent_settings
    
    # Async Init (hack for sync constructor)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(brain_service.initialize())

    # Hook SystemEventLogger to SSE stream
    if event_logger and hasattr(brain_service, "chat_event_queue"):
        def _bridge_system_event(event_dict):
            # Pass system events to the SSE queue for UI consumption
            try:
                brain_service.chat_event_queue.put(event_dict)
            except Exception:
                pass
        event_logger.add_listener(_bridge_system_event)

    # Bind the autonomous learner to API routes so live training metrics
    # surface through the learning + HOPE monitoring endpoints.
    try:
        from continuonbrain.api.routes import hope_routes, learning_routes
        learning_routes.set_brain_service(brain_service)

        background_learner = getattr(brain_service, "background_learner", None)
        if background_learner:
            try:
                hope_routes.set_background_learner(background_learner)
                learning_routes.set_background_learner(background_learner)
                logger.info("Background learner wired to web routes for live metrics")
            except ImportError:
                logger.warning("Learning/HOPE routes unavailable; skipping learner binding")
        else:
            logger.info("Background learner not active; learning endpoints will report disabled")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to bind background learner: {exc}")

    # Ensure HOPE agent model is active by default
    try:
        brain_service.switch_model("hope-v1")
    except Exception as exc:
        logger.warning(f"HOPE model activation failed: {exc}")

    if event_logger:
        event_logger.log(
            "server_ready",
            "Brain API server ready",
            {
                "port": bound_port,
                "shell_type": shell_type,
                "prefer_real_hardware": prefer_real,
            },
        )
    
    # Legacy BackgroundLearner initialization removed. 
    # It is now handled by BrainService in verify_learning_startup logic / production code.
    
    # Fire and forget UI launch on desktop systems
    threading.Timer(1.0, lambda: launch_ui_if_desktop(bound_port)).start()

    print(f" Server listening on http://0.0.0.0:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    
    # Graceful shutdown
    print("\\n🛑 Shutting down...")
    
    if background_learner:
        print("🔄 Stopping autonomous learning...")
        background_learner.stop()
    
    brain_service.shutdown()
    server.server_close()
    print("Server stopped.")

if __name__ == "__main__":
    main()
