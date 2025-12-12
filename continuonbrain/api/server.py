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

    return {
        "status": "ok",
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

class BrainRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Brain API."""

    def do_GET(self):
        try:
            if self.path in ("/", "/ui", "/ui/"):
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
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_dashboard_html().encode("utf-8"))

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
                self.wfile.write(ui_routes.get_manual_html().encode("utf-8"))

            elif self.path == "/ui/tasks":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_tasks_html().encode("utf-8"))
            
            # HOPE Monitoring Pages
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
                self.wfile.write(ui_routes.get_brain_map_html().encode("utf-8"))

            elif self.path == "/api/hope/structure":
                data = brain_service.get_brain_structure()
                self.send_json(data)


            elif self.path == "/api/tasks/library":
                # Return task library with eligibility checks
                tasks = brain_service.get_task_library()
                self.send_json({"tasks": tasks})

            elif self.path == "/api/status/introspection":
                # Introspection endpoint for Brain Status page
                # identity_service.self_report() # Removed to prevent log spam/heavy IO on polling
                data = identity_service.identity
                self.send_json(data)

            elif self.path.startswith("/api/status"):
                # Enriched robot status for UI
                self.send_json(_build_status_payload())

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
                        "provenance": getattr(skill, "provenance", ""),
                    })
                if not include_ineligible:
                    skills = [s for s in skills if s.get("eligibility", {}).get("eligible", False)]
                self.send_json({"skills": skills})

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
                except Exception:
                    pass
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

            elif self.path == "/api/camera/stream":
                self.handle_mjpeg_stream()

            elif self.path == "/api/camera/frame":
                self.handle_single_frame()

            elif self.path == "/api/settings":
                store = SettingsStore(Path(brain_service.config_dir))
                self.send_json({"success": True, "settings": store.load()})
            
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

            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Request error: {e}")
            # self.send_error(500) # Generating error during stream breaks things

    def do_POST(self):
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len).decode('utf-8')
            
            if self.path == "/api/chat":
                data = json.loads(body)
                msg = data.get("message", "")
                
                result = brain_service.ChatWithGemma(msg, [])
                result = brain_service.ChatWithGemma(msg, [])
                self.send_json(result)
            
            elif self.path == "/api/chat/history/clear":
                brain_service.clear_chat_history()
                self.send_json({"success": True})

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

            elif self.path == "/api/tasks/select":
                data = json.loads(body) if body else {}
                task_id = data.get("task_id")
                if task_id:
                    global selected_task_id
                    selected_task_id = task_id
                    self.send_json({"success": True, "selected_task": task_id})
                else:
                    self.send_json({"success": False, "message": "task_id required"}, status=400)

            # HOPE API POST Endpoints
            elif self.path.startswith("/api/hope/"):
                try:
                    from continuonbrain.api.routes import hope_routes
                    hope_routes.handle_hope_request(self, body)
                except ImportError:
                    self.send_json({"error": "HOPE implementation not available"}, status=503)

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
                data = json.loads(body)
                steering = float(data.get("steering", 0.0))
                throttle = float(data.get("throttle", 0.0))
                
                if brain_service.drivetrain:
                    brain_service.drivetrain.apply_drive(steering, throttle)
                    self.send_json({"success": True})
                else:
                    self.send_json({"success": False, "message": "No drivetrain"})
            
            elif self.path == "/api/robot/joints":
                data = json.loads(body)
                joint_idx = data.get("joint_index")
                val = data.get("value")
                
                if brain_service.arm and joint_idx is not None:
                    # Get current state first
                    current = brain_service.arm.get_normalized_state()
                    # Determine target
                    target = list(current)
                    if 0 <= joint_idx < 6:
                        target[joint_idx] = float(val)
                        brain_service.arm.set_normalized_action(target)
                        self.send_json({"success": True})
                    else:
                        self.send_json({"success": False, "message": "Invalid joint index"})
                else:
                    self.send_json({"success": False, "message": "No arm or invalid data"})
            
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
                validated = data.get('validated', True)
                correction = data.get('correction')
                
                if not timestamp:
                    self.send_json({"success": False, "error": "timestamp required"}, status=400)
                else:
                    try:
                        updated = brain_service.experience_logger.validate_conversation(
                            timestamp=timestamp,
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

            elif self.path == "/api/network/bluetooth/connect":
                data = json.loads(body) if body else {}
                address = data.get("address")
                result = bluetooth_connect(address)
                status = 200 if result.get("success") else 500
                self.send_json(result, status=status)

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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

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
    global brain_service, identity_service, event_logger, background_learner
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/tmp/continuonbrain_demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--real-hardware", action="store_true", help="Prefer real hardware")
    parser.add_argument("--mock-hardware", action="store_true", help="Force mock hardware")
    args = parser.parse_args()
    
    prefer_real = args.real_hardware and not args.mock_hardware
    desired_port = args.port
    event_logger = SystemEventLogger(args.config_dir)

    try:
        server, bound_port = bind_http_server(desired_port)
    except OSError as exc:
        print(f"❌ Failed to bind HTTP server on port {desired_port}: {exc}")
        raise

    if bound_port != desired_port:
        logger.warning(f"Port {desired_port} in use; bound to {bound_port} instead")
    print(f"🧠 Starting ContinuonBrain Server on port {bound_port}...")
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
    
    print(f"📋 Agent Manager Settings:")
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
    
    # Store settings in brain_service for access by ChatWithGemma
    brain_service.agent_settings = agent_settings
    
    # Async Init (hack for sync constructor)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(brain_service.initialize())

    # Bind the autonomous learner to API routes so live training metrics
    # surface through the learning + HOPE monitoring endpoints.
    try:
        background_learner = getattr(brain_service, "background_learner", None)
        if background_learner:
            try:
                from continuonbrain.api.routes import hope_routes, learning_routes

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

    print(f"🚀 Server listening on http://0.0.0.0:{bound_port}")
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
