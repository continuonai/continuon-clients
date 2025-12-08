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
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
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
from continuonbrain.agent_identity import AgentIdentity
from continuonbrain.api.routes import ui_routes
from continuonbrain.settings_manager import SettingsStore, SettingsValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrainServer")

# Global Service Instance
brain_service: BrainService = None
identity_service: AgentIdentity = None
background_learner = None  # Autonomous learning service

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
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_dashboard_html().encode("utf-8"))

            elif self.path == "/ui/chat":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(ui_routes.get_chat_html().encode("utf-8"))

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

            elif self.path == "/api/status":
                # Legacy robot status
                # TODO: Implement full status serialization
                self.send_json({"status": "ok", "mode": "idle"})

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
            
            elif self.path == "/api/agent/info":
                # Aggregate info from chat agent and learning agent
                chat_info = brain_service.get_chat_agent_info()
                
                learning_info = {"enabled": False, "status": "disabled"}
                if background_learner:
                    learning_info = background_learner.get_status()
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
                    # Check if model is changing
                    old_settings = store.load()
                    old_model = old_settings.get("agent_manager", {}).get("agent_model", "mock")
                    new_model = data.get("agent_manager", {}).get("agent_model", "mock")
                    
                    # Save settings first
                    settings = store.save(data)
                    
                    # If model changed, switch it dynamically
                    switch_result = None
                    if old_model != new_model:
                        logger.info(f"Model change detected: {old_model} -> {new_model}")
                        switch_result = brain_service.switch_model(new_model)
                    
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
                    detector.detect_all()
                    devices = detector.generate_config()
                    
                    response = {
                        "success": True,
                        "device_count": len(devices.get("devices", {})),
                        "devices": {
                            "camera": "depth_camera" in devices.get("primary", {}),
                            "arm": "servo_controller" in devices.get("primary", {}) or "servo_controller" in devices.get("devices", {}),
                            "drivetrain": "servo_controller" in devices.get("primary", {}) or "servo_controller" in devices.get("devices", {})
                        },
                        "message": "Scan complete"
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

            else:
                self.send_error(404)

        except Exception as e:
            logger.error(f"POST error: {e}")
            self.send_error(500)

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
    
    # Legacy BackgroundLearner initialization removed. 
    # It is now handled by BrainService in verify_learning_startup logic / production code.
    
    server = ThreadedHTTPServer(("0.0.0.0", args.port), BrainRequestHandler)
    print(f"🚀 Server listening on http://0.0.0.0:{args.port}")
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
