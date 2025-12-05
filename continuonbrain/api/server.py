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

            elif self.path == "/api/tasks/library":
                # Return task library with eligibility checks
                tasks = brain_service.get_task_library()
                self.send_json({"tasks": tasks})

            elif self.path == "/api/status/introspection":
                # Introspection endpoint for Brain Status page
                identity_service.self_report() # Updates internal state
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
                
                response = brain_service.ChatWithGemma(msg, [])
                self.send_json({"response": response})
            
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
                data = json.loads(body)
                # Save to disk
                settings_path = Path(brain_service.config_dir) / "settings.json"
                with open(settings_path, "w") as f:
                    json.dump(data, f, indent=2)
                self.send_json({"success": True})
            
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
                except Exception as e:
                    self.send_json({"success": False, "message": str(e)})
            
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
    
    # If Desktop Station, force mock hardware for robot components?
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
