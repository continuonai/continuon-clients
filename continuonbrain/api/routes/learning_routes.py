"""
Learning API Routes

Endpoints for controlling autonomous learning.
"""

import json
from typing import Dict, Any


# Global reference to background learner (set by server)
# Global reference to background learner (set by server)
_background_learner = None
_brain_service = None


def set_background_learner(learner):
    """Set the global background learner instance."""
    global _background_learner
    _background_learner = learner


def set_brain_service(service):
    """Set the global BrainService instance."""
    global _brain_service
    _brain_service = service


def handle_learning_request(handler):
    """Handle learning API requests."""
    path = handler.path
    
    try:
        if path == "/api/learning/status":
            # Get learning status with comprehensive metrics
            if _background_learner:
                data = _background_learner.get_status()
            else:
                data = {"enabled": False, "running": False, "error": "Learner not initialized"}
            handler.send_json(data)
            
        elif path == "/api/learning/progress":
            # Get detailed learning progress metrics
            if _background_learner:
                status = _background_learner.get_status()
                data = {
                    'total_steps': status['total_steps'],
                    'total_episodes': status['total_episodes'],
                    'learning_updates': status['learning_updates'],
                    'avg_parameter_change': status['avg_parameter_change'],
                    'recent_parameter_change': status['recent_parameter_change'],
                    'avg_episode_reward': status['avg_episode_reward'],
                    'recent_episode_reward': status['recent_episode_reward'],
                    'learning_rate': status['learning_rate'],
                    'is_stable': status['is_stable'],
                }
            else:
                data = {"error": "Learner not initialized"}
            handler.send_json(data)
            
        elif path == "/api/learning/metrics":
            # Get all metrics including curiosity and stability
            if _background_learner:
                data = _background_learner.get_status()
            else:
                data = {"error": "Learner not initialized"}
            handler.send_json(data)
            
        elif path == "/api/learning/pause":
            # Pause learning
            if _background_learner:
                _background_learner.pause()
                handler.send_json({"success": True, "message": "Learning paused"})
            else:
                handler.send_json({"success": False, "error": "Learner not initialized"}, status=503)
            
        elif path == "/api/learning/resume":
            # Resume learning
            if _background_learner:
                _background_learner.resume()
                handler.send_json({"success": True, "message": "Learning resumed"})
            else:
                handler.send_json({"success": False, "error": "Learner not initialized"}, status=503)
            
        elif path == "/api/learning/reset":
            # Reset learning (stop and restart)
            if _background_learner:
                _background_learner.stop()
                _background_learner.start()
                handler.send_json({"success": True, "message": "Learning reset"})
            else:
                handler.send_json({"success": False, "error": "Learner not initialized"}, status=503)

        elif path == "/api/learning/chat_learn":
            # Trigger ad-hoc chat learning session (e.g. with Gemini)
            with open("/tmp/debug_route_hit", "w") as f:
                 f.write(f"Hit at {time.time()}\n")
            
            if not _brain_service:
                with open("/tmp/debug_route_error", "w") as f: f.write("BrainService None\n")
                handler.send_json({"success": False, "error": "BrainService not initialized"}, status=503)
                return

            # Parse payload
            content_len = int(handler.headers.get('Content-Length', 0))
            body = handler.rfile.read(content_len).decode('utf-8')
            with open("/tmp/debug_route_payload", "w") as f: f.write(body)
            payload = json.loads(body) if body else {}

            try:
                # Use asyncio to run the async method properly
                import asyncio
                # We need to run this on the event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(_brain_service.RunChatLearn(payload))
                loop.close()
                
                handler.send_json({"success": True, "result": result})
            except Exception as e:
                handler.send_json({"success": False, "error": str(e)}, status=500)
        
        else:
            handler.send_json({"error": "Unknown endpoint"}, status=404)
            
    except Exception as e:
        with open("/tmp/debug_route_exception", "w") as f:
            import traceback
            traceback.print_exc(file=f)
        handler.send_json({"error": str(e)}, status=500)
