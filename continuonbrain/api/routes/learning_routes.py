"""
Learning API Routes

Endpoints for controlling autonomous learning.
"""

import json
from typing import Dict, Any


# Global reference to background learner (set by server)
_background_learner = None


def set_background_learner(learner):
    """Set the global background learner instance."""
    global _background_learner
    _background_learner = learner


def handle_learning_request(handler):
    """Handle learning API requests."""
    path = handler.path
    
    try:
        if path == "/api/learning/status":
            # Get learning status
            if _background_learner:
                data = _background_learner.get_status()
            else:
                data = {"enabled": False, "running": False, "error": "Learner not initialized"}
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
        
        else:
            handler.send_json({"error": "Unknown endpoint"}, status=404)
            
    except Exception as e:
        handler.send_json({"error": str(e)}, status=500)
