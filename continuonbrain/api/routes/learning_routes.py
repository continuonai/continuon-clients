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
        
        else:
            handler.send_json({"error": "Unknown endpoint"}, status=404)
            
    except Exception as e:
        handler.send_json({"error": str(e)}, status=500)
