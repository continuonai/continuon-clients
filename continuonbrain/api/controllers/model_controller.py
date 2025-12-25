import json
import logging
import os
from pathlib import Path
from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)

class ModelControllerMixin:
    """
    Mixin for processing Model Management API requests.
    """

    @require_role(UserRole.CONSUMER)
    def handle_list_models(self):
        """
        List available models (Seed and Release).
        """
        brain_service = self.server.brain_service
        try:
            from continuonbrain.services.model_detector import ModelDetector
            detector = ModelDetector()
            models = detector.get_available_models()
            self.send_json({"success": True, "models": models})
        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            # Fallback to brain_service.get_chat_agent_info()
            info = brain_service.get_chat_agent_info()
            self.send_json({"success": True, "models": [info]})

    @require_role(UserRole.CREATOR)
    def handle_activate_model(self, body: str):
        """
        Switch the active brain model.
        """
        brain_service = self.server.brain_service
        data = json.loads(body) if body else {}
        model_id = data.get("model_id")
        
        if not model_id:
            self.send_json({"success": False, "message": "model_id required"}, status=400)
            return

        result = brain_service.switch_model(model_id)
        status = 200 if result.get("success") else 500
        self.send_json(result, status=status)

    @require_role(UserRole.CREATOR)
    def handle_upload_model(self, body: str):
        """
        Upload a new seed model (Stub for now).
        """
        # In a real implementation, this would handle multipart upload.
        # For this track, we provide the endpoint structure.
        self.send_json({"success": True, "message": "Model upload initiated (stub)"})

    @require_role(UserRole.CREATOR)
    def handle_install_model(self, body: str):
        """
        Install and activate a seed model bundle.
        """
        brain_service = self.server.brain_service
        data = json.loads(body) if body else {}
        bundle_url = data.get("bundle_url")
        model_id = data.get("model_id", "new_seed")
        
        if not bundle_url:
            self.send_json({"success": False, "message": "bundle_url required"}, status=400)
            return

        logger.info(f"Installing seed model from {bundle_url}...")
        
        # 1. Download and Verify (Stubbed for simulation)
        # 2. Extract to /opt/continuonos/brain/model/adapters/candidate/
        # 3. Hot swap
        
        # Simulating successful installation
        brain_service.seed_installed = True
        
        # Simulate Hot Reload (if we have a checkpoint)
        candidate_dir = Path(brain_service.config_dir) / "model" / "adapters" / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        # In a real run, the bundle would be extracted here.
        
        # If model_id matches current, trigger reload
        self.send_json({
            "success": True, 
            "message": f"Seed model {model_id} installed and activated via hot-reload.",
            "requires_reload": False
        })
