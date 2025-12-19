import json
import logging
import os
from pathlib import Path
from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)

class DataControllerMixin:
    """
    Mixin for processing Data/RLDS Management API requests.
    """

    @require_role(UserRole.DEVELOPER)
    def handle_list_episodes(self):
        """
        List RLDS episodes collected on device.
        """
        brain_service = self.server.brain_service
        config_dir = Path(brain_service.config_dir)
        rlds_dir = config_dir / "recordings" / "episodes"
        
        episodes = []
        if rlds_dir.exists():
            for f in rlds_dir.glob("*.json"):
                try:
                    # Just return metadata for listing
                    episodes.append({
                        "id": f.stem,
                        "timestamp": f.stat().st_mtime,
                        "size": f.stat().st_size
                    })
                except Exception:
                    pass
        
        self.send_json({"success": True, "episodes": episodes})

    @require_role(UserRole.CREATOR)
    def handle_tag_episode(self, body: str):
        """
        Tag an episode for training quality (e.g. 'Gold', 'Failure').
        """
        data = json.loads(body) if body else {}
        episode_id = data.get("episode_id")
        tag = data.get("tag")
        
        if not episode_id or not tag:
            self.send_json({"success": False, "message": "episode_id and tag required"}, status=400)
            return

        # Implementation would update the metadata or move the file
        # For now, acknowledge the tag.
        logger.info(f"Tagged episode {episode_id} as {tag}")
        self.send_json({"success": True, "episode_id": episode_id, "tag": tag})
