import json
from pathlib import Path
from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role
from continuonbrain.services.reset_manager import ResetManager, ResetProfile, ResetRequest
from continuonbrain.services.promotion_manager import PromotionManager

class AdminControllerMixin:
    """
    Mixin for processing Admin API requests.
    Expects self to be a BaseHTTPRequestHandler with access to:
    - self.server.brain_service
    - self.headers
    - self.rfile
    - self.send_json
    - self.send_error
    """

    @require_role(UserRole.CREATOR)
    def handle_factory_reset(self, body: str):
        brain_service = self.server.brain_service
        payload = json.loads(body) if body else {}
        if not isinstance(payload, dict):
            self.send_json({"success": False, "message": "invalid payload"}, status=400)
            return

        profile = str(payload.get("profile") or "factory").strip()
        confirm = str(payload.get("confirm") or "").strip()
        token = payload.get("token") or self.headers.get("x-continuon-admin-token")
        dry_run = bool(payload.get("dry_run", False))

        # Gate on mode and allow_motion
        gates = brain_service.mode_manager.get_gate_snapshot() if brain_service.mode_manager else {}
        mode = (gates or {}).get("mode") or "unknown"
        allow_motion = (gates or {}).get("allow_motion")
        
        if mode not in ("idle", "emergency_stop") or allow_motion not in (False, None):
            self.send_json(
                {
                    "success": False,
                    "message": "reset blocked: requires mode=idle or emergency_stop and allow_motion=false",
                    "mode": mode,
                    "allow_motion": allow_motion,
                },
                status=409,
            )
            return

        manager = ResetManager()
        try:
            prof = ResetProfile(profile)
        except Exception:
            self.send_json({"success": False, "message": f"unknown profile: {profile}"}, status=400)
            return

        expected_confirm = manager.CONFIRM_FACTORY if prof == ResetProfile.FACTORY else manager.CONFIRM_MEMORIES
        if confirm != expected_confirm:
            self.send_json({"success": False, "message": "confirmation required", "confirm_expected": expected_confirm}, status=400)
            return

        runtime_root = Path("/opt/continuonos/brain")
        config_dir = Path(getattr(brain_service, "config_dir", ""))
        
        # Token check inside manager.authorize using provided token
        if not manager.authorize(provided_token=token, runtime_root=runtime_root, config_dir=config_dir):
            self.send_json(
                {
                    "success": False,
                    "message": "not authorized (set CONTINUON_ADMIN_TOKEN or CONTINUON_ALLOW_UNSAFE_RESET=1 for dev)",
                },
                status=403,
            )
            return

        req = ResetRequest(profile=prof, dry_run=dry_run, config_dir=config_dir, runtime_root=runtime_root)
        res = manager.run(req)
        self.send_json({"success": res.success, "result": res.__dict__}, status=200 if res.success else 500)

    @require_role(UserRole.CREATOR)
    def handle_promote_candidate(self, body: str):
        brain_service = self.server.brain_service
        payload = json.loads(body) if body else {}
        if not isinstance(payload, dict):
            self.send_json({"success": False, "message": "invalid payload"}, status=400)
            return

        token = payload.get("token") or self.headers.get("x-continuon-admin-token")
        dry_run = bool(payload.get("dry_run", False))

        gates = brain_service.mode_manager.get_gate_snapshot() if brain_service.mode_manager else {}
        mode = (gates or {}).get("mode") or "unknown"
        allow_motion = (gates or {}).get("allow_motion")
        
        if mode not in ("idle", "emergency_stop") or allow_motion not in (False, None):
            self.send_json(
                {
                    "success": False,
                    "message": "promotion blocked: requires mode=idle or emergency_stop and allow_motion=false",
                    "mode": mode,
                    "allow_motion": allow_motion,
                },
                status=409,
            )
            return

        runtime_root = Path("/opt/continuonos/brain")
        config_dir = Path(getattr(brain_service, "config_dir", ""))
        
        mgr = PromotionManager()
        if not mgr.authorize(provided_token=token, runtime_root=runtime_root, config_dir=config_dir):
            self.send_json(
                {
                    "success": False,
                    "message": "not authorized (set CONTINUON_ADMIN_TOKEN or CONTINUON_ALLOW_UNSAFE_RESET=1 for dev)",
                },
                status=403,
            )
            return

        res = mgr.promote(runtime_root=runtime_root, dry_run=dry_run)
        self.send_json({"success": res.success, "result": res.__dict__}, status=200 if res.success else 500)
