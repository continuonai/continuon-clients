"""
Update Controller - REST API endpoints for OTA model updates.

Provides endpoints for:
- Checking for updates
- Downloading updates
- Activating updates
- Rolling back to previous version
- Getting update status

Usage:
    Add UpdateControllerMixin to the BrainRequestHandler class:

    from continuonbrain.api.controllers.update_controller import UpdateControllerMixin

    class BrainRequestHandler(BaseHTTPRequestHandler, UpdateControllerMixin, ...):
        ...
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)

# Global OTA updater and scheduler instances
_ota_updater = None
_update_scheduler = None


def get_ota_updater():
    """Get or create the global OTA updater instance."""
    global _ota_updater
    if _ota_updater is None:
        try:
            from continuonbrain.services.ota_updater import OTAUpdater
            # Try to get config_dir from brain_service
            config_dir = Path("/opt/continuonos/brain")
            try:
                from continuonbrain.api import server
                if hasattr(server, 'brain_service') and server.brain_service:
                    config_dir = Path(server.brain_service.config_dir)
            except Exception:
                pass
            _ota_updater = OTAUpdater(config_dir=config_dir)
            logger.info(f"OTA Updater initialized with config_dir: {config_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize OTA Updater: {e}")
    return _ota_updater


def get_update_scheduler():
    """Get or create the global update scheduler instance."""
    global _update_scheduler
    if _update_scheduler is None:
        try:
            from continuonbrain.services.update_scheduler import UpdateScheduler
            updater = get_ota_updater()
            if updater:
                _update_scheduler = UpdateScheduler(
                    updater=updater,
                    check_interval_hours=24,
                    auto_download=False,
                    auto_activate=False,
                )
                logger.info("Update Scheduler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Update Scheduler: {e}")
    return _update_scheduler


def init_update_services(config_dir: Path = None, auto_start_scheduler: bool = False):
    """
    Initialize update services with custom configuration.

    Args:
        config_dir: Configuration directory path
        auto_start_scheduler: Start the background scheduler immediately
    """
    global _ota_updater, _update_scheduler

    try:
        from continuonbrain.services.ota_updater import OTAUpdater
        from continuonbrain.services.update_scheduler import UpdateScheduler

        if config_dir is None:
            config_dir = Path("/opt/continuonos/brain")

        _ota_updater = OTAUpdater(config_dir=config_dir)
        _update_scheduler = UpdateScheduler(
            updater=_ota_updater,
            check_interval_hours=24,
            auto_download=False,
            auto_activate=False,
        )

        if auto_start_scheduler:
            _update_scheduler.start()

        logger.info(f"Update services initialized (config_dir: {config_dir})")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize update services: {e}")
        return False


class UpdateControllerMixin:
    """
    Mixin for processing OTA Update API requests.

    Endpoints:
    - GET /api/updates/check - Check for available updates
    - POST /api/updates/download - Download an update
    - POST /api/updates/activate - Activate a downloaded update
    - POST /api/updates/rollback - Rollback to previous version
    - GET /api/updates/status - Get current update status
    - GET /api/updates/scheduler - Get scheduler status
    - POST /api/updates/scheduler/start - Start background scheduler
    - POST /api/updates/scheduler/stop - Stop background scheduler
    """

    def send_json(self, payload: dict, status: int = 200) -> None:
        """Send JSON response (expected to be defined by base handler)."""
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))

    @require_role(UserRole.CONSUMER)
    def handle_check_updates(self):
        """
        GET /api/updates/check
        Check for available model updates.

        Response:
        {
            "success": true,
            "update_available": true/false,
            "current_version": "1.0.0",
            "update": {
                "model_id": "seed",
                "version": "1.1.0",
                "size_bytes": 123456,
                "release_notes": "Bug fixes and improvements"
            }
        }
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            # Run async check in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                update = loop.run_until_complete(updater.check_for_updates())
            finally:
                loop.close()

            current_version = updater._get_current_version()

            response = {
                "success": True,
                "update_available": update is not None,
                "current_version": current_version,
            }

            if update:
                response["update"] = {
                    "model_id": update.model_id,
                    "version": update.version,
                    "size_bytes": update.size_bytes,
                    "release_notes": update.release_notes,
                    "min_brain_version": update.min_brain_version,
                    "download_url": update.download_url,
                }

            self.send_json(response)

        except Exception as e:
            logger.error(f"Update check failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_download_update(self, body: str):
        """
        POST /api/updates/download
        Download an available update to the candidate directory.

        Request body (optional):
        {
            "model_id": "seed",
            "version": "1.1.0"
        }

        If no body provided, downloads the latest available update.

        Response:
        {
            "success": true,
            "message": "Update downloaded successfully",
            "version": "1.1.0",
            "candidate_path": "/opt/continuonos/brain/model/candidate"
        }
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            data = json.loads(body) if body else {}
            model_id = data.get("model_id", "seed")
            requested_version = data.get("version")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Get update info
                if requested_version:
                    update = loop.run_until_complete(
                        updater.registry.get_model_info(model_id, requested_version)
                    )
                else:
                    update = loop.run_until_complete(updater.check_for_updates(model_id))

                if not update:
                    self.send_json({
                        "success": False,
                        "error": "No update available"
                    }, status=404)
                    return

                # Download update
                candidate_path = loop.run_until_complete(
                    updater.download_update(update)
                )

                self.send_json({
                    "success": True,
                    "message": "Update downloaded successfully",
                    "version": update.version,
                    "candidate_path": str(candidate_path),
                })

            finally:
                loop.close()

        except ValueError as e:
            logger.error(f"Download verification failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=400)

        except Exception as e:
            logger.error(f"Download failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_activate_update(self, body: str):
        """
        POST /api/updates/activate
        Activate a downloaded update (candidate -> current).

        Request body (optional):
        {
            "run_health_check": true,
            "force": false
        }

        Response:
        {
            "success": true,
            "message": "Update activated successfully",
            "new_version": "1.1.0",
            "rolled_back": false
        }
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            data = json.loads(body) if body else {}
            run_health_check = data.get("run_health_check", True)
            force = data.get("force", False)

            # Check if candidate exists
            if not (updater.candidate_dir / "manifest.json").exists():
                self.send_json({
                    "success": False,
                    "error": "No candidate update to activate. Download an update first."
                }, status=400)
                return

            # Get candidate version before activation
            with open(updater.candidate_dir / "manifest.json") as f:
                manifest = json.load(f)
                candidate_version = manifest.get("version", "unknown")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                success = loop.run_until_complete(
                    updater.activate_update(run_health_check=run_health_check and not force)
                )

                if success:
                    self.send_json({
                        "success": True,
                        "message": "Update activated successfully",
                        "new_version": candidate_version,
                        "rolled_back": False,
                    })
                else:
                    self.send_json({
                        "success": False,
                        "message": "Update activation failed health check, rolled back to previous version",
                        "attempted_version": candidate_version,
                        "rolled_back": True,
                    }, status=422)

            finally:
                loop.close()

        except ValueError as e:
            logger.error(f"Activation failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=400)

        except Exception as e:
            logger.error(f"Activation failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_rollback_update(self, body: str):
        """
        POST /api/updates/rollback
        Rollback to the previous version.

        Response:
        {
            "success": true,
            "message": "Rolled back successfully",
            "version": "1.0.0"
        }
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            # Check if rollback is available
            if not updater.rollback_dir.exists():
                self.send_json({
                    "success": False,
                    "error": "No rollback available"
                }, status=400)
                return

            # Get rollback version
            rollback_version = "unknown"
            rollback_manifest = updater.rollback_dir / "manifest.json"
            if rollback_manifest.exists():
                with open(rollback_manifest) as f:
                    manifest = json.load(f)
                    rollback_version = manifest.get("version", "unknown")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                success = loop.run_until_complete(updater.rollback())

                if success:
                    self.send_json({
                        "success": True,
                        "message": "Rolled back successfully",
                        "version": rollback_version,
                    })
                else:
                    self.send_json({
                        "success": False,
                        "error": "Rollback failed"
                    }, status=500)

            finally:
                loop.close()

        except ValueError as e:
            logger.error(f"Rollback failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=400)

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_update_status(self):
        """
        GET /api/updates/status
        Get current update system status.

        Response:
        {
            "success": true,
            "status": {
                "state": "idle",
                "current_version": "1.0.0",
                "available_update": {...},
                "progress_percent": 0,
                "error_message": null,
                "last_check": "2025-01-05T12:00:00Z",
                "last_update": "2025-01-01T10:00:00Z",
                "rollback_available": true
            },
            "installed_versions": {
                "current": "1.0.0",
                "candidate": null,
                "rollback": "0.9.0"
            }
        }
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            status = updater.get_status()
            versions = updater.get_installed_versions()

            self.send_json({
                "success": True,
                "status": status.to_dict(),
                "installed_versions": versions,
            })

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_scheduler_status(self):
        """
        GET /api/updates/scheduler
        Get scheduler status.

        Response:
        {
            "success": true,
            "scheduler": {
                "state": "running",
                "next_check": "2025-01-06T12:00:00Z",
                "last_check": "2025-01-05T12:00:00Z",
                "checks_completed": 10,
                "updates_found": 2,
                "auto_download": false,
                "auto_activate": false
            }
        }
        """
        scheduler = get_update_scheduler()
        if not scheduler:
            self.send_json({
                "success": False,
                "error": "Update Scheduler not available"
            }, status=503)
            return

        try:
            status = scheduler.get_status()

            self.send_json({
                "success": True,
                "scheduler": status.to_dict(),
            })

        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_scheduler_start(self, body: str):
        """
        POST /api/updates/scheduler/start
        Start the background update scheduler.

        Request body (optional):
        {
            "check_interval_hours": 24,
            "auto_download": false,
            "auto_activate": false
        }
        """
        scheduler = get_update_scheduler()
        if not scheduler:
            self.send_json({
                "success": False,
                "error": "Update Scheduler not available"
            }, status=503)
            return

        try:
            data = json.loads(body) if body else {}

            # Update scheduler settings
            if "check_interval_hours" in data:
                scheduler.set_check_interval(data["check_interval_hours"])
            if "auto_download" in data:
                scheduler.set_auto_download(data["auto_download"])
            if "auto_activate" in data:
                scheduler.set_auto_activate(data["auto_activate"])

            scheduler.start()

            self.send_json({
                "success": True,
                "message": "Update scheduler started",
                "scheduler": scheduler.get_status().to_dict(),
            })

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_scheduler_stop(self, body: str):
        """
        POST /api/updates/scheduler/stop
        Stop the background update scheduler.
        """
        scheduler = get_update_scheduler()
        if not scheduler:
            self.send_json({
                "success": False,
                "error": "Update Scheduler not available"
            }, status=503)
            return

        try:
            scheduler.stop()

            self.send_json({
                "success": True,
                "message": "Update scheduler stopped",
            })

        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_scheduler_trigger(self, body: str):
        """
        POST /api/updates/scheduler/trigger
        Trigger an immediate update check.
        """
        scheduler = get_update_scheduler()
        if not scheduler:
            # If scheduler not available, still allow manual check
            return self.handle_check_updates()

        try:
            scheduler.trigger_check_now()

            self.send_json({
                "success": True,
                "message": "Update check triggered",
            })

        except Exception as e:
            logger.error(f"Failed to trigger check: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_cleanup_candidate(self, body: str):
        """
        POST /api/updates/cleanup/candidate
        Clean up the candidate directory.
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(updater.cleanup_candidate())
            finally:
                loop.close()

            self.send_json({
                "success": True,
                "message": "Candidate directory cleaned up",
            })

        except Exception as e:
            logger.error(f"Failed to cleanup candidate: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)

    @require_role(UserRole.CREATOR)
    def handle_cleanup_rollback(self, body: str):
        """
        POST /api/updates/cleanup/rollback
        Clean up the rollback directory to save space.
        Warning: This removes the ability to rollback!
        """
        updater = get_ota_updater()
        if not updater:
            self.send_json({
                "success": False,
                "error": "OTA Updater not available"
            }, status=503)
            return

        try:
            data = json.loads(body) if body else {}
            confirm = data.get("confirm", False)

            if not confirm:
                self.send_json({
                    "success": False,
                    "error": "Confirmation required. Set 'confirm': true to proceed. "
                             "Warning: This removes the ability to rollback!"
                }, status=400)
                return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(updater.cleanup_rollback())
            finally:
                loop.close()

            self.send_json({
                "success": True,
                "message": "Rollback directory cleaned up",
            })

        except Exception as e:
            logger.error(f"Failed to cleanup rollback: {e}")
            self.send_json({
                "success": False,
                "error": str(e)
            }, status=500)
