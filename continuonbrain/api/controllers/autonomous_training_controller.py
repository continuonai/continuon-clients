"""
Autonomous Training Controller - API endpoints for autonomous training management.

This controller provides API endpoints for the autonomous training scheduler,
model validation, model distribution, and robot registration.

Endpoints:
    /api/autonomous/status              - Get autonomous scheduler status
    /api/autonomous/start               - Start autonomous training scheduler
    /api/autonomous/stop                - Stop autonomous training scheduler
    /api/autonomous/trigger             - Manually trigger training
    /api/autonomous/episodes            - Get episode quality scores
    /api/autonomous/gaps                - Get capability gaps
    /api/autonomous/validate            - Validate a model
    /api/autonomous/distribute          - Distribute model to robots
    /api/autonomous/robots              - Robot registration management
    /api/autonomous/updates             - OTA update management
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)


# Lazy-loaded services
_autonomous_scheduler: Optional[Any] = None
_model_validator: Optional[Any] = None
_distribution_service: Optional[Any] = None


def _get_autonomous_scheduler():
    """Lazy load the autonomous training scheduler."""
    global _autonomous_scheduler
    if _autonomous_scheduler is None:
        try:
            from continuonbrain.services.autonomous_training_scheduler import (
                AutonomousTrainingScheduler,
                TrainingTriggerConfig,
            )
            config_path = Path("/opt/continuonos/brain/config/training_scheduler.json")
            _autonomous_scheduler = AutonomousTrainingScheduler(config_path=config_path)
        except ImportError as e:
            logger.warning(f"Autonomous training scheduler not available: {e}")
            _autonomous_scheduler = None
    return _autonomous_scheduler


def _get_model_validator():
    """Lazy load the model validator."""
    global _model_validator
    if _model_validator is None:
        try:
            from continuonbrain.services.model_validator import (
                ModelValidator,
                ValidationConfig,
            )
            baseline_dir = Path("/opt/continuonos/brain/model/current")
            _model_validator = ModelValidator(baseline_dir=baseline_dir)
        except ImportError as e:
            logger.warning(f"Model validator not available: {e}")
            _model_validator = None
    return _model_validator


def _get_distribution_service():
    """Lazy load the model distribution service."""
    global _distribution_service
    if _distribution_service is None:
        try:
            from continuonbrain.services.model_distribution import (
                ModelDistributionService,
                DistributionConfig,
            )
            config_path = Path("/opt/continuonos/brain/config/distribution.json")
            _distribution_service = ModelDistributionService(config_path=config_path)
        except ImportError as e:
            logger.warning(f"Model distribution service not available: {e}")
            _distribution_service = None
    return _distribution_service


class AutonomousTrainingControllerMixin:
    """
    Mixin for autonomous training API endpoints.
    """

    # =========================================================================
    # Autonomous Scheduler Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_autonomous_status(self):
        """
        GET /api/autonomous/status
        Get autonomous training scheduler status.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": True,
                    "status": {
                        "available": False,
                        "message": "Autonomous training scheduler not available",
                    }
                })
                return

            status = scheduler.get_status()
            self.send_json({
                "success": True,
                "status": status.to_dict(),
            })
        except Exception as e:
            logger.error(f"Autonomous status error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_autonomous_start(self, body: str = None):
        """
        POST /api/autonomous/start
        Start the autonomous training scheduler.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": False,
                    "error": "Autonomous training scheduler not available"
                }, status=503)
                return

            scheduler.start()
            self.send_json({
                "success": True,
                "message": "Autonomous training scheduler started",
                "is_running": scheduler._running,
            })
        except Exception as e:
            logger.error(f"Autonomous start error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_autonomous_stop(self, body: str = None):
        """
        POST /api/autonomous/stop
        Stop the autonomous training scheduler.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": False,
                    "error": "Autonomous training scheduler not available"
                }, status=503)
                return

            scheduler.stop()
            self.send_json({
                "success": True,
                "message": "Autonomous training scheduler stopped",
                "is_running": scheduler._running,
            })
        except Exception as e:
            logger.error(f"Autonomous stop error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_autonomous_trigger(self, body: str):
        """
        POST /api/autonomous/trigger
        Manually trigger training.

        Request body:
        {
            "mode": "local_slow"  // local_fast, local_mid, local_slow, cloud_slow, cloud_full
        }
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": False,
                    "error": "Autonomous training scheduler not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}
            mode_str = data.get("mode", "local_slow")

            from continuonbrain.services.autonomous_training_scheduler import TrainingMode
            try:
                mode = TrainingMode(mode_str)
            except ValueError:
                self.send_json({
                    "success": False,
                    "error": f"Invalid mode: {mode_str}. Valid modes: {[m.value for m in TrainingMode]}"
                }, status=400)
                return

            success = scheduler.trigger_training_now(mode=mode)
            self.send_json({
                "success": success,
                "message": f"Training triggered with mode {mode.value}" if success else "Failed to trigger training",
            })
        except Exception as e:
            logger.error(f"Autonomous trigger error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_autonomous_config(self, body: str = None):
        """
        GET/POST /api/autonomous/config
        Get or update autonomous training configuration.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": False,
                    "error": "Autonomous training scheduler not available"
                }, status=503)
                return

            if body:
                # POST - update config
                data = json.loads(body)
                config = scheduler.config

                # Update allowed fields
                if "min_episodes_for_local" in data:
                    config.min_episodes_for_local = int(data["min_episodes_for_local"])
                if "min_episodes_for_cloud" in data:
                    config.min_episodes_for_cloud = int(data["min_episodes_for_cloud"])
                if "min_quality_score" in data:
                    config.min_quality_score = float(data["min_quality_score"])
                if "max_hours_without_training" in data:
                    config.max_hours_without_training = float(data["max_hours_without_training"])
                if "enable_cloud_training" in data:
                    config.enable_cloud_training = bool(data["enable_cloud_training"])
                if "auto_deploy" in data:
                    config.auto_deploy = bool(data["auto_deploy"])

                # Save config
                config_path = Path("/opt/continuonos/brain/config/training_scheduler.json")
                config.save(config_path)

                self.send_json({
                    "success": True,
                    "message": "Configuration updated",
                })
            else:
                # GET - return current config
                from dataclasses import asdict
                self.send_json({
                    "success": True,
                    "config": asdict(scheduler.config),
                })
        except Exception as e:
            logger.error(f"Autonomous config error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    # =========================================================================
    # Episode Quality Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_autonomous_episodes(self):
        """
        GET /api/autonomous/episodes
        Get episode quality scores.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": True,
                    "episodes": [],
                    "message": "Autonomous training scheduler not available",
                })
                return

            # Get scored episodes
            episodes_data = []
            for score in scheduler._episode_scores:
                episodes_data.append(score.to_dict())

            # Sort by quality score
            episodes_data.sort(key=lambda e: e.get("overall_score", 0), reverse=True)

            self.send_json({
                "success": True,
                "episodes": episodes_data[:100],  # Limit to 100
                "total_count": len(episodes_data),
            })
        except Exception as e:
            logger.error(f"Episodes error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_autonomous_episodes_ready(self):
        """
        GET /api/autonomous/episodes/ready
        Get episodes ready for training.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": True,
                    "episodes": [],
                })
                return

            ready_episodes = scheduler.scorer.get_training_ready_episodes(
                min_quality=scheduler.config.min_quality_score,
                limit=50,
            )

            episodes_data = [e.to_dict() for e in ready_episodes]

            self.send_json({
                "success": True,
                "episodes": episodes_data,
                "count": len(episodes_data),
            })
        except Exception as e:
            logger.error(f"Ready episodes error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    # =========================================================================
    # Capability Gap Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_autonomous_gaps(self):
        """
        GET /api/autonomous/gaps
        Get capability gaps analysis.
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": True,
                    "gaps": [],
                })
                return

            gaps = scheduler.gap_detector.get_capability_gaps()
            recommendations = scheduler.gap_detector.get_training_recommendations()

            self.send_json({
                "success": True,
                "gaps": gaps,
                "recommendations": recommendations,
            })
        except Exception as e:
            logger.error(f"Gaps error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_autonomous_gaps_record(self, body: str):
        """
        POST /api/autonomous/gaps/record
        Record a capability failure or success.

        Request body:
        {
            "task_type": "navigation",
            "success": false,
            "context": {"environment": "indoor"},
            "error_type": "timeout",
            "severity": 0.7
        }
        """
        try:
            scheduler = _get_autonomous_scheduler()
            if scheduler is None:
                self.send_json({
                    "success": False,
                    "error": "Autonomous training scheduler not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            task_type = data.get("task_type")
            if not task_type:
                self.send_json({
                    "success": False,
                    "error": "task_type is required"
                }, status=400)
                return

            if data.get("success", False):
                scheduler.gap_detector.record_success(task_type)
            else:
                scheduler.gap_detector.record_failure(
                    task_type=task_type,
                    context=data.get("context", {}),
                    error_type=data.get("error_type", "unknown"),
                    severity=data.get("severity", 0.5),
                )

            self.send_json({
                "success": True,
                "message": "Recorded",
            })
        except Exception as e:
            logger.error(f"Record gap error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    # =========================================================================
    # Model Validation Endpoints
    # =========================================================================

    @require_role(UserRole.CREATOR)
    def handle_autonomous_validate(self, body: str):
        """
        POST /api/autonomous/validate
        Validate a model before deployment.

        Request body:
        {
            "model_path": "/path/to/model",
            "level": "standard"  // quick, standard, full
        }
        """
        try:
            validator = _get_model_validator()
            if validator is None:
                self.send_json({
                    "success": False,
                    "error": "Model validator not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            model_path = data.get("model_path")
            if not model_path:
                self.send_json({
                    "success": False,
                    "error": "model_path is required"
                }, status=400)
                return

            level_str = data.get("level", "standard")
            from continuonbrain.services.model_validator import ValidationLevel
            try:
                level = ValidationLevel(level_str)
            except ValueError:
                level = ValidationLevel.STANDARD

            # Run validation asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    validator.validate_model(Path(model_path), level=level)
                )
            finally:
                loop.close()

            self.send_json({
                "success": True,
                "validation": result.to_dict(),
            })
        except Exception as e:
            logger.error(f"Validate error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    # =========================================================================
    # Model Distribution Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_distribution_status(self):
        """
        GET /api/autonomous/distribution/status
        Get model distribution service status.
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": True,
                    "status": {
                        "available": False,
                        "message": "Distribution service not available",
                    }
                })
                return

            status = service.get_status()
            self.send_json({
                "success": True,
                "status": status,
            })
        except Exception as e:
            logger.error(f"Distribution status error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_distribution_upload(self, body: str):
        """
        POST /api/autonomous/distribution/upload
        Upload a model for distribution.

        Request body:
        {
            "model_path": "/path/to/model",
            "version": "1.0.0",
            "model_id": "seed",
            "release_notes": "Initial release"
        }
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            model_path = data.get("model_path")
            version = data.get("version")

            if not model_path or not version:
                self.send_json({
                    "success": False,
                    "error": "model_path and version are required"
                }, status=400)
                return

            # Run upload asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.upload_model(
                        model_path=Path(model_path),
                        version=version,
                        model_id=data.get("model_id", "seed"),
                        release_notes=data.get("release_notes", ""),
                        name=data.get("name"),
                        description=data.get("description"),
                    )
                )
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=500)

        except Exception as e:
            logger.error(f"Distribution upload error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_distribution_distribute(self, body: str):
        """
        POST /api/autonomous/distribution/distribute
        Distribute a model update to robots.

        Request body:
        {
            "model_id": "seed",
            "version": "1.0.0",
            "target_robots": ["robot_001", "robot_002"],  // optional, all if not specified
            "priority": "normal",
            "rollout_percentage": 100.0
        }
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            model_id = data.get("model_id", "seed")
            version = data.get("version")

            if not version:
                self.send_json({
                    "success": False,
                    "error": "version is required"
                }, status=400)
                return

            from continuonbrain.services.model_distribution import UpdatePriority
            priority_str = data.get("priority", "normal")
            try:
                priority = UpdatePriority(priority_str)
            except ValueError:
                priority = UpdatePriority.NORMAL

            # Run distribution asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.distribute_update(
                        model_id=model_id,
                        version=version,
                        target_robots=data.get("target_robots"),
                        priority=priority,
                        rollout_percentage=data.get("rollout_percentage", 100.0),
                    )
                )
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=500)

        except Exception as e:
            logger.error(f"Distribution distribute error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    # =========================================================================
    # Robot Registration Endpoints
    # =========================================================================

    @require_role(UserRole.CREATOR)
    def handle_robots_register(self, body: str):
        """
        POST /api/autonomous/robots/register
        Register a new robot.

        Request body:
        {
            "device_id": "robot_001",
            "robot_model": "donkeycar",
            "owner_uid": "user_abc",
            "hardware_profile": "pi5",
            "name": "My Robot",
            "auto_update": true
        }
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            device_id = data.get("device_id")
            robot_model = data.get("robot_model")
            owner_uid = data.get("owner_uid")

            if not device_id or not robot_model or not owner_uid:
                self.send_json({
                    "success": False,
                    "error": "device_id, robot_model, and owner_uid are required"
                }, status=400)
                return

            # Run registration asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.register_robot(
                        device_id=device_id,
                        robot_model=robot_model,
                        owner_uid=owner_uid,
                        hardware_profile=data.get("hardware_profile", "pi5"),
                        name=data.get("name"),
                        auto_update=data.get("auto_update", True),
                    )
                )
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=400)

        except Exception as e:
            logger.error(f"Robot register error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_robots_list(self):
        """
        GET /api/autonomous/robots
        List registered robots.
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": True,
                    "robots": [],
                })
                return

            # Run list asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                robots = loop.run_until_complete(service.list_robots(limit=100))
            finally:
                loop.close()

            robots_data = [r.to_dict() for r in robots]

            self.send_json({
                "success": True,
                "robots": robots_data,
                "count": len(robots_data),
            })
        except Exception as e:
            logger.error(f"Robots list error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_robots_get(self, device_id: str):
        """
        GET /api/autonomous/robots/{device_id}
        Get information about a specific robot.
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            # Run get asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                robot = loop.run_until_complete(service.get_robot(device_id))
            finally:
                loop.close()

            if robot is None:
                self.send_json({
                    "success": False,
                    "error": f"Robot not found: {device_id}"
                }, status=404)
                return

            self.send_json({
                "success": True,
                "robot": robot.to_dict(),
            })
        except Exception as e:
            logger.error(f"Robots get error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_robots_heartbeat(self, body: str):
        """
        POST /api/autonomous/robots/heartbeat
        Robot heartbeat (called by robot).

        Request body:
        {
            "device_id": "robot_001",
            "brain_version": "0.1.0",
            "current_model_version": "1.0.0",
            "ip_address": "192.168.1.100"
        }
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            device_id = data.get("device_id")
            if not device_id:
                self.send_json({
                    "success": False,
                    "error": "device_id is required"
                }, status=400)
                return

            # Run update asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                success = loop.run_until_complete(
                    service.update_robot_status(
                        device_id=device_id,
                        brain_version=data.get("brain_version"),
                        current_model_version=data.get("current_model_version"),
                        ip_address=data.get("ip_address"),
                    )
                )

                # Also get pending updates
                pending_updates = loop.run_until_complete(
                    service.get_pending_updates(device_id)
                )
            finally:
                loop.close()

            self.send_json({
                "success": success,
                "pending_updates": pending_updates,
            })
        except Exception as e:
            logger.error(f"Robots heartbeat error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_robots_ack_update(self, body: str):
        """
        POST /api/autonomous/robots/ack-update
        Acknowledge an update (called by robot).

        Request body:
        {
            "device_id": "robot_001",
            "update_id": "update_abc123",
            "action": "installed"  // downloaded, installed, rejected
        }
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            device_id = data.get("device_id")
            update_id = data.get("update_id")
            action = data.get("action")

            if not device_id or not update_id or not action:
                self.send_json({
                    "success": False,
                    "error": "device_id, update_id, and action are required"
                }, status=400)
                return

            # Run acknowledge asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                success = loop.run_until_complete(
                    service.acknowledge_update(
                        device_id=device_id,
                        update_id=update_id,
                        action=action,
                    )
                )
            finally:
                loop.close()

            self.send_json({
                "success": success,
            })
        except Exception as e:
            logger.error(f"Robots ack update error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    # =========================================================================
    # Model Registry Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_registry_latest(self, model_id: str = "seed"):
        """
        GET /api/autonomous/registry/latest/{model_id}
        Get latest version of a model.
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": True,
                    "latest_version": None,
                })
                return

            # Run get asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                version = loop.run_until_complete(service.get_latest_version(model_id))
            finally:
                loop.close()

            self.send_json({
                "success": True,
                "model_id": model_id,
                "latest_version": version,
            })
        except Exception as e:
            logger.error(f"Registry latest error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_registry_info(self, model_id: str, version: str = None):
        """
        GET /api/autonomous/registry/info/{model_id}/{version}
        Get model information from registry.
        """
        try:
            service = _get_distribution_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Distribution service not available"
                }, status=503)
                return

            # Run get asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                info = loop.run_until_complete(service.get_model_info(model_id, version))
            finally:
                loop.close()

            if info is None:
                self.send_json({
                    "success": False,
                    "error": f"Model not found: {model_id} v{version}"
                }, status=404)
                return

            self.send_json({
                "success": True,
                "model": info,
            })
        except Exception as e:
            logger.error(f"Registry info error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)
