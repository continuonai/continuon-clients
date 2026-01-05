"""
Training Controller - API endpoints for brain training management.

Simple API for the SimpleBrainTrainer (JAX CoreModel only).
Also includes cloud training endpoints for GCP-based training.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)

# Cloud training service (lazy loaded)
_cloud_training_service: Optional[Any] = None


def _get_cloud_training_service():
    """Lazy load the cloud training service."""
    global _cloud_training_service
    if _cloud_training_service is None:
        try:
            from continuonbrain.services.cloud_training import (
                CloudTrainingService,
                CloudTrainingConfig,
            )
            config_path = Path("/opt/continuonos/brain/config/cloud_training.yaml")
            _cloud_training_service = CloudTrainingService(config_path=config_path)
        except ImportError as e:
            logger.warning(f"Cloud training not available: {e}")
            _cloud_training_service = None
    return _cloud_training_service


class TrainingControllerMixin:
    """
    Mixin for brain training API requests.
    """

    @require_role(UserRole.CONSUMER)
    def handle_training_status(self):
        """
        GET /api/training/status
        Get current training status.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({
                    "success": True,
                    "status": {
                        "enabled": False,
                        "running": False,
                        "message": "Brain trainer not initialized",
                    }
                })
                return

            status = trainer.get_status()
            self.send_json({"success": True, "status": status})
        except Exception as e:
            logger.error(f"Training status error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_training_benchmarks(self):
        """
        GET /api/training/benchmarks
        Get benchmark history.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({
                    "success": True,
                    "benchmarks": [],
                    "message": "No benchmark data available",
                })
                return

            benchmarks = trainer.get_benchmark_history(limit=50)
            self.send_json({
                "success": True,
                "benchmarks": benchmarks,
                "best_score": trainer.best_score,
            })
        except Exception as e:
            logger.error(f"Benchmarks error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_training_progress(self):
        """
        GET /api/training/progress
        Get training progress.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({
                    "success": True,
                    "progress": {
                        "running": False,
                        "total_cycles": 0,
                        "total_steps": 0,
                    }
                })
                return

            progress = {
                "running": trainer.running,
                "paused": trainer.paused,
                "total_cycles": trainer.total_cycles,
                "total_steps": trainer.total_steps_trained,
                "best_score": trainer.best_score,
            }
            self.send_json({"success": True, "progress": progress})
        except Exception as e:
            logger.error(f"Progress error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_training_trigger(self, body: str):
        """
        POST /api/training/trigger
        Trigger training action.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({
                    "success": False,
                    "error": "Brain trainer not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}
            action = data.get("action", "train")

            if action == "train":
                result = trainer.trigger_train()
                self.send_json({
                    "success": result.success,
                    "steps_trained": result.steps_trained,
                    "loss": result.final_loss,
                    "duration": result.duration_seconds,
                    "error": result.error,
                })
            elif action == "benchmark":
                result = trainer.trigger_benchmark()
                if result:
                    self.send_json({
                        "success": True,
                        "score": result.overall_score,
                        "level": result.level_passed,
                        "is_new_best": result.is_new_best,
                    })
                else:
                    self.send_json({"success": False, "error": "Benchmark failed"})
            elif action == "start":
                if not trainer.running:
                    trainer.start()
                self.send_json({"success": True, "running": trainer.running})
            elif action == "stop":
                trainer.stop()
                self.send_json({"success": True, "running": trainer.running})
            elif action == "pause":
                trainer.pause()
                self.send_json({"success": True, "paused": trainer.paused})
            elif action == "resume":
                trainer.resume()
                self.send_json({"success": True, "paused": trainer.paused})
            else:
                self.send_json({
                    "success": False,
                    "error": f"Unknown action: {action}"
                }, status=400)

        except Exception as e:
            logger.error(f"Training trigger error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_training_decisions(self):
        """
        GET /api/training/decisions
        Get training history (decisions/logs).
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({"success": True, "history": []})
                return

            history = trainer.get_training_history(limit=50)
            self.send_json({"success": True, "history": history})
        except Exception as e:
            logger.error(f"Decisions error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_training_models(self):
        """
        GET /api/training/models
        Get model info.
        """
        try:
            brain_service = getattr(self.server, 'brain_service', None)

            model_info = {
                "name": "JAX CoreModel (WaveCore)",
                "framework": "JAX/Flax",
                "available": False,
            }

            if brain_service and brain_service.jax_adapter:
                config = brain_service.jax_adapter.get("config")
                model_info["available"] = True
                if config:
                    model_info["d_s"] = getattr(config, "d_s", None)
                    model_info["d_w"] = getattr(config, "d_w", None)

            trainer = self._get_brain_trainer()
            if trainer:
                model_info["best_score"] = trainer.best_score
                model_info["total_steps_trained"] = trainer.total_steps_trained

            self.send_json({"success": True, "model": model_info})
        except Exception as e:
            logger.error(f"Models error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_training_history(self, body: str = None):
        """
        GET /api/training/history
        Get training history.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({"success": True, "history": []})
                return

            history = trainer.get_training_history(limit=100)
            self.send_json({"success": True, "history": history})
        except Exception as e:
            logger.error(f"History error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_training_trends(self):
        """
        GET /api/training/trends
        Get training trends.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({"success": True, "trends": {}})
                return

            # Calculate simple trends from history
            history = trainer.training_history
            benchmarks = trainer.benchmark_history

            trends = {
                "training_cycles": len(history),
                "benchmark_runs": len(benchmarks),
                "best_score": trainer.best_score,
            }

            if history:
                losses = [h["loss"] for h in history if h.get("loss")]
                if losses:
                    trends["avg_loss"] = sum(losses) / len(losses)
                    trends["recent_loss"] = losses[-1] if losses else None

            if benchmarks:
                scores = [b["score"] for b in benchmarks if b.get("score")]
                if scores:
                    trends["avg_score"] = sum(scores) / len(scores)
                    trends["recent_score"] = scores[-1] if scores else None

            self.send_json({"success": True, "trends": trends})
        except Exception as e:
            logger.error(f"Trends error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_training_config(self, body: str = None):
        """
        GET/POST /api/training/config
        Get or update training config.
        """
        try:
            trainer = self._get_brain_trainer()

            if body:
                # POST - update config
                if trainer is None:
                    self.send_json({"success": False, "error": "Trainer not available"}, status=503)
                    return

                data = json.loads(body)
                if "train_interval_minutes" in data:
                    trainer.config.train_interval_minutes = int(data["train_interval_minutes"])
                if "max_steps_per_cycle" in data:
                    trainer.config.max_steps_per_cycle = int(data["max_steps_per_cycle"])
                if "learning_rate" in data:
                    trainer.config.learning_rate = float(data["learning_rate"])

                self.send_json({"success": True, "message": "Config updated"})
            else:
                # GET
                if trainer is None:
                    self.send_json({"success": True, "config": {"enabled": False}})
                    return

                config = {
                    "enabled": trainer.config.enabled,
                    "train_interval_minutes": trainer.config.train_interval_minutes,
                    "max_steps_per_cycle": trainer.config.max_steps_per_cycle,
                    "learning_rate": trainer.config.learning_rate,
                    "batch_size": trainer.config.batch_size,
                }
                self.send_json({"success": True, "config": config})

        except Exception as e:
            logger.error(f"Config error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_run_benchmark(self, body: str = None):
        """
        POST /api/training/benchmark
        Run a benchmark.
        """
        try:
            trainer = self._get_brain_trainer()
            if trainer is None:
                self.send_json({"success": False, "error": "Trainer not available"}, status=503)
                return

            result = trainer.trigger_benchmark()
            if result:
                self.send_json({
                    "success": True,
                    "result": {
                        "score": result.overall_score,
                        "level": result.level_passed,
                        "tests_passed": result.tests_passed,
                        "total_tests": result.total_tests,
                        "is_new_best": result.is_new_best,
                    }
                })
            else:
                self.send_json({"success": False, "error": "Benchmark failed"})

        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    def _get_brain_trainer(self):
        """Get the brain trainer from brain_service."""
        brain_service = getattr(self.server, 'brain_service', None)
        if brain_service is None:
            return None
        return getattr(brain_service, 'brain_trainer', None)

    # =========================================================================
    # Cloud Training Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_cloud_training_status(self):
        """
        GET /api/training/cloud/status
        Get cloud training service status.
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": True,
                    "status": {
                        "available": False,
                        "message": "Cloud training service not available",
                    }
                })
                return

            status = service.get_status()
            self.send_json({"success": True, "status": status})
        except Exception as e:
            logger.error(f"Cloud training status error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_cloud_training_upload(self, body: str):
        """
        POST /api/training/cloud/upload
        Upload episodes for cloud training.

        Request body:
        {
            "episodes_dir": "/path/to/episodes",  // optional
            "episode_ids": ["ep1", "ep2"],        // optional, filter by IDs
            "compress": true                       // optional, default true
        }
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Cloud training service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            episodes_dir = None
            if data.get("episodes_dir"):
                episodes_dir = Path(data["episodes_dir"])

            episode_ids = data.get("episode_ids")
            compress = data.get("compress", True)

            # Run async upload
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.upload_episodes(
                        local_dir=episodes_dir,
                        episode_ids=episode_ids,
                        compress=compress,
                    )
                )
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=500)

        except Exception as e:
            logger.error(f"Cloud upload error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_cloud_training_trigger(self, body: str):
        """
        POST /api/training/cloud/trigger
        Start a cloud training job.

        Request body:
        {
            "episodes_uri": "gs://bucket/path",   // required if not auto-uploading
            "auto_upload": true,                   // upload local episodes first
            "config": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "model_type": "wavecore",
                "arch_preset": "cloud"
            }
        }
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Cloud training service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            episodes_uri = data.get("episodes_uri")
            auto_upload = data.get("auto_upload", False)
            config_data = data.get("config", {})

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Auto-upload if requested
                if auto_upload and not episodes_uri:
                    upload_result = loop.run_until_complete(
                        service.upload_episodes(compress=True)
                    )
                    if not upload_result.get("success"):
                        self.send_json({
                            "success": False,
                            "error": f"Episode upload failed: {upload_result.get('error')}",
                        }, status=500)
                        return
                    episodes_uri = upload_result.get("gcs_uri")

                if not episodes_uri:
                    self.send_json({
                        "success": False,
                        "error": "episodes_uri required (or use auto_upload=true)",
                    }, status=400)
                    return

                # Create training config
                from continuonbrain.services.cloud_training import TrainingJobConfig
                training_config = TrainingJobConfig(
                    model_type=config_data.get("model_type", "wavecore"),
                    epochs=config_data.get("epochs", 100),
                    batch_size=config_data.get("batch_size", 32),
                    learning_rate=config_data.get("learning_rate", 1e-4),
                    obs_dim=config_data.get("obs_dim", 128),
                    action_dim=config_data.get("action_dim", 32),
                    output_dim=config_data.get("output_dim", 32),
                    arch_preset=config_data.get("arch_preset", "cloud"),
                    use_tpu=config_data.get("use_tpu", False),
                    use_gpu=config_data.get("use_gpu", True),
                    mixed_precision=config_data.get("mixed_precision", True),
                    sparsity_lambda=config_data.get("sparsity_lambda", 0.0),
                )

                # Trigger training
                result = loop.run_until_complete(
                    service.trigger_training(
                        episodes_uri=episodes_uri,
                        config=training_config,
                    )
                )
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=500)

        except Exception as e:
            logger.error(f"Cloud trigger error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_cloud_training_job_status(self, job_id: str):
        """
        GET /api/training/cloud/status/{job_id}
        Get status of a specific training job.
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Cloud training service not available"
                }, status=503)
                return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(service.get_job_status(job_id))
            finally:
                loop.close()

            self.send_json({
                "success": True,
                "job_id": result.job_id,
                "status": result.status.value,
                "model_uri": result.model_uri,
                "final_loss": result.final_loss,
                "training_time_seconds": result.training_time_seconds,
                "epochs_completed": result.epochs_completed,
                "metrics": result.metrics,
                "error_message": result.error_message,
                "created_at": result.created_at,
                "completed_at": result.completed_at,
            })

        except Exception as e:
            logger.error(f"Cloud job status error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_cloud_training_jobs(self):
        """
        GET /api/training/cloud/jobs
        List recent training jobs.
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": True,
                    "jobs": [],
                    "message": "Cloud training service not available",
                })
                return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                jobs = loop.run_until_complete(service.list_jobs(limit=20))
            finally:
                loop.close()

            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "model_uri": job.model_uri,
                    "final_loss": job.final_loss,
                    "training_time_seconds": job.training_time_seconds,
                    "created_at": job.created_at,
                    "completed_at": job.completed_at,
                })

            self.send_json({"success": True, "jobs": jobs_data})

        except Exception as e:
            logger.error(f"Cloud jobs list error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_cloud_training_download(self, body: str):
        """
        POST /api/training/cloud/download/{job_id}
        Download trained model from completed job.

        Request body:
        {
            "job_id": "job_abc123",
            "dest_dir": "/path/to/download",  // optional
            "install": true                    // optional, install to adapters
        }
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Cloud training service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            job_id = data.get("job_id")
            if not job_id:
                self.send_json({
                    "success": False,
                    "error": "job_id is required"
                }, status=400)
                return

            dest_dir = None
            if data.get("dest_dir"):
                dest_dir = Path(data["dest_dir"])

            install = data.get("install", True)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.download_result(
                        job_id=job_id,
                        dest_dir=dest_dir,
                        install=install,
                    )
                )
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=500)

        except Exception as e:
            logger.error(f"Cloud download error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)

    @require_role(UserRole.CREATOR)
    def handle_cloud_training_cancel(self, body: str):
        """
        POST /api/training/cloud/cancel
        Cancel a running or queued training job.

        Request body:
        {
            "job_id": "job_abc123"
        }
        """
        try:
            service = _get_cloud_training_service()
            if service is None:
                self.send_json({
                    "success": False,
                    "error": "Cloud training service not available"
                }, status=503)
                return

            data = json.loads(body) if body else {}

            job_id = data.get("job_id")
            if not job_id:
                self.send_json({
                    "success": False,
                    "error": "job_id is required"
                }, status=400)
                return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(service.cancel_job(job_id))
            finally:
                loop.close()

            if result.get("success"):
                self.send_json(result)
            else:
                self.send_json(result, status=400)

        except Exception as e:
            logger.error(f"Cloud cancel error: {e}")
            self.send_json({"success": False, "error": str(e)}, status=500)
