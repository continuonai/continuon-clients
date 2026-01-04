"""
Training Controller - API endpoints for brain training management.

Simple API for the SimpleBrainTrainer (JAX CoreModel only).
"""

import json
import logging
from typing import Any, Dict

from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)


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
