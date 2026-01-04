"""
Simple Brain Trainer - Single-model autonomous training for JAX CoreModel.

This is THE training system for ContinuonBrain. It trains one model:
the JAX CoreModel (WaveCore seed).

Training loop:
1. Collect episodes from RLDS
2. Train CoreModel on episodes
3. Run benchmarks
4. If improved, save as new best
5. Repeat
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Simple training configuration."""
    enabled: bool = True

    # Training intervals
    train_interval_minutes: int = 30
    benchmark_interval_minutes: int = 60

    # Training parameters
    max_steps_per_cycle: int = 32
    batch_size: int = 4
    learning_rate: float = 2e-4

    # Paths
    rlds_dir: Path = Path("/opt/continuonos/brain/rlds/episodes")
    checkpoint_dir: Path = Path("/opt/continuonos/brain/checkpoints")
    model_dir: Path = Path("/opt/continuonos/brain/model/seed_stable")

    # Resource limits
    max_memory_percent: float = 70.0
    pause_on_high_temp: bool = True
    max_temp_c: float = 75.0


@dataclass
class TrainingResult:
    """Result of a training cycle."""
    success: bool
    steps_trained: int
    final_loss: float
    duration_seconds: float
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    overall_score: float
    level_passed: str
    tests_passed: int
    total_tests: int
    is_new_best: bool


class SimpleBrainTrainer:
    """
    Simple autonomous trainer for the JAX CoreModel brain.

    One brain. One training loop. Simple.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        resource_monitor=None,
    ):
        self.config = config or TrainingConfig()
        self.resource_monitor = resource_monitor

        # State
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self.total_cycles = 0
        self.total_steps_trained = 0
        self.best_score = 0.0
        self.start_time: Optional[float] = None
        self.last_train_time: Optional[float] = None
        self.last_benchmark_time: Optional[float] = None

        # History
        self.training_history: List[Dict] = []
        self.benchmark_history: List[Dict] = []

        # Components (lazy loaded)
        self._trainer = None
        self._benchmark_tracker = None

        logger.info("SimpleBrainTrainer initialized")

    def _ensure_trainer(self):
        """Lazy load the WaveCore trainer."""
        if self._trainer is None:
            from continuonbrain.services.wavecore_trainer import WavecoreTrainer
            self._trainer = WavecoreTrainer(
                default_rlds_dir=self.config.rlds_dir,
                checkpoint_dir=self.config.checkpoint_dir,
                export_dir=self.config.model_dir / "adapters" / "candidate",
            )
            logger.info("WavecoreTrainer loaded")

    def _ensure_benchmark_tracker(self):
        """Lazy load the benchmark tracker."""
        if self._benchmark_tracker is None:
            from continuonbrain.services.benchmark_tracker import BenchmarkTracker
            self._benchmark_tracker = BenchmarkTracker()
            logger.info("BenchmarkTracker loaded")

    def start(self):
        """Start autonomous training."""
        if self.running:
            logger.warning("Trainer already running")
            return

        if not self.config.enabled:
            logger.info("Training is disabled in config")
            return

        self._ensure_trainer()
        self._ensure_benchmark_tracker()

        self.running = True
        self.paused = False
        self.start_time = time.time()
        self._stop_event.clear()

        self.thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="SimpleBrainTrainer"
        )
        self.thread.start()

        logger.info("SimpleBrainTrainer started")

    def stop(self):
        """Stop training."""
        if not self.running:
            return

        logger.info("Stopping SimpleBrainTrainer...")
        self.running = False
        self._stop_event.set()

        if self.thread:
            self.thread.join(timeout=10.0)

        logger.info("SimpleBrainTrainer stopped")

    def pause(self):
        """Pause training."""
        self.paused = True
        logger.info("Training paused")

    def resume(self):
        """Resume training."""
        self.paused = False
        logger.info("Training resumed")

    def trigger_train(self) -> TrainingResult:
        """Manually trigger a training cycle."""
        logger.info("Manual training triggered")
        return self._run_training_cycle()

    def trigger_benchmark(self) -> Optional[BenchmarkResult]:
        """Manually trigger a benchmark."""
        logger.info("Manual benchmark triggered")
        return self._run_benchmark()

    def _check_resources(self) -> bool:
        """Check if resources allow training."""
        if not self.resource_monitor:
            return True

        try:
            res = self.resource_monitor.check_resources()

            # Check resource level
            if hasattr(res, 'level'):
                level = res.level.value if hasattr(res.level, 'value') else str(res.level)
                if level in ["critical", "emergency"]:
                    logger.debug(f"Resource constraint: {level}")
                    return False

            # Check memory
            if hasattr(res, 'memory_percent'):
                if res.memory_percent > self.config.max_memory_percent:
                    logger.debug(f"Memory too high: {res.memory_percent:.1f}%")
                    return False

            # Check temperature
            if self.config.pause_on_high_temp and hasattr(res, 'cpu_temp'):
                if res.cpu_temp and res.cpu_temp > self.config.max_temp_c:
                    logger.debug(f"CPU temp too high: {res.cpu_temp:.1f}°C")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True

    def _training_loop(self):
        """Main training loop."""
        logger.info("Training loop started")

        while self.running:
            try:
                # Check if paused
                if self.paused:
                    self._stop_event.wait(timeout=5.0)
                    continue

                # Check resources
                if not self._check_resources():
                    self._stop_event.wait(timeout=30.0)
                    continue

                # Check if it's time to train
                train_interval = self.config.train_interval_minutes * 60
                if (self.last_train_time is None or
                    time.time() - self.last_train_time > train_interval):

                    # Run training cycle
                    result = self._run_training_cycle()
                    self.last_train_time = time.time()

                    if result.success:
                        self.total_steps_trained += result.steps_trained
                        self.total_cycles += 1

                        # Run benchmark after training
                        benchmark_result = self._run_benchmark()
                        if benchmark_result and benchmark_result.is_new_best:
                            logger.info(f"New best score: {benchmark_result.overall_score:.3f}")
                            self.best_score = benchmark_result.overall_score

                # Sleep between checks
                self._stop_event.wait(timeout=60.0)

            except Exception as e:
                logger.error(f"Training loop error: {e}", exc_info=True)
                self._stop_event.wait(timeout=60.0)

        logger.info("Training loop ended")

    def _run_training_cycle(self) -> TrainingResult:
        """Run a single training cycle."""
        start_time = time.time()

        try:
            self._ensure_trainer()

            # Run training with WaveCore trainer
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._trainer.run_loops({
                        "fast": {"max_steps": 8, "learning_rate": 1e-3},
                        "mid": {"max_steps": 16, "learning_rate": 5e-4},
                        "slow": {
                            "max_steps": self.config.max_steps_per_cycle,
                            "learning_rate": self.config.learning_rate,
                            "batch_size": self.config.batch_size,
                        },
                        "compact_export": True,
                    })
                )
            finally:
                loop.close()

            duration = time.time() - start_time

            # Extract results
            slow_result = result.get("slow", {}).get("result", {})
            final_loss = slow_result.get("final_loss", 0.0)
            steps = slow_result.get("steps_trained", 0)
            checkpoint_path = result.get("slow", {}).get("checkpoint_dir")

            training_result = TrainingResult(
                success=result.get("status") == "ok",
                steps_trained=steps,
                final_loss=final_loss,
                duration_seconds=duration,
                checkpoint_path=checkpoint_path,
            )

            # Record in history
            self.training_history.append({
                "timestamp": time.time(),
                "steps": steps,
                "loss": final_loss,
                "duration": duration,
                "success": training_result.success,
            })

            # Keep only last 100 entries
            if len(self.training_history) > 100:
                self.training_history = self.training_history[-100:]

            logger.info(
                f"Training cycle complete: {steps} steps, "
                f"loss={final_loss:.4f}, {duration:.1f}s"
            )

            return training_result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Training cycle failed: {e}")
            return TrainingResult(
                success=False,
                steps_trained=0,
                final_loss=0.0,
                duration_seconds=duration,
                error=str(e),
            )

    def _run_benchmark(self) -> Optional[BenchmarkResult]:
        """Run benchmark evaluation."""
        try:
            from continuonbrain.eval.benchmark_suite import BenchmarkSuite

            suite = BenchmarkSuite(
                model_dir=self.config.model_dir,
                tracker=self._benchmark_tracker,
                use_lightweight_encoder=True,
            )

            result = suite.run_quick_benchmark()

            is_new_best = result.overall_score > self.best_score

            benchmark_result = BenchmarkResult(
                overall_score=result.overall_score,
                level_passed=result.highest_level_passed,
                tests_passed=result.tests_passed,
                total_tests=result.total_tests,
                is_new_best=is_new_best,
            )

            # Record in history
            self.benchmark_history.append({
                "timestamp": time.time(),
                "score": result.overall_score,
                "level": result.highest_level_passed,
                "is_new_best": is_new_best,
            })

            if len(self.benchmark_history) > 100:
                self.benchmark_history = self.benchmark_history[-100:]

            self.last_benchmark_time = time.time()

            logger.info(
                f"Benchmark: score={result.overall_score:.3f}, "
                f"level={result.highest_level_passed}, "
                f"new_best={is_new_best}"
            )

            return benchmark_result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get trainer status."""
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "enabled": self.config.enabled,
            "running": self.running,
            "paused": self.paused,
            "total_cycles": self.total_cycles,
            "total_steps_trained": self.total_steps_trained,
            "best_score": self.best_score,
            "uptime_hours": uptime / 3600,
            "last_train": datetime.fromtimestamp(self.last_train_time).isoformat() if self.last_train_time else None,
            "last_benchmark": datetime.fromtimestamp(self.last_benchmark_time).isoformat() if self.last_benchmark_time else None,
            "config": {
                "train_interval_minutes": self.config.train_interval_minutes,
                "max_steps_per_cycle": self.config.max_steps_per_cycle,
                "learning_rate": self.config.learning_rate,
            },
            "recent_training": self.training_history[-5:] if self.training_history else [],
            "recent_benchmarks": self.benchmark_history[-5:] if self.benchmark_history else [],
        }

    def get_training_history(self, limit: int = 50) -> List[Dict]:
        """Get training history."""
        return self.training_history[-limit:]

    def get_benchmark_history(self, limit: int = 50) -> List[Dict]:
        """Get benchmark history."""
        return self.benchmark_history[-limit:]
