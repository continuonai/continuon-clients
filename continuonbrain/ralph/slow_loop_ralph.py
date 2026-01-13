"""
Slow Loop Ralph Layer
=====================

Wraps the cloud training/planning loop with Ralph's context rotation and guardrails.

Characteristics:
- 1s+ latency (cloud operations)
- Planning and goal management
- Weight updates via OTA
- Aggregated learning from RLDS
- Knowledge base updates
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import (
    RalphLayer,
    RalphConfig,
    RalphState,
    LoopType,
    Guardrail,
    MetaLayerContext,
)

logger = logging.getLogger(__name__)


@dataclass
class SlowLoopInput:
    """Input for slow loop iteration."""
    rlds_episodes: List[Dict[str, Any]] = field(default_factory=list)
    user_corrections: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_updates: List[Dict[str, Any]] = field(default_factory=list)
    goal_updates: List[str] = field(default_factory=list)
    trigger: str = "scheduled"  # scheduled, user_request, error_threshold


@dataclass
class SlowLoopOutput:
    """Output from slow loop iteration."""
    training_started: bool = False
    model_updated: bool = False
    new_model_version: Optional[str] = None
    knowledge_integrated: int = 0
    goals_updated: List[str] = field(default_factory=list)
    next_scheduled: Optional[str] = None


@dataclass
class TrainingJob:
    """A cloud training job."""
    job_id: str
    status: str  # pending, running, completed, failed
    started_at: str
    completed_at: Optional[str] = None
    episodes_used: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


class SlowLoopRalph(RalphLayer):
    """
    Ralph layer for the cloud training/planning loop.

    This loop handles:
    - Aggregating RLDS episodes for training
    - Cloud TPU training orchestration
    - OTA model weight updates
    - Long-term goal management
    - Knowledge base integration

    Guardrails for this loop focus on:
    - Training quality thresholds
    - Model regression prevention
    - Safe weight update procedures
    - Resource usage limits
    """

    def __init__(
        self,
        cloud_trainer: Any = None,
        knowledge_base: Any = None,
        ota_manager: Any = None,
        **kwargs
    ):
        config = RalphConfig(
            loop_type=LoopType.SLOW,
            target_latency_ms=5000.0,  # 5 seconds for cloud ops
            max_iterations=100,
            context_window_tokens=32768,
        )
        super().__init__(config, **kwargs)

        self.cloud_trainer = cloud_trainer
        self.knowledge_base = knowledge_base
        self.ota_manager = ota_manager

        # Training state
        self._current_job: Optional[TrainingJob] = None
        self._training_history: List[TrainingJob] = []
        self._episode_queue: List[Dict[str, Any]] = []

        # Goal management
        self._active_goals: List[str] = []
        self._completed_goals: List[str] = []

        # Scheduling
        self._last_training: Optional[datetime] = None
        self._training_interval = timedelta(hours=4)

    async def execute_iteration(self, state: RalphState, input_data: Any) -> RalphState:
        """
        Execute a single slow loop iteration.
        """
        start_time = time.perf_counter()

        if not isinstance(input_data, SlowLoopInput):
            input_data = SlowLoopInput(trigger="manual")

        # Queue new episodes
        if input_data.rlds_episodes:
            self._episode_queue.extend(input_data.rlds_episodes)
            state.context_summary = f"Queued {len(input_data.rlds_episodes)} new episodes"

        output = SlowLoopOutput()

        # Check if training should run
        should_train = await self._should_train(input_data, state)

        if should_train:
            # Check guardrails before training
            triggered = self.check_guardrails(
                action="start_training",
                context={"episodes": len(self._episode_queue)}
            )

            if not any(g.severity == "critical" for g in triggered):
                output = await self._run_training_cycle(input_data, state)
            else:
                state.errors.append(f"Training blocked: {triggered[0].instruction}")

        # Process knowledge updates
        if input_data.knowledge_updates:
            integrated = await self._integrate_knowledge(input_data.knowledge_updates)
            output.knowledge_integrated = integrated

        # Update goals
        if input_data.goal_updates:
            output.goals_updated = await self._update_goals(input_data.goal_updates)

        # Update state
        latency_ms = (time.perf_counter() - start_time) * 1000
        state.last_action = f"slow_loop:{input_data.trigger}"
        state.last_result = f"trained={output.training_started}, knowledge={output.knowledge_integrated}"
        state.metrics["last_latency_ms"] = latency_ms
        state.metrics["episode_queue_size"] = len(self._episode_queue)
        state.metrics["active_goals"] = len(self._active_goals)

        # Schedule next iteration
        output.next_scheduled = self._schedule_next()

        return state

    async def _should_train(self, input_data: SlowLoopInput, state: RalphState) -> bool:
        """Determine if training should run."""

        # Explicit request
        if input_data.trigger == "user_request":
            return True

        # Error threshold exceeded
        if input_data.trigger == "error_threshold":
            return len(self._episode_queue) > 0

        # Scheduled training
        if self._last_training is None:
            return len(self._episode_queue) > 10

        time_since = datetime.now() - self._last_training
        if time_since > self._training_interval:
            return len(self._episode_queue) > 10

        # Enough episodes accumulated
        if len(self._episode_queue) > 100:
            return True

        return False

    async def _run_training_cycle(
        self,
        input_data: SlowLoopInput,
        state: RalphState
    ) -> SlowLoopOutput:
        """Run a cloud training cycle."""

        output = SlowLoopOutput()

        # Create training job
        job = TrainingJob(
            job_id=f"train_{int(time.time())}",
            status="pending",
            started_at=datetime.now().isoformat(),
            episodes_used=len(self._episode_queue)
        )

        self._current_job = job
        output.training_started = True

        try:
            # Include user corrections as high-priority examples
            if input_data.user_corrections:
                self._episode_queue.extend([
                    {**c, "priority": "high"} for c in input_data.user_corrections
                ])

            # Run training (simulated or real)
            if self.cloud_trainer:
                result = await self._train_on_cloud(job)
            else:
                result = await self._simulate_training(job)

            if result.get("success"):
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.metrics = result.get("metrics", {})

                # Apply model update via OTA
                if self.ota_manager and result.get("model_path"):
                    updated = await self._apply_ota_update(result["model_path"])
                    if updated:
                        output.model_updated = True
                        output.new_model_version = result.get("version")

                # Clear used episodes
                self._episode_queue = []
                self._last_training = datetime.now()

            else:
                job.status = "failed"
                state.errors.append(f"Training failed: {result.get('error', 'unknown')}")

                # Add guardrail for failure pattern
                self.add_guardrail(
                    trigger=f"training_failure_{result.get('error_type', 'unknown')}",
                    instruction=f"Investigate training failure: {result.get('error', 'unknown')}",
                    severity="error",
                    iteration=state.iteration
                )

        except Exception as e:
            job.status = "failed"
            state.errors.append(str(e))
            logger.exception("Training cycle failed")

        self._training_history.append(job)
        self._current_job = None

        return output

    async def _train_on_cloud(self, job: TrainingJob) -> Dict[str, Any]:
        """Submit training job to cloud TPU."""

        if not hasattr(self.cloud_trainer, 'train'):
            return {"success": False, "error": "trainer not configured"}

        try:
            result = await self.cloud_trainer.train(
                episodes=self._episode_queue,
                job_id=job.job_id
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _simulate_training(self, job: TrainingJob) -> Dict[str, Any]:
        """Simulate training for testing."""

        await asyncio.sleep(0.5)  # Simulate some work

        return {
            "success": True,
            "metrics": {
                "loss": 0.15,
                "accuracy": 0.92,
                "episodes_processed": job.episodes_used
            },
            "version": f"v{int(time.time())}",
            "model_path": None  # No actual model in simulation
        }

    async def _apply_ota_update(self, model_path: str) -> bool:
        """Apply OTA model update safely."""

        if not self.ota_manager:
            return False

        try:
            # Check guardrails for model updates
            triggered = self.check_guardrails(
                action="ota_update",
                context={"path": model_path}
            )

            if any(g.severity == "critical" for g in triggered):
                logger.warning(f"OTA update blocked by guardrail: {triggered[0].instruction}")
                return False

            # Apply update
            if hasattr(self.ota_manager, 'apply_update'):
                result = await self.ota_manager.apply_update(model_path)
                return result.get("success", False)

        except Exception as e:
            logger.error(f"OTA update failed: {e}")
            self.add_guardrail(
                trigger="ota_failure",
                instruction=f"OTA update failed: {e}",
                severity="error",
                iteration=self.load_fresh_context().iteration
            )

        return False

    async def _integrate_knowledge(self, updates: List[Dict[str, Any]]) -> int:
        """Integrate knowledge updates into the knowledge base."""

        if not self.knowledge_base:
            return 0

        integrated = 0
        for update in updates:
            try:
                if hasattr(self.knowledge_base, 'add'):
                    await self.knowledge_base.add(update)
                    integrated += 1
            except Exception as e:
                logger.warning(f"Failed to integrate knowledge: {e}")

        return integrated

    async def _update_goals(self, goal_updates: List[str]) -> List[str]:
        """Update active goals."""

        updated = []
        for goal in goal_updates:
            if goal.startswith("+"):
                # Add goal
                new_goal = goal[1:].strip()
                if new_goal not in self._active_goals:
                    self._active_goals.append(new_goal)
                    updated.append(f"added: {new_goal}")
            elif goal.startswith("-"):
                # Complete goal
                done_goal = goal[1:].strip()
                if done_goal in self._active_goals:
                    self._active_goals.remove(done_goal)
                    self._completed_goals.append(done_goal)
                    updated.append(f"completed: {done_goal}")
            else:
                # Add as new goal
                if goal not in self._active_goals:
                    self._active_goals.append(goal)
                    updated.append(f"added: {goal}")

        return updated

    def _schedule_next(self) -> str:
        """Schedule the next slow loop iteration."""

        if self._last_training:
            next_time = self._last_training + self._training_interval
        else:
            next_time = datetime.now() + timedelta(hours=1)

        return next_time.isoformat()

    async def should_continue(self, state: RalphState) -> bool:
        """
        Slow loop continues indefinitely on schedule.
        """
        if state.status == "failed":
            # Allow recovery
            return state.iteration < 3

        return True

    # ========== Meta-Layer Features ==========

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""

        successful = [j for j in self._training_history if j.status == "completed"]
        failed = [j for j in self._training_history if j.status == "failed"]

        return {
            "total_jobs": len(self._training_history),
            "successful": len(successful),
            "failed": len(failed),
            "episode_queue_size": len(self._episode_queue),
            "last_training": self._last_training.isoformat() if self._last_training else None,
            "active_goals": self._active_goals,
            "completed_goals": len(self._completed_goals)
        }

    def get_model_health(self) -> Dict[str, Any]:
        """Get model health metrics."""

        if not self._training_history:
            return {"status": "no_training"}

        recent = self._training_history[-5:]
        success_rate = sum(1 for j in recent if j.status == "completed") / len(recent)

        return {
            "success_rate": success_rate,
            "recent_jobs": len(recent),
            "status": "healthy" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "unhealthy"
        }

    def queue_episode(self, episode: Dict[str, Any]) -> None:
        """Queue an episode for training."""
        self._episode_queue.append(episode)

    def add_goal(self, goal: str) -> None:
        """Add a new goal."""
        if goal not in self._active_goals:
            self._active_goals.append(goal)

    def complete_goal(self, goal: str) -> bool:
        """Mark a goal as completed."""
        if goal in self._active_goals:
            self._active_goals.remove(goal)
            self._completed_goals.append(goal)
            return True
        return False
