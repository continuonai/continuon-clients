"""
Base Ralph Layer Implementation
===============================

Core concepts:
1. Context Rotation - Each iteration starts fresh, reads state from files
2. Guardrails (Signs) - Lessons learned persist across context resets
3. State Persistence - Progress saved to disk/git, not memory
4. Meta Layer - Each loop has introspection and self-improvement capabilities
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class LoopType(Enum):
    """The three loops plus safety kernel."""
    FAST = "fast"      # 10ms - reflexes, motor control
    MID = "mid"        # 100ms - skills, attention
    SLOW = "slow"      # 1s+ - planning, learning
    SAFETY = "safety"  # Ring 0 - constitutional constraints


@dataclass
class RalphConfig:
    """Configuration for a Ralph layer."""
    loop_type: LoopType
    max_iterations: int = 100
    context_window_tokens: int = 32000
    guardrails_path: Path = field(default_factory=lambda: Path(".ralph/guardrails"))
    state_path: Path = field(default_factory=lambda: Path(".ralph/state"))
    progress_path: Path = field(default_factory=lambda: Path(".ralph/progress"))

    # Loop-specific timing
    target_latency_ms: float = 10.0  # Default for fast loop

    # Meta-layer settings
    enable_introspection: bool = True
    enable_self_improvement: bool = True
    log_decisions: bool = True

    def __post_init__(self):
        """Set loop-specific defaults."""
        if self.loop_type == LoopType.FAST:
            self.target_latency_ms = 10.0
            self.max_iterations = 1000  # Many fast iterations
        elif self.loop_type == LoopType.MID:
            self.target_latency_ms = 100.0
            self.max_iterations = 100
        elif self.loop_type == LoopType.SLOW:
            self.target_latency_ms = 1000.0
            self.max_iterations = 20
        elif self.loop_type == LoopType.SAFETY:
            self.target_latency_ms = 1.0  # Safety must be fastest
            self.max_iterations = float('inf')  # Never stop checking


@dataclass
class Guardrail:
    """A lesson learned (sign) that prevents repeated mistakes."""
    id: str
    trigger: str
    instruction: str
    severity: str  # "warning", "error", "critical"
    loop_type: LoopType
    created_at: str
    iteration_added: int
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "loop_type": self.loop_type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Guardrail":
        data["loop_type"] = LoopType(data["loop_type"])
        return cls(**data)


@dataclass
class RalphState:
    """Persisted state for a Ralph layer."""
    loop_type: LoopType
    iteration: int = 0
    status: str = "idle"  # idle, running, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    last_action: str = ""
    last_result: str = ""
    errors: List[str] = field(default_factory=list)
    guardrails: List[Guardrail] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    context_summary: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_type": self.loop_type.value,
            "iteration": self.iteration,
            "status": self.status,
            "progress": self.progress,
            "last_action": self.last_action,
            "last_result": self.last_result,
            "errors": self.errors,
            "guardrails": [g.to_dict() for g in self.guardrails],
            "metrics": self.metrics,
            "context_summary": self.context_summary,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RalphState":
        data["loop_type"] = LoopType(data["loop_type"])
        data["guardrails"] = [Guardrail.from_dict(g) for g in data.get("guardrails", [])]
        return cls(**data)


@dataclass
class MetaLayerContext:
    """Context for the meta-layer introspection."""
    loop_type: LoopType
    parent_state: Optional[RalphState] = None
    sibling_states: List[RalphState] = field(default_factory=list)
    global_guardrails: List[Guardrail] = field(default_factory=list)
    performance_history: List[Dict[str, float]] = field(default_factory=list)

    def get_cross_loop_insights(self) -> List[str]:
        """Get insights from other loops that may be relevant."""
        insights = []
        for state in self.sibling_states:
            if state.errors:
                insights.append(
                    f"[{state.loop_type.value}] Recent errors: {state.errors[-3:]}"
                )
            if state.guardrails:
                for g in state.guardrails[-3:]:
                    insights.append(
                        f"[{state.loop_type.value}] Guardrail: {g.trigger} -> {g.instruction}"
                    )
        return insights


class RalphLayer(ABC):
    """
    Base class for Ralph layers wrapping each loop.

    Each layer implements:
    1. Context rotation - Fresh start each iteration
    2. Guardrails - Persistent lessons learned
    3. State persistence - Save progress to disk
    4. Meta-layer - Self-introspection and improvement
    """

    def __init__(self, config: RalphConfig, base_path: Optional[Path] = None):
        self.config = config
        self.base_path = base_path or Path.cwd()

        # Set up paths
        self.guardrails_path = self.base_path / config.guardrails_path / config.loop_type.value
        self.state_path = self.base_path / config.state_path / f"{config.loop_type.value}.json"
        self.progress_path = self.base_path / config.progress_path / f"{config.loop_type.value}.md"

        # Ensure directories exist
        self.guardrails_path.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

        # Meta-layer context
        self.meta_context: Optional[MetaLayerContext] = None

        # Callbacks for cross-loop communication
        self._on_guardrail_added: List[Callable[[Guardrail], None]] = []
        self._on_state_changed: List[Callable[[RalphState], None]] = []

    # ========== Context Rotation ==========

    def load_fresh_context(self) -> RalphState:
        """
        Load state from disk for a fresh context.
        This is the key to Ralph's context rotation.
        """
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                state = RalphState.from_dict(data)
                logger.info(f"[{self.config.loop_type.value}] Loaded state: iteration={state.iteration}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        # Return fresh state
        return RalphState(loop_type=self.config.loop_type)

    def save_state(self, state: RalphState) -> None:
        """Persist state to disk."""
        state.updated_at = datetime.now().isoformat()
        with open(self.state_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)

        # Notify callbacks
        for callback in self._on_state_changed:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def rotate_context(self, state: RalphState) -> RalphState:
        """
        Perform context rotation - save current state and prepare for next iteration.
        """
        # Save current state
        self.save_state(state)

        # Update progress file
        self._update_progress_file(state)

        # Increment iteration
        state.iteration += 1
        state.last_action = ""
        state.last_result = ""

        logger.info(f"[{self.config.loop_type.value}] Context rotated to iteration {state.iteration}")

        return state

    # ========== Guardrails (Signs) ==========

    def load_guardrails(self) -> List[Guardrail]:
        """Load all guardrails for this loop."""
        guardrails = []

        # Load loop-specific guardrails
        guardrail_file = self.guardrails_path / "guardrails.json"
        if guardrail_file.exists():
            try:
                with open(guardrail_file, 'r') as f:
                    data = json.load(f)
                guardrails = [Guardrail.from_dict(g) for g in data]
            except Exception as e:
                logger.warning(f"Failed to load guardrails: {e}")

        return guardrails

    def add_guardrail(
        self,
        trigger: str,
        instruction: str,
        severity: str = "warning",
        context: str = "",
        iteration: int = 0
    ) -> Guardrail:
        """
        Add a new guardrail (sign) based on a lesson learned.
        """
        guardrail = Guardrail(
            id=f"{self.config.loop_type.value}_{int(time.time())}",
            trigger=trigger,
            instruction=instruction,
            severity=severity,
            loop_type=self.config.loop_type,
            created_at=datetime.now().isoformat(),
            iteration_added=iteration,
            context=context
        )

        # Load existing and append
        guardrails = self.load_guardrails()
        guardrails.append(guardrail)

        # Save
        guardrail_file = self.guardrails_path / "guardrails.json"
        with open(guardrail_file, 'w') as f:
            json.dump([g.to_dict() for g in guardrails], f, indent=2)

        # Notify callbacks
        for callback in self._on_guardrail_added:
            try:
                callback(guardrail)
            except Exception as e:
                logger.error(f"Guardrail callback error: {e}")

        logger.info(f"[{self.config.loop_type.value}] Added guardrail: {trigger}")

        return guardrail

    def check_guardrails(self, action: str, context: Dict[str, Any]) -> List[Guardrail]:
        """
        Check if any guardrails are triggered by the proposed action.
        Returns list of triggered guardrails.
        """
        guardrails = self.load_guardrails()
        triggered = []

        action_lower = action.lower()
        context_str = json.dumps(context).lower()

        for g in guardrails:
            trigger_lower = g.trigger.lower()
            if trigger_lower in action_lower or trigger_lower in context_str:
                triggered.append(g)
                logger.warning(
                    f"[{self.config.loop_type.value}] Guardrail triggered: {g.trigger} -> {g.instruction}"
                )

        return triggered

    # ========== Meta-Layer ==========

    def set_meta_context(self, context: MetaLayerContext) -> None:
        """Set the meta-layer context for cross-loop introspection."""
        self.meta_context = context

    def get_meta_insights(self) -> List[str]:
        """Get insights from the meta-layer."""
        insights = []

        if self.meta_context:
            # Get cross-loop insights
            insights.extend(self.meta_context.get_cross_loop_insights())

            # Get performance insights
            if self.meta_context.performance_history:
                recent = self.meta_context.performance_history[-5:]
                avg_latency = sum(p.get("latency_ms", 0) for p in recent) / len(recent)
                if avg_latency > self.config.target_latency_ms * 1.5:
                    insights.append(
                        f"Performance degraded: avg latency {avg_latency:.1f}ms > target {self.config.target_latency_ms}ms"
                    )

        return insights

    def introspect(self, state: RalphState) -> Dict[str, Any]:
        """
        Meta-layer introspection - analyze current state and performance.
        """
        return {
            "loop_type": self.config.loop_type.value,
            "iteration": state.iteration,
            "progress": state.progress,
            "error_rate": len(state.errors) / max(state.iteration, 1),
            "guardrail_count": len(state.guardrails),
            "meta_insights": self.get_meta_insights(),
            "performance": state.metrics,
            "health": self._assess_health(state)
        }

    def _assess_health(self, state: RalphState) -> str:
        """Assess the health of this loop."""
        error_rate = len(state.errors) / max(state.iteration, 1)

        if error_rate > 0.5:
            return "critical"
        elif error_rate > 0.2:
            return "degraded"
        elif error_rate > 0.1:
            return "warning"
        else:
            return "healthy"

    # ========== Progress Tracking ==========

    def _update_progress_file(self, state: RalphState) -> None:
        """Update the human-readable progress file."""
        content = f"""# {self.config.loop_type.value.title()} Loop Progress

**Updated:** {state.updated_at}
**Iteration:** {state.iteration}
**Status:** {state.status}
**Progress:** {state.progress * 100:.1f}%

## Last Action
{state.last_action or "None"}

## Last Result
{state.last_result or "None"}

## Errors ({len(state.errors)})
{chr(10).join(f"- {e}" for e in state.errors[-10:]) if state.errors else "None"}

## Guardrails ({len(state.guardrails)})
{chr(10).join(f"- **{g.trigger}**: {g.instruction}" for g in state.guardrails[-10:]) if state.guardrails else "None"}

## Metrics
{chr(10).join(f"- {k}: {v}" for k, v in state.metrics.items()) if state.metrics else "None"}

## Context Summary
{state.context_summary or "No context available"}
"""
        with open(self.progress_path, 'w') as f:
            f.write(content)

    # ========== Abstract Methods ==========

    @abstractmethod
    async def execute_iteration(self, state: RalphState, input_data: Any) -> RalphState:
        """
        Execute a single iteration of this loop.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def should_continue(self, state: RalphState) -> bool:
        """
        Determine if the loop should continue or stop.
        Must be implemented by subclasses.
        """
        pass

    # ========== Main Loop ==========

    async def run(self, input_data: Any = None, max_iterations: Optional[int] = None) -> RalphState:
        """
        Run the Ralph loop with context rotation.
        """
        max_iter = max_iterations or self.config.max_iterations

        # Load fresh context
        state = self.load_fresh_context()
        state.status = "running"
        state.guardrails = self.load_guardrails()

        logger.info(f"[{self.config.loop_type.value}] Starting Ralph loop from iteration {state.iteration}")

        try:
            while state.iteration < max_iter:
                # Check if we should continue
                if not await self.should_continue(state):
                    state.status = "completed"
                    break

                # Execute iteration with timing
                start_time = time.time()
                state = await self.execute_iteration(state, input_data)
                latency_ms = (time.time() - start_time) * 1000

                # Track performance
                state.metrics["last_latency_ms"] = latency_ms
                if "avg_latency_ms" not in state.metrics:
                    state.metrics["avg_latency_ms"] = latency_ms
                else:
                    # Exponential moving average
                    state.metrics["avg_latency_ms"] = (
                        0.9 * state.metrics["avg_latency_ms"] + 0.1 * latency_ms
                    )

                # Check latency guardrail
                if latency_ms > self.config.target_latency_ms * 2:
                    self.add_guardrail(
                        trigger=f"latency_exceeded_{int(latency_ms)}ms",
                        instruction=f"Optimize: iteration took {latency_ms:.1f}ms, target is {self.config.target_latency_ms}ms",
                        severity="warning",
                        iteration=state.iteration
                    )

                # Context rotation
                state = self.rotate_context(state)

        except Exception as e:
            state.status = "failed"
            state.errors.append(str(e))
            self.add_guardrail(
                trigger=f"exception_{type(e).__name__}",
                instruction=f"Handle error: {str(e)}",
                severity="error",
                iteration=state.iteration
            )
            logger.exception(f"[{self.config.loop_type.value}] Loop failed")

        # Final save
        self.save_state(state)

        return state

    # ========== Event Subscriptions ==========

    def on_guardrail_added(self, callback: Callable[[Guardrail], None]) -> None:
        """Subscribe to guardrail additions."""
        self._on_guardrail_added.append(callback)

    def on_state_changed(self, callback: Callable[[RalphState], None]) -> None:
        """Subscribe to state changes."""
        self._on_state_changed.append(callback)
