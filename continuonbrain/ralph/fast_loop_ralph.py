"""
Fast Loop Ralph Layer
=====================

Wraps the 10ms reflex loop with Ralph's context rotation and guardrails.

Characteristics:
- 10ms target latency (real-time capable)
- Reflexive control (obstacle avoidance, balance)
- Direct safety kernel integration
- Minimal memory footprint
- Constant-time inference
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
class FastLoopInput:
    """Input for fast loop iteration."""
    sensor_readings: Dict[str, float]  # IMU, proximity, etc.
    motor_state: Dict[str, float]      # Current motor positions
    emergency_stop: bool = False
    timestamp_ms: float = 0.0


@dataclass
class FastLoopOutput:
    """Output from fast loop iteration."""
    motor_commands: Dict[str, float]   # Target positions/velocities
    safety_status: str                  # ok, warning, halt
    reflex_triggered: Optional[str]     # Name of triggered reflex
    latency_ms: float = 0.0


class FastLoopRalph(RalphLayer):
    """
    Ralph layer for the 10ms reflex loop.

    This loop handles:
    - Immediate obstacle avoidance
    - Balance control
    - Emergency stop detection
    - Motor safety limits

    Guardrails for this loop focus on:
    - Collision patterns (learn from near-misses)
    - Motor limit violations
    - Timing violations (must stay under 10ms)
    """

    def __init__(self, hope_brain: Any = None, safety_kernel: Any = None, **kwargs):
        config = RalphConfig(
            loop_type=LoopType.FAST,
            target_latency_ms=10.0,
            max_iterations=10000,  # Many fast iterations
            context_window_tokens=4096,  # Small context for speed
        )
        super().__init__(config, **kwargs)

        self.hope_brain = hope_brain
        self.safety_kernel = safety_kernel

        # Reflex patterns (learned from guardrails)
        self._reflex_patterns: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self._latency_buffer: List[float] = []
        self._max_buffer_size = 100

    async def execute_iteration(self, state: RalphState, input_data: Any) -> RalphState:
        """
        Execute a single 10ms iteration.
        """
        start_time = time.perf_counter()

        if not isinstance(input_data, FastLoopInput):
            # Wrap raw data
            input_data = FastLoopInput(
                sensor_readings=input_data if isinstance(input_data, dict) else {},
                motor_state={},
                timestamp_ms=time.time() * 1000
            )

        # Check emergency stop first (highest priority)
        if input_data.emergency_stop:
            state.last_action = "emergency_stop"
            state.last_result = "All motors halted"
            return state

        # Check guardrails before proceeding
        triggered = self.check_guardrails(
            action="motor_command",
            context={"sensors": input_data.sensor_readings}
        )

        if any(g.severity == "critical" for g in triggered):
            state.last_action = "blocked_by_guardrail"
            state.last_result = f"Critical guardrail: {triggered[0].instruction}"
            state.errors.append(f"Blocked: {triggered[0].trigger}")
            return state

        # Execute reflexive control
        output = await self._execute_reflex(input_data, state)

        # Track performance
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._latency_buffer.append(latency_ms)
        if len(self._latency_buffer) > self._max_buffer_size:
            self._latency_buffer.pop(0)

        # Update state
        state.last_action = f"reflex:{output.reflex_triggered or 'none'}"
        state.last_result = f"status={output.safety_status}, latency={latency_ms:.2f}ms"
        state.metrics["last_latency_ms"] = latency_ms
        state.metrics["avg_latency_ms"] = sum(self._latency_buffer) / len(self._latency_buffer)

        # Learn from near-misses (meta-layer)
        if output.safety_status == "warning":
            self._learn_from_near_miss(input_data, output, state)

        return state

    async def _execute_reflex(
        self,
        input_data: FastLoopInput,
        state: RalphState
    ) -> FastLoopOutput:
        """Execute reflexive control logic."""

        motor_commands = {}
        safety_status = "ok"
        reflex_triggered = None

        # Check proximity sensors for obstacle avoidance
        for sensor, value in input_data.sensor_readings.items():
            if "proximity" in sensor.lower() or "distance" in sensor.lower():
                if value < 0.1:  # 10cm threshold
                    reflex_triggered = "obstacle_avoidance"
                    safety_status = "warning"
                    # Back away from obstacle
                    motor_commands["drive_left"] = -0.5
                    motor_commands["drive_right"] = -0.5
                    break

        # Check IMU for balance
        if "imu_pitch" in input_data.sensor_readings:
            pitch = input_data.sensor_readings["imu_pitch"]
            if abs(pitch) > 30:  # Degrees
                reflex_triggered = "balance_recovery"
                safety_status = "warning"

        # Check motor limits
        for motor, position in input_data.motor_state.items():
            limit = self._get_motor_limit(motor)
            if abs(position) > limit * 0.95:
                reflex_triggered = "motor_limit"
                safety_status = "warning"
                # Stop that motor
                motor_commands[motor] = 0

        # Pass through safety kernel if available
        if self.safety_kernel and motor_commands:
            validated = await self._validate_with_safety_kernel(motor_commands)
            motor_commands = validated.get("commands", motor_commands)
            if validated.get("blocked"):
                safety_status = "halt"

        return FastLoopOutput(
            motor_commands=motor_commands,
            safety_status=safety_status,
            reflex_triggered=reflex_triggered,
            latency_ms=0  # Filled in by caller
        )

    async def _validate_with_safety_kernel(
        self,
        commands: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate commands with the safety kernel."""
        if self.safety_kernel is None:
            return {"commands": commands, "blocked": False}

        try:
            # Call safety kernel validation
            if hasattr(self.safety_kernel, 'validate_actuation'):
                result = await self.safety_kernel.validate_actuation(commands)
                return result
            else:
                return {"commands": commands, "blocked": False}
        except Exception as e:
            logger.error(f"Safety kernel validation failed: {e}")
            return {"commands": {}, "blocked": True}

    def _get_motor_limit(self, motor: str) -> float:
        """Get the position limit for a motor."""
        # Default limits (should be loaded from config)
        default_limits = {
            "servo": 180.0,
            "drive": 1.0,
            "arm": 270.0,
            "gripper": 100.0
        }
        for key, limit in default_limits.items():
            if key in motor.lower():
                return limit
        return 180.0

    def _learn_from_near_miss(
        self,
        input_data: FastLoopInput,
        output: FastLoopOutput,
        state: RalphState
    ) -> None:
        """
        Learn from near-miss situations (meta-layer learning).
        Creates guardrails to prevent future issues.
        """
        if output.reflex_triggered:
            # Check if we already have a guardrail for this pattern
            existing = [g for g in state.guardrails if output.reflex_triggered in g.trigger]

            if len(existing) < 3:  # Allow up to 3 similar guardrails
                # Create a new guardrail with context
                context = {
                    "sensors": input_data.sensor_readings,
                    "motors": input_data.motor_state,
                    "reflex": output.reflex_triggered
                }

                self.add_guardrail(
                    trigger=f"{output.reflex_triggered}_pattern",
                    instruction=f"Watch for {output.reflex_triggered} with similar sensor readings",
                    severity="warning",
                    context=str(context),
                    iteration=state.iteration
                )

    async def should_continue(self, state: RalphState) -> bool:
        """
        Fast loop should always continue (it's the heartbeat).
        Only stops on critical failure.
        """
        if state.status == "failed":
            return False

        # Check for too many errors
        if len(state.errors) > 100:
            error_rate = len(state.errors) / max(state.iteration, 1)
            if error_rate > 0.5:
                logger.error("Fast loop error rate too high, stopping")
                return False

        return True

    # ========== Meta-Layer Features ==========

    def get_reflex_statistics(self) -> Dict[str, Any]:
        """Get statistics about reflex activations."""
        state = self.load_fresh_context()
        guardrails = self.load_guardrails()

        reflex_counts = {}
        for g in guardrails:
            if "_pattern" in g.trigger:
                reflex_name = g.trigger.replace("_pattern", "")
                reflex_counts[reflex_name] = reflex_counts.get(reflex_name, 0) + 1

        return {
            "total_iterations": state.iteration,
            "avg_latency_ms": state.metrics.get("avg_latency_ms", 0),
            "reflex_counts": reflex_counts,
            "guardrail_count": len(guardrails),
            "health": self._assess_health(state)
        }

    def optimize_for_latency(self) -> None:
        """
        Meta-layer optimization: analyze and optimize for latency.
        """
        if len(self._latency_buffer) < 10:
            return

        avg = sum(self._latency_buffer) / len(self._latency_buffer)
        max_latency = max(self._latency_buffer)

        if max_latency > self.config.target_latency_ms * 3:
            self.add_guardrail(
                trigger="latency_spike",
                instruction=f"Reduce computation: max latency {max_latency:.1f}ms exceeds 3x target",
                severity="warning",
                context=f"avg={avg:.1f}ms, max={max_latency:.1f}ms",
                iteration=self.load_fresh_context().iteration
            )
