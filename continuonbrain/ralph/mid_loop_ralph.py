"""
Mid Loop Ralph Layer
====================

Wraps the 100ms skill/attention loop with Ralph's context rotation and guardrails.

Characteristics:
- 100ms target latency
- Skill sequencing and attention
- Context integration (CMS Level 1)
- Intent inference
- RLDS episode logging
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

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
class MidLoopInput:
    """Input for mid loop iteration."""
    user_intent: Optional[str] = None
    fast_loop_state: Optional[Dict[str, Any]] = None
    sensor_context: Dict[str, Any] = field(default_factory=dict)
    current_skill: Optional[str] = None
    skill_progress: float = 0.0
    timestamp_ms: float = 0.0


@dataclass
class MidLoopOutput:
    """Output from mid loop iteration."""
    next_skill: Optional[str] = None
    skill_parameters: Dict[str, Any] = field(default_factory=dict)
    attention_target: Optional[str] = None
    context_update: Dict[str, Any] = field(default_factory=dict)
    response_text: Optional[str] = None
    confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Skill:
    """A learned skill in the task library."""
    name: str
    description: str
    preconditions: List[str]
    effects: List[str]
    parameters: Dict[str, Any]
    success_rate: float = 1.0


class MidLoopRalph(RalphLayer):
    """
    Ralph layer for the 100ms skill/attention loop.

    This loop handles:
    - Skill sequencing ("What skill should execute next?")
    - Intent inference from user commands
    - Context integration with CMS Level 1
    - Attention management
    - RLDS episode logging for learning

    Guardrails for this loop focus on:
    - Skill failure patterns
    - Intent misinterpretation
    - Context staleness
    - Skill sequencing errors
    """

    def __init__(
        self,
        hope_brain: Any = None,
        cms: Any = None,
        skill_library: Optional[Dict[str, Skill]] = None,
        **kwargs
    ):
        config = RalphConfig(
            loop_type=LoopType.MID,
            target_latency_ms=100.0,
            max_iterations=1000,
            context_window_tokens=16384,
        )
        super().__init__(config, **kwargs)

        self.hope_brain = hope_brain
        self.cms = cms
        self.skill_library = skill_library or {}

        # Current skill execution state
        self._current_skill: Optional[str] = None
        self._skill_start_time: float = 0
        self._skill_history: List[Dict[str, Any]] = []

        # Intent inference cache
        self._intent_cache: Dict[str, str] = {}

        # RLDS episode buffer
        self._episode_buffer: List[Dict[str, Any]] = []

    async def execute_iteration(self, state: RalphState, input_data: Any) -> RalphState:
        """
        Execute a single 100ms iteration.
        """
        start_time = time.perf_counter()

        if not isinstance(input_data, MidLoopInput):
            input_data = MidLoopInput(
                user_intent=input_data if isinstance(input_data, str) else None,
                timestamp_ms=time.time() * 1000
            )

        # Check guardrails for intent
        if input_data.user_intent:
            triggered = self.check_guardrails(
                action=f"intent:{input_data.user_intent}",
                context={"skill": self._current_skill}
            )
            if triggered:
                state.last_action = "guardrail_check"
                for g in triggered:
                    if g.severity == "critical":
                        state.errors.append(f"Blocked intent: {g.instruction}")
                        return state

        # Execute skill/attention logic
        output = await self._execute_skill_loop(input_data, state)

        # Log to RLDS episode buffer
        self._log_episode_step(input_data, output, state)

        # Update state
        latency_ms = (time.perf_counter() - start_time) * 1000
        state.last_action = f"skill:{output.next_skill or 'none'}"
        state.last_result = output.response_text or f"confidence={output.confidence:.2f}"
        state.metrics["last_latency_ms"] = latency_ms
        state.metrics["skill_success_rate"] = self._calculate_success_rate()

        # Meta-layer learning
        if output.confidence < 0.5:
            self._learn_from_low_confidence(input_data, output, state)

        return state

    async def _execute_skill_loop(
        self,
        input_data: MidLoopInput,
        state: RalphState
    ) -> MidLoopOutput:
        """Execute the skill sequencing and attention logic."""

        output = MidLoopOutput()

        # 1. Infer intent if provided
        if input_data.user_intent:
            intent, confidence = await self._infer_intent(input_data.user_intent)
            output.attention_target = intent
            output.confidence = confidence

            # Map intent to skill
            skill = self._map_intent_to_skill(intent)
            if skill:
                output.next_skill = skill.name
                output.skill_parameters = skill.parameters

        # 2. Check current skill progress
        if input_data.current_skill and input_data.skill_progress < 1.0:
            # Continue current skill
            output.next_skill = input_data.current_skill
            output.response_text = f"Executing {input_data.current_skill}: {input_data.skill_progress*100:.0f}%"

        # 3. Query CMS for context
        if self.cms:
            context = await self._query_cms(input_data, output)
            output.context_update = context

        # 4. Generate response
        if output.next_skill:
            output.response_text = f"Starting skill: {output.next_skill}"
        elif input_data.user_intent:
            output.response_text = self._generate_response(input_data.user_intent, output.confidence)

        return output

    async def _infer_intent(self, user_input: str) -> tuple[str, float]:
        """Infer user intent from natural language."""

        # Check cache first
        if user_input in self._intent_cache:
            return self._intent_cache[user_input], 0.9

        # Intent patterns (should be learned, this is a fallback)
        intent_patterns = {
            "move": ["go", "move", "walk", "drive", "forward", "backward"],
            "look": ["look", "see", "watch", "observe", "find"],
            "grab": ["grab", "pick", "take", "hold", "grasp"],
            "stop": ["stop", "halt", "freeze", "wait"],
            "greet": ["hello", "hi", "hey", "greet"],
            "status": ["status", "how are you", "what", "where"],
        }

        user_lower = user_input.lower()
        best_intent = "unknown"
        best_confidence = 0.3

        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in user_lower:
                    confidence = 0.7 + (0.1 if user_lower.startswith(pattern) else 0)
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence

        # Use HOPE brain if available for better inference
        if self.hope_brain and best_confidence < 0.7:
            try:
                if hasattr(self.hope_brain, 'infer_intent'):
                    result = await self.hope_brain.infer_intent(user_input)
                    if result.get("confidence", 0) > best_confidence:
                        best_intent = result.get("intent", best_intent)
                        best_confidence = result.get("confidence", best_confidence)
            except Exception as e:
                logger.warning(f"HOPE intent inference failed: {e}")

        # Cache result
        self._intent_cache[user_input] = best_intent

        return best_intent, best_confidence

    def _map_intent_to_skill(self, intent: str) -> Optional[Skill]:
        """Map an intent to an executable skill."""

        # Check skill library
        if intent in self.skill_library:
            return self.skill_library[intent]

        # Default skill mappings
        default_skills = {
            "move": Skill(
                name="navigate",
                description="Move to a location",
                preconditions=["localized", "no_obstacles"],
                effects=["at_target"],
                parameters={"speed": 0.5}
            ),
            "look": Skill(
                name="scan_environment",
                description="Look around and identify objects",
                preconditions=["camera_active"],
                effects=["environment_scanned"],
                parameters={"angle": 360}
            ),
            "grab": Skill(
                name="pick_object",
                description="Grasp an object",
                preconditions=["object_detected", "arm_ready"],
                effects=["holding_object"],
                parameters={"force": 0.3}
            ),
            "stop": Skill(
                name="halt",
                description="Stop all motion",
                preconditions=[],
                effects=["stopped"],
                parameters={}
            ),
        }

        return default_skills.get(intent)

    async def _query_cms(
        self,
        input_data: MidLoopInput,
        output: MidLoopOutput
    ) -> Dict[str, Any]:
        """Query the Continuous Memory System for context."""

        if self.cms is None:
            return {}

        try:
            # Query working memory (Level 1)
            if hasattr(self.cms, 'query'):
                query = output.attention_target or input_data.user_intent or "current_task"
                results = await self.cms.query(query, level=1)
                return {"memory_results": results}
        except Exception as e:
            logger.warning(f"CMS query failed: {e}")

        return {}

    def _generate_response(self, user_intent: str, confidence: float) -> str:
        """Generate a response for the user."""

        if confidence < 0.3:
            return f"I'm not sure what you mean by '{user_intent}'. Could you clarify?"
        elif confidence < 0.6:
            return f"I think you want me to do something related to '{user_intent}'?"
        else:
            return f"Understood: {user_intent}"

    def _calculate_success_rate(self) -> float:
        """Calculate skill success rate from history."""
        if not self._skill_history:
            return 1.0

        recent = self._skill_history[-20:]
        successes = sum(1 for s in recent if s.get("success", False))
        return successes / len(recent)

    def _log_episode_step(
        self,
        input_data: MidLoopInput,
        output: MidLoopOutput,
        state: RalphState
    ) -> None:
        """Log a step for RLDS episode recording."""

        step = {
            "timestamp_ms": input_data.timestamp_ms,
            "iteration": state.iteration,
            "user_intent": input_data.user_intent,
            "inferred_intent": output.attention_target,
            "skill": output.next_skill,
            "confidence": output.confidence,
            "context": output.context_update,
        }

        self._episode_buffer.append(step)

        # Limit buffer size
        if len(self._episode_buffer) > 1000:
            self._episode_buffer = self._episode_buffer[-500:]

    def _learn_from_low_confidence(
        self,
        input_data: MidLoopInput,
        output: MidLoopOutput,
        state: RalphState
    ) -> None:
        """Meta-layer: learn from low-confidence situations."""

        if input_data.user_intent:
            self.add_guardrail(
                trigger=f"low_confidence_intent:{input_data.user_intent[:20]}",
                instruction=f"Clarify intent for inputs like '{input_data.user_intent}'",
                severity="warning",
                context=f"confidence={output.confidence:.2f}",
                iteration=state.iteration
            )

    async def should_continue(self, state: RalphState) -> bool:
        """
        Mid loop should continue unless explicitly stopped.
        """
        if state.status == "failed":
            return False

        if state.status == "completed":
            return False

        return True

    # ========== Meta-Layer Features ==========

    def get_skill_statistics(self) -> Dict[str, Any]:
        """Get statistics about skill execution."""
        state = self.load_fresh_context()

        skill_counts = {}
        for step in self._skill_history:
            skill = step.get("skill", "unknown")
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

        return {
            "total_iterations": state.iteration,
            "skill_counts": skill_counts,
            "success_rate": self._calculate_success_rate(),
            "episode_buffer_size": len(self._episode_buffer),
            "intent_cache_size": len(self._intent_cache),
            "health": self._assess_health(state)
        }

    def get_pending_questions(self) -> List[str]:
        """Get questions the mid loop needs answered (for teaching interface)."""

        questions = []
        state = self.load_fresh_context()

        # Check for low-confidence patterns
        low_conf_guardrails = [
            g for g in state.guardrails
            if "low_confidence" in g.trigger
        ]

        for g in low_conf_guardrails[-5:]:
            questions.append(f"How should I interpret: {g.context}")

        return questions

    def teach_intent(self, user_input: str, correct_intent: str) -> None:
        """Teach the correct intent for a user input."""

        # Update cache
        self._intent_cache[user_input] = correct_intent

        # Add positive guardrail
        self.add_guardrail(
            trigger=f"learned_intent:{user_input[:20]}",
            instruction=f"Map '{user_input}' to intent '{correct_intent}'",
            severity="info",
            context=f"taught by user",
            iteration=self.load_fresh_context().iteration
        )

        logger.info(f"Learned intent: '{user_input}' -> '{correct_intent}'")

    def add_skill(self, skill: Skill) -> None:
        """Add a new skill to the library."""
        self.skill_library[skill.name] = skill
        logger.info(f"Added skill: {skill.name}")
