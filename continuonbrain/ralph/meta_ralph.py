"""
Meta Ralph Agent
================

The primary user interaction layer for ContinuonBrain.
Orchestrates all four Ralph layers and provides the agent chat interface.

Architecture:
                    ┌─────────────────────────────────────┐
                    │        META RALPH AGENT              │
                    │   (Primary User Interaction)         │
                    ├─────────────────────────────────────┤
                    │                                       │
                    │   ┌─────────┐ ┌─────────┐ ┌────────┐ │
                    │   │  Fast   │ │   Mid   │ │  Slow  │ │
                    │   │  Ralph  │ │  Ralph  │ │ Ralph  │ │
                    │   └────┬────┘ └────┬────┘ └────┬───┘ │
                    │        │           │           │      │
                    │        └───────────┼───────────┘      │
                    │                    │                  │
                    │              ┌─────┴─────┐            │
                    │              │  Safety   │            │
                    │              │   Ralph   │            │
                    │              └───────────┘            │
                    │                                       │
                    ├─────────────────────────────────────┤
                    │     Claude Code CLI (Swappable)      │
                    │   [Opus 4.5 | Gemini | Ollama]       │
                    └─────────────────────────────────────┘
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, AsyncIterator

from .base import RalphState, Guardrail, LoopType, MetaLayerContext
from .fast_loop_ralph import FastLoopRalph, FastLoopInput
from .mid_loop_ralph import MidLoopRalph, MidLoopInput
from .slow_loop_ralph import SlowLoopRalph, SlowLoopInput
from .safety_ralph import SafetyRalph, SafetyCheckResult
from .claude_code_cli import ClaudeCodeCLI, CLIProvider, CLIResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """A message in the agent chat."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from the Meta Ralph Agent."""
    text: str
    loop_states: Dict[str, Dict[str, Any]]
    guardrails_triggered: List[Guardrail]
    actions_taken: List[str]
    cli_provider: Optional[CLIProvider] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


class MetaRalphAgent:
    """
    Meta Ralph Agent - The primary user interaction layer for ContinuonBrain.

    This agent:
    1. Orchestrates all four Ralph layers (Fast, Mid, Slow, Safety)
    2. Provides the chat interface for user interaction
    3. Integrates swappable Claude Code CLI (Opus 4.5, Gemini, Ollama)
    4. Maintains global guardrails across all loops
    5. Provides meta-layer introspection for self-improvement
    """

    def __init__(
        self,
        brain_service: Any = None,
        hope_brain: Any = None,
        safety_kernel: Any = None,
        cms: Any = None,
        base_path: Optional[Path] = None,
        default_cli_provider: CLIProvider = CLIProvider.CLAUDE_CODE
    ):
        self.brain_service = brain_service
        self.hope_brain = hope_brain
        self.base_path = base_path or Path.cwd()

        # Initialize Ralph layers
        self.fast_ralph = FastLoopRalph(
            hope_brain=hope_brain,
            safety_kernel=safety_kernel,
            base_path=self.base_path
        )

        self.mid_ralph = MidLoopRalph(
            hope_brain=hope_brain,
            cms=cms,
            base_path=self.base_path
        )

        self.slow_ralph = SlowLoopRalph(
            base_path=self.base_path
        )

        self.safety_ralph = SafetyRalph(
            safety_kernel=safety_kernel,
            base_path=self.base_path
        )

        # Initialize Claude Code CLI
        self.cli = ClaudeCodeCLI(default_provider=default_cli_provider)

        # Chat history
        self._chat_history: List[AgentMessage] = []

        # Global guardrails (aggregated from all layers)
        self._global_guardrails: List[Guardrail] = []

        # Cross-layer communication
        self._setup_cross_layer_communication()

        # System prompt for CLI
        self._setup_system_prompt()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Callbacks
        self._on_message: List[Callable[[AgentMessage], None]] = []
        self._on_loop_state_change: List[Callable[[LoopType, RalphState], None]] = []

    def _setup_cross_layer_communication(self) -> None:
        """Set up callbacks for cross-layer communication."""

        # Guardrail propagation
        def propagate_guardrail(guardrail: Guardrail) -> None:
            self._global_guardrails.append(guardrail)
            # Update meta contexts
            self._update_meta_contexts()

        self.fast_ralph.on_guardrail_added(propagate_guardrail)
        self.mid_ralph.on_guardrail_added(propagate_guardrail)
        self.slow_ralph.on_guardrail_added(propagate_guardrail)
        self.safety_ralph.on_guardrail_added(propagate_guardrail)

        # State change propagation
        def handle_state_change(state: RalphState) -> None:
            for callback in self._on_loop_state_change:
                try:
                    callback(state.loop_type, state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

        self.fast_ralph.on_state_changed(handle_state_change)
        self.mid_ralph.on_state_changed(handle_state_change)
        self.slow_ralph.on_state_changed(handle_state_change)

    def _update_meta_contexts(self) -> None:
        """Update meta-layer contexts for all loops."""

        # Get current states
        states = {
            LoopType.FAST: self.fast_ralph.load_fresh_context(),
            LoopType.MID: self.mid_ralph.load_fresh_context(),
            LoopType.SLOW: self.slow_ralph.load_fresh_context(),
        }

        for loop_type, ralph in [
            (LoopType.FAST, self.fast_ralph),
            (LoopType.MID, self.mid_ralph),
            (LoopType.SLOW, self.slow_ralph),
        ]:
            sibling_states = [s for t, s in states.items() if t != loop_type]
            context = MetaLayerContext(
                loop_type=loop_type,
                sibling_states=sibling_states,
                global_guardrails=self._global_guardrails
            )
            ralph.set_meta_context(context)

    def _setup_system_prompt(self) -> None:
        """Set up the system prompt for the CLI."""

        system_prompt = """You are the Meta Ralph Agent for ContinuonBrain, an embodied AI system.

You orchestrate four control loops:
1. Fast Loop (10ms) - Reflexes and motor control
2. Mid Loop (100ms) - Skills and attention
3. Slow Loop (cloud) - Learning and planning
4. Safety Kernel (Ring 0) - Constitutional constraints

Your role is to:
- Process user messages and route them appropriately
- Coordinate between loops for complex tasks
- Learn from mistakes via guardrails
- Maintain safety at all times

Current guardrails: {guardrail_count}
Safety status: {safety_status}

Respond conversationally while being precise about robot capabilities."""

        self.cli.set_system_prompt(system_prompt.format(
            guardrail_count=len(self._global_guardrails),
            safety_status=self.safety_ralph.get_safety_status().get("halted", False)
        ))

    # ========== Primary Interface ==========

    async def process_user_message(
        self,
        message: str,
        use_cli: bool = True,
        route_to_loops: bool = True
    ) -> AgentResponse:
        """
        Process a user message through the Meta Ralph Agent.

        This is the primary interaction method.
        """
        start_time = time.time()

        # Add to chat history
        user_msg = AgentMessage(role="user", content=message)
        self._chat_history.append(user_msg)
        self._notify_message(user_msg)

        response = AgentResponse(
            text="",
            loop_states={},
            guardrails_triggered=[],
            actions_taken=[]
        )

        try:
            # 1. Safety check first
            safety_check = await self._check_safety(message)
            if not safety_check.allowed:
                response.text = f"Safety blocked: {safety_check.message}"
                response.error = "safety_violation"
                return response

            # 2. Route to appropriate loops
            if route_to_loops:
                loop_results = await self._route_to_loops(message)
                response.loop_states = loop_results.get("states", {})
                response.guardrails_triggered = loop_results.get("guardrails", [])
                response.actions_taken = loop_results.get("actions", [])

            # 3. Generate response via CLI
            if use_cli:
                # Build context for CLI
                context = self._build_cli_context(message, response)

                # Send to CLI
                cli_response = await self.cli.send(context)
                response.text = cli_response.text
                response.cli_provider = cli_response.provider
                response.latency_ms = cli_response.latency_ms

                if cli_response.error:
                    response.error = cli_response.error
            else:
                # Generate response without CLI
                response.text = self._generate_local_response(message, response)

            # 4. Post-process and learn
            await self._post_process(message, response)

        except Exception as e:
            logger.exception("Error processing message")
            response.text = f"Error: {str(e)}"
            response.error = str(e)

        # Add assistant response to history
        assistant_msg = AgentMessage(
            role="assistant",
            content=response.text,
            metadata={"actions": response.actions_taken}
        )
        self._chat_history.append(assistant_msg)
        self._notify_message(assistant_msg)

        response.latency_ms = (time.time() - start_time) * 1000

        return response

    async def stream_response(
        self,
        message: str
    ) -> AsyncIterator[str]:
        """
        Stream a response to the user message.
        """
        # Safety check
        safety_check = await self._check_safety(message)
        if not safety_check.allowed:
            yield f"Safety blocked: {safety_check.message}"
            return

        # Route to loops (don't wait)
        asyncio.create_task(self._route_to_loops(message))

        # Stream from CLI
        context = self._build_cli_context(message, AgentResponse(
            text="",
            loop_states={},
            guardrails_triggered=[],
            actions_taken=[]
        ))

        async for chunk in self.cli.stream(context):
            yield chunk

    # ========== Loop Routing ==========

    async def _check_safety(self, message: str) -> SafetyCheckResult:
        """Check message against safety kernel."""

        # Convert message to a command structure
        command = {
            "type": "user_message",
            "content": message,
            "timestamp": time.time()
        }

        return await self.safety_ralph.validate_command(command)

    async def _route_to_loops(self, message: str) -> Dict[str, Any]:
        """Route the message to appropriate loops."""

        results = {
            "states": {},
            "guardrails": [],
            "actions": []
        }

        # Mid loop handles intent inference
        mid_input = MidLoopInput(user_intent=message)
        mid_state = await self.mid_ralph.execute_iteration(
            self.mid_ralph.load_fresh_context(),
            mid_input
        )
        results["states"]["mid"] = mid_state.to_dict()
        results["guardrails"].extend(mid_state.guardrails)

        if mid_state.last_action:
            results["actions"].append(mid_state.last_action)

        # Fast loop may need to react to commands
        if any(word in message.lower() for word in ["stop", "halt", "wait", "pause"]):
            fast_input = FastLoopInput(
                sensor_readings={},
                motor_state={},
                emergency_stop="stop" in message.lower()
            )
            fast_state = await self.fast_ralph.execute_iteration(
                self.fast_ralph.load_fresh_context(),
                fast_input
            )
            results["states"]["fast"] = fast_state.to_dict()

        # Slow loop for learning/planning requests
        if any(word in message.lower() for word in ["learn", "remember", "goal", "plan"]):
            slow_input = SlowLoopInput(
                trigger="user_request",
                goal_updates=[message] if "goal" in message.lower() else []
            )
            slow_state = await self.slow_ralph.execute_iteration(
                self.slow_ralph.load_fresh_context(),
                slow_input
            )
            results["states"]["slow"] = slow_state.to_dict()

        return results

    def _build_cli_context(self, message: str, response: AgentResponse) -> str:
        """Build context string for the CLI."""

        context_parts = [message]

        # Add loop state summaries
        if response.loop_states:
            context_parts.append("\n[Loop States]")
            for loop, state in response.loop_states.items():
                context_parts.append(f"- {loop}: {state.get('last_result', 'no result')}")

        # Add triggered guardrails
        if response.guardrails_triggered:
            context_parts.append("\n[Active Guardrails]")
            for g in response.guardrails_triggered[-3:]:
                context_parts.append(f"- {g.trigger}: {g.instruction}")

        # Add recent actions
        if response.actions_taken:
            context_parts.append(f"\n[Actions Taken]: {', '.join(response.actions_taken)}")

        return "\n".join(context_parts)

    def _generate_local_response(self, message: str, response: AgentResponse) -> str:
        """Generate a response without using the CLI."""

        parts = []

        if response.actions_taken:
            parts.append(f"Actions: {', '.join(response.actions_taken)}")

        if response.guardrails_triggered:
            parts.append(f"Guardrails active: {len(response.guardrails_triggered)}")

        if not parts:
            parts.append("Message received.")

        return " | ".join(parts)

    async def _post_process(self, message: str, response: AgentResponse) -> None:
        """Post-process the response for learning."""

        # Queue for slow loop if there's something to learn
        if response.guardrails_triggered:
            episode = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "response": response.text,
                "guardrails": [g.to_dict() for g in response.guardrails_triggered]
            }
            self.slow_ralph.queue_episode(episode)

    # ========== CLI Management ==========

    def switch_cli_provider(self, provider: CLIProvider) -> bool:
        """Switch the CLI provider."""
        return self.cli.switch_provider(provider)

    def get_available_providers(self) -> List[CLIProvider]:
        """Get available CLI providers."""
        return self.cli.get_available_providers()

    def get_active_provider(self) -> Optional[CLIProvider]:
        """Get the active CLI provider."""
        return self.cli.get_active_provider()

    # ========== State Management ==========

    def get_all_loop_states(self) -> Dict[str, RalphState]:
        """Get current states of all loops."""
        return {
            "fast": self.fast_ralph.load_fresh_context(),
            "mid": self.mid_ralph.load_fresh_context(),
            "slow": self.slow_ralph.load_fresh_context(),
            "safety": self.safety_ralph.load_fresh_context()
        }

    def get_global_guardrails(self) -> List[Guardrail]:
        """Get all global guardrails."""
        return self._global_guardrails.copy()

    def get_chat_history(self) -> List[AgentMessage]:
        """Get chat history."""
        return self._chat_history.copy()

    def clear_chat_history(self) -> None:
        """Clear chat history."""
        self._chat_history = []
        self.cli.clear_history()

    # ========== Introspection ==========

    def introspect(self) -> Dict[str, Any]:
        """Get full introspection of the Meta Ralph Agent."""

        states = self.get_all_loop_states()

        return {
            "agent_type": "meta_ralph",
            "cli_provider": self.get_active_provider().value if self.get_active_provider() else None,
            "available_providers": [p.value for p in self.get_available_providers()],
            "loop_states": {
                loop: {
                    "iteration": state.iteration,
                    "status": state.status,
                    "health": self._assess_loop_health(state),
                    "guardrails": len(state.guardrails)
                }
                for loop, state in states.items()
            },
            "global_guardrails": len(self._global_guardrails),
            "chat_history_length": len(self._chat_history),
            "safety_status": self.safety_ralph.get_safety_status()
        }

    def _assess_loop_health(self, state: RalphState) -> str:
        """Assess the health of a loop."""
        if state.status == "failed":
            return "critical"
        error_rate = len(state.errors) / max(state.iteration, 1)
        if error_rate > 0.3:
            return "degraded"
        return "healthy"

    # ========== Teaching Interface ==========

    async def teach(self, teaching_input: str, correct_response: str) -> bool:
        """Teach the agent a correct response."""

        # Add to mid loop as learned intent
        self.mid_ralph.teach_intent(teaching_input, correct_response)

        # Queue as high-priority episode for slow loop
        episode = {
            "type": "teaching",
            "input": teaching_input,
            "correct_response": correct_response,
            "timestamp": datetime.now().isoformat(),
            "priority": "high"
        }
        self.slow_ralph.queue_episode(episode)

        return True

    def get_pending_questions(self) -> List[str]:
        """Get questions the agent needs answered."""
        return self.mid_ralph.get_pending_questions()

    # ========== Event Subscriptions ==========

    def on_message(self, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe to message events."""
        self._on_message.append(callback)

    def on_loop_state_change(
        self,
        callback: Callable[[LoopType, RalphState], None]
    ) -> None:
        """Subscribe to loop state changes."""
        self._on_loop_state_change.append(callback)

    def _notify_message(self, message: AgentMessage) -> None:
        """Notify message subscribers."""
        for callback in self._on_message:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

    # ========== Lifecycle ==========

    async def start(self) -> None:
        """Start the Meta Ralph Agent."""
        logger.info("Starting Meta Ralph Agent")

        # Update meta contexts
        self._update_meta_contexts()

        # Start background tasks if needed
        # (e.g., periodic slow loop triggers)

    async def stop(self) -> None:
        """Stop the Meta Ralph Agent."""
        logger.info("Stopping Meta Ralph Agent")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Save all states
        for ralph in [self.fast_ralph, self.mid_ralph, self.slow_ralph]:
            state = ralph.load_fresh_context()
            ralph.save_state(state)
