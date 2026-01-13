"""
Chat Integration for Meta Ralph Agent
======================================

Integrates the Meta Ralph Agent with ContinuonBrain's existing chat interface.
This module bridges the Ralph layer to the production API server.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator

from .meta_ralph import MetaRalphAgent, AgentResponse, AgentMessage
from .claude_code_cli import CLIProvider

logger = logging.getLogger(__name__)


class RalphChatAdapter:
    """
    Adapter that connects Meta Ralph Agent to ContinuonBrain's chat routes.

    This replaces or augments the existing chat integration with Ralph's
    context rotation and guardrails system.
    """

    def __init__(
        self,
        brain_service: Any = None,
        hope_brain: Any = None,
        safety_kernel: Any = None,
        cms: Any = None,
        base_path: Optional[Path] = None,
        default_provider: CLIProvider = CLIProvider.CLAUDE_CODE
    ):
        # Initialize Meta Ralph Agent
        self.agent = MetaRalphAgent(
            brain_service=brain_service,
            hope_brain=hope_brain,
            safety_kernel=safety_kernel,
            cms=cms,
            base_path=base_path,
            default_cli_provider=default_provider
        )

        # Session tracking
        self._sessions: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the chat adapter."""
        await self.agent.start()
        logger.info("RalphChatAdapter started")

    async def stop(self) -> None:
        """Stop the chat adapter."""
        await self.agent.stop()
        logger.info("RalphChatAdapter stopped")

    # ========== Chat API ==========

    async def process_chat_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and return the response.

        Compatible with the existing /api/chat endpoint format.
        """
        # Ensure session exists
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "user_id": user_id,
                "message_count": 0
            }

        # Process through Meta Ralph
        response = await self.agent.process_user_message(message)

        # Update session
        self._sessions[session_id]["message_count"] += 1
        self._sessions[session_id]["last_message_at"] = datetime.now().isoformat()

        # Format response for API
        return {
            "session_id": session_id,
            "response": {
                "text": response.text,
                "actions": response.actions_taken,
                "guardrails_triggered": [
                    {"trigger": g.trigger, "instruction": g.instruction}
                    for g in response.guardrails_triggered
                ],
                "loop_states": response.loop_states,
                "provider": response.cli_provider.value if response.cli_provider else None,
                "latency_ms": response.latency_ms,
                "error": response.error
            },
            "timestamp": datetime.now().isoformat()
        }

    async def stream_chat_message(
        self,
        session_id: str,
        message: str
    ) -> AsyncIterator[str]:
        """
        Stream a chat response.

        Compatible with SSE/streaming endpoints.
        """
        async for chunk in self.agent.stream_response(message):
            yield json.dumps({
                "type": "chunk",
                "content": chunk,
                "session_id": session_id
            }) + "\n"

        # Send final state
        yield json.dumps({
            "type": "done",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }) + "\n"

    # ========== Teaching API ==========

    async def teach(
        self,
        session_id: str,
        user_input: str,
        correct_response: str
    ) -> Dict[str, Any]:
        """
        Teach the agent a correct response.

        Compatible with /api/teacher/correct endpoint.
        """
        success = await self.agent.teach(user_input, correct_response)

        return {
            "success": success,
            "session_id": session_id,
            "input": user_input,
            "timestamp": datetime.now().isoformat()
        }

    async def get_questions(self) -> Dict[str, Any]:
        """
        Get pending questions from the agent.

        Compatible with /api/teacher/questions endpoint.
        """
        questions = self.agent.get_pending_questions()

        return {
            "questions": questions,
            "count": len(questions),
            "timestamp": datetime.now().isoformat()
        }

    # ========== Provider Management ==========

    def switch_provider(self, provider_name: str) -> Dict[str, Any]:
        """
        Switch the CLI provider.

        Compatible with /api/ralph/provider endpoint.
        """
        try:
            provider = CLIProvider(provider_name)
            success = self.agent.switch_cli_provider(provider)

            return {
                "success": success,
                "provider": provider_name,
                "available": [p.value for p in self.agent.get_available_providers()]
            }
        except ValueError:
            return {
                "success": False,
                "error": f"Unknown provider: {provider_name}",
                "available": [p.value for p in self.agent.get_available_providers()]
            }

    def get_providers(self) -> Dict[str, Any]:
        """
        Get available CLI providers.
        """
        active = self.agent.get_active_provider()

        return {
            "active": active.value if active else None,
            "available": [p.value for p in self.agent.get_available_providers()]
        }

    # ========== Introspection API ==========

    def get_status(self) -> Dict[str, Any]:
        """
        Get full status of the Ralph system.

        Compatible with /api/ralph/status endpoint.
        """
        introspection = self.agent.introspect()

        return {
            "status": "healthy" if not introspection["safety_status"].get("halted") else "halted",
            "introspection": introspection,
            "sessions": {
                "count": len(self._sessions),
                "active": [
                    sid for sid, sess in self._sessions.items()
                    if sess.get("last_message_at", "") > (
                        datetime.now().isoformat()[:10]  # Today
                    )
                ]
            }
        }

    def get_loop_state(self, loop_name: str) -> Dict[str, Any]:
        """
        Get state of a specific loop.
        """
        states = self.agent.get_all_loop_states()

        if loop_name in states:
            return {"loop": loop_name, "state": states[loop_name].to_dict()}

        return {"error": f"Unknown loop: {loop_name}"}

    def get_guardrails(self) -> Dict[str, Any]:
        """
        Get all global guardrails.
        """
        guardrails = self.agent.get_global_guardrails()

        return {
            "count": len(guardrails),
            "guardrails": [g.to_dict() for g in guardrails]
        }

    # ========== Session Management ==========

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return self._sessions.get(session_id)

    def get_chat_history(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Get chat history."""
        history = self.agent.get_chat_history()

        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in history
        ]

    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self.agent.clear_chat_history()
            return True
        return False


# ========== Factory Function ==========

def create_ralph_chat_adapter(
    brain_service: Any = None,
    hope_brain: Any = None,
    safety_kernel: Any = None,
    cms: Any = None,
    base_path: Optional[str] = None,
    default_provider: str = "claude"
) -> RalphChatAdapter:
    """
    Factory function to create a RalphChatAdapter.

    Use this in the ContinuonBrain server initialization.
    """
    provider_map = {
        "claude": CLIProvider.CLAUDE_CODE,
        "gemini": CLIProvider.GEMINI,
        "ollama": CLIProvider.OLLAMA,
    }

    provider = provider_map.get(default_provider, CLIProvider.CLAUDE_CODE)

    return RalphChatAdapter(
        brain_service=brain_service,
        hope_brain=hope_brain,
        safety_kernel=safety_kernel,
        cms=cms,
        base_path=Path(base_path) if base_path else None,
        default_provider=provider
    )


# ========== API Route Handlers ==========

class RalphAPIRoutes:
    """
    API route handlers for the Ralph chat adapter.

    These can be registered with the ContinuonBrain API server.
    """

    def __init__(self, adapter: RalphChatAdapter):
        self.adapter = adapter

    async def handle_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle POST /api/chat"""
        return await self.adapter.process_chat_message(
            session_id=request.get("session_id", "default"),
            message=request.get("message", ""),
            user_id=request.get("user_id")
        )

    async def handle_teach(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle POST /api/teacher/correct"""
        return await self.adapter.teach(
            session_id=request.get("session_id", "default"),
            user_input=request.get("input", ""),
            correct_response=request.get("correct_response", "")
        )

    async def handle_questions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GET /api/teacher/questions"""
        return await self.adapter.get_questions()

    def handle_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GET /api/ralph/status"""
        return self.adapter.get_status()

    def handle_providers(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GET /api/ralph/providers"""
        return self.adapter.get_providers()

    def handle_switch_provider(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle POST /api/ralph/provider"""
        return self.adapter.switch_provider(request.get("provider", "claude"))

    def handle_guardrails(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GET /api/ralph/guardrails"""
        return self.adapter.get_guardrails()

    def handle_loop_state(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GET /api/ralph/loops/:loop"""
        return self.adapter.get_loop_state(request.get("loop", "mid"))

    def handle_history(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GET /api/ralph/history"""
        return {
            "history": self.adapter.get_chat_history(request.get("session_id"))
        }
