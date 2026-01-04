"""
Chat Controller - API endpoints for HOPE Chat conversational interface.

This controller provides the API for the HopeChat system, enabling
both the web UI (Brain Studio) and Flutter app (ContinuonAI) to
have conversations with the HOPE brain.
"""

import json
import logging
from typing import Any, Dict

from continuonbrain.core.security import UserRole
from continuonbrain.api.middleware.auth import require_role

logger = logging.getLogger(__name__)


class ChatControllerMixin:
    """
    Mixin for chat API requests.

    Provides endpoints for conversational interaction with the HOPE brain.
    """

    @require_role(UserRole.CONSUMER)
    def handle_chat(self, body: str):
        """
        POST /api/chat
        Send a message to the HOPE brain and get a response.

        Request body:
        {
            "message": "Hello, HOPE!",
            "session_id": "optional-session-id",
            "history": [{"role": "user", "content": "..."}]  // optional
        }

        Response:
        {
            "success": true,
            "response": "Hello! I'm HOPE...",
            "session_id": "abc123",
            "agent": "hope_brain",
            "duration_ms": 45.2,
            "turn_count": 2
        }
        """
        try:
            hope_chat = self._get_hope_chat()
            if hope_chat is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Chat not available",
                    "response": "I'm still initializing. Please try again in a moment.",
                }, status=503)
                return

            data = json.loads(body) if body else {}
            message = data.get("message", "").strip()

            if not message:
                self.send_json({
                    "success": False,
                    "error": "Message is required",
                }, status=400)
                return

            session_id = data.get("session_id")
            history = data.get("history")

            result = hope_chat.chat(
                message=message,
                session_id=session_id,
                history=history,
            )

            self.send_json(result)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chat request: {e}")
            self.send_json({
                "success": False,
                "error": "Invalid JSON in request body",
            }, status=400)
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            self.send_json({
                "success": False,
                "error": str(e),
                "response": "I encountered an error. Please try again.",
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_chat_status(self):
        """
        GET /api/chat/status
        Get HOPE Chat service status.

        Response:
        {
            "success": true,
            "status": {
                "ready": true,
                "encoder_loaded": true,
                "decoder_loaded": true,
                "active_sessions": 3,
                "model_dir": "/opt/continuonos/brain/model/seed_stable"
            }
        }
        """
        try:
            hope_chat = self._get_hope_chat()
            if hope_chat is None:
                self.send_json({
                    "success": True,
                    "status": {
                        "ready": False,
                        "message": "HOPE Chat not initialized",
                    }
                })
                return

            status = hope_chat.get_status()
            self.send_json({"success": True, "status": status})

        except Exception as e:
            logger.error(f"Chat status error: {e}")
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_chat_session(self, session_id: str):
        """
        GET /api/chat/session/<session_id>
        Get session history.

        Response:
        {
            "success": true,
            "session": {
                "session_id": "abc123",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ],
                "created_at": 1704387600.0,
                "last_activity": 1704388000.0
            }
        }
        """
        try:
            hope_chat = self._get_hope_chat()
            if hope_chat is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Chat not available",
                }, status=503)
                return

            session = hope_chat.get_session(session_id)
            if session is None:
                self.send_json({
                    "success": False,
                    "error": f"Session not found: {session_id}",
                }, status=404)
                return

            self.send_json({
                "success": True,
                "session": {
                    "session_id": session.session_id,
                    "messages": session.get_history(limit=100),
                    "created_at": session.created_at,
                    "last_activity": session.last_activity,
                }
            })

        except Exception as e:
            logger.error(f"Chat session error: {e}")
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_chat_clear_session(self, body: str):
        """
        POST /api/chat/clear
        Clear a chat session.

        Request body:
        {
            "session_id": "abc123"
        }

        Response:
        {
            "success": true,
            "message": "Session cleared"
        }
        """
        try:
            hope_chat = self._get_hope_chat()
            if hope_chat is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Chat not available",
                }, status=503)
                return

            data = json.loads(body) if body else {}
            session_id = data.get("session_id")

            if not session_id:
                self.send_json({
                    "success": False,
                    "error": "session_id is required",
                }, status=400)
                return

            hope_chat.clear_session(session_id)
            self.send_json({
                "success": True,
                "message": f"Session {session_id} cleared",
            })

        except Exception as e:
            logger.error(f"Chat clear session error: {e}")
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    def _get_hope_chat(self):
        """Get the HopeChat instance from brain_service."""
        brain_service = getattr(self.server, 'brain_service', None)
        if brain_service is None:
            return None
        return getattr(brain_service, 'hope_chat', None)
