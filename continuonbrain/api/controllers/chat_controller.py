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

    def _get_hope_agent(self):
        """Get the HOPEAgent instance from brain_service, auto-initializing if needed."""
        brain_service = getattr(self.server, 'brain_service', None)
        if brain_service is None:
            return None

        # Try to get existing hope_agent
        agent = getattr(brain_service, 'hope_agent', None)
        if agent is not None:
            return agent

        # Auto-initialize HOPE brain if not ready
        if hasattr(brain_service, 'initialize_hope_brain'):
            if brain_service.initialize_hope_brain():
                return getattr(brain_service, 'hope_agent', None)

        return None

    # =========================================================================
    # HOPE Active Learning Endpoints
    # =========================================================================

    @require_role(UserRole.CONSUMER)
    def handle_hope_analyze_scene(self, body: str):
        """
        POST /api/hope/analyze-scene
        Analyze the current scene and generate learning questions.

        This allows HOPE to proactively learn about what it sees by
        asking the user questions about unfamiliar objects or situations.

        Request body:
        {
            "include_visual": true  // optional, include segmentation data
        }

        Response:
        {
            "success": true,
            "scene_description": "I can see 3 objects...",
            "objects_detected": 3,
            "learning_opportunities": [...],
            "questions": [
                {
                    "question": "What is this object?",
                    "type": "object_identification",
                    "priority": "high"
                }
            ]
        }
        """
        try:
            hope_agent = self._get_hope_agent()
            if hope_agent is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Agent not available",
                }, status=503)
                return

            data = json.loads(body) if body else {}

            # Get current segmentation data if vision is available
            segmentation_data = None
            if data.get("include_visual", True):
                brain_service = getattr(self.server, 'brain_service', None)
                if brain_service and hasattr(brain_service, 'last_segmentation'):
                    segmentation_data = brain_service.last_segmentation

            result = hope_agent.analyze_scene_for_learning(segmentation_data)
            result["success"] = True
            self.send_json(result)

        except Exception as e:
            logger.error(f"Analyze scene error: {e}", exc_info=True)
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_hope_should_ask(self, body: str):
        """
        POST /api/hope/should-ask
        Check if HOPE should ask a clarifying question before responding.

        This is used by the chat interface to determine if HOPE needs
        more information before it can confidently respond.

        Request body:
        {
            "message": "Pick up that thing over there",
            "context": {
                "visual_perception": {...}  // optional
            }
        }

        Response:
        {
            "success": true,
            "should_ask": true,
            "question": {
                "reason": "needs_clarification",
                "confidence": 0.4,
                "question": "When you say 'that thing', which object do you mean?",
                "options": [...]
            }
        }
        """
        try:
            hope_agent = self._get_hope_agent()
            if hope_agent is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Agent not available",
                }, status=503)
                return

            data = json.loads(body) if body else {}
            message = data.get("message", "")
            context = data.get("context", {})

            # Enrich context with current visual perception if available
            if "visual_perception" not in context:
                brain_service = getattr(self.server, 'brain_service', None)
                if brain_service and hasattr(brain_service, 'last_segmentation'):
                    context["visual_perception"] = hope_agent.get_visual_perception(
                        brain_service.last_segmentation
                    )

            should_ask, question_info = hope_agent.should_ask_question(message, context)

            self.send_json({
                "success": True,
                "should_ask": should_ask,
                "question": question_info,
            })

        except Exception as e:
            logger.error(f"Should ask error: {e}", exc_info=True)
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_hope_learn_correction(self, body: str):
        """
        POST /api/hope/learn-correction
        Submit a correction to help HOPE learn.

        This allows users to correct HOPE's responses, enabling
        the agent to improve over time from human feedback.

        Request body:
        {
            "original_response": "I think that's a cup",
            "correction": "That's actually a mug",
            "context": {...}  // optional
        }

        Response:
        {
            "success": true,
            "learned": true,
            "memory_stored": true,
            "message": "Thank you for the correction..."
        }
        """
        try:
            hope_agent = self._get_hope_agent()
            if hope_agent is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Agent not available",
                }, status=503)
                return

            data = json.loads(body) if body else {}
            original = data.get("original_response", "")
            correction = data.get("correction", "")
            context = data.get("context", {})

            if not original or not correction:
                self.send_json({
                    "success": False,
                    "error": "Both original_response and correction are required",
                }, status=400)
                return

            # Get experience logger if available
            brain_service = getattr(self.server, 'brain_service', None)
            experience_logger = None
            if brain_service:
                experience_logger = getattr(brain_service, 'experience_logger', None)

            result = hope_agent.learn_from_correction(
                original_response=original,
                correction=correction,
                context=context,
                experience_logger=experience_logger,
            )
            result["success"] = True
            self.send_json(result)

        except Exception as e:
            logger.error(f"Learn correction error: {e}", exc_info=True)
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_hope_knowledge_gaps(self, body: str):
        """
        POST /api/hope/knowledge-gaps
        Get HOPE's current knowledge gaps.

        This returns areas where HOPE is uncertain or lacks knowledge,
        which can be used to guide training or user assistance.

        Request body:
        {
            "context": {...}  // optional current context
        }

        Response:
        {
            "success": true,
            "gaps": [
                {
                    "type": "unknown_objects",
                    "description": "I see 2 objects I can't identify",
                    "priority": "high",
                    "suggested_question": "What are these objects?"
                }
            ]
        }
        """
        try:
            hope_agent = self._get_hope_agent()
            if hope_agent is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Agent not available",
                }, status=503)
                return

            data = json.loads(body) if body else {}
            context = data.get("context", {})

            # Enrich context with visual perception
            if "visual_perception" not in context:
                brain_service = getattr(self.server, 'brain_service', None)
                if brain_service and hasattr(brain_service, 'last_segmentation'):
                    context["visual_perception"] = hope_agent.get_visual_perception(
                        brain_service.last_segmentation
                    )

            gaps = hope_agent.identify_knowledge_gaps(context)

            self.send_json({
                "success": True,
                "gaps": gaps,
                "total_gaps": len(gaps),
                "high_priority": len([g for g in gaps if g.get("priority") == "high"]),
            })

        except Exception as e:
            logger.error(f"Knowledge gaps error: {e}", exc_info=True)
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)

    @require_role(UserRole.CONSUMER)
    def handle_hope_clarify(self, body: str):
        """
        POST /api/hope/clarify
        Get clarifying questions for a message.

        This can be used by the chat interface to proactively
        show clarification options before sending to HOPE.

        Request body:
        {
            "message": "Move it over there",
            "context": {...}  // optional
        }

        Response:
        {
            "success": true,
            "questions": [
                {
                    "type": "object_reference",
                    "question": "When you say 'it', which object do you mean?",
                    "options": [...]
                }
            ]
        }
        """
        try:
            hope_agent = self._get_hope_agent()
            if hope_agent is None:
                self.send_json({
                    "success": False,
                    "error": "HOPE Agent not available",
                }, status=503)
                return

            data = json.loads(body) if body else {}
            message = data.get("message", "")
            context = data.get("context", {})

            # Enrich context with visual perception
            if "visual_perception" not in context:
                brain_service = getattr(self.server, 'brain_service', None)
                if brain_service and hasattr(brain_service, 'last_segmentation'):
                    context["visual_perception"] = hope_agent.get_visual_perception(
                        brain_service.last_segmentation
                    )

            questions = hope_agent.generate_clarifying_questions(message, context)

            self.send_json({
                "success": True,
                "questions": questions,
                "needs_clarification": len(questions) > 0,
            })

        except Exception as e:
            logger.error(f"Clarify error: {e}", exc_info=True)
            self.send_json({
                "success": False,
                "error": str(e),
            }, status=500)
