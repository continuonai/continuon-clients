"""
Teacher API Routes for Claude Code Integration

These routes enable Claude Code (or other teaching agents) to:
1. Get HOPE's pending questions (knowledge gaps)
2. Provide answers to HOPE's questions
3. Correct HOPE's mistakes
4. Demonstrate action sequences
5. Get teaching interaction summaries
6. Suggest teaching focus areas
"""

import json
import logging
from typing import Dict, Any, Optional
from http.server import BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


def register_teacher_routes(handler_class):
    """
    Register teacher API routes on the handler class.

    Call this from the main server setup.
    """
    original_do_GET = handler_class.do_GET
    original_do_POST = handler_class.do_POST

    def new_do_GET(self):
        if self.path == "/api/teacher/questions":
            return handle_get_questions(self)
        elif self.path == "/api/teacher/summary":
            return handle_get_summary(self)
        elif self.path == "/api/teacher/suggestions":
            return handle_get_suggestions(self)
        return original_do_GET(self)

    def new_do_POST(self):
        if self.path == "/api/teacher/answer":
            return handle_provide_answer(self)
        elif self.path == "/api/teacher/correct":
            return handle_provide_correction(self)
        elif self.path == "/api/teacher/demonstrate":
            return handle_demonstrate_action(self)
        elif self.path == "/api/teacher/validate":
            return handle_validate_knowledge(self)
        return original_do_POST(self)

    handler_class.do_GET = new_do_GET
    handler_class.do_POST = new_do_POST


def get_teacher_interface(handler):
    """Get or create the TeacherInterface from brain_service."""
    brain_service = getattr(handler, 'brain_service', None)
    if not brain_service:
        # Try to get from global
        import continuonbrain.api.server as server_module
        brain_service = getattr(server_module, 'brain_service', None)

    if not brain_service:
        return None

    # Get or create teacher interface
    if not hasattr(brain_service, '_teacher_interface') or brain_service._teacher_interface is None:
        from continuonbrain.services.world_model_integration import create_world_model_integration
        wm, teacher = create_world_model_integration(brain_service)
        brain_service._world_model_integration = wm
        brain_service._teacher_interface = teacher

    return brain_service._teacher_interface


def send_json_response(handler, data: Dict[str, Any], status: int = 200):
    """Send a JSON response."""
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps(data).encode())


def read_json_body(handler) -> Optional[Dict[str, Any]]:
    """Read and parse JSON body from request."""
    try:
        content_length = int(handler.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = handler.rfile.read(content_length)
        return json.loads(body.decode())
    except Exception as e:
        logger.error(f"Error reading JSON body: {e}")
        return None


def handle_get_questions(handler):
    """
    GET /api/teacher/questions

    Returns HOPE's pending questions that need answers from the teacher.

    Response:
    {
        "questions": [
            {
                "id": 0,
                "type": "unknown_concept",
                "question": "What is the purpose of this object?",
                "priority": "high",
                "context": {...}
            }
        ],
        "count": 1
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "questions": [],
            "count": 0
        }, 503)
        return

    try:
        questions = teacher.get_pending_questions()
        send_json_response(handler, {
            "questions": questions,
            "count": len(questions)
        })
    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        send_json_response(handler, {
            "error": str(e),
            "questions": [],
            "count": 0
        }, 500)


def handle_get_summary(handler):
    """
    GET /api/teacher/summary

    Returns a summary of teaching interactions.

    Response:
    {
        "total_interactions": 10,
        "corrections": 3,
        "answers": 5,
        "demonstrations": 2,
        "recent": [...]
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "total_interactions": 0
        }, 503)
        return

    try:
        summary = teacher.get_teaching_summary()
        send_json_response(handler, summary)
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        send_json_response(handler, {"error": str(e)}, 500)


def handle_get_suggestions(handler):
    """
    GET /api/teacher/suggestions

    Returns suggested areas where teaching would be most helpful.

    Response:
    {
        "suggestions": [
            {
                "type": "knowledge_gap",
                "description": "HOPE has 3 high-priority questions",
                "action": "Answer pending questions",
                "questions": [...]
            }
        ]
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "suggestions": []
        }, 503)
        return

    try:
        suggestions = teacher.suggest_teaching_focus()
        send_json_response(handler, {"suggestions": suggestions})
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        send_json_response(handler, {"error": str(e), "suggestions": []}, 500)


def handle_provide_answer(handler):
    """
    POST /api/teacher/answer

    Provide an answer to one of HOPE's questions.

    Request:
    {
        "question_id": 0,
        "answer": "This is the answer...",
        "confidence": 0.9
    }

    Response:
    {
        "success": true,
        "message": "Thank you for teaching me: This is the answer...",
        "stored": true
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "success": False
        }, 503)
        return

    body = read_json_body(handler)
    if body is None:
        send_json_response(handler, {
            "error": "Invalid JSON body",
            "success": False
        }, 400)
        return

    question_id = body.get("question_id", 0)
    answer = body.get("answer", "")
    confidence = body.get("confidence", 0.9)

    if not answer:
        send_json_response(handler, {
            "error": "Answer is required",
            "success": False
        }, 400)
        return

    try:
        result = teacher.provide_answer(question_id, answer, confidence)
        send_json_response(handler, result)
    except Exception as e:
        logger.error(f"Error providing answer: {e}")
        send_json_response(handler, {"error": str(e), "success": False}, 500)


def handle_provide_correction(handler):
    """
    POST /api/teacher/correct

    Correct a mistake HOPE made.

    Request:
    {
        "original_response": "What HOPE said",
        "correct_response": "What it should have been",
        "context": {...}  // optional
    }

    Response:
    {
        "success": true,
        "learned": true,
        "message": "Thank you for the correction..."
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "success": False
        }, 503)
        return

    body = read_json_body(handler)
    if body is None:
        send_json_response(handler, {
            "error": "Invalid JSON body",
            "success": False
        }, 400)
        return

    original = body.get("original_response", "")
    correct = body.get("correct_response", "")
    context = body.get("context")

    if not original or not correct:
        send_json_response(handler, {
            "error": "original_response and correct_response are required",
            "success": False
        }, 400)
        return

    try:
        result = teacher.provide_correction(original, correct, context)
        send_json_response(handler, result)
    except Exception as e:
        logger.error(f"Error providing correction: {e}")
        send_json_response(handler, {"error": str(e), "success": False}, 500)


def handle_demonstrate_action(handler):
    """
    POST /api/teacher/demonstrate

    Demonstrate how to perform an action.

    Request:
    {
        "action_name": "pick up cup",
        "action_steps": [
            "Move arm above cup",
            "Lower gripper",
            "Close gripper",
            "Lift arm"
        ],
        "context": {...}  // optional
    }

    Response:
    {
        "success": true,
        "message": "Learned demonstration for 'pick up cup' with 4 steps",
        "stored": true
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "success": False
        }, 503)
        return

    body = read_json_body(handler)
    if body is None:
        send_json_response(handler, {
            "error": "Invalid JSON body",
            "success": False
        }, 400)
        return

    action_name = body.get("action_name", "")
    action_steps = body.get("action_steps", [])
    context = body.get("context")

    if not action_name or not action_steps:
        send_json_response(handler, {
            "error": "action_name and action_steps are required",
            "success": False
        }, 400)
        return

    try:
        result = teacher.demonstrate_action(action_name, action_steps, context)
        send_json_response(handler, result)
    except Exception as e:
        logger.error(f"Error demonstrating action: {e}")
        send_json_response(handler, {"error": str(e), "success": False}, 500)


def handle_validate_knowledge(handler):
    """
    POST /api/teacher/validate

    Validate HOPE's knowledge on a topic.

    Request:
    {
        "topic": "object manipulation",
        "test_questions": [
            {"question": "How do you pick up a cup?", "expected_concepts": ["gripper", "lift"]},
            ...
        ]
    }

    Response:
    {
        "topic": "object manipulation",
        "score": 0.75,
        "passed": 3,
        "failed": 1,
        "details": [...]
    }
    """
    teacher = get_teacher_interface(handler)
    if not teacher:
        send_json_response(handler, {
            "error": "Teacher interface not available",
            "success": False
        }, 503)
        return

    body = read_json_body(handler)
    if body is None:
        send_json_response(handler, {
            "error": "Invalid JSON body",
            "success": False
        }, 400)
        return

    topic = body.get("topic", "general")
    test_questions = body.get("test_questions", [])

    # For now, return a placeholder validation
    # This would be enhanced to actually test HOPE's knowledge
    try:
        results = {
            "topic": topic,
            "score": 0.5,
            "passed": 0,
            "failed": 0,
            "details": [],
            "message": "Knowledge validation is in development. Use /api/teacher/suggestions to identify gaps."
        }

        if test_questions:
            results["passed"] = len(test_questions) // 2
            results["failed"] = len(test_questions) - results["passed"]
            results["score"] = results["passed"] / len(test_questions) if test_questions else 0

        send_json_response(handler, results)
    except Exception as e:
        logger.error(f"Error validating knowledge: {e}")
        send_json_response(handler, {"error": str(e)}, 500)
