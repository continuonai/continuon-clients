"""Curriculum Manager for autonomous skill-teaching."""
from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Optional
from continuonbrain.studio_server import StateAggregator

logger = logging.getLogger("CurriculumManager")

class LessonChallenge:
    def __init__(self, id: str, title: str, tool: str, args: Dict[str, Any], expected_result_contains: Optional[str] = None):
        self.id = id
        self.title = title
        self.tool = tool
        self.args = args
        self.expected_result_contains = expected_result_contains

class Lesson:
    def __init__(self, id: str, title: str, challenges: List[LessonChallenge]):
        self.id = id
        self.title = title
        self.challenges = challenges

class CurriculumManager:
    """Orchestrates autonomous 'Lessons' to verify skill acquisition."""

    def __init__(self, brain_service: Any, aggregator: StateAggregator):
        self.brain_service = brain_service
        self.aggregator = aggregator
        self.lessons: Dict[str, Lesson] = self._init_lessons()
        self.active_lesson_id: Optional[str] = None

    def _init_lessons(self) -> Dict[str, Lesson]:
        return {
            "math-basics": Lesson("math-basics", "Deterministic Logic (Math)", [
                LessonChallenge("calc-1", "Basic Addition", "calculator", {"expression": "123 + 456"}, "579"),
                LessonChallenge("calc-2", "Square Root", "calculator", {"expression": "math.sqrt(144)"}, "12.0")
            ]),
            "world-knowledge": Lesson("world-knowledge", "Global Knowledge (Wikipedia)", [
                LessonChallenge("wiki-1", "General Knowledge", "wikipedia", {"query": "Raspberry Pi 5"}, "Raspberry Pi"),
                LessonChallenge("wiki-2", "Scientific Fact", "wikipedia", {"query": "Mars"}, "planet")
            ])
        }

    async def run_lesson(self, lesson_id: str) -> Dict[str, Any]:
        """Execute all challenges in a lesson."""
        lesson = self.lessons.get(lesson_id)
        if not lesson:
            return {"success": False, "message": f"Lesson {lesson_id} not found."}

        self.active_lesson_id = lesson_id
        self.aggregator.push_thought(f"Starting Lesson: {lesson.title}", source="system")
        
        results = []
        for challenge in lesson.challenges:
            self.aggregator.push_thought(f"Challenge: {challenge.title}", source="system")
            
            # Call Brain tool
            res = await self.brain_service.CallBrainTool(challenge.tool, challenge.args)
            
            passed = res.get("success", False)
            if passed and challenge.expected_result_contains:
                result_str = str(res.get("result", ""))
                if challenge.expected_result_contains.lower() not in result_str.lower():
                    passed = False
                    self.aggregator.push_thought(f"Verification failed: Expected '{challenge.expected_result_contains}' in result.", source="system")

            results.append({
                "challenge_id": challenge.id,
                "title": challenge.title,
                "passed": passed,
                "result": res
            })
            
            await asyncio.sleep(1.0) # Small pause between challenges

        self.active_lesson_id = None
        all_passed = all(r["passed"] for r in results)
        summary = f"Lesson {lesson.title} {'COMPLETED' if all_passed else 'FAILED'}"
        self.aggregator.push_thought(summary, source="system")
        
        return {
            "success": True,
            "lesson_id": lesson_id,
            "all_passed": all_passed,
            "results": results
        }

    def list_curriculum(self) -> List[Dict[str, Any]]:
        return [{
            "id": l.id,
            "title": l.title,
            "challenge_count": len(l.challenges)
        } for l in self.lessons.values()]
