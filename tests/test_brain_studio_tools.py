import os
import sys
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuonbrain.tools.calculator import CalculatorTool
from continuonbrain.tools.wikipedia import WikipediaTool
from continuonbrain.services.curriculum_manager import CurriculumManager
from continuonbrain.studio_server import StateAggregator

class TestBrainStudioTools(unittest.IsolatedAsyncioTestCase):
    async def test_calculator_tool(self):
        tool = CalculatorTool()
        res = await tool.execute("10 + 20 * 2")
        self.assertEqual(res["result"], 50)
        
        res = await tool.execute("math.sqrt(16)")
        self.assertEqual(res["result"], 4.0)

    async def test_wikipedia_tool(self):
        tool = WikipediaTool()
        # Mock wikipedia summary if not installed or to avoid network
        import continuonbrain.tools.wikipedia as wp_module
        original_wp = wp_module.wikipedia
        wp_module.wikipedia = MagicMock()
        wp_module.wikipedia.summary.return_value = "Python is a programming language."
        
        res = await tool.execute("Python")
        self.assertIn("Python", res["summary"])
        
        # Restore
        wp_module.wikipedia = original_wp

    async def test_curriculum_manager(self):
        aggregator = StateAggregator(asyncio.Queue())
        brain_service = MagicMock()
        brain_service.CallBrainTool = AsyncMock(return_value={"success": True, "result": 579})
        
        manager = CurriculumManager(brain_service, aggregator)
        lessons = manager.list_curriculum()
        self.assertGreater(len(lessons), 0)
        
        # Run math lesson
        res = await manager.run_lesson("math-basics")
        self.assertTrue(res["success"])
        self.assertTrue(res["all_passed"])
        
        # Verify queue got messages
        q = aggregator._event_queue
        self.assertGreater(q.qsize(), 0)
        
        msg = await q.get()
        self.assertEqual(msg["type"], "thought")
        self.assertIn("Starting Lesson", msg["text"])

if __name__ == "__main__":
    unittest.main()
