#!/usr/bin/env python3
"""
Orchestrator End-to-End Tests

Tests the complete orchestrator system including:
- Task submission and execution
- Workflow execution
- Event handling
- State persistence
- Health monitoring
"""

import sys
import time
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from continuonbrain.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    Task,
    TaskType,
    TaskPriority,
    TaskStatus,
    TaskResult,
    Workflow,
    WorkflowStep,
    EventBus,
    EventType,
    create_orchestrator,
)


class TestTaskQueue(unittest.TestCase):
    """Test task queue functionality."""

    def test_task_priority_ordering(self):
        """Tasks should be ordered by priority."""
        from continuonbrain.orchestrator.task import TaskQueue

        queue = TaskQueue()

        # Submit tasks in reverse priority order
        low = Task(task_type=TaskType.CUSTOM, priority=TaskPriority.LOW)
        high = Task(task_type=TaskType.CUSTOM, priority=TaskPriority.HIGH)
        critical = Task(task_type=TaskType.CUSTOM, priority=TaskPriority.CRITICAL)

        queue.put(low)
        queue.put(high)
        queue.put(critical)

        # Should come out in priority order
        self.assertEqual(queue.get_nowait().priority, TaskPriority.CRITICAL)
        self.assertEqual(queue.get_nowait().priority, TaskPriority.HIGH)
        self.assertEqual(queue.get_nowait().priority, TaskPriority.LOW)

    def test_task_cancellation(self):
        """Tasks can be cancelled while queued."""
        from continuonbrain.orchestrator.task import TaskQueue

        queue = TaskQueue()
        task = Task(task_type=TaskType.CUSTOM)
        queue.put(task)

        self.assertTrue(queue.cancel(task.task_id))
        self.assertEqual(task.status, TaskStatus.CANCELLED)


class TestEventBus(unittest.TestCase):
    """Test event bus functionality."""

    def test_event_subscription(self):
        """Events should be delivered to subscribers."""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.TASK_COMPLETED, handler)
        bus.emit(EventType.TASK_COMPLETED, {"task_id": "test"})

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data["task_id"], "test")

    def test_global_subscription(self):
        """Global handlers receive all events."""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe_all(handler)
        bus.emit(EventType.TASK_COMPLETED, {"id": 1})
        bus.emit(EventType.TASK_FAILED, {"id": 2})

        self.assertEqual(len(received), 2)


class TestOrchestrator(unittest.TestCase):
    """Test orchestrator core functionality."""

    def setUp(self):
        """Create test orchestrator."""
        self.orch = create_orchestrator(development=True)

    def tearDown(self):
        """Stop orchestrator."""
        if self.orch.is_running():
            self.orch.stop()

    def test_orchestrator_start_stop(self):
        """Orchestrator should start and stop cleanly."""
        self.assertFalse(self.orch.is_running())
        self.orch.start()
        self.assertTrue(self.orch.is_running())
        self.orch.stop()
        self.assertFalse(self.orch.is_running())

    def test_task_submission(self):
        """Tasks should be submitted and tracked."""
        self.orch.start()

        task_id = self.orch.submit_task(TaskType.HEALTH_CHECK)
        self.assertIsNotNone(task_id)

        task = self.orch.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.task_type, TaskType.HEALTH_CHECK)

    def test_custom_handler(self):
        """Custom task handlers should be executed."""
        executed = []

        def custom_handler(task):
            executed.append(task.task_id)
            return TaskResult(success=True, data={"executed": True})

        self.orch.register_handler(TaskType.TRAIN, custom_handler)
        self.orch.start()

        task_id = self.orch.submit_task(TaskType.TRAIN, {"test": True})
        time.sleep(0.5)  # Wait for execution

        self.assertIn(task_id, executed)
        task = self.orch.get_task(task_id)
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_health_check(self):
        """Health check should return valid status."""
        self.orch.start()
        self.orch._monitor.run_all_checks()

        health = self.orch.get_health()
        self.assertIn("overall", health)
        self.assertIn("checks", health)
        self.assertIn("cpu", health["checks"])
        self.assertIn("memory", health["checks"])

    def test_status(self):
        """Status should include all components."""
        self.orch.start()

        status = self.orch.get_status()
        self.assertTrue(status["running"])
        self.assertIn("tasks", status)
        self.assertIn("workers", status)
        self.assertIn("workflows", status)


class TestWorkflow(unittest.TestCase):
    """Test workflow functionality."""

    def setUp(self):
        """Create test orchestrator with custom handlers."""
        self.orch = create_orchestrator(development=True)

        # Register handlers for all workflow task types
        def make_handler(name):
            def handler(task):
                time.sleep(0.05)
                return TaskResult(success=True, data={f"{name}_done": True})
            return handler

        for task_type in [TaskType.LOAD_DATA, TaskType.PREPROCESS, TaskType.TRAIN,
                          TaskType.VALIDATE, TaskType.CHECKPOINT]:
            self.orch.register_handler(task_type, make_handler(task_type.value))

    def tearDown(self):
        """Stop orchestrator."""
        if self.orch.is_running():
            self.orch.stop()

    def test_workflow_execution(self):
        """Workflow should execute all steps in order."""
        self.orch.start()

        wf_id = self.orch.run_workflow("training_pipeline")
        self.assertIsNotNone(wf_id)

        # Wait for workflow completion
        timeout = 10
        start = time.time()
        while time.time() - start < timeout:
            workflow = self.orch.get_workflow(wf_id)
            if workflow.status.name in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)

        workflow = self.orch.get_workflow(wf_id)
        self.assertEqual(workflow.status.name, "COMPLETED")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTaskQueue))
    suite.addTests(loader.loadTestsFromTestCase(TestEventBus))
    suite.addTests(loader.loadTestsFromTestCase(TestOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflow))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
