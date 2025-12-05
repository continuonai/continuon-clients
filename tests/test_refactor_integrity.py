"""
Verify Refactor Integrity.
Checks if new modules can be imported and instantiated.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

class TestRefactor(unittest.TestCase):
    def test_imports(self):
        print("Testing Imports...")
        try:
            from continuonbrain.core.models import TaskDefinition
            print("✅ continuonbrain.core.models")
            
            from continuonbrain.services.brain_service import BrainService
            print("✅ continuonbrain.services.brain_service")
            
            from continuonbrain.api.routes import ui_routes
            print("✅ continuonbrain.api.routes.ui_routes")
            
            # Check basic instantiation
            task_lib = TaskDefinition(id="test", title="Test", description="Desc", group="Test")
            self.assertEqual(task_lib.id, "test")
            
        except ImportError as e:
            self.fail(f"Import failed: {e}")

if __name__ == "__main__":
    unittest.main()
