"""
Verify Nested Learning Sidecar Launch and Gemma Readiness.
"""
import sys
import unittest
import subprocess
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.startup_manager import StartupManager

class TestNestedLearningStartup(unittest.TestCase):

    def setUp(self):
        self.config_dir = REPO_ROOT / "continuonbrain" / "tests" / "fixtures" / "brain_config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @patch("subprocess.Popen")
    @patch("continuonbrain.startup_manager.StartupManager.launch_ui")
    @patch("continuonbrain.startup_manager.AgentIdentity")
    @patch("continuonbrain.startup_manager.RobotModeManager")
    @patch("continuonbrain.startup_manager.LANDiscoveryService")
    def test_sidecar_launch(self, MockLAN, MockMode, MockIdentity, MockLaunchUI, MockPopen):
        """Verify run_trainer is launched."""
        print("\nTesting Sidecar Launch...")
        
        # Setup mocks
        MockPopen.return_value.pid = 1234
        
        manager = StartupManager(config_dir=str(self.config_dir), start_services=True)
        # Mock discovery service behavior
        manager.discovery_service = MockLAN.return_value
        manager.discovery_service.get_robot_info.return_value = {'ip_address': '127.0.0.1'}
        
        # Run startup
        manager.startup(force_health_check=False)
        
        # Check Popen calls
        # Expected calls: Robot API, then Run Trainer (UI is mocked/skipped)
        # We need to find the call with "continuonbrain.run_trainer"
        
        sidecar_called = False
        for call_args in MockPopen.call_args_list:
            args, kwargs = call_args
            cmd_list = args[0]
            if "continuonbrain.run_trainer" in cmd_list or "continuonbrain.run_trainer" in str(cmd_list):
                sidecar_called = True
                print("✅ Found sidecar launch command:", cmd_list)
                break
        
        self.assertTrue(sidecar_called, "Nested Learning Sidecar (run_trainer) was not launched")

    def test_gemma_import_ready(self):
        """Verify we can import transformers and instantiate GemmaChat (mocking load)."""
        print("\nTesting Gemma Software Readiness...")
        try:
            import transformers
            import torch
            print(f"✅ Transformers {transformers.__version__} installed")
            print(f"✅ Torch {torch.__version__} installed")
        except ImportError:
            self.fail("Transformers/Torch not installed!")

        # Now try to create GemmaChat and see if it picks non-mock
        from continuonbrain.gemma_chat import create_gemma_chat, GemmaChat, MockGemmaChat
        
        chat = create_gemma_chat(use_mock=False)
        
        if isinstance(chat, MockGemmaChat):
            print("⚠️ create_gemma_chat returned MockGemmaChat - transformers import inside module failed?")
            # This fails the user requirement "actually on device"
            self.fail("GemmaChat reverted to Mock despite libraries being present.")
        
        self.assertIsInstance(chat, GemmaChat)
        print("✅ GemmaChat instantiated in Real Mode")

if __name__ == "__main__":
    unittest.main()
