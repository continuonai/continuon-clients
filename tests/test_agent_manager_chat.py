"""
Tests for Agent Manager Chat Persona and Real Mode Enforcement.
"""
import sys
import unittest
import asyncio
import json
import shutil
from unittest.mock import MagicMock, patch
from pathlib import Path
import datetime

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Patch before import to control gemma creation
with patch("continuonbrain.robot_api_server.create_gemma_chat") as mock_create_chat:
    from continuonbrain.robot_api_server import RobotService

class TestAgentManagerChat(unittest.TestCase):
    
    def setUp(self):
        self.config_dir = REPO_ROOT / "continuonbrain" / "tests" / "fixtures" / "brain_config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # Clean logs
        self.log_dir = self.config_dir / "memories" / "chat_logs"
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir)

    def test_real_hardware_default(self):
        """Verify RobotService now enforces real hardware by default."""
        # Note: We need to patch hardware controllers to avoid actual init attempts or check the flags
        with patch("continuonbrain.robot_api_server.PCA9685ArmController"), \
             patch("continuonbrain.robot_api_server.OAKDepthCapture"), \
             patch("continuonbrain.robot_api_server.DrivetrainController"):
            
            service = RobotService(config_dir=str(self.config_dir))
            
            self.assertTrue(service.prefer_real_hardware)
            # allow_mock_fallback should be False by default now
            self.assertFalse(service.allow_mock_fallback)
            print("✅ RobotService defaults obey 'Always Non-Mock'")

    async def async_test_agent_manager_prompt(self):
        """Verify ChatWithGemma uses Agent Manager persona and logs chat."""
        print("\nTesting Agent Manager Chat...")
        
        # Setup mocks
        mock_gemma = MagicMock()
        mock_gemma.chat.return_value = "Hello, I am the Agent Manager. How can I help you improve?"
        
        with patch("continuonbrain.robot_api_server.create_gemma_chat", return_value=mock_gemma), \
             patch("continuonbrain.robot_api_server.PCA9685ArmController"), \
             patch("continuonbrain.robot_api_server.OAKDepthCapture"), \
             patch("continuonbrain.robot_api_server.DrivetrainController") as MockDrivetrain:
            
            # Allow mock fallback explicitly for this test only, 
            # or rely on patched controllers pretending to be real.
            # Since we patched the controllers to succeed, real mode init should succeed.
             
            MockDrivetrain.return_value.initialize.return_value = True

            service = RobotService(config_dir=str(self.config_dir))
            # Manually inject mock gemma if init re-created it (init calls create_gemma_chat)
            service.gemma_chat = mock_gemma
            
            # Call Chat
            user_msg = "Create a new learning agent"
            response = await service.ChatWithGemma(user_msg, [])
            
            # Verify Response
            self.assertEqual(response["response"], "Hello, I am the Agent Manager. How can I help you improve?")
            
            # Verify Prompt Context
            # Check the call arguments
            args, kwargs = mock_gemma.chat.call_args
            system_prompt = kwargs.get("system_context", "")
            
            print(f"Captured System Prompt:\n{system_prompt}")
            
            self.assertIn("You are the Agent Manager", system_prompt)
            self.assertIn("manage sub-agents", system_prompt)
            self.assertIn("Current Status:", system_prompt)
            print("✅ Agent Manager System Prompt confirmed")
            
            # Verify Logging
            log_files = list(self.log_dir.glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            
            with open(log_files[0], 'r') as f:
                log_entry = json.loads(f.readline())
                self.assertEqual(log_entry["user_message"], user_msg)
                self.assertEqual(log_entry["agent_response"], response["response"])
                self.assertIn("timestamp", log_entry)
            print("✅ Chat logged to memory")

    def test_chat(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_test_agent_manager_prompt())
        loop.close()

if __name__ == "__main__":
    unittest.main()
