
import unittest
from unittest.mock import MagicMock, patch
import json
import logging
import sys
import os

# Ensure proper path
sys.path.append("/home/craigm26/ContinuonXR")

from continuonbrain.services.brain_service import BrainService

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestVisionChain(unittest.TestCase):
    def setUp(self):
        # Mock components to run without full hardware
        self.mock_camera = MagicMock()
        # Mock capture frame to return a valid dummy image structure (black pixel)
        import numpy as np
        dummy_frame = {"rgb": np.zeros((100, 100, 3), dtype=np.uint8)}
        self.mock_camera.capture_frame.return_value = dummy_frame
        
        self.brain = BrainService(
            config_dir="/tmp/continuonbrain_vision_test",
            prefer_real_hardware=False,
            auto_detect=False
        )
        self.brain.camera = self.mock_camera
        
        # Mock Gemma Chat to simulate Agent responses
        self.brain.gemma_chat = MagicMock()
        
        # Bypass Safety Protocol for Unit Test
        self.brain._check_safety_protocol = MagicMock(return_value=(True, "Safe for testing"))
        
    def test_vision_loop(self):
        """
        Verify the chain: 
        1. User asks "What do you see?"
        2. Agent calls [TOOL: CAPTURE_IMAGE]
        3. System returns image path
        4. Agent calls [TOOL: ASK_GEMINI ...]
        """
        print("\n--- Testing Vision Tool Chain ---")
        
        # Turn 1: User asks
        user_msg = "What is in front of you?"
        print(f"User: {user_msg}")
        
        # Mock Agent deciding to capture image
        # This simulates the LLM's decision making
        self.brain.gemma_chat.chat.side_effect = [
            "I will take a look. [TOOL: CAPTURE_IMAGE]", # Response 1
            "I see an object. [TOOL: ASK_GEMINI \"Describe this image\" /tmp/vision_capture.jpg]" # Response 2
        ]
        
        # --- Step 1: Agent decides to capture ---
        response_1 = self.brain.ChatWithGemma(user_msg, [])
        agent_text_1 = response_1["response"]
        print(f"Agent: {agent_text_1}")
        
        # Check if tool was parsed and executed
        self.assertIn("[TOOL: CAPTURE_IMAGE]", agent_text_1)
        self.mock_camera.capture_frame.assert_called_once()
        print("✅ Agent invoked CAPTURE_IMAGE")
        
        # Verify file creation (mocked but logic runs)
        # In real run, cv2.imwrite would be called.
        # We can verify the status updates in response
        updates = response_1["status_updates"]
        self.assertTrue(any("Captured visible world" in u for u in updates), "Tool execution status missing")
        print("✅ Tool execution confirmed via status")
        
        # --- Step 2: Agent analyzes image ---
        # We assume the chat history now contains the tool result (in loop) or the agent decides next step
        # In this unit test, we manually trigger next turn or mock conversation flow.
        # Here we just verify the tool parsing logic for the second step.
        
        # Mock subprocess to avoid actual Gemini call
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "I see a red ball on a table."
            
            response_2 = self.brain.ChatWithGemma("CONTINUE", []) # Simulating loop continuation
            agent_text_2 = response_2["response"]
            print(f"Agent: {agent_text_2}")
            
            # Check parsing of ASK_GEMINI
            # The tool parsing logic in BrainService handles the tool execution
            # Expected: subprocess.run called with correct args
            
            executed_cmds = [call_args[0][0] for call_args in mock_run.call_args_list]
            found_gemini_call = any("gemini_cli.py" in cmd for cmd in executed_cmds)
            self.assertTrue(found_gemini_call, "ASK_GEMINI tool did not trigger CLI")
            print("✅ Agent invoked ASK_GEMINI via CLI")


if __name__ == "__main__":
    unittest.main()
