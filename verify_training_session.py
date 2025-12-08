
import unittest
from unittest.mock import MagicMock, patch
import logging
import sys
import os
import shutil
import time

# Ensure proper path
sys.path.append("/home/craigm26/ContinuonXR")

from continuonbrain.services.brain_service import BrainService

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestTrainingSession(unittest.TestCase):
    def setUp(self):
        # Mock camera and gemma
        self.mock_camera = MagicMock()
        
        # Temp dir for checkpoints
        self.ckpt_dir = "/tmp/continuonbrain_test_training"
        if os.path.exists(self.ckpt_dir):
            shutil.rmtree(self.ckpt_dir)
            
        self.brain = BrainService(config_dir="/tmp/test_config", prefer_real_hardware=False, auto_detect=False)
        self.brain.camera = self.mock_camera
        
        # Mock Safety Protocol to allow TRAIN_VISION
        self.brain._check_safety_protocol = MagicMock(return_value=(True, "Safe"))
        self.brain.gemma_chat = MagicMock()

    def tearDown(self):
        if os.path.exists(self.ckpt_dir):
            shutil.rmtree(self.ckpt_dir)

    def test_training_trigger(self):
        print("\n--- Testing Training Trigger ---")
        
        # 1. User asks to train
        user_msg = "Please refine your vision model."
        print(f"User: {user_msg}")
        
        # 2. Agent invokes tool
        self.brain.gemma_chat.chat.return_value = "Starting training session. [TOOL: TRAIN_VISION]"
        
        # IMPORTANT: We need to patch where run_training_session looks for config?
        # No, the tool call uses default config or we can edit brain_service to accept args.
        # But wait, run_training_session defaults to ./checkpoints/aina_vision. 
        # For TEST isolation, we should probably patch 'continuonbrain.aina_impl.train.run_training_session'
        # so we don't actually run a heavy training loop in unit test, OR we verify it runs by checking side effects.
        # Let's run the REAL training (since it's dummy data + short epochs) to verify integration fully.
        # We just need to make sure it thinks it's writing to /tmp.
        # However, the tool implementation hardcodes run_training_session(config={"epochs": 5}).
        # It uses default checkpoint dir. We'll check ./checkpoints/aina_vision exists after.
        
        # Patching run_training_session to *just* override the checkpoint dir for hygiene
        with patch("continuonbrain.aina_impl.train.run_training_session") as mock_train:
            # We want the real logic but redirect dir... actually mocking the whole thing confirms the tool HANDLER works.
            # Verifying train.py logic was done via manual scrutiny/logic.
            # Let's just verify the tool handler calls the function.
            
            response = self.brain.ChatWithGemma(user_msg, [])
            print(f"Agent: {response['response']}")
            print(f"Status Updates: {response['status_updates']}")
            
            # Assert tool invocation
            self.assertIn("[TOOL: TRAIN_VISION]", response['response'])
            
            # Assert handlers fired
            # Wait a moment for thread to start (it's fast but async)
            time.sleep(0.5)
            
            mock_train.assert_called()
            print("✅ Training function called by Tool Handler")
            
            # Check status update
            self.assertTrue(any("Started AINA Training" in u for u in response["status_updates"]))
            print("✅ Status updated")

if __name__ == "__main__":
    unittest.main()
