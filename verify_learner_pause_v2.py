
import unittest
from unittest.mock import MagicMock, patch
import sys
import logging
import time

# Add repo root to path
sys.path.append("/home/craigm26/ContinuonXR")

from continuonbrain.services.background_learner import BackgroundLearner
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel, ResourceStatus

class TestLearnerPause(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_brain = MagicMock()
        self.mock_brain.obs_dim = 10
        self.mock_brain.action_dim = 2
        
        self.mock_resource_monitor = MagicMock(spec=ResourceMonitor)
        
        # Patch internal components
        # We use a context manager in setUp, but need to stop it in tearDown?
        # Simpler to just patch broadly for the class.
        pass

    @patch('continuonbrain.services.background_learner.CuriosityEnvironment')
    @patch('continuonbrain.services.background_learner.CheckpointManager')
    @patch('continuonbrain.services.background_learner.time.sleep')
    @patch('continuonbrain.services.background_learner.logger')
    def test_pause_on_critical_resource(self, mock_logger, mock_sleep, mock_ckpt, mock_env):
        print("Test started")
        
        # Initialize Learner with mocks
        learner = BackgroundLearner(
            brain=self.mock_brain,
            resource_monitor=self.mock_resource_monitor
        )
        print("Learner initialized")
        
        # Setup critical status
        critical_status = ResourceStatus(
            timestamp=0, level=ResourceLevel.CRITICAL,
            total_memory_mb=1000, used_memory_mb=900, available_memory_mb=100, memory_percent=90,
            total_swap_mb=0, used_swap_mb=0, swap_percent=0,
            system_reserve_mb=200, max_brain_mb=500, can_allocate=False, message="Critical"
        )
        self.mock_resource_monitor.check_resources.return_value = critical_status
        
        # Setup loop control
        learner.running = True
        
        # Define side effect to break the loop
        def sleep_side_effect(seconds):
            print(f"Sleep called with {seconds}s")
            if seconds == 5.0:
                print("Triggering break exception")
                learner.running = False # Stop the loop gracefully
                return None # The loop will continue, check running, and exit
            return None
            
        mock_sleep.side_effect = sleep_side_effect
        
        # Run loop
        print("Starting learning loop")
        learner._learning_loop()
        print("Learning loop finished")
        
        # Verify
        # 1. Check resources was called
        # The loop runs indefinitely, checking resources at start of iteration
        self.mock_resource_monitor.check_resources.assert_called()
        
        # 2. Verify logger warning was called
        # Note: total_steps is 0, so 0 % 100 == 0 is true
        mock_logger.warning.assert_called_with("Resource constraint (critical): Pausing autonomous learning.")
        
        # 3. Verify sleep(5.0) was called
        mock_sleep.assert_called_with(5.0)
        
        print("Success: Verified pause on critical resources.")

if __name__ == '__main__':
    unittest.main()
