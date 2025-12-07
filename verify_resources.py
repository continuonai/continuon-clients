
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.append("/home/craigm26/ContinuonXR")

from continuonbrain.services.brain_service import BrainService
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel, ResourceStatus

class TestResourceAwareness(unittest.TestCase):
    def setUp(self):
        self.brain = BrainService(auto_detect=False, allow_mock_fallback=True)
        # Mock resource monitor
        self.brain.resource_monitor = MagicMock(spec=ResourceMonitor)
        
    def test_agent_warning_on_critical(self):
        # Setup critical status
        critical_status = ResourceStatus(
            timestamp=0, level=ResourceLevel.CRITICAL,
            total_memory_mb=1000, used_memory_mb=900, available_memory_mb=100, memory_percent=90,
            total_swap_mb=0, used_swap_mb=0, swap_percent=0,
            system_reserve_mb=200, max_brain_mb=500, can_allocate=False, message="Critical"
        )
        self.brain.resource_monitor.check_resources.return_value = critical_status
        
        # Test ChatWithGemma context injection (we can check the args passed to gemma_chat.chat)
        self.brain.gemma_chat = MagicMock()
        self.brain.gemma_chat.chat.return_value = "I advise sleep mode."
        
        self.brain.ChatWithGemma("Status?", [])
        
        # Verify the system context contained the warning
        call_args = self.brain.gemma_chat.chat.call_args
        system_context = call_args[1].get('system_context', '')
        
        print(f"Captured System Context:\n{system_context}")
        
        self.assertIn("!!! CRITICAL RESOURCE NOTICE !!!", system_context)
        self.assertIn("ADVICE TO AGENT", system_context)
        self.assertIn("Sleep Mode", system_context)

if __name__ == '__main__':
    unittest.main()
