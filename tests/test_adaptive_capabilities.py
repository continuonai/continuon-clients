"""
Tests for Adaptive UI Capability Negotiation.
"""
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.robot_api_server import RobotService, TaskDefinition

class TestAdaptiveCapabilities(unittest.TestCase):
    
    def setUp(self):
        self.config_dir = REPO_ROOT / "continuonbrain" / "tests" / "fixtures" / "brain_config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    async def async_test_missing_arm(self):
        """Verify that missing arm blocks manipulation tasks."""
        print("\nTesting Missing Arm Scenario...")
        
        # Initialize service with NO arm (force mock failure or partial init)
        service = RobotService(config_dir=str(self.config_dir), prefer_real_hardware=False, auto_detect=False)
        
        # Manually force partial initialization state
        service.arm = None # No arm
        service.camera = MagicMock() # Has camera
        service.drivetrain = MagicMock()
        service.drivetrain.initialized = True # Has base
        
        # Check Capabilities
        caps = service.capabilities
        print(f"Capabilities: {caps}")
        self.assertFalse(caps["has_manipulator"])
        self.assertTrue(caps["has_vision"])
        self.assertTrue(caps["has_mobile_base"])
        
        # Check Task Eligibility (Pick & Place requires arm)
        pick_task = TaskDefinition(
            id="test-pick", title="Test Pick", description="...", group="test",
            required_modalities=["arm", "vision"]
        )
        
        eligibility = service._build_task_eligibility(pick_task)
        print(f"Pick Task Eligible: {eligibility.eligible}")
        self.assertFalse(eligibility.eligible)
        self.assertTrue(any(m.code == "MISSING_ARM" for m in eligibility.markers))
        
        # Check Task Eligibility (Inspection requires vision)
        inspect_task = TaskDefinition(
            id="test-inspect", title="Test Inspect", description="...", group="test",
            required_modalities=["vision"]
        )
        
        eligibility_inspect = service._build_task_eligibility(inspect_task)
        print(f"Inspect Task Eligible: {eligibility_inspect.eligible}")
        self.assertTrue(eligibility_inspect.eligible)

    async def async_test_full_capabilities(self):
        """Verify full capabilities allow all tasks."""
        print("\nTesting Full Hardware Scenario...")
        service = RobotService(config_dir=str(self.config_dir), prefer_real_hardware=False, auto_detect=False)
        
        # Force full mock
        service.arm = MagicMock()
        service.camera = MagicMock()
        service.drivetrain = MagicMock()
        service.drivetrain.initialized = True
        
        caps = service.capabilities
        self.assertTrue(caps["has_manipulator"])
        
        pick_task = TaskDefinition(
            id="test-pick", title="Test Pick", description="...", group="test",
            required_modalities=["arm"]
        )
        # Note: Mode check might fail if not autonomous, but hardware check should pass
        # We Mock mode manager to allow autonomy or ignore mode hint (which is non-blocking usually)
        
        eligibility = service._build_task_eligibility(pick_task)
        # Filter for HARDWARE markers
        hardware_markers = [m for m in eligibility.markers if m.code.startswith("MISSING_")]
        self.assertEqual(len(hardware_markers), 0)
        print("✅ No missing hardware markers found")

    def test_capabilities(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_test_missing_arm())
        loop.run_until_complete(self.async_test_full_capabilities())
        loop.close()

if __name__ == "__main__":
    unittest.main()
