"""
Verification script for Startup Readiness & Auto-Agent Checks.
"""
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch

# Ensure repo root is on path
from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.agent_identity import AgentIdentity, MemoryStats, BodyStats, DesignStats
from continuonbrain.robot_api_server import RobotService

class TestStartupReadiness(unittest.TestCase):
    
    def setUp(self):
        self.config_dir = REPO_ROOT / "continuonbrain" / "tests" / "fixtures" / "brain_config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        (self.config_dir / "rlds" / "episodes").mkdir(parents=True, exist_ok=True)
        (self.config_dir / "model").mkdir(parents=True, exist_ok=True)

    def test_agent_identity_report(self):
        """Verify AgentIdentity can inspect the system and generate a report."""
        print("\nTesting Agent Identity Report...")
        identity = AgentIdentity(config_dir=str(self.config_dir))
        
        # Test introspection methods
        memories = identity.check_memories()
        self.assertIsInstance(memories, MemoryStats)
        print(f"✅ Memories checked: {memories.local_episodes} local episodes")
        
        body = identity.check_body()
        self.assertIsInstance(body, BodyStats)
        print(f"✅ Body checked: {body.actuators}, {body.sensors}")
        
        design = identity.check_design()
        self.assertIsInstance(design, DesignStats)
        print(f"✅ Design checked: {design.name}, {design.mission}")
        
        # Capture stdout to verify narrative
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        identity.self_report()
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("AGENT SELF-ACTIVATION REPORT", output)
        self.assertIn("Status: Awaiting owner instructions", output)
        print("✅ Narrative report generated successfully")

    async def async_test_ui_buttons(self):
        """Simulate UI button presses against RobotService."""
        print("\nTesting UI Backend (RobotService)...")
        
        # Mock hardware to avoid real initialization constraints
        with patch("continuonbrain.robot_api_server.PCA9685ArmController"), \
             patch("continuonbrain.robot_api_server.OAKDepthCapture"), \
             patch("continuonbrain.robot_api_server.DrivetrainController") as MockDrivetrain:
            
            # Setup Drivetrain mock
            mock_dt = MockDrivetrain.return_value
            mock_dt.initialize.return_value = True
            mock_dt.mode = "mock"
            mock_dt.apply_drive.return_value = {"success": True, "mode": "mock"}
            
            service = RobotService(
                config_dir=str(self.config_dir),
                prefer_real_hardware=False, # Use mock mode
                auto_detect=False
            )
            await service.initialize()
            
            # 1. Test Drive (Steering/Accel)
            # Must switch to manual_control first
            print("1. Set Manual Mode...")
            res = await service.SetRobotMode("manual_control")
            self.assertTrue(res["success"], f"Failed to set mode: {res.get('message')}")
            
            print("2. Testing Drivetrain Control...")
            result = await service.Drive(steering=0.5, throttle=0.8)
            self.assertTrue(result["success"], f"Drive failed: {result.get('message')}")
            mock_dt.apply_drive.assert_called_with(0.5, 0.8)
            print("✅ Drive command processed")
            
            # 3. Test Autonomy Mode Toggle
            print("3. Testing Mode Switching...")
            # Should fail if system instructions not loaded? Service loads a default if missing.
            # We need to register instructions first? Service does it.
            
            # Manual -> Autonomous
            result = await service.SetRobotMode("autonomous")
            self.assertTrue(result["success"])
            self.assertEqual(result["mode"], "autonomous")
            print("✅ Switched to Autonomous Mode")
            
            # Autonomous -> Idle
            result = await service.SetRobotMode("idle")
            self.assertTrue(result["success"])
            self.assertEqual(result["mode"], "idle")
            print("✅ Switched to Idle Mode")

    def test_ui_backend(self):
        """Wrapper for async UI test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_test_ui_buttons())
        loop.close()

if __name__ == "__main__":
    unittest.main()
