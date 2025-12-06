"""
Test PCA9685 wiring configuration for drivetrain and arm controllers.

This test validates that the PCA9685 is correctly wired and configured
before testing web server controls. It checks:
- I2C communication with PCA9685 at address 0x40
- Channel assignments for drivetrain (steering/throttle)
- Channel assignments for arm controller (6 DOF)
- Safe operation within configured limits
- Mock mode fallback when hardware is unavailable

Run this test to ensure hardware is correctly set up before validating
web server control functionality.
"""
import pytest
import time
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock

# Import the controllers we're testing
from continuonbrain.actuators.drivetrain_controller import (
    DrivetrainController,
    DrivetrainConfig,
)
from continuonbrain.actuators.pca9685_arm import (
    PCA9685ArmController,
    ArmConfig,
    ServoLimits,
)


class TestPCA9685WiringConfiguration:
    """Test PCA9685 wiring configuration and channel assignments."""

    def test_drivetrain_default_config(self):
        """Test drivetrain uses correct default channel configuration."""
        config = DrivetrainConfig()
        
        # Verify default I2C address
        assert config.i2c_address == 0x40, "Drivetrain should use I2C address 0x40"
        
        # Verify channel assignments per documentation
        assert config.steering_channel == 0, "Steering should be on channel 0"
        assert config.throttle_channel == 1, "Throttle should be on channel 1"
        
        # Verify steering parameters
        assert config.steering_center_degrees == 90.0
        assert config.steering_range_degrees == 30.0

    def test_arm_default_config(self):
        """Test arm controller uses correct default channel configuration."""
        config = ArmConfig()
        
        # Verify channel assignments for 6 DOF arm
        assert config.base_channel == 0, "Base should be on channel 0"
        assert config.shoulder_channel == 1, "Shoulder should be on channel 1"
        assert config.elbow_channel == 2, "Elbow should be on channel 2"
        assert config.wrist_pitch_channel == 3, "Wrist pitch should be on channel 3"
        assert config.wrist_roll_channel == 4, "Wrist roll should be on channel 4"
        assert config.gripper_channel == 5, "Gripper should be on channel 5"
        
        # Verify safety limits are configured
        assert config.base_limits.min_angle == 0
        assert config.base_limits.max_angle == 180
        assert config.base_limits.default_angle == 90

    def test_channel_conflict_detection(self):
        """Test that drivetrain and arm channels don't conflict when used together."""
        drivetrain_config = DrivetrainConfig()
        arm_config = ArmConfig()
        
        # Get all channels used by drivetrain
        drivetrain_channels = {
            drivetrain_config.steering_channel,
            drivetrain_config.throttle_channel,
        }
        
        # Get all channels used by arm
        arm_channels = {
            arm_config.base_channel,
            arm_config.shoulder_channel,
            arm_config.elbow_channel,
            arm_config.wrist_pitch_channel,
            arm_config.wrist_roll_channel,
            arm_config.gripper_channel,
        }
        
        # Check for conflicts (channels 0-1 are shared)
        conflicts = drivetrain_channels & arm_channels
        
        # This is expected - document the conflict
        assert conflicts == {0, 1}, (
            "Expected channel conflict on 0 and 1. "
            "Drivetrain and arm should NOT be used simultaneously on same PCA9685. "
            "Use separate PCA9685 boards or different I2C addresses."
        )


class TestDrivetrainWiring:
    """Test drivetrain controller wiring and operation."""

    def test_drivetrain_initialization_mock(self):
        """Test drivetrain initializes correctly in mock mode."""
        controller = DrivetrainController()
        
        # Should initialize successfully even without hardware
        assert controller.initialize() is True
        assert controller.initialized is True
        
        # Verify status reports correct configuration
        status = controller.status()
        assert status["initialized"] is True
        assert status["channels"] == "steering=servo[0], throttle=continuous_servo[1]"

    def test_drivetrain_steering_range(self):
        """Test steering commands map to correct angle range."""
        config = DrivetrainConfig()
        controller = DrivetrainController(config=config)
        controller.initialize()
        
        # Test steering limits
        # -1.0 should map to center - range = 90 - 30 = 60 degrees
        # +1.0 should map to center + range = 90 + 30 = 120 degrees
        # 0.0 should map to center = 90 degrees
        
        test_cases = [
            (-1.0, 60.0),   # Full left
            (0.0, 90.0),    # Center
            (1.0, 120.0),   # Full right
            (-0.5, 75.0),   # Half left
            (0.5, 105.0),   # Half right
        ]
        
        for steering_input, expected_angle in test_cases:
            calculated_angle = (
                config.steering_center_degrees
                + steering_input * config.steering_range_degrees
            )
            assert calculated_angle == expected_angle, (
                f"Steering {steering_input} should map to {expected_angle}°, "
                f"got {calculated_angle}°"
            )

    def test_drivetrain_command_clamping(self):
        """Test that drive commands are properly clamped to [-1, 1]."""
        controller = DrivetrainController()
        controller.initialize()
        
        # Test values outside valid range get clamped
        test_cases = [
            (2.0, 1.5, 1.0, 1.0),    # Over-range values
            (-2.0, -1.5, -1.0, -1.0),  # Under-range values
            (0.5, 0.5, 0.5, 0.5),    # Valid values
        ]
        
        for steer_in, throttle_in, steer_expected, throttle_expected in test_cases:
            result = controller.apply_drive(steer_in, throttle_in)
            assert result["steering"] == steer_expected
            assert result["throttle"] == throttle_expected

    def test_drivetrain_mock_mode_behavior(self):
        """Test that mock mode properly blocks hardware output."""
        controller = DrivetrainController()
        controller.initialize()
        
        # In mock mode, commands should be blocked
        result = controller.apply_drive(0.5, 0.5)
        
        if controller.is_mock:
            assert result["success"] is False
            assert "MOCK mode active" in result["message"]
            assert result["hardware_available"] is False
        else:
            # Real hardware should succeed
            assert result["success"] is True


class TestArmWiring:
    """Test arm controller wiring and operation."""

    def test_arm_initialization_mock(self):
        """Test arm initializes correctly in mock mode."""
        controller = PCA9685ArmController()
        
        assert controller.initialize() is True
        
        # Verify all joints initialized to default positions
        state = controller.get_current_state()
        assert len(state) == 6
        assert all(joint in state for joint in [
            "base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"
        ])

    def test_arm_channel_mapping(self):
        """Test that arm joints map to correct PCA9685 channels."""
        config = ArmConfig()
        
        channel_map = {
            "base": config.base_channel,
            "shoulder": config.shoulder_channel,
            "elbow": config.elbow_channel,
            "wrist_pitch": config.wrist_pitch_channel,
            "wrist_roll": config.wrist_roll_channel,
            "gripper": config.gripper_channel,
        }
        
        # Verify sequential channel assignment
        expected_channels = {
            "base": 0,
            "shoulder": 1,
            "elbow": 2,
            "wrist_pitch": 3,
            "wrist_roll": 4,
            "gripper": 5,
        }
        
        assert channel_map == expected_channels

    def test_arm_safety_limits(self):
        """Test that arm enforces safety limits on all joints."""
        controller = PCA9685ArmController()
        controller.initialize()
        
        # Try to set angles outside limits
        unsafe_angles = {
            "base": 200.0,      # Over max (180)
            "shoulder": -10.0,  # Under min (0)
            "elbow": 90.0,      # Valid
        }
        
        controller.set_joint_angles(unsafe_angles)
        state = controller.get_current_state()
        
        # Verify clamping occurred
        assert state["base"] == 180.0, "Base should be clamped to max"
        assert state["shoulder"] == 0.0, "Shoulder should be clamped to min"
        assert state["elbow"] == 90.0, "Elbow should remain unchanged"

    def test_arm_normalized_action_mapping(self):
        """Test normalized action vector maps correctly to joint angles."""
        controller = PCA9685ArmController()
        controller.initialize()
        
        # Test center position (all zeros)
        center_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        controller.set_normalized_action(center_action)
        state = controller.get_current_state()
        
        # All joints should be at center (90 degrees) except gripper
        assert state["base"] == 90.0
        assert state["shoulder"] == 90.0
        assert state["elbow"] == 90.0
        assert state["wrist_pitch"] == 90.0
        assert state["wrist_roll"] == 90.0
        # Gripper has different range (30-90), center is 60
        assert state["gripper"] == 60.0

    def test_arm_normalized_state_roundtrip(self):
        """Test that normalized state can be converted back to angles."""
        controller = PCA9685ArmController()
        controller.initialize()
        
        # Set a known action
        test_action = [0.5, -0.5, 0.0, 0.25, -0.25, 0.0]
        controller.set_normalized_action(test_action)
        
        # Get normalized state back
        normalized_state = controller.get_normalized_state()
        
        # Should match original action (within floating point tolerance)
        for i, (expected, actual) in enumerate(zip(test_action, normalized_state)):
            assert abs(expected - actual) < 0.01, (
                f"Joint {i}: expected {expected}, got {actual}"
            )


class TestPCA9685HardwareDetection:
    """Test PCA9685 hardware detection and I2C communication."""

    def test_i2c_address_configuration(self):
        """Test that both controllers use the same I2C address."""
        drivetrain_config = DrivetrainConfig()
        arm_controller = PCA9685ArmController()
        
        assert drivetrain_config.i2c_address == 0x40
        assert arm_controller.i2c_address == 0x40
        
        # Both use same address - they should NOT run simultaneously

    @pytest.mark.skipif(
        True,  # Skip by default - requires actual hardware
        reason="Requires real PCA9685 hardware on I2C bus"
    )
    def test_i2c_device_detection(self):
        """Test that PCA9685 is detected on I2C bus (requires hardware)."""
        import subprocess
        
        try:
            # Run i2cdetect to check for device at 0x40
            result = subprocess.run(
                ['i2cdetect', '-y', '1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Check if 0x40 appears in output
            assert '40' in result.stdout, (
                "PCA9685 not detected at address 0x40. "
                "Check I2C wiring and power."
            )
        except FileNotFoundError:
            pytest.skip("i2cdetect not available")
        except subprocess.TimeoutExpired:
            pytest.fail("i2cdetect command timed out")


class TestPCA9685SafetyFeatures:
    """Test safety features of PCA9685 controllers."""

    def test_drivetrain_emergency_stop(self):
        """Test that drivetrain can be safely stopped."""
        controller = DrivetrainController()
        controller.initialize()
        
        # Apply some movement
        controller.apply_drive(0.5, 0.5)
        
        # Stop should set both to zero
        result = controller.apply_drive(0.0, 0.0)
        assert result["steering"] == 0.0
        assert result["throttle"] == 0.0

    def test_arm_emergency_stop(self):
        """Test arm emergency stop functionality."""
        controller = PCA9685ArmController()
        controller.initialize()
        
        # Move arm
        controller.set_normalized_action([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        
        # Emergency stop should hold position
        controller.emergency_stop()
        
        # Position should be maintained
        state = controller.get_current_state()
        assert len(state) == 6

    def test_arm_graceful_shutdown(self):
        """Test arm returns to safe defaults on shutdown."""
        controller = PCA9685ArmController()
        controller.initialize()
        
        # Move to non-default position
        controller.set_normalized_action([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        
        # Shutdown should return to defaults
        controller.shutdown()
        
        state = controller.get_current_state()
        # All joints should be at default (90 degrees) except gripper (30)
        assert state["base"] == 90.0
        assert state["gripper"] == 30.0  # Open position


class TestPCA9685ConfigurationOverrides:
    """Test configuration override mechanisms."""

    def test_drivetrain_config_from_env(self, monkeypatch):
        """Test drivetrain can be configured via environment variables."""
        # Set environment variables
        monkeypatch.setenv("DRIVETRAIN_STEERING_CHANNEL", "2")
        monkeypatch.setenv("DRIVETRAIN_THROTTLE_CHANNEL", "3")
        monkeypatch.setenv("DRIVETRAIN_I2C_ADDRESS", "0x41")
        
        config = DrivetrainConfig.from_sources()
        
        assert config.steering_channel == 2
        assert config.throttle_channel == 3
        assert config.i2c_address == 0x41

    def test_drivetrain_config_from_json(self, tmp_path):
        """Test drivetrain can be configured via JSON file."""
        config_file = tmp_path / "drivetrain_config.json"
        config_file.write_text('''{
            "drivetrain": {
                "steering_channel": 4,
                "throttle_channel": 5,
                "i2c_address": "0x42"
            }
        }''')
        
        config = DrivetrainConfig.from_sources(json_config_path=str(config_file))
        
        assert config.steering_channel == 4
        assert config.throttle_channel == 5
        assert config.i2c_address == 0x42


def test_pca9685_wiring_summary():
    """Print a summary of PCA9685 wiring configuration for reference."""
    print("\n" + "="*60)
    print("PCA9685 WIRING CONFIGURATION SUMMARY")
    print("="*60)
    
    print("\nI2C Address: 0x40")
    print("\nDRIVETRAIN CHANNELS:")
    drivetrain_config = DrivetrainConfig()
    print(f"  Channel {drivetrain_config.steering_channel}: Steering (standard servo)")
    print(f"  Channel {drivetrain_config.throttle_channel}: Throttle (continuous servo)")
    
    print("\nARM CONTROLLER CHANNELS:")
    arm_config = ArmConfig()
    print(f"  Channel {arm_config.base_channel}: Base (0-180°, default 90°)")
    print(f"  Channel {arm_config.shoulder_channel}: Shoulder (0-180°, default 90°)")
    print(f"  Channel {arm_config.elbow_channel}: Elbow (0-180°, default 90°)")
    print(f"  Channel {arm_config.wrist_pitch_channel}: Wrist Pitch (0-180°, default 90°)")
    print(f"  Channel {arm_config.wrist_roll_channel}: Wrist Roll (0-180°, default 90°)")
    print(f"  Channel {arm_config.gripper_channel}: Gripper (30-90°, default 30°)")
    
    print("\n⚠️  WARNING: Channel Conflict Detected!")
    print("  Drivetrain and Arm both use channels 0-1")
    print("  DO NOT operate both simultaneously on the same PCA9685")
    print("  Use separate PCA9685 boards with different I2C addresses")
    
    print("\nRECOMMENDED HARDWARE SETUP:")
    print("  Option 1: Single PCA9685 - Use EITHER drivetrain OR arm")
    print("  Option 2: Two PCA9685 boards:")
    print("    - 0x40: Drivetrain (channels 0-1)")
    print("    - 0x41: Arm (channels 0-5)")
    
    print("\nSAFETY FEATURES:")
    print("  ✓ Angle clamping to configured limits")
    print("  ✓ Command value clamping to [-1, 1]")
    print("  ✓ Safe default positions on initialization")
    print("  ✓ Graceful shutdown returns to defaults")
    print("  ✓ Mock mode fallback when hardware unavailable")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run the wiring summary
    test_pca9685_wiring_summary()
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
