#!/usr/bin/env python3
"""
Standalone PCA9685 wiring validation script.

This script validates the PCA9685 wiring configuration for both drivetrain
and arm controllers without requiring pytest. Run this before testing web
server controls to ensure hardware is correctly configured.

Usage:
    python validate_pca9685_wiring.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from continuonbrain.actuators.drivetrain_controller import (
    DrivetrainController,
    DrivetrainConfig,
)
from continuonbrain.actuators.pca9685_arm import (
    PCA9685ArmController,
    ArmConfig,
)


class TestRunner:
    """Simple test runner that doesn't require pytest."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name, func):
        """Run a test function."""
        try:
            func()
            self.passed += 1
            print(f"✓ {name}")
            return True
        except AssertionError as e:
            self.failed += 1
            print(f"✗ {name}")
            print(f"  Error: {e}")
            return False
        except Exception as e:
            self.failed += 1
            print(f"✗ {name}")
            print(f"  Unexpected error: {e}")
            return False
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"⚠️  {self.failed} tests failed")
        else:
            print("✅ All tests passed!")
        print(f"{'='*60}\n")
        return self.failed == 0


def print_header(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def test_drivetrain_config():
    """Test drivetrain configuration."""
    config = DrivetrainConfig()
    
    assert config.i2c_address == 0x40, "Drivetrain should use I2C address 0x40"
    assert config.steering_channel == 0, "Steering should be on channel 0"
    assert config.throttle_channel == 1, "Throttle should be on channel 1"
    assert config.steering_center_degrees == 90.0
    assert config.steering_range_degrees == 30.0


def test_arm_config():
    """Test arm controller configuration."""
    config = ArmConfig()
    
    assert config.base_channel == 0, "Base should be on channel 0"
    assert config.shoulder_channel == 1, "Shoulder should be on channel 1"
    assert config.elbow_channel == 2, "Elbow should be on channel 2"
    assert config.wrist_pitch_channel == 3, "Wrist pitch should be on channel 3"
    assert config.wrist_roll_channel == 4, "Wrist roll should be on channel 4"
    assert config.gripper_channel == 5, "Gripper should be on channel 5"


def test_drivetrain_initialization():
    """Test drivetrain initializes correctly."""
    controller = DrivetrainController()
    assert controller.initialize() is True, "Drivetrain should initialize"
    assert controller.initialized is True, "Drivetrain should be marked initialized"


def test_arm_initialization():
    """Test arm initializes correctly."""
    controller = PCA9685ArmController()
    assert controller.initialize() is True, "Arm should initialize"
    
    state = controller.get_current_state()
    assert len(state) == 6, "Arm should have 6 joints"
    assert all(joint in state for joint in [
        "base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"
    ]), "All joints should be present"


def test_drivetrain_command_clamping():
    """Test drivetrain clamps commands to valid range."""
    controller = DrivetrainController()
    controller.initialize()
    
    # Test over-range values get clamped
    result = controller.apply_drive(2.0, 1.5)
    assert result["steering"] == 1.0, "Steering should be clamped to 1.0"
    assert result["throttle"] == 1.0, "Throttle should be clamped to 1.0"
    
    # Test under-range values get clamped
    result = controller.apply_drive(-2.0, -1.5)
    assert result["steering"] == -1.0, "Steering should be clamped to -1.0"
    assert result["throttle"] == -1.0, "Throttle should be clamped to -1.0"


def test_arm_safety_limits():
    """Test arm enforces safety limits."""
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
    
    assert state["base"] == 180.0, "Base should be clamped to max (180)"
    assert state["shoulder"] == 0.0, "Shoulder should be clamped to min (0)"
    assert state["elbow"] == 90.0, "Elbow should remain unchanged"


def test_arm_normalized_actions():
    """Test arm normalized action mapping."""
    controller = PCA9685ArmController()
    controller.initialize()
    
    # Test center position (all zeros)
    center_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    controller.set_normalized_action(center_action)
    state = controller.get_current_state()
    
    assert state["base"] == 90.0, "Base should be at center (90°)"
    assert state["shoulder"] == 90.0, "Shoulder should be at center (90°)"
    assert state["elbow"] == 90.0, "Elbow should be at center (90°)"
    assert state["wrist_pitch"] == 90.0, "Wrist pitch should be at center (90°)"
    assert state["wrist_roll"] == 90.0, "Wrist roll should be at center (90°)"
    assert state["gripper"] == 60.0, "Gripper should be at center (60° for 30-90 range)"


def test_steering_angle_mapping():
    """Test steering angle calculations."""
    config = DrivetrainConfig()
    
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


def print_wiring_summary():
    """Print comprehensive wiring summary."""
    print_header("PCA9685 WIRING CONFIGURATION SUMMARY")
    
    print("I2C Address: 0x40")
    print("\nDRIVETRAIN CHANNELS:")
    drivetrain_config = DrivetrainConfig()
    print(f"  Channel {drivetrain_config.steering_channel}: Steering (standard servo)")
    print(f"    - Range: {drivetrain_config.steering_center_degrees - drivetrain_config.steering_range_degrees}° to "
          f"{drivetrain_config.steering_center_degrees + drivetrain_config.steering_range_degrees}°")
    print(f"    - Center: {drivetrain_config.steering_center_degrees}°")
    print(f"  Channel {drivetrain_config.throttle_channel}: Throttle (continuous servo)")
    print(f"    - Range: -1.0 to +1.0 (normalized)")
    
    print("\nARM CONTROLLER CHANNELS:")
    arm_config = ArmConfig()
    joints = [
        ("Base", arm_config.base_channel, arm_config.base_limits),
        ("Shoulder", arm_config.shoulder_channel, arm_config.shoulder_limits),
        ("Elbow", arm_config.elbow_channel, arm_config.elbow_limits),
        ("Wrist Pitch", arm_config.wrist_pitch_channel, arm_config.wrist_pitch_limits),
        ("Wrist Roll", arm_config.wrist_roll_channel, arm_config.wrist_roll_limits),
        ("Gripper", arm_config.gripper_channel, arm_config.gripper_limits),
    ]
    
    for name, channel, limits in joints:
        print(f"  Channel {channel}: {name}")
        print(f"    - Range: {limits.min_angle}° to {limits.max_angle}°")
        print(f"    - Default: {limits.default_angle}°")
    
    print("\n⚠️  WARNING: Channel Conflict Detected!")
    print("  Drivetrain uses channels 0-1")
    print("  Arm uses channels 0-5")
    print("  DO NOT operate both simultaneously on the same PCA9685!")
    
    print("\nRECOMMENDED HARDWARE SETUP:")
    print("  Option 1: Single PCA9685 - Use EITHER drivetrain OR arm")
    print("  Option 2: Two PCA9685 boards:")
    print("    - 0x40: Drivetrain (channels 0-1)")
    print("    - 0x41: Arm (channels 0-5)")
    print("    - Modify arm controller: PCA9685ArmController(i2c_address=0x41)")
    
    print("\nSAFETY FEATURES:")
    print("  ✓ Angle clamping to configured limits")
    print("  ✓ Command value clamping to [-1, 1]")
    print("  ✓ Safe default positions on initialization")
    print("  ✓ Graceful shutdown returns to defaults")
    print("  ✓ Mock mode fallback when hardware unavailable")
    
    print("\nCONFIGURATION OPTIONS:")
    print("  Environment variables:")
    print("    - DRIVETRAIN_STEERING_CHANNEL")
    print("    - DRIVETRAIN_THROTTLE_CHANNEL")
    print("    - DRIVETRAIN_I2C_ADDRESS")
    print("  JSON config file (set DRIVETRAIN_CONFIG_PATH)")


def test_hardware_modes():
    """Test and report hardware mode status."""
    print_header("HARDWARE MODE STATUS")
    
    drivetrain = DrivetrainController()
    drivetrain.initialize()
    
    arm = PCA9685ArmController()
    arm.initialize()
    
    print(f"Drivetrain Mode: {drivetrain.mode}")
    print(f"  - Mock mode: {drivetrain.is_mock}")
    print(f"  - Initialized: {drivetrain.initialized}")
    
    print(f"\nArm Mode: {'mock' if arm.is_mock else 'real'}")
    print(f"  - Mock mode: {arm.is_mock}")
    
    if drivetrain.is_mock or arm.is_mock:
        print("\n⚠️  Running in MOCK mode - hardware drivers not available")
        print("   This is normal for development/testing without physical hardware")
        print("   Install adafruit-servokit for real hardware support:")
        print("   pip install adafruit-circuitpython-pca9685 adafruit-servokit")
    else:
        print("\n✅ Real hardware mode - PCA9685 drivers available")


def main():
    """Run all validation tests."""
    print_header("PCA9685 WIRING VALIDATION")
    print("This script validates PCA9685 configuration before web server testing\n")
    
    # Print wiring summary first
    print_wiring_summary()
    
    # Check hardware mode
    test_hardware_modes()
    
    # Run tests
    print_header("RUNNING VALIDATION TESTS")
    
    runner = TestRunner()
    
    # Configuration tests
    runner.test("Drivetrain default configuration", test_drivetrain_config)
    runner.test("Arm default configuration", test_arm_config)
    
    # Initialization tests
    runner.test("Drivetrain initialization", test_drivetrain_initialization)
    runner.test("Arm initialization", test_arm_initialization)
    
    # Functional tests
    runner.test("Drivetrain command clamping", test_drivetrain_command_clamping)
    runner.test("Arm safety limits", test_arm_safety_limits)
    runner.test("Arm normalized actions", test_arm_normalized_actions)
    runner.test("Steering angle mapping", test_steering_angle_mapping)
    
    # Print summary
    success = runner.summary()
    
    if success:
        print("✅ PCA9685 wiring configuration is CORRECT")
        print("   You can proceed to test web server controls")
        return 0
    else:
        print("❌ PCA9685 wiring validation FAILED")
        print("   Fix configuration issues before testing web server")
        return 1


if __name__ == "__main__":
    sys.exit(main())
