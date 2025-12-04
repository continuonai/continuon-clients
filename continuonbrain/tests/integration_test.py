#!/usr/bin/env python3
"""Integration test for Pi5 robot arm with OAK-D depth camera.
Tests the full ContinuonBrain OS stack: sensors → actuators → RLDS recording.
Prints capability awareness on boot so we stay within safe hardware boundaries.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_DEPTHAI = importlib.util.find_spec("depthai") is not None
HAS_SERVOKIT = importlib.util.find_spec("adafruit_servokit") is not None

if not HAS_NUMPY:
    pytest.skip("numpy required for integration test", allow_module_level=True)

import numpy as np

# Add repository root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from continuonbrain.sensors.oak_depth import OAKDepthCapture, CameraConfig
from continuonbrain.actuators.pca9685_arm import PCA9685ArmController, ArmConfig
from continuonbrain.recording.arm_episode_recorder import ArmEpisodeRecorder
from continuonbrain.sensors.hardware_detector import HardwareDetector, HardwareDevice


@dataclass
class CapabilityReport:
    """Summarizes detected hardware and software readiness."""

    has_numpy: bool
    has_depthai: bool
    has_servokit: bool
    detected_devices: List[HardwareDevice]

    @classmethod
    def collect(cls, auto_detect: bool = True) -> "CapabilityReport":
        detector = HardwareDetector() if auto_detect else None
        devices: List[HardwareDevice] = detector.detect_all() if detector else []
        return cls(
            has_numpy=HAS_NUMPY,
            has_depthai=HAS_DEPTHAI,
            has_servokit=HAS_SERVOKIT,
            detected_devices=devices,
        )

    def has_camera(self) -> bool:
        return any(d.device_type in {"depth_camera", "camera"} for d in self.detected_devices)

    def has_servo_controller(self) -> bool:
        return any(d.device_type == "servo_controller" for d in self.detected_devices)

    def ready_for_real_hardware(self) -> bool:
        return self.has_depthai and self.has_servokit and self.has_camera() and self.has_servo_controller()

    def log_summary(self) -> None:
        print("Capability Check (software + hardware)")
        print(f"  numpy: {'✅' if self.has_numpy else '❌'}")
        print(f"  depthai: {'✅' if self.has_depthai else '❌'}")
        print(f"  adafruit-servokit: {'✅' if self.has_servokit else '❌'}")

        if not self.detected_devices:
            print("  hardware: ❓ No devices detected (mock mode unless forced)")
        else:
            for device in self.detected_devices:
                print(f"  hardware: {device.device_type} → {device.name} ({device.interface})")

        if not self.ready_for_real_hardware():
            print("  safety: Real hardware control disabled; missing capabilities")
        else:
            print("  safety: Real hardware path available (all prerequisites met)")


def test_hardware_detection():
    """Test hardware auto-detection."""
    print("=" * 60)
    print("Hardware Detection Test")
    print("=" * 60)
    print()

    report = CapabilityReport.collect(auto_detect=True)
    report.log_summary()
    devices = report.detected_devices

    if not devices:
        print("⚠️  No devices detected - will run in mock mode")
        return False

    detector = HardwareDetector()
    detector.detected_devices = devices
    detector.print_summary()
    config = detector.generate_config()
    
    # Check for required devices
    has_camera = "depth_camera" in config.get("devices", {}) or "camera" in config.get("devices", {})
    has_servo = "servo_controller" in config.get("devices", {})
    
    print("Hardware Readiness:")
    print(f"  Camera: {'✅' if has_camera else '❌'}")
    print(f"  Servo Controller: {'✅' if has_servo else '❌'}")
    print()
    
    return has_camera and has_servo


def test_full_stack(use_real_hardware: bool = False, auto_detect: bool = True):
    """
    Test the complete ContinuonBrain OS stack.
    
    Args:
        use_real_hardware: If True, uses real OAK-D and PCA9685.
                          If False, runs in mock mode.
    """
    print("=" * 60)
    print("ContinuonBrain OS Integration Test")
    print("=" * 60)

    report = CapabilityReport.collect(auto_detect=auto_detect)
    report.log_summary()

    if use_real_hardware and not report.ready_for_real_hardware():
        print("⚠️  Real hardware requested but prerequisites missing; using mock mode for safety")
        use_real_hardware = False

    print(f"Mode: {'REAL HARDWARE' if use_real_hardware else 'MOCK/SIMULATION'}")
    print()
    
    # Initialize recorder
    recorder = ArmEpisodeRecorder(
        episodes_dir="/tmp/continuonbrain_test",
        max_steps=50,
    )
    
    print("1. Initializing hardware...")
    if not recorder.initialize_hardware(use_mock=not use_real_hardware, auto_detect=auto_detect):
        print("⚠️  Hardware initialization had warnings, continuing...")
    
    print()
    print("2. Starting test episode...")
    episode_id = recorder.start_episode(
        episode_id=f"integration_test_{int(time.time())}",
        language_instruction="Move arm through test positions and grasp",
        action_source="human_teleop_xr",
    )
    
    # Define test trajectory
    test_actions = [
        # Move to ready position
        ([0.0, 0.2, -0.2, 0.1, 0.0, -1.0], "Move to ready position (gripper open)"),
        ([0.0, 0.2, -0.2, 0.1, 0.0, -1.0], "Hold ready position"),
        
        # Reach forward
        ([0.0, 0.5, 0.3, 0.2, 0.0, -1.0], "Reach forward"),
        ([0.0, 0.5, 0.3, 0.2, 0.0, -1.0], "Hold reach position"),
        
        # Close gripper
        ([0.0, 0.5, 0.3, 0.2, 0.0, 0.8], "Close gripper (grasp)"),
        ([0.0, 0.5, 0.3, 0.2, 0.0, 0.8], "Hold grasp"),
        
        # Lift object
        ([0.0, 0.0, -0.3, 0.0, 0.0, 0.8], "Lift object"),
        ([0.0, 0.0, -0.3, 0.0, 0.0, 0.8], "Hold lifted position"),
        
        # Move to target
        ([0.5, 0.2, -0.2, 0.1, 0.3, 0.8], "Move to target location"),
        ([0.5, 0.2, -0.2, 0.1, 0.3, 0.8], "Hold at target"),
        
        # Release
        ([0.5, 0.2, -0.2, 0.1, 0.3, -1.0], "Release object"),
        ([0.5, 0.2, -0.2, 0.1, 0.3, -1.0], "Hold release position"),
        
        # Return home
        ([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], "Return to home"),
    ]
    
    print()
    print("3. Executing manipulation sequence...")
    for i, (action, description) in enumerate(test_actions):
        is_terminal = (i == len(test_actions) - 1)
        
        print(f"   Step {i+1:2d}/{len(test_actions)}: {description}")
        
        success = recorder.record_step(
            action=action,
            action_source="human_teleop_xr",
            language_instruction=description,
            is_terminal=is_terminal,
        )
        
        if not success:
            print(f"   ⚠️  Step {i+1} failed")
        
        time.sleep(0.05)  # 20Hz control rate
    
    print()
    print("4. Saving episode...")
    episode_path = recorder.end_episode(success=True)
    
    if episode_path:
        print(f"✅ Episode saved successfully!")
        print(f"   Path: {episode_path}")
        
        # Show statistics
        print()
        print("5. Episode Statistics:")
        
        # Count files
        rgb_files = list(episode_path.glob("step_*_rgb.npy"))
        depth_files = list(episode_path.glob("step_*_depth.npy"))
        
        print(f"   RGB frames: {len(rgb_files)}")
        print(f"   Depth frames: {len(depth_files)}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in episode_path.glob("*"))
        print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")
        
        # Read metadata
        import json
        with open(episode_path / "episode.json") as f:
            metadata = json.load(f)
        
        print(f"   Total steps: {metadata['metadata']['total_steps']}")
        duration_ns = (metadata['metadata']['end_timestamp_ns'] - 
                      metadata['metadata']['start_timestamp_ns'])
        duration_s = duration_ns / 1e9
        print(f"   Duration: {duration_s:.2f}s")
        print(f"   Control rate: {len(test_actions) / duration_s:.1f} Hz")
    else:
        print("❌ Failed to save episode")
    
    print()
    print("6. Shutting down...")
    recorder.shutdown()
    
    print()
    print("=" * 60)
    print("✅ Integration test complete!")
    print("=" * 60)
    print()
    
    if use_real_hardware:
        print("Next steps:")
        print("  1. View recorded episode in: " + str(episode_path))
        print("  2. Move episode to: /opt/continuonos/brain/rlds/episodes/")
        print("  3. Run trainer when 16+ episodes collected:")
        print("     python -m continuonbrain.trainer.local_lora_trainer \\")
        print("       --config continuonbrain/configs/pi5-donkey.json")
    else:
        print("To test with real hardware:")
        print("  python continuonbrain/tests/integration_test.py --real-hardware")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ContinuonBrain OS integration test")
    parser.add_argument(
        "--real-hardware",
        action="store_true",
        help="Use real OAK-D and PCA9685 hardware (default: mock mode)"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only run hardware detection (don't run full test)"
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable hardware auto-detection"
    )
    
    args = parser.parse_args()
    
    try:
        if args.detect_only:
            test_hardware_detection()
        else:
            test_full_stack(
                use_real_hardware=args.real_hardware,
                auto_detect=not args.no_auto_detect
            )
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

