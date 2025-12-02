#!/usr/bin/env python3
"""
Example: Using hardware auto-detection in your application.
Shows how to detect and configure devices automatically.
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from continuonbrain.sensors.hardware_detector import HardwareDetector


def example_basic_detection():
    """Example 1: Basic hardware detection."""
    print("Example 1: Basic Detection")
    print("=" * 60)
    
    detector = HardwareDetector()
    devices = detector.detect_all()
    
    if not devices:
        print("No hardware detected!")
        return
    
    # Print summary
    detector.print_summary()
    
    print()


def example_conditional_initialization():
    """Example 2: Conditional initialization based on detected hardware."""
    print("\nExample 2: Conditional Initialization")
    print("=" * 60)
    
    detector = HardwareDetector()
    devices = detector.detect_all()
    config = detector.generate_config()
    
    # Check what's available
    has_depth_camera = "depth_camera" in config.get("devices", {})
    has_servo = "servo_controller" in config.get("devices", {})
    has_ai_accel = "ai_accelerator" in config.get("devices", {})
    
    print("\nAvailable Hardware:")
    print(f"  Depth Camera: {has_depth_camera}")
    print(f"  Servo Controller: {has_servo}")
    print(f"  AI Accelerator: {has_ai_accel}")
    
    # Initialize based on what's available
    if has_depth_camera:
        camera_name = config["primary"]["depth_camera"]
        driver = config["primary"]["depth_camera_driver"]
        print(f"\n✅ Would initialize {camera_name} with {driver} driver")
    
    if has_servo:
        servo_name = config["primary"]["servo_controller"]
        address = config["primary"]["servo_controller_address"]
        print(f"✅ Would initialize {servo_name} at {address}")
    
    if has_ai_accel:
        accel_name = config["primary"]["ai_accelerator"]
        print(f"✅ Would use {accel_name} for inference")
    else:
        print("⚠️  No AI accelerator - would use CPU inference")
    
    print()


def example_device_filtering():
    """Example 3: Filtering devices by type and capabilities."""
    print("\nExample 3: Device Filtering")
    print("=" * 60)
    
    detector = HardwareDetector()
    devices = detector.detect_all()
    
    # Find all cameras with depth capability
    depth_cameras = [
        d for d in devices 
        if "depth" in d.capabilities
    ]
    
    print(f"\nFound {len(depth_cameras)} depth camera(s):")
    for cam in depth_cameras:
        print(f"  • {cam.name} ({cam.interface})")
        print(f"    Capabilities: {', '.join(cam.capabilities)}")
    
    # Find all I2C devices
    i2c_devices = [
        d for d in devices 
        if d.interface == "i2c"
    ]
    
    print(f"\nFound {len(i2c_devices)} I2C device(s):")
    for dev in i2c_devices:
        print(f"  • {dev.name} at {dev.address}")
    
    print()


def example_save_and_load_config():
    """Example 4: Saving and loading hardware configuration."""
    print("\nExample 4: Save/Load Configuration")
    print("=" * 60)
    
    import json
    
    # Detect and save
    detector = HardwareDetector()
    devices = detector.detect_all()
    
    config_path = "/tmp/my_robot_config.json"
    detector.save_config(config_path)
    
    # Load saved config
    with open(config_path) as f:
        loaded_config = json.load(f)
    
    print(f"\nLoaded configuration from {config_path}")
    print(f"Platform: {loaded_config['platform']}")
    print(f"Detected: {loaded_config['detected_timestamp']}")
    
    if loaded_config.get("primary"):
        print("\nPrimary devices:")
        for key, value in loaded_config["primary"].items():
            print(f"  {key}: {value}")
    
    print()


def example_graceful_degradation():
    """Example 5: Graceful degradation when hardware is missing."""
    print("\nExample 5: Graceful Degradation")
    print("=" * 60)
    
    detector = HardwareDetector()
    devices = detector.detect_all()
    config = detector.generate_config()
    
    # Try to get camera, fall back to mock
    camera = None
    camera_type = "mock"
    
    if "depth_camera" in config.get("devices", {}):
        camera_info = config["devices"]["depth_camera"][0]
        camera_type = camera_info["name"]
        print(f"\n✅ Using real camera: {camera_type}")
        # camera = initialize_camera(camera_info)
    else:
        print("\n⚠️  No depth camera detected - using mock camera")
        # camera = MockCamera()
    
    # Try to get servo controller, fall back to simulation
    servo = None
    servo_type = "simulated"
    
    if "servo_controller" in config.get("devices", {}):
        servo_info = config["devices"]["servo_controller"][0]
        servo_type = servo_info["name"]
        print(f"✅ Using real servos: {servo_type}")
        # servo = initialize_servo(servo_info)
    else:
        print("⚠️  No servo controller detected - using simulation")
        # servo = SimulatedServo()
    
    print(f"\nRobot configuration:")
    print(f"  Camera: {camera_type}")
    print(f"  Servos: {servo_type}")
    print()


def example_multi_camera_setup():
    """Example 6: Handling multiple cameras."""
    print("\nExample 6: Multi-Camera Setup")
    print("=" * 60)
    
    detector = HardwareDetector()
    devices = detector.detect_all()
    
    # Get all cameras
    cameras = [d for d in devices if "camera" in d.device_type]
    
    if len(cameras) == 0:
        print("\nNo cameras detected!")
    elif len(cameras) == 1:
        print(f"\nSingle camera setup: {cameras[0].name}")
    else:
        print(f"\nMulti-camera setup ({len(cameras)} cameras):")
        for i, cam in enumerate(cameras):
            print(f"  Camera {i+1}: {cam.name}")
            print(f"    Type: {cam.device_type}")
            print(f"    Capabilities: {', '.join(cam.capabilities)}")
            
            # Suggest usage
            if "depth" in cam.capabilities and "stereo" in cam.capabilities:
                print("    → Suggested use: Primary depth perception")
            elif "rgb" in cam.capabilities:
                print("    → Suggested use: Secondary RGB view")
    
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_detection()
    example_conditional_initialization()
    example_device_filtering()
    example_save_and_load_config()
    example_graceful_degradation()
    example_multi_camera_setup()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
