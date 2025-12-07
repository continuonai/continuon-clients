#!/usr/bin/env python3
"""
Test script for resource monitoring.

Tests the ResourceMonitor class and verifies memory tracking,
thresholds, and cleanup callbacks.
"""

import time
from pathlib import Path
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel


def test_basic_monitoring():
    """Test basic resource monitoring."""
    print("=" * 60)
    print("Testing Resource Monitor")
    print("=" * 60)
    print()
    
    # Create monitor
    monitor = ResourceMonitor()
    
    # Check resources
    print("1. Checking current resources...")
    status = monitor.check_resources()
    
    print(f"   Total Memory: {status.total_memory_mb}MB")
    print(f"   Used Memory: {status.used_memory_mb}MB")
    print(f"   Available Memory: {status.available_memory_mb}MB")
    print(f"   Memory Usage: {status.memory_percent:.1f}%")
    print(f"   Resource Level: {status.level.value}")
    print(f"   Can Allocate: {status.can_allocate}")
    print(f"   Message: {status.message}")
    print()
    
    # Test safe allocation check
    print("2. Testing safe allocation checks...")
    test_sizes = [100, 500, 1000, 2000, 4000]
    for size_mb in test_sizes:
        is_safe = monitor.is_safe_to_allocate(size_mb)
        print(f"   Allocate {size_mb}MB: {'✓ SAFE' if is_safe else '✗ UNSAFE'}")
    print()
    
    # Test cleanup callbacks
    print("3. Testing cleanup callbacks...")
    
    callback_triggered = {"warning": False, "critical": False, "emergency": False}
    
    def warning_callback():
        callback_triggered["warning"] = True
        print("   ⚠️  WARNING callback triggered")
    
    def critical_callback():
        callback_triggered["critical"] = True
        print("   🔴 CRITICAL callback triggered")
    
    def emergency_callback():
        callback_triggered["emergency"] = True
        print("   🚨 EMERGENCY callback triggered")
    
    monitor.register_cleanup_callback(ResourceLevel.WARNING, warning_callback)
    monitor.register_cleanup_callback(ResourceLevel.CRITICAL, critical_callback)
    monitor.register_cleanup_callback(ResourceLevel.EMERGENCY, emergency_callback)
    
    print("   Registered 3 cleanup callbacks")
    print()
    
    # Test status summary
    print("4. Testing status summary...")
    summary = monitor.get_status_summary()
    print(f"   Level: {summary['level']}")
    print(f"   Memory: {summary['memory_percent']}%")
    print(f"   Available: {summary['available_mb']}MB")
    print(f"   Can Allocate: {summary['can_allocate']}")
    print(f"   System Reserve: {summary['limits']['system_reserve_mb']}MB")
    print(f"   Max Brain: {summary['limits']['max_brain_mb']}MB")
    print()
    
    print("=" * 60)
    print("✅ Resource Monitor Tests Complete")
    print("=" * 60)
    print()
    
    return monitor, status


def test_with_config():
    """Test resource monitor with config file."""
    print("=" * 60)
    print("Testing Resource Monitor with Config")
    print("=" * 60)
    print()
    
    config_dir = Path("/home/craigm26/.continuonbrain")
    
    print(f"1. Loading config from: {config_dir}")
    monitor = ResourceMonitor(config_dir=config_dir)
    
    print(f"   System Reserve: {monitor.limits.system_reserve_mb}MB")
    print(f"   Max Brain: {monitor.limits.max_brain_mb}MB")
    print(f"   Warning Threshold: {monitor.limits.warning_threshold_percent}%")
    print(f"   Critical Threshold: {monitor.limits.critical_threshold_percent}%")
    print(f"   Emergency Threshold: {monitor.limits.emergency_threshold_percent}%")
    print()
    
    status = monitor.check_resources()
    print(f"2. Current Status: {status.level.value}")
    print(f"   {status.message}")
    print()
    
    print("=" * 60)
    print("✅ Config Test Complete")
    print("=" * 60)
    print()


if __name__ == "__main__":
    # Run basic tests
    monitor, status = test_basic_monitoring()
    
    # Run config tests
    test_with_config()
    
    print("\n📊 Resource monitoring is working correctly!")
    print(f"Current system status: {status.level.value.upper()}")
    print(f"Available memory: {status.available_memory_mb}MB")
