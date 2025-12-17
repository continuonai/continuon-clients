#!/usr/bin/env python3
"""
Example: Complete robot wake-up sequence with health checks.
Demonstrates the full startup → run → sleep → wake → health check cycle.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from continuonbrain.startup_manager import StartupManager
from continuonbrain.system_health import SystemHealthChecker, HealthStatus


def simulate_robot_lifecycle():
    """Simulate complete robot lifecycle with sleep/wake."""
    
    print("=" * 70)
    print("🤖 Robot Lifecycle Simulation")
    print("=" * 70)
    print()
    
    # Use temp directory for testing
    config_dir = "/tmp/robot_lifecycle_test"
    
    # === 1. Initial Cold Boot ===
    print("📍 Phase 1: Cold Boot")
    print("-" * 70)
    
    manager = StartupManager(config_dir=config_dir)
    success = manager.startup()
    
    if not success:
        print("❌ Startup failed!")
        return 1
    
    print("✅ Robot is awake and operational")
    print()
    time.sleep(2)
    
    # === 2. First Task ===
    print("📍 Phase 2: Performing Task")
    print("-" * 70)
    print("🏃 Picking up object...")
    time.sleep(1)
    print("✅ Task complete")
    print()
    time.sleep(1)
    
    # === 3. Go to Sleep ===
    print("📍 Phase 3: Going to Sleep")
    print("-" * 70)
    manager.prepare_sleep()
    print("💤 Robot is sleeping...")
    print()
    time.sleep(2)
    
    # === 4. Wake from Sleep (with automatic health check) ===
    print("📍 Phase 4: Wake from Sleep")
    print("-" * 70)
    print("⏰ Waking up...")
    print()
    
    manager = StartupManager(config_dir=config_dir)
    success = manager.startup()  # Automatic health check!
    
    if not success:
        print("❌ Wake failed - critical issues detected")
        return 1
    
    print("✅ Robot is awake and healthy")
    print()
    time.sleep(2)
    
    # === 5. Second Task ===
    print("📍 Phase 5: Performing Another Task")
    print("-" * 70)
    print("🏃 Moving to location...")
    time.sleep(1)
    print("✅ Task complete")
    print()
    time.sleep(1)
    
    # === 6. Clean Shutdown ===
    print("📍 Phase 6: Clean Shutdown")
    print("-" * 70)
    manager.prepare_shutdown()
    print("🛑 Robot powered down")
    print()
    
    print("=" * 70)
    print("✅ Lifecycle simulation complete!")
    print("=" * 70)
    print()
    
    return 0


def demonstrate_health_monitoring():
    """Demonstrate continuous health monitoring."""
    
    print("=" * 70)
    print("🏥 Continuous Health Monitoring Demo")
    print("=" * 70)
    print()
    
    checker = SystemHealthChecker(config_dir="/tmp/health_demo")
    
    print("Running periodic health checks...")
    print()
    
    for i in range(3):
        print(f"Check {i+1}/3 at {time.strftime('%H:%M:%S')}")
        print("-" * 70)
        
        overall_status, results = checker.run_all_checks(quick_mode=True)
        
        # Count issues
        warnings = sum(1 for r in results if r.status == HealthStatus.WARNING)
        critical = sum(1 for r in results if r.status == HealthStatus.CRITICAL)
        
        print(f"\n📊 Status: {overall_status.value.upper()}")
        if warnings > 0:
            print(f"   ⚠️  {warnings} warning(s)")
        if critical > 0:
            print(f"   ❌ {critical} critical issue(s)")
        
        if overall_status == HealthStatus.CRITICAL:
            print("\n🚨 CRITICAL FAILURE - Would stop robot")
            break
        
        print()
        
        if i < 2:
            print("Waiting 5 seconds...")
            time.sleep(5)
            print()
    
    print("=" * 70)
    print("✅ Health monitoring complete!")
    print("=" * 70)
    print()


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation with warnings."""
    
    print("=" * 70)
    print("🎯 Graceful Degradation Demo")
    print("=" * 70)
    print()
    
    checker = SystemHealthChecker(config_dir="/tmp/degradation_test")
    overall_status, results = checker.run_all_checks(quick_mode=True)
    
    # Analyze what's available
    print("📋 System Capabilities:")
    print("-" * 70)
    
    # Check hardware availability
    has_camera = any(
        r.component == "Camera" and r.status == HealthStatus.HEALTHY
        for r in results
    )
    
    has_servo = any(
        r.component == "Servo Controller" and r.status == HealthStatus.HEALTHY
        for r in results
    )
    
    has_ai_accel = any(
        r.component == "AI Accelerator" and r.status == HealthStatus.HEALTHY
        for r in results
    )
    
    # Determine operational mode
    if has_camera and has_servo:
        mode = "Full Operation"
        print("✅ Camera: Available")
        print("✅ Servo Controller: Available")
    elif has_camera:
        mode = "Vision Only (no actuation)"
        print("✅ Camera: Available")
        print("⚠️  Servo Controller: Not available - simulated mode")
    elif has_servo:
        mode = "Blind Actuation (no vision)"
        print("⚠️  Camera: Not available - using prior map")
        print("✅ Servo Controller: Available")
    else:
        mode = "Simulation Mode (no hardware)"
        print("⚠️  Camera: Not available - using simulation")
        print("⚠️  Servo Controller: Not available - using simulation")
    
    if has_ai_accel:
        print("✅ AI Accelerator: Using hardware acceleration")
    else:
        print("⚠️  AI Accelerator: Using CPU (slower inference)")
    
    print()
    print(f"🎯 Operational Mode: {mode}")
    print()
    
    if overall_status == HealthStatus.WARNING:
        print("⚠️  System running in degraded mode")
        print("   Some features may be limited or simulated")
    elif overall_status == HealthStatus.HEALTHY:
        print("✅ All systems nominal")
    else:
        print("❌ Critical issues - cannot operate safely")
    
    print()
    print("=" * 70)
    print()


def demonstrate_crash_recovery():
    """Demonstrate crash detection and recovery."""
    
    print("=" * 70)
    print("🚑 Crash Recovery Demo")
    print("=" * 70)
    print()
    
    config_dir = "/tmp/crash_recovery_test"
    
    # Simulate a crash
    print("📍 Phase 1: Simulating Crash")
    print("-" * 70)
    
    manager = StartupManager(config_dir=config_dir)
    manager.record_crash("Simulated hardware fault")
    
    print("💥 Crash recorded: Simulated hardware fault")
    print()
    time.sleep(2)
    
    # Recovery startup
    print("📍 Phase 2: Recovery Startup")
    print("-" * 70)
    print("🔄 Attempting recovery...")
    print()
    
    manager = StartupManager(config_dir=config_dir)
    success = manager.startup()  # Will run FULL health check
    
    if success:
        print("✅ Recovery successful - system operational")
    else:
        print("❌ Recovery failed - manual intervention needed")
    
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("🤖 ContinuonBrain Lifecycle & Health Check Examples")
    print("=" * 70)
    print("\n")
    
    # Run all demonstrations
    print("Running demonstrations...\n")
    
    # 1. Full lifecycle
    simulate_robot_lifecycle()
    time.sleep(1)
    
    # 2. Health monitoring (commented out - takes time)
    # demonstrate_health_monitoring()
    # time.sleep(1)
    
    # 3. Graceful degradation
    demonstrate_graceful_degradation()
    time.sleep(1)
    
    # 4. Crash recovery
    demonstrate_crash_recovery()
    
    print("\n")
    print("=" * 70)
    print("✅ All demonstrations complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Health checks run automatically on wake from sleep")
    print("  2. System gracefully degrades when hardware unavailable")
    print("  3. Critical issues block startup for safety")
    print("  4. Crash recovery runs full validation")
    print()
