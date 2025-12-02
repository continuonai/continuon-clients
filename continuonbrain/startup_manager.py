"""
Startup/wake manager for ContinuonBrain OS.
Handles system initialization, health checks, recovery, and service startup.
"""
import time
import json
import subprocess
from pathlib import Path
from typing import Optional
from enum import Enum

from continuonbrain.system_health import SystemHealthChecker, HealthStatus
from continuonbrain.robot_modes import RobotModeManager, RobotMode
from continuonbrain.network_discovery import LANDiscoveryService


class StartupMode(Enum):
    """System startup mode."""
    COLD_BOOT = "cold_boot"  # Initial power-on
    WAKE_FROM_SLEEP = "wake_from_sleep"  # Resume from sleep
    RECOVERY = "recovery"  # Recovery from crash/error


class StartupManager:
    """
    Manages ContinuonBrain startup sequence.
    Always runs health checks on wake from sleep.
    """
    
    def __init__(
        self, 
        config_dir: str = "/opt/continuonos/brain",
        start_services: bool = True,
        robot_name: str = "ContinuonBot"
    ):
        self.config_dir = Path(config_dir)
        self.state_file = self.config_dir / ".startup_state"
        self.last_wake_time: Optional[int] = None
        self.start_services = start_services
        self.robot_name = robot_name
        
        # Services
        self.discovery_service: Optional[LANDiscoveryService] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.robot_api_process: Optional[subprocess.Popen] = None
        
    def detect_startup_mode(self) -> StartupMode:
        """
        Detect how the system is starting up.
        
        Returns:
            StartupMode indicating cold boot, wake, or recovery
        """
        # Check for state file
        if not self.state_file.exists():
            return StartupMode.COLD_BOOT
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            last_shutdown = state.get("last_shutdown")
            shutdown_type = state.get("shutdown_type", "unknown")
            
            if shutdown_type == "sleep":
                return StartupMode.WAKE_FROM_SLEEP
            elif shutdown_type == "crash":
                return StartupMode.RECOVERY
            else:
                return StartupMode.COLD_BOOT
        
        except Exception:
            return StartupMode.COLD_BOOT
    
    def startup(self, force_health_check: bool = False) -> bool:
        """
        Run startup sequence with health checks.
        
        Args:
            force_health_check: Run health check even on cold boot
        
        Returns:
            True if startup succeeded, False if critical issues
        """
        startup_mode = self.detect_startup_mode()
        
        print("=" * 60)
        print("🚀 ContinuonBrain Startup")
        print("=" * 60)
        print(f"Startup Mode: {startup_mode.value}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Always run health check on wake from sleep
        should_check = (
            startup_mode == StartupMode.WAKE_FROM_SLEEP or
            startup_mode == StartupMode.RECOVERY or
            force_health_check
        )
        
        if should_check:
            print("🏥 Running health check...")
            print()
            
            checker = SystemHealthChecker(config_dir=str(self.config_dir))
            
            # Quick check for wake, full check for recovery
            quick_mode = (startup_mode == StartupMode.WAKE_FROM_SLEEP)
            
            overall_status, results = checker.run_all_checks(quick_mode=quick_mode)
            
            # Save health report
            report_path = self.config_dir / "logs" / f"health_{int(time.time())}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            checker.save_report(str(report_path))
            
            # Handle critical issues
            if overall_status == HealthStatus.CRITICAL:
                print("❌ CRITICAL ISSUES DETECTED")
                print("System cannot start safely.")
                print()
                print("Recommended actions:")
                print("  1. Review health report")
                print("  2. Fix critical issues")
                print("  3. Re-run startup")
                print()
                return False
            
            elif overall_status == HealthStatus.WARNING:
                print("⚠️  WARNINGS DETECTED")
                print("System will start in degraded mode.")
                print()
        
        else:
            print("ℹ️  Skipping health check (cold boot)")
            print("  Run with --force-health-check to check anyway")
            print()
        
        # Record successful startup
        self._record_startup(startup_mode)
        
        # Start services if requested
        if self.start_services:
            print("🚀 Starting robot services...")
            print()
            self._start_services()
        
        return True
    
    def prepare_sleep(self, enable_learning: bool = True):
        """
        Prepare system for sleep mode.
        Optionally starts self-training on saved memories.
        
        Args:
            enable_learning: If True, robot will self-train during sleep
        """
        print("💤 Preparing for sleep...")
        
        # Shutdown services
        self.shutdown_services()
        
        # Set mode to sleep learning if enabled
        if enable_learning and self.mode_manager:
            print("🧠 Enabling sleep learning mode...")
            self.mode_manager.start_sleep_learning()
            print("   Robot will self-train on saved memories")
            print("   Using Gemma-3 for knowledge extraction")
    def _start_services(self):
        """Start robot services (discovery, API server, mode manager)."""
        import sys
        from pathlib import Path
        
        # Start LAN discovery for iPhone/web browser
        print("📡 Starting LAN discovery...")
        self.discovery_service = LANDiscoveryService(
            robot_name=self.robot_name,
            service_port=8080
        )
        self.discovery_service.start()
        
        # Initialize mode manager
        print("🎮 Initializing mode manager...")
        self.mode_manager = RobotModeManager(config_dir=str(self.config_dir))
        
        # Restore last mode or default to idle
        last_mode = self.mode_manager.load_state()
        if last_mode == RobotMode.SLEEP_LEARNING:
            # If we were sleep learning, return to idle on wake
            self.mode_manager.return_to_idle()
        elif last_mode:
            print(f"   Restored mode: {last_mode.value}")
            self.mode_manager.current_mode = last_mode
        else:
            self.mode_manager.return_to_idle()
        
        # Start Robot API server (production entry point)
        print("🌐 Starting Robot API server...")
        try:
            repo_root = Path(__file__).parent.parent
            server_module = "continuonbrain.robot_api_server"
            server_path = repo_root / "continuonbrain" / "robot_api_server.py"
            
            if server_path.exists():
                # Start in background
                self.robot_api_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        server_module,
                        "--port",
                        "8080",
                        "--real-hardware",  # production path uses real controllers; fail fast if missing
                    ],
                    env={**subprocess.os.environ, "PYTHONPATH": str(repo_root)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"   Robot API started (PID: {self.robot_api_process.pid})")
                print(f"   Endpoint: http://localhost:8080")
            else:
                print(f"   ⚠️  Robot API module not found: {server_path}")
        except Exception as e:
            print(f"   ⚠️  Could not start Robot API: {e}")
        
        print()
        print("=" * 60)
        print("📱 Robot Ready for Control")
        print("=" * 60)
        print(f"🌐 Open in browser: http://{self.discovery_service.get_robot_info()['ip_address']}:8080/ui")
        print("📱 Or find '{0}' on iPhone app".format(self.robot_name))
        print()
        print("Available modes:")
        print("  • Manual Training - Control robot for training data")
        print("  • Autonomous - VLA policy control")
        print("  • Sleep Learning - Self-train on memories")
        print("=" * 60)
        print()
    
    def shutdown_services(self):
        """Shutdown all robot services."""
        print("🛑 Shutting down services...")
        
        # Stop discovery
        if self.discovery_service:
            self.discovery_service.stop()
        
        # Stop Robot API server
        if self.robot_api_process:
            try:
                self.robot_api_process.terminate()
                self.robot_api_process.wait(timeout=5)
                print("   Robot API stopped")
            except Exception as e:
                print(f"   Error stopping Robot API: {e}")
        
        print("✅ Services shutdown complete")
    
    def prepare_sleep(self, enable_learning: bool = True):
        """
        Prepare system for sleep mode.
        Optionally starts self-training on saved memories.
        
        Args:
            enable_learning: If True, robot will self-train during sleep
        """
        print("💤 Preparing for sleep...")
        
        # Shutdown services
        self.shutdown_services()
        
        # Set mode to sleep learning if enabled
        if enable_learning and self.mode_manager:
            print("🧠 Enabling sleep learning mode...")
            self.mode_manager.start_sleep_learning()
            print("   Robot will self-train on saved memories")
            print("   Using Gemma-3 for knowledge extraction")
        
        state = {
            "last_shutdown": time.time(),
            "shutdown_type": "sleep",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print("✅ System ready for sleep")
    
    def prepare_shutdown(self):
        """
        Prepare system for clean shutdown.
        """
        print("🛑 Preparing for shutdown...")
        
        # Shutdown services
        self.shutdown_services()
        
        state = {
            "last_shutdown": time.time(),
            "shutdown_type": "shutdown",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print("✅ System ready for shutdown")
    
    def record_crash(self, error: str):
        """
        Record a crash for recovery on next boot.
        """
        state = {
            "last_shutdown": time.time(),
            "shutdown_type": "crash",
            "error": error,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _record_startup(self, mode: StartupMode):
        """Record successful startup."""
        state = {
            "last_startup": time.time(),
            "startup_mode": mode.value,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)


def main():
    """Run startup manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ContinuonBrain Startup Manager")
    parser.add_argument(
        "--force-health-check",
        action="store_true",
        help="Force health check even on cold boot"
    )
    parser.add_argument(
        "--config-dir",
        default="/tmp/continuonbrain_demo",
        help="Configuration directory (default: /tmp/continuonbrain_demo)"
    )
    parser.add_argument(
        "--prepare-sleep",
        action="store_true",
        help="Prepare system for sleep (instead of starting)"
    )
    parser.add_argument(
        "--prepare-shutdown",
        action="store_true",
        help="Prepare system for shutdown (instead of starting)"
    )
    parser.add_argument(
        "--no-services",
        action="store_true",
        help="Don't start services (discovery, API server)"
    )
    parser.add_argument(
        "--robot-name",
        default="ContinuonBot",
        help="Robot name for LAN discovery (default: ContinuonBot)"
    )
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Don't enable sleep learning when preparing for sleep"
    )
    
    args = parser.parse_args()
    
    manager = StartupManager(
        config_dir=args.config_dir,
        start_services=not args.no_services,
        robot_name=args.robot_name
    )
    
    if args.prepare_sleep:
        manager.prepare_sleep(enable_learning=not args.no_learning)
    elif args.prepare_shutdown:
        manager.prepare_shutdown()
    else:
        try:
            success = manager.startup(force_health_check=args.force_health_check)
            
            if success and not args.no_services:
                # Keep running until interrupted
                print("Robot services running. Press Ctrl+C to shutdown...")
                import signal
                signal.signal(signal.SIGINT, lambda s, f: None)
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            manager.shutdown_services()
        finally:
            exit(0 if success else 1)


if __name__ == "__main__":
    main()
