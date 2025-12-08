"""
Startup/wake manager for ContinuonBrain OS.
Handles system initialization, health checks, recovery, and service startup.
"""
import time
import json
import subprocess
import socket
from pathlib import Path
from typing import Optional
from enum import Enum

from continuonbrain.system_health import SystemHealthChecker, HealthStatus
from continuonbrain.robot_modes import RobotModeManager, RobotMode
from continuonbrain.network_discovery import LANDiscoveryService
from continuonbrain.system_context import SystemContext
from continuonbrain.system_instructions import SystemInstructions
from continuonbrain.agent_identity import AgentIdentity
from continuonbrain.system_events import SystemEventLogger


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
        self.service_port = 8080
        
        # Services
        self.discovery_service: Optional[LANDiscoveryService] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.robot_api_process: Optional[subprocess.Popen] = None
        self.system_instructions: Optional[SystemInstructions] = None
        self.event_logger = SystemEventLogger(config_dir=str(self.config_dir))
        
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
        overall_status = HealthStatus.UNKNOWN
        
        print("=" * 60)
        print("🚀 ContinuonBrain Startup")
        print("=" * 60)
        print(f"Startup Mode: {startup_mode.value}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        self._load_boot_protocols()
        
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
        
        # Check resource availability
        print("📊 Checking system resources...")
        from continuonbrain.resource_monitor import ResourceMonitor
        resource_monitor = ResourceMonitor(config_dir=self.config_dir)
        resource_status = resource_monitor.check_resources()
        
        print(f"  Memory: {resource_status.available_memory_mb}MB available ({resource_status.memory_percent:.1f}% used)")
        print(f"  Level: {resource_status.level.value.upper()}")
        
        if resource_status.level.value in ['critical', 'emergency']:
            print("⚠️  WARNING: Low memory detected")
            print(f"  {resource_status.message}")
            print("  Services will run in resource-constrained mode")
        print()
        
        # Record successful startup
        self._record_startup(startup_mode)

        self._log_event(
            "reboot",
            f"Startup complete in {startup_mode.value} mode",
            {
                "startup_mode": startup_mode.value,
                "health_status": overall_status.value,
                "memory_percent": resource_status.memory_percent,
                "available_memory_mb": resource_status.available_memory_mb,
                "resource_level": resource_status.level.value,
                "services_requested": self.start_services,
                "service_port": self.service_port,
            },
        )
        
        # Start services if requested
        if self.start_services:
            print("🚀 Starting robot services...")
            print()
            self._start_services()
            self._log_event(
                "services_started",
                "Robot services activated",
                {
                    "lan_discovery": bool(self.discovery_service),
                    "robot_api_pid": getattr(self.robot_api_process, "pid", None),
                    "mode": getattr(getattr(self, "mode_manager", None), "current_mode", None).value
                    if getattr(getattr(self, "mode_manager", None), "current_mode", None)
                    else None,
                    "service_port": self.service_port,
                },
            )
        
        return True
    
    def _start_services(self):
        """Start robot services (discovery, API server, mode manager)."""
        import sys
        from pathlib import Path
        
        # Choose service port (fallback if 8080 is busy)
        port = self._find_available_port(preferred=self.service_port)
        if port is None:
            print("❌ No available service port in range 8080-8085")
            self._log_event("port_unavailable", "No free port for robot services", {"preferred": self.service_port})
            return
        self.service_port = port

        # Start LAN discovery for iPhone/web browser
        print(f"📡 Starting LAN discovery on port {self.service_port}...")
        self.discovery_service = LANDiscoveryService(
            robot_name=self.robot_name,
            service_port=self.service_port
        )
        self.discovery_service.start()
        
        # Initialize mode manager
        print("🎮 Initializing mode manager...")
        self.mode_manager = RobotModeManager(
            config_dir=str(self.config_dir),
            system_instructions=self.system_instructions,
        )
        
        # Always start in AUTONOMOUS mode for production (motion + inference enabled)
        print("🤖 Activating AUTONOMOUS mode (motion + inference + training enabled)")
        self.mode_manager.set_mode(
            RobotMode.AUTONOMOUS,
            metadata={
                "startup_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "auto_activated": True,
                "self_training_enabled": True
            }
        )
        
        # Start Robot API server (production entry point)
        print(f"🌐 Starting Robot API server on port {self.service_port}...")
        try:
            repo_root = Path(__file__).parent.parent
            server_module = "continuonbrain.api.server"
            server_path = repo_root / "continuonbrain" / "api" / "server.py"
            
            if server_path.exists():
                # Start in background
                env = {**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
                # Inject HF Token for Gemma 3n
                env["HUGGINGFACE_TOKEN"] = "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"

                instructions_path = SystemContext.get_persist_path()
                if instructions_path:
                    env["CONTINUON_SYSTEM_INSTRUCTIONS_PATH"] = str(instructions_path)
                
                # Force usage of venv python if available/detected relative to repo root
                venv_python = repo_root / ".venv" / "bin" / "python3"
                python_exec = str(venv_python) if venv_python.exists() else sys.executable

                self.robot_api_process = subprocess.Popen(
                    [
                        python_exec,
                        "-m",
                        server_module,
                        "--port",
                        str(self.service_port),
                        "--config-dir",
                        self.config_dir,
                        "--real-hardware",  # production path uses real controllers; fail fast if missing
                    ],
                    env=env,
                    # Inherit stdout/stderr to see logs in systemd
                    # stdout=subprocess.PIPE,
                    # stderr=subprocess.PIPE
                )
                print(f"   Robot API started (PID: {self.robot_api_process.pid})")
                print(f"   Endpoint: http://localhost:{self.service_port}")
                
                # Start Nested Learning Sidecar
                print("🧠 Starting Nested Learning Sidecar...")
                trainer_path = repo_root / "continuonbrain" / "run_trainer.py"
                if trainer_path.exists():
                    self.trainer_process = subprocess.Popen(
                        [sys.executable, "-m", "continuonbrain.run_trainer"],
                        env=env,
                        stdout=subprocess.DEVNULL,  # Keep console clean
                        stderr=subprocess.DEVNULL
                    )
                    print(f"   Sidecar Trainer started (PID: {self.trainer_process.pid})")
                else:
                    print(f"   ⚠️ Trainer script not found: {trainer_path}")

            else:
                print(f"   ⚠️  Robot API module not found: {server_path}")
        except Exception as e:
            print(f"   ⚠️  Could not start Robot API: {e}")
        
        print()
        print("=" * 60)
        print("📱 Robot Ready for Control")
        print("=" * 60)
        print(f"🌐 Open in browser: http://{self.discovery_service.get_robot_info()['ip_address']}:{self.service_port}/ui")
        print("📱 Or find '{0}' on iPhone app".format(self.robot_name))
        print()
        print("Available modes:")
        print("  • Manual Training - Control robot for training data")
        print("  • Autonomous - VLA policy control")
        print("  • Sleep Learning - Self-train on memories")
        print("=" * 60)
        print("=" * 60)
        print()
        
        # Run Auto-Agent Checks (Self-Activation)
        print("🤖 Running Self-Activation Checks...")
        identity = AgentIdentity(config_dir=str(self.config_dir))
        identity.self_report()

        # Launch UI if configured
        self.launch_ui()
        
    def launch_ui(self):
        """Launch the web UI in the default browser if configured."""
        import webbrowser
        import shutil
        
        ui_config_path = self.config_dir / "ui_config.json"
        auto_launch = True  # Default to True
        
        if ui_config_path.exists():
            try:
                with open(ui_config_path, 'r') as f:
                    config = json.load(f)
                    auto_launch = config.get("auto_launch", True)
            except Exception as e:
                print(f"⚠️  Could not read UI config: {e}")
        
        if auto_launch:
            url = f"http://{self.discovery_service.get_robot_info()['ip_address']}:{self.service_port}/ui"
            print(f"🌐 Launching UI: {url}")
            
            # Try to launch known browsers with flags to avoid keyring prompts
            browser_cmd = None
            for browser in ["chromium-browser", "chromium", "google-chrome", "google-chrome-stable"]:
                if shutil.which(browser):
                    browser_cmd = browser
                    break
            
            if browser_cmd:
                print(f"   Using browser: {browser_cmd} (with --password-store=basic)")
                try:
                    # Use a temp user data dir to avoid locking main profile and further suppress prompts
                    # user_data_dir = f"/tmp/continuon_browser_{int(time.time())}"
                    # os.makedirs(user_data_dir, exist_ok=True)
                    
                    cmd = [
                        browser_cmd, 
                        "--password-store=basic", 
                        "--no-default-browser-check",
                        "--no-first-run",
                        # "--user-data-dir=" + user_data_dir, # Optional: isolate session
                        url
                    ]
                    
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL,
                        start_new_session=True # Detach from parent
                    )
                    return
                except Exception as e:
                    print(f"   ⚠️  Failed to launch {browser_cmd}: {e}")
            
            # Fallback to standard webbrowser module
            print("   Using default system browser...")
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"⚠️  Could not launch browser: {e}")
        else:
            print("ℹ️  UI auto-launch disabled in config")
    
    def shutdown_services(self):
        """Shutdown all robot services."""
        print("🛑 Shutting down services...")
        
        # Stop discovery
        if self.discovery_service:
            self.discovery_service.stop()
        
        # Stop Robot API server
        if hasattr(self, 'robot_api_process') and self.robot_api_process:
            self.robot_api_process.terminate()
            try:
                self.robot_api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.robot_api_process.kill()
        
        if hasattr(self, 'trainer_process') and self.trainer_process:
            self.trainer_process.terminate()
            try:
                self.trainer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.trainer_process.kill()
        
        print("✅ Services stopped.")
    
    def prepare_sleep(
        self,
        enable_learning: bool = True,
        *,
        max_sleep_training_hours: float = 6.0,
        max_download_bytes: int = 1024 * 1024 * 1024,
    ):
        """
        Prepare system for sleep mode.
        Optionally starts self-training on saved memories.

        Args:
            enable_learning: If True, robot will self-train during sleep
            max_sleep_training_hours: Wall-clock ceiling for sleep training.
            max_download_bytes: Download budget for model/assets during training.
        """
        print("💤 Preparing for sleep...")
        
        # Shutdown services
        self.shutdown_services()
        
        # Set mode to sleep learning if enabled
        if enable_learning and self.mode_manager:
            print("🧠 Enabling sleep learning mode...")
            self.mode_manager.start_sleep_learning(
                max_sleep_training_hours=max_sleep_training_hours,
                max_download_bytes=max_download_bytes,
            )
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

    def _load_boot_protocols(self) -> None:
        """Load system instructions and safety protocol during startup."""

        try:
            self.system_instructions = SystemInstructions.load(self.config_dir)
            persist_path = self.config_dir / "logs" / "system_instructions_merged.json"
            SystemContext.register_instructions(
                self.system_instructions,
                persist_path=persist_path,
            )

            print("🛡️  Safety protocol loaded (base rules cannot be overridden):")
            for rule in self.system_instructions.safety_protocol.rules:
                print(f"  - {rule}")

            print("📜 System instructions:")
            for instruction in self.system_instructions.instructions:
                print(f"  - {instruction}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load system instructions: {exc}") from exc
    
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
    
    def _log_event(self, event_type: str, message: str, data: Optional[dict] = None) -> None:
        """Persist a lifecycle event without breaking startup on errors."""
        try:
            self.event_logger.log(event_type, message, data or {})
        except Exception as exc:  # Best-effort; never block startup
            print(f"⚠️  Could not log event '{event_type}': {exc}")

    def _find_available_port(self, preferred: int = 8080, max_tries: int = 6) -> Optional[int]:
        """Pick the first open TCP port starting at preferred."""
        for offset in range(max_tries):
            candidate = preferred + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = sock.connect_ex(("0.0.0.0", candidate))
                if result != 0:  # Non-zero means no listener
                    return candidate
        return None

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
    parser.add_argument(
        "--max-sleep-training-hours",
        type=float,
        default=6.0,
        help="Max hours to allow self-training during sleep (default: 6)",
    )
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="Download ceiling for model/assets during sleep training (default: 1GiB)",
    )
    
    args = parser.parse_args()
    
    manager = StartupManager(
        config_dir=args.config_dir,
        start_services=not args.no_services,
        robot_name=args.robot_name
    )
    
    if args.prepare_sleep:
        manager.prepare_sleep(
            enable_learning=not args.no_learning,
            max_sleep_training_hours=args.max_sleep_training_hours,
            max_download_bytes=args.max_download_bytes,
        )
    elif args.prepare_shutdown:
        manager.prepare_shutdown()
    else:
        success = False
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
