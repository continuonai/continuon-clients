"""
Startup/wake manager for ContinuonBrain OS.
Handles system initialization, health checks, recovery, and service startup.
"""
import os
import sys
import time
import json
import subprocess
import socket
import platform
import threading
import psutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass, field

from continuonbrain.system_health import SystemHealthChecker, HealthStatus
from continuonbrain.robot_modes import RobotModeManager, RobotMode
from continuonbrain.network_discovery import LANDiscoveryService
from continuonbrain.system_context import SystemContext
from continuonbrain.system_instructions import SystemInstructions
from continuonbrain.agent_identity import AgentIdentity
from continuonbrain.system_events import SystemEventLogger


@dataclass
class ServiceDefinition:
    """
    Declarative definition of a service to be launched at startup.
    
    Attributes:
        name: Human-readable service name
        module: Python module path (e.g., "continuonbrain.api.server")
        enabled: Whether this service should be started
        priority: Startup order (lower = earlier)
        log_file: Optional log file path (relative to config_dir/logs)
        log_mode: "text" or "binary" for log file handling
        args: Additional command-line arguments
        env_vars: Additional environment variables
        stdout: Where to redirect stdout (file handle, DEVNULL, or None)
        stderr: Where to redirect stderr (file handle, DEVNULL, or None)
        watch_exit: If True, monitor process exit and trigger system restart
        condition: Optional callable that returns True if service should start
    """
    name: str
    module: str
    enabled: bool = True
    priority: int = 100
    log_file: Optional[str] = None
    log_mode: str = "text"  # "text" or "binary"
    args: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    stdout: Any = None  # file handle, subprocess.DEVNULL, or None
    stderr: Any = None  # file handle, subprocess.DEVNULL, or None
    watch_exit: bool = False
    condition: Optional[Callable[[], bool]] = None


class ServiceRegistry:
    """
    Registry for managing service definitions and their lifecycle.
    Provides a declarative way to define and launch brain modules at boot.
    """
    
    def __init__(self, config_dir: Path, repo_root: Path, python_exec: str, base_env: Dict[str, str]):
        self.config_dir = config_dir
        self.repo_root = repo_root
        self.python_exec = python_exec
        self.base_env = base_env
        self.services: List[ServiceDefinition] = []
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_handles: Dict[str, Any] = {}
    
    def register(self, service: ServiceDefinition):
        """Register a service definition."""
        self.services.append(service)
    
    def get_enabled_services(self) -> List[ServiceDefinition]:
        """Get all enabled services, sorted by priority."""
        enabled = [s for s in self.services if s.enabled]
        # Apply conditions
        enabled = [s for s in enabled if s.condition is None or s.condition()]
        return sorted(enabled, key=lambda s: s.priority)
    
    def start_service(self, service: ServiceDefinition) -> Optional[subprocess.Popen]:
        """
        Start a single service based on its definition.
        
        Returns:
            subprocess.Popen if started successfully, None otherwise
        """
        print(f"🚀 Starting {service.name}...")
        
        # Prepare command
        cmd = [self.python_exec, "-m", service.module] + service.args
        
        # Prepare environment
        env = {**self.base_env, **service.env_vars}
        
        # Prepare log file if specified
        stdout_handle = service.stdout
        stderr_handle = service.stderr
        
        if service.log_file:
            log_dir = self.config_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / service.log_file
            
            if service.log_mode == "binary":
                log_fh = log_path.open("ab")
                # Write header
                try:
                    header = f"\n=== {service.name} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    log_fh.write(header.encode("utf-8", errors="replace"))
                except Exception:
                    pass
            else:
                log_fh = log_path.open("a", buffering=1)
                log_fh.write(f"\n=== {service.name} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            
            self.log_handles[service.name] = log_fh
            stdout_handle = log_fh
            stderr_handle = log_fh
        
        # Default to DEVNULL if not specified
        if stdout_handle is None:
            stdout_handle = subprocess.DEVNULL
        if stderr_handle is None:
            stderr_handle = subprocess.DEVNULL
        
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            print(f"   {service.name} started with PID {proc.pid}")
            self.processes[service.name] = proc
            
            # Setup exit watcher if requested
            if service.watch_exit:
                def _watch_child(proc: subprocess.Popen, name: str) -> None:
                    code = proc.wait()
                    try:
                        if name in self.log_handles:
                            log_fh = self.log_handles[name]
                            if service.log_mode == "binary":
                                msg = f"\n=== {name} exited code={{code}} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                                log_fh.write(msg.encode("utf-8", errors="replace"))
                            else:
                                log_fh.write(f"\n=== {name} exited code={{code}} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                            log_fh.flush()
                    except Exception:
                        pass
                    # If a watched service dies, force a non-zero exit so systemd restarts the whole startup sequence.
                    os._exit(1)  # noqa: WPS437
                
                threading.Thread(target=_watch_child, args=(proc, service.name), daemon=True).start()
            
            return proc
        except Exception as e:
            print(f"   ⚠️  Failed to start {service.name}: {{e}}")
            if service.name in self.log_handles:
                try:
                    self.log_handles[service.name].close()
                except Exception:
                    pass
                del self.log_handles[service.name]
            return None
    
    def start_all(self):
        """Start all enabled services in priority order."""
        services = self.get_enabled_services()
        for service in services:
            self.start_service(service)
    
    def get_process(self, name: str) -> Optional[subprocess.Popen]:
        """Get a process by service name."""
        return self.processes.get(name)
    
    def close_log_handles(self):
        """Close all log file handles."""
        for name, handle in self.log_handles.items():
            try:
                handle.close()
            except Exception:
                pass
        self.log_handles.clear()


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
        config_dir: Optional[str] = None,
        start_services: bool = True,
        robot_name: str = "ContinuonBot",
        headless: bool = False,
        port: int = 8081,
    ):
        self.config_dir = Path(config_dir or os.environ.get("CONTINUON_CONFIG_DIR", "/opt/continuonos/brain"))
        self.state_file = self.config_dir / ".startup_state"
        self.lock_file = self.config_dir / ".startup_lock"
        self.last_wake_time: Optional[int] = None
        self.start_services = start_services
        self.robot_name = robot_name
        env_headless = self._env_flag("CONTINUON_HEADLESS", default=self._default_headless())
        self.headless = headless or env_headless
        self.service_port = port
        self.processes: List[subprocess.Popen] = []
        # Background trainer defaults to off on Pi-class boards to keep boot fast and memory low.
        self.enable_background_trainer = self._env_flag(
            "CONTINUON_ENABLE_BACKGROUND_TRAINER",
            default=not self._default_headless(),
        )

        # Optional "curiosity" learning: run small offline Wikipedia sessions at boot if corpus is present.
        # Default is OFF unless a corpus path is explicitly configured.
        self.enable_wiki_curiosity = self._env_flag(
            "CONTINUON_ENABLE_WIKI_CURIOSITY",
            default=bool(os.environ.get("CONTINUON_WIKI_JSONL")),
        )
        self.wiki_jsonl = os.environ.get("CONTINUON_WIKI_JSONL")
        
        # Services
        self.discovery_service: Optional[LANDiscoveryService] = None
        self.mode_manager: Optional[RobotModeManager] = None
        self.safety_kernel_process: Optional[subprocess.Popen] = None
        self.robot_api_process: Optional[subprocess.Popen] = None
        self._robot_api_log_fh = None
        self.wiki_curiosity_process: Optional[subprocess.Popen] = None
        self._wiki_curiosity_log_fh = None
        self.system_instructions: Optional[SystemInstructions] = None
        self.event_logger = SystemEventLogger(config_dir=str(self.config_dir))
        self._service_registry: Optional[ServiceRegistry] = None
    
    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        """Return boolean env flag with safe default."""
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _default_headless() -> bool:
        """
        Prefer headless mode on ARM SBCs (e.g., Pi5) to avoid launching browsers
        and heavy UI paths during boot.
        """
        machine = platform.machine().lower()
        return machine.startswith("arm") or machine.startswith("aarch64")
        
    def _resolve_module_path(self, module_name: str) -> Optional[str]:
        """Resolve a python module string to a concrete file path."""
        import importlib.util
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                return spec.origin
        except ImportError:
            pass
        return None

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
        # Check for lock file
        if self.lock_file.exists():
            try:
                pid = int(self.lock_file.read_text().strip())
                if psutil.pid_exists(pid):
                    print(f"❌ StartupManager is already running (PID: {{pid}})")
                    return False
            except Exception:
                pass
        
        # Acquire lock
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.write_text(str(os.getpid()))

        startup_mode = self.detect_startup_mode()
        overall_status = HealthStatus.UNKNOWN
        
        print("=" * 60)
        print("🚀 ContinuonBrain Startup")
        print("=" * 60)
        print(f"Startup Mode: {startup_mode.value}")
        
        # Memory-aware model selection info
        try:
            from continuonbrain.services.memory_aware_model_selector import get_memory_status
            mem_status = get_memory_status()
            print(f"System RAM: {mem_status['total_mb']:,}MB ({mem_status['tier'].upper()})")
            print(f"Recommended Model: {mem_status['recommended_model']}")
        except Exception:
            pass  # Non-fatal if memory detection fails
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        self._load_boot_protocols()
        
        # Check battery voltage before proceeding (Low Voltage Cutoff)
        if not self._check_battery_lvc():
            print("❌ BATTERY VOLTAGE TOO LOW - EMERGENCY SHUTDOWN")
            print("System cannot start safely with battery voltage < 9.9V")
            print()
            return False
        
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
            report_path = self.config_dir / "logs" / f"health_{{int(time.time())}}.json"
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
            print(f"  {{resource_status.message}}")
            print("  Services will run in resource-constrained mode")
        print()
        
        # Initialize runtime context and detect inference hardware
        print("🔍 Detecting inference hardware...")
        try:
            from continuonbrain.services.runtime_context import (
                get_runtime_context_manager,
                PrimaryMode,
                SubMode,
            )
            runtime_mgr = get_runtime_context_manager(str(self.config_dir))
            runtime_mgr.detect_hardware()
            
            # Set default mode: HYBRID + AUTONOMOUS for production
            runtime_mgr.set_mode(
                primary=PrimaryMode.HYBRID,
                sub=SubMode.AUTONOMOUS,
            )
            
            # Print summary
            runtime_mgr.print_summary()
            
            self.runtime_context = runtime_mgr.get_context()
        except Exception as e:
            print(f"  ⚠️ Runtime context detection failed: {e}")
            self.runtime_context = None
        
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
        """Start robot services using ServiceRegistry pattern."""
        import sys
        from pathlib import Path
        
        # Choose service port (fallback if 8081 is busy)
        port = self._find_available_port(preferred=self.service_port)
        if port is None:
            print("❌ No available service port in range 8081-8086")
            self._log_event("port_unavailable", "No free port for robot services", {"preferred": self.service_port})
            return
        self.service_port = port

        # Setup base environment
        repo_root = Path(__file__).parent.parent
        env = {**subprocess.os.environ, "PYTHONPATH": str(repo_root)}
        env["PYTHONUNBUFFERED"] = "1"

        instructions_path = SystemContext.get_persist_path()
        if instructions_path:
            env["CONTINUON_SYSTEM_INSTRUCTIONS_PATH"] = str(instructions_path)

        if self.headless:
            env.setdefault("CONTINUON_PREFER_JAX", "1")

        venv_python = repo_root / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python3")
        python_exec = str(venv_python) if venv_python.exists() else sys.executable
        env["CONTINUON_PYTHON"] = python_exec

        # Initialize ServiceRegistry
        registry = ServiceRegistry(
            config_dir=self.config_dir,
            repo_root=repo_root,
            python_exec=python_exec,
            base_env=env
        )

        # Register services in priority order
        # Priority 0: Safety Kernel (Ring 0)
        registry.register(ServiceDefinition(
            name="Safety Kernel",
            module="continuonbrain.kernel.safety_kernel",
            priority=0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        ))

        # Priority 10: Robot API Server (Unified Brain Server)
        robot_api_args = ["--config-dir", str(self.config_dir), "--port", str(self.service_port)]
        registry.register(ServiceDefinition(
            name="Unified Brain Server",
            module="continuonbrain.api.server",
            priority=10,
            log_file="robot_api_server.log",
            log_mode="text",
            args=robot_api_args,
            watch_exit=True,  # Critical service - restart on exit
        ))

        # Priority 20: Background Trainer (optional)
        registry.register(ServiceDefinition(
            name="Nested Learning Sidecar",
            module="continuonbrain.run_trainer",
            enabled=self.enable_background_trainer,
            priority=20,
            args=["--trainer", "auto", "--mode", "local"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            condition=lambda: (repo_root / "continuonbrain" / "run_trainer.py").exists(),
        ))

        # Priority 30: Wiki Curiosity (optional)
        registry.register(ServiceDefinition(
            name="Wiki Curiosity",
            module="continuonbrain.eval.wiki_curiosity_boot",
            enabled=self.enable_wiki_curiosity and bool(self.wiki_jsonl),
            priority=30,
            log_file="wiki_curiosity.log",
            log_mode="binary",
            args=["--config-dir", str(self.config_dir)],
            condition=lambda: bool(self.wiki_jsonl),
        ))

        # Start all registered services
        registry.start_all()

        # Store process references for shutdown and status
        self.safety_kernel_process = registry.get_process("Safety Kernel")
        self.robot_api_process = registry.get_process("Unified Brain Server")
        self.trainer_process = registry.get_process("Nested Learning Sidecar")
        self.wiki_curiosity_process = registry.get_process("Wiki Curiosity")
        
        # Store log handles for status reporting
        if "Unified Brain Server" in registry.log_handles:
            self._robot_api_log_fh = registry.log_handles["Unified Brain Server"]
        if "Wiki Curiosity" in registry.log_handles:
            self._wiki_curiosity_log_fh = registry.log_handles["Wiki Curiosity"]
        
        # Store registry for shutdown
        self._service_registry = registry

        # Start LAN discovery for iPhone/web browser (not a subprocess service)
        print(f"📡 Starting LAN discovery on port {self.service_port}...")
        self.discovery_service = LANDiscoveryService(
            robot_name=self.robot_name,
            service_port=self.service_port
        )
        self.discovery_service.start()
        
        # Initialize mode manager (not a subprocess service)
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
        
        print()
        print("=" * 60)
        print("📱 Robot Ready for Control")
        print("=" * 60)
        print(f"🌐 Open in browser: http://{{self.discovery_service.get_robot_info()['ip_address']}}:{self.service_port}/ui")
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
        # Default to False on headless targets unless explicitly enabled
        auto_launch = self._env_flag("CONTINUON_UI_AUTOLAUNCH", default=not self.headless)
        
        if ui_config_path.exists():
            try:
                with open(ui_config_path, 'r') as f:
                    config = json.load(f)
                    auto_launch = config.get("auto_launch", True)
            except Exception as e:
                print(f"⚠️  Could not read UI config: {{e}}")
        
        if auto_launch:
            url = f"http://{{self.discovery_service.get_robot_info()['ip_address']}}:{self.service_port}/ui"
            print(f"🌐 Launching UI: {{url}}")
            
            # Try to launch known browsers with flags to avoid keyring prompts
            browser_cmd = None
            for browser in ["chromium-browser", "chromium", "google-chrome", "google-chrome-stable"]:
                if shutil.which(browser):
                    browser_cmd = browser
                    break
            
            if browser_cmd:
                print(f"   Using browser: {{browser_cmd}} (with --password-store=basic)")
                try:
                    cmd = [
                        browser_cmd, 
                        "--kiosk",
                        "--password-store=basic", 
                        "--no-default-browser-check",
                        "--no-first-run",
                        "--noerrdialogs",
                        "--disable-infobars",
                        "--check-for-update-interval=31536000",
                        "--simulated-keyring",
                        url
                    ]
                    
                    # Check for kiosk override from env
                    if not self._env_flag("CONTINUON_BROWSER_KIOSK", default=False):
                         cmd.remove("--kiosk")

                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL,
                        start_new_session=True # Detach from parent
                    )
                    return
                except Exception as e:
                    print(f"   ⚠️  Failed to launch {{browser_cmd}}: {{e}}")
            
            # Fallback to standard webbrowser module
            print("   Using default system browser...")
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"⚠️  Could not launch browser: {{e}}")
        else:
            print("ℹ️  UI auto-launch disabled in config")
    
    def shutdown_services(self):
        """Shutdown all robot services."""
        print("🛑 Shutting down services...")
        
        # Remove lock file
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except Exception:
                pass

        # Stop discovery
        if self.discovery_service:
            self.discovery_service.stop()
        
        # Stop all services via registry if available
        if self._service_registry:
            registry = self._service_registry
            
            # Stop processes in reverse priority order
            services = sorted(registry.services, key=lambda s: s.priority, reverse=True)
            for service in services:
                proc = registry.get_process(service.name)
                if proc:
                    print(f"🛑 Stopping {service.name}...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    except Exception:
                        pass
            
            # Close all log handles
            registry.close_log_handles()
        else:
            # Fallback: stop known processes directly if registry not available
            for proc_attr in ['safety_kernel_process', 'robot_api_process', 'trainer_process', 'wiki_curiosity_process']:
                proc = getattr(self, proc_attr, None)
                if proc:
                    print(f"🛑 Stopping {proc_attr}...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            
            # Close log handles manually
            for log_attr in ['_robot_api_log_fh', '_wiki_curiosity_log_fh']:
                log_fh = getattr(self, log_attr, None)
                if log_fh:
                    try:
                        log_fh.close()
                    except Exception:
                        pass
        
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
                print(f"  - {{rule}}")

            print("📜 System instructions:")
            for instruction in self.system_instructions.instructions:
                print(f"  - {{instruction}}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load system instructions: {{exc}}") from exc
    
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
            print(f"⚠️  Could not log event '{{event_type}}': {{exc}}")

    def _find_available_port(self, preferred: int = 8081, max_tries: int = 6) -> Optional[int]:
        """Pick the first open TCP port starting at preferred."""
        for offset in range(max_tries):
            candidate = preferred + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = sock.connect_ex(("0.0.0.0", candidate))
                if result != 0:  # Non-zero means no listener
                    return candidate
        return None

    def _check_battery_lvc(self) -> bool:
        """
        Check battery voltage for Low Voltage Cutoff (LVC).
        """
        try:
            from continuonbrain.sensors.battery_monitor import BatteryMonitor
            
            monitor = BatteryMonitor()
            status = monitor.read_status()
            
            if not status:
                # Battery monitor unavailable - allow startup (may be tethered)
                print("ℹ️  Battery monitor unavailable (may be tethered power)")
                return True
            
            voltage_v = status.voltage_v
            
            # Emergency shutdown threshold: < 9.9V (3.3V/cell)
            if voltage_v < 9.9:
                print(f"❌ CRITICAL: Battery voltage {{voltage_v:.2f}}V < 9.9V (3.3V/cell)")
                print("   Emergency shutdown required to protect battery")
                self._log_event(
                    "battery_lvc_shutdown",
                    f"Emergency shutdown due to low voltage: {{voltage_v:.2f}}V",
                    {"voltage_v": voltage_v, "threshold_v": 9.9}
                )
                return False
            
            # Warning threshold: < 10.5V (3.5V/cell)
            if voltage_v < 10.5:
                print(f"⚠️  WARNING: Battery voltage {{voltage_v:.2f}}V < 10.5V (3.5V/cell)")
                print("   Battery is low but safe to continue")
            
            return True
            
        except Exception as e:
            # If battery check fails, allow startup (fail-safe)
            print(f"⚠️  Battery LVC check failed: {{e}}")
            print("   Allowing startup (may be tethered power)")
            return True
    
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
        help="Max duration for sleep training",
    )
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="Download ceiling for model/assets during sleep training (default: 1GiB)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Service port (default: 8081)"
    )
    
    args = parser.parse_args()
    
    # If --config-dir passed, set expected env var so sub-processes see it
    if args.config_dir:
        os.environ["CONTINUON_CONFIG_DIR"] = args.config_dir

    manager = StartupManager(
        config_dir=args.config_dir,
        start_services=not args.no_services,
        robot_name=args.robot_name,
        port=args.port,
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