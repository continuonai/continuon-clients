"""
System health checker for ContinuonBrain OS.
Runs comprehensive hardware and software validation on startup/wake.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    status: HealthStatus
    message: str
    details: Dict = None
    timestamp_ns: int = 0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()


class SystemHealthChecker:
    """
    Comprehensive system health checker for ContinuonBrain.
    Validates hardware, software, and dependencies on startup/wake.
    """
    
    def __init__(self, config_dir: str = "/opt/continuonos/brain"):
        self.config_dir = Path(config_dir)
        self.results: List[HealthCheckResult] = []
        self.start_time_ns = 0
        self.end_time_ns = 0
    
    def run_all_checks(self, quick_mode: bool = False) -> Tuple[HealthStatus, List[HealthCheckResult]]:
        """
        Run all health checks.
        
        Args:
            quick_mode: Skip expensive checks (sensor data capture, AI inference)
        
        Returns:
            (overall_status, list of check results)
        """
        self.results = []
        self.start_time_ns = time.time_ns()
        
        print("=" * 60)
        print("🏥 ContinuonBrain System Health Check")
        print("=" * 60)
        print(f"Mode: {'Quick' if quick_mode else 'Full'}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Hardware checks
        print("🔧 Hardware Checks")
        print("-" * 60)
        self._check_hardware_detection()
        self._check_camera_health()
        self._check_servo_controller()
        self._check_battery()
        self._check_i2c_bus()
        self._check_usb_devices()
        if not quick_mode:
            self._check_ai_accelerator()
        print()
        
        # Software checks
        print("💾 Software Checks")
        print("-" * 60)
        self._check_python_environment()
        self._check_critical_dependencies()
        self._check_model_files()
        self._check_rlds_storage()
        self._check_developer_tooling()
        self._check_self_update_tooling()
        self._check_api_budget()
        print()
        
        # System resources
        print("📊 System Resources")
        print("-" * 60)
        self._check_disk_space()
        self._check_memory()
        self._check_cpu_temperature()
        if not quick_mode:
            self._check_network_connectivity()
        print()
        
        # Configuration checks
        print("⚙️  Configuration Checks")
        print("-" * 60)
        self._check_directory_structure()
        self._check_permissions()
        self._check_mission_statement()
        self._check_safety_config()
        print()
        
        self.end_time_ns = time.time_ns()
        
        # Compute overall status
        overall_status = self._compute_overall_status()
        
        self._print_summary(overall_status)
        
        return overall_status, self.results
    
    def _add_result(self, component: str, status: HealthStatus, message: str, details: Dict = None):
        """Add a health check result."""
        result = HealthCheckResult(component, status, message, details or {})
        self.results.append(result)
        
        # Print status
        status_symbol = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️ ",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓",
        }[status]
        
        print(f"  {status_symbol} {component}: {message}")
    
    def _check_hardware_detection(self):
        """Check if hardware auto-detection works."""
        try:
            from continuonbrain.sensors.hardware_detector import HardwareDetector
            
            detector = HardwareDetector()
            devices = detector.detect_all()
            
            if devices:
                self._add_result(
                    "Hardware Detection",
                    HealthStatus.HEALTHY,
                    f"Detected {len(devices)} device(s)",
                    {"device_count": len(devices)}
                )
            else:
                self._add_result(
                    "Hardware Detection",
                    HealthStatus.WARNING,
                    "No devices detected - will run in mock mode",
                    {"device_count": 0}
                )
        except Exception as e:
            self._add_result(
                "Hardware Detection",
                HealthStatus.CRITICAL,
                f"Detection failed: {str(e)}",
                {"error": str(e)}
            )
    
    def _check_camera_health(self):
        """Check camera availability and basic operation."""
        try:
            from continuonbrain.sensors.hardware_detector import HardwareDetector
            
            detector = HardwareDetector()
            devices = detector.detect_all()
            cameras = [d for d in devices if "camera" in d.device_type]
            
            if cameras:
                primary_cam = cameras[0]
                self._add_result(
                    "Camera",
                    HealthStatus.HEALTHY,
                    f"{primary_cam.name} available",
                    {"name": primary_cam.name, "capabilities": primary_cam.capabilities}
                )
            else:
                self._add_result(
                    "Camera",
                    HealthStatus.WARNING,
                    "No camera detected",
                    {}
                )
        except Exception as e:
            self._add_result(
                "Camera",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def _check_servo_controller(self):
        """Check servo controller availability."""
        try:
            from continuonbrain.sensors.hardware_detector import HardwareDetector
            
            detector = HardwareDetector()
            devices = detector.detect_all()
            servos = [d for d in devices if d.device_type == "servo_controller"]
            
            if servos:
                servo = servos[0]
                self._add_result(
                    "Servo Controller",
                    HealthStatus.HEALTHY,
                    f"{servo.name} at {servo.address}",
                    {"name": servo.name, "address": servo.address}
                )
            else:
                self._add_result(
                    "Servo Controller",
                    HealthStatus.WARNING,
                    "No servo controller detected",
                    {}
                )
        except Exception as e:
            self._add_result(
                "Servo Controller",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )

    def _check_developer_tooling(self):
        """Detect visual code editors and MCP agentic chat tooling."""

        editors, editor_detail = self._detect_code_editors()
        mcp_tooling = self._detect_mcp_tooling()
        gemini_tooling = self._detect_gemini_cli()

        details = {
            "visual_editors": editors,
            "editor_detail": editor_detail or "",
            "mcp_tooling": mcp_tooling,
            "gemini_cli": gemini_tooling,
        }

        if editors or mcp_tooling["available"] or gemini_tooling["available"]:
            message = "Developer tooling detected; prefer onboard tools before internet access"
            status = HealthStatus.HEALTHY
        else:
            message = "No visual editor or MCP tooling detected; self-maintenance will be limited"
            status = HealthStatus.WARNING

        self._add_result("Developer Tooling", status, message, details)

    def _detect_code_editors(self) -> Tuple[List[str], Optional[str]]:
        """Identify available visual code editors from common CLI shims and running processes."""

        editor_commands = {
            "VS Code": ["code", "code-insiders"],
            "Cursor": ["cursor"],
            "VSCodium": ["vscodium"],
        }

        found: List[str] = []
        for name, binaries in editor_commands.items():
            if any(shutil.which(binary) for binary in binaries):
                found.append(name)

        process_hint: Optional[str] = None
        if not found:
            try:
                result = subprocess.run(
                    ["ps", "-A", "-o", "comm="], capture_output=True, text=True, check=False
                )
                processes = result.stdout.lower().splitlines()
                if any(
                    hint in proc
                    for proc in processes
                    for hint in ("code", "cursor", "vscodium", "vscode")
                ):
                    process_hint = "Found running visual editor process"
            except Exception as exc:  # noqa: BLE001
                process_hint = f"Process scan failed: {exc}"

        return found, process_hint

    def _detect_mcp_tooling(self) -> Dict[str, object]:
        """Detect MCP command-line tooling and configuration for agentic chat."""

        binaries = [binary for binary in ("mcp", "mcp-cli", "mcp-client") if shutil.which(binary)]
        config_candidates = [Path.home() / ".config" / "mcp", self.config_dir / "mcp"]
        configs = [str(path) for path in config_candidates if path.exists()]

        return {
            "available": bool(binaries or configs),
            "binaries": binaries,
            "configs": configs,
        }

    def _detect_gemini_cli(self) -> Dict[str, object]:
        """Detect Gemini CLI to supervise agentic self-updates."""

        binaries = [binary for binary in ("gemini", "gemini-cli", "gcloud") if shutil.which(binary)]
        config_candidates = [Path.home() / ".config" / "gemini", self.config_dir / "gemini"]
        configs = [str(path) for path in config_candidates if path.exists()]

        return {
            "available": bool(binaries or configs),
            "binaries": binaries,
            "configs": configs,
        }

    def _check_self_update_tooling(self):
        """Ensure the robot can refresh its brain safely with an offline-first plan."""

        gemini_cli = self._detect_gemini_cli()
        local_brain_present = self.config_dir.exists()

        details = {
            "gemini_cli": gemini_cli,
            "local_brain_dir": str(self.config_dir),
            "local_brain_present": local_brain_present,
        }

        if local_brain_present and gemini_cli["available"]:
            status = HealthStatus.HEALTHY
            message = (
                "Self-update channel ready via Gemini CLI supervising antigravity agentic routines"
            )
        elif local_brain_present:
            status = HealthStatus.WARNING
            message = (
                "Local brain present; Gemini CLI unavailable so updates stay offline-first"
            )
        else:
            status = HealthStatus.CRITICAL
            message = f"Brain directory missing at {self.config_dir}; cannot self-update safely"

        self._add_result("Self-Update", status, message, details)

    def _load_daily_api_budget(self) -> Tuple[float, str, Optional[str]]:
        """Load the daily API/network spend ceiling with a $5/day default."""

        default_limit = 5.0
        warning: Optional[str] = None

        env_value = os.environ.get("BRAIN_DAILY_API_BUDGET_USD")
        if env_value:
            try:
                return float(env_value), "env:BRAIN_DAILY_API_BUDGET_USD", warning
            except ValueError:
                warning = f"Invalid env budget '{env_value}', using default ${default_limit:.2f}"

        config_path = self.config_dir / "budgets" / "api_budget.json"
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text())
                limit = float(payload.get("daily_limit_usd", default_limit))
                return limit, f"config:{config_path}", warning
            except Exception as exc:  # noqa: BLE001
                warning = f"Failed to read {config_path}: {exc}"

        return default_limit, "default", warning

    def _check_api_budget(self):
        """Validate that API/network usage stays within the $5/day ceiling."""

        limit, source, warning = self._load_daily_api_budget()
        details = {
            "daily_limit_usd": limit,
            "source": source,
            "offline_first": True,
            "onboard_first_models": ["Gemma 3n"],
            "internet_tools": ["Gemini CLI", "MCP clients"],
        }
        if warning:
            details["warning"] = warning

        if limit <= 5.0:
            status = HealthStatus.HEALTHY
            message = (
                f"API/network spend capped at ${limit:.2f}/day; preferring Gemma/local tools before internet calls"
            )
        else:
            status = HealthStatus.WARNING
            message = (
                f"API budget set to ${limit:.2f}/day (source={source}), exceeds recommended $5 ceiling; reduce spend and favor onboard models"
            )

        self._add_result("API Budget", status, message, details)
    
    def _check_battery(self):
        """Check battery monitor and charge level."""
        try:
            from continuonbrain.sensors.battery_monitor import BatteryMonitor
            
            monitor = BatteryMonitor()
            status = monitor.read_status()
            
            if status:
                # Determine health based on charge level
                if status.charge_percent < 10:
                    health = HealthStatus.CRITICAL
                    message = f"Battery critical: {status.charge_percent:.1f}%"
                elif status.charge_percent < 20:
                    health = HealthStatus.WARNING
                    message = f"Battery low: {status.charge_percent:.1f}%"
                else:
                    health = HealthStatus.HEALTHY
                    message = f"Battery OK: {status.charge_percent:.1f}%"
                
                self._add_result(
                    "Battery",
                    health,
                    message,
                    {
                        "voltage_v": round(status.voltage_v, 2),
                        "charge_percent": round(status.charge_percent, 1),
                        "is_charging": status.is_charging,
                        "current_ma": round(status.current_ma, 1),
                    }
                )
            else:
                self._add_result(
                    "Battery",
                    HealthStatus.WARNING,
                    "Battery monitor unavailable",
                    {}
                )
        except Exception as e:
            self._add_result(
                "Battery",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def _check_i2c_bus(self):
        """Check I2C bus is accessible."""
        try:
            result = subprocess.run(['i2cdetect', '-y', '1'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Count devices
                device_count = result.stdout.count('\n') - 1  # Subtract header
                self._add_result(
                    "I2C Bus",
                    HealthStatus.HEALTHY,
                    "Bus accessible",
                    {"bus": 1}
                )
            else:
                self._add_result(
                    "I2C Bus",
                    HealthStatus.WARNING,
                    "Bus not accessible",
                    {}
                )
        except FileNotFoundError:
            self._add_result(
                "I2C Bus",
                HealthStatus.WARNING,
                "i2cdetect not installed",
                {}
            )
        except Exception as e:
            self._add_result(
                "I2C Bus",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def _check_usb_devices(self):
        """Check USB device enumeration."""
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                device_count = len(result.stdout.strip().split('\n'))
                self._add_result(
                    "USB Devices",
                    HealthStatus.HEALTHY,
                    f"{device_count} USB device(s) enumerated",
                    {"device_count": device_count}
                )
            else:
                self._add_result(
                    "USB Devices",
                    HealthStatus.WARNING,
                    "USB enumeration failed",
                    {}
                )
        except Exception as e:
            self._add_result(
                "USB Devices",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def _check_ai_accelerator(self):
        """Check AI accelerator availability (Hailo, Coral, etc.)."""
        try:
            from continuonbrain.sensors.hardware_detector import HardwareDetector
            
            detector = HardwareDetector()
            devices = detector.detect_all()
            ai_devices = [d for d in devices if d.device_type == "ai_accelerator"]
            
            if ai_devices:
                accel = ai_devices[0]
                self._add_result(
                    "AI Accelerator",
                    HealthStatus.HEALTHY,
                    f"{accel.name} available",
                    {"name": accel.name}
                )
            else:
                self._add_result(
                    "AI Accelerator",
                    HealthStatus.WARNING,
                    "No accelerator - using CPU",
                    {}
                )
        except Exception as e:
            self._add_result(
                "AI Accelerator",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def _check_python_environment(self):
        """Check Python version and environment."""
        import sys
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            self._add_result(
                "Python",
                HealthStatus.HEALTHY,
                f"Python {version.major}.{version.minor}.{version.micro}",
                {"version": f"{version.major}.{version.minor}.{version.micro}"}
            )
        else:
            self._add_result(
                "Python",
                HealthStatus.WARNING,
                f"Python {version.major}.{version.minor} (recommend 3.9+)",
                {"version": f"{version.major}.{version.minor}.{version.micro}"}
            )
    
    def _check_critical_dependencies(self):
        """Check critical Python dependencies are installed."""
        critical_deps = {
            "numpy": "numpy",
            "depthai": "depthai (OAK camera)",
        }
        
        optional_deps = {
            "adafruit_servokit": "adafruit-servokit (PCA9685)",
        }
        
        missing_critical = []
        missing_optional = []
        
        for module, name in critical_deps.items():
            try:
                __import__(module)
            except ImportError:
                missing_critical.append(name)
        
        for module, name in optional_deps.items():
            try:
                __import__(module)
            except ImportError:
                missing_optional.append(name)
        
        if not missing_critical:
            msg = "All critical dependencies installed"
            if missing_optional:
                msg += f" (optional: {', '.join(missing_optional)} not installed)"
            self._add_result(
                "Dependencies",
                HealthStatus.HEALTHY,
                msg,
                {"missing_optional": missing_optional}
            )
        else:
            self._add_result(
                "Dependencies",
                HealthStatus.CRITICAL,
                f"Missing critical: {', '.join(missing_critical)}",
                {"missing_critical": missing_critical}
            )
    
    def _check_model_files(self):
        """Check model files exist and are valid."""
        model_dir = self.config_dir / "model"
        
        if not model_dir.exists():
            self._add_result(
                "Model Files",
                HealthStatus.WARNING,
                f"Model directory not found: {model_dir}",
                {}
            )
            return
        
        # Check for base model or manifest
        has_base = (model_dir / "base_model").exists()
        has_manifest = list(model_dir.glob("manifest*.json"))
        
        if has_base or has_manifest:
            self._add_result(
                "Model Files",
                HealthStatus.HEALTHY,
                "Model files present",
                {"has_base": has_base, "manifests": len(has_manifest)}
            )
        else:
            self._add_result(
                "Model Files",
                HealthStatus.WARNING,
                "No model files found",
                {}
            )
    
    def _check_rlds_storage(self):
        """Check RLDS storage is accessible and has space."""
        rlds_dir = self.config_dir / "rlds" / "episodes"
        
        if not rlds_dir.exists():
            self._add_result(
                "RLDS Storage",
                HealthStatus.WARNING,
                f"RLDS directory not found: {rlds_dir}",
                {}
            )
            return
        
        # Count episodes
        episodes = list(rlds_dir.glob("*/episode.json"))
        
        self._add_result(
            "RLDS Storage",
            HealthStatus.HEALTHY,
            f"{len(episodes)} episode(s) stored",
            {"episode_count": len(episodes), "path": str(rlds_dir)}
        )
    
    def _check_disk_space(self):
        """Check available disk space."""
        try:
            result = subprocess.run(['df', '-h', str(self.config_dir.parent)],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        used_pct = parts[4].rstrip('%')
                        available = parts[3]
                        
                        if int(used_pct) > 90:
                            status = HealthStatus.CRITICAL
                            msg = f"Disk {used_pct}% full (only {available} available)"
                        elif int(used_pct) > 75:
                            status = HealthStatus.WARNING
                            msg = f"Disk {used_pct}% full ({available} available)"
                        else:
                            status = HealthStatus.HEALTHY
                            msg = f"{available} available ({used_pct}% used)"
                        
                        self._add_result(
                            "Disk Space",
                            status,
                            msg,
                            {"used_percent": used_pct, "available": available}
                        )
                        return
            
            self._add_result("Disk Space", HealthStatus.UNKNOWN, "Could not check", {})
        except Exception as e:
            self._add_result("Disk Space", HealthStatus.UNKNOWN, f"Check failed: {str(e)}", {})
    
    def _check_memory(self):
        """Check available memory."""
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
            
            mem_total = 0
            mem_available = 0
            
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1])  # kB
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1])  # kB
            
            if mem_total > 0:
                used_pct = int(100 * (mem_total - mem_available) / mem_total)
                available_mb = mem_available // 1024
                
                if used_pct > 90:
                    status = HealthStatus.CRITICAL
                    msg = f"Memory {used_pct}% used (only {available_mb}MB available)"
                elif used_pct > 75:
                    status = HealthStatus.WARNING
                    msg = f"Memory {used_pct}% used ({available_mb}MB available)"
                else:
                    status = HealthStatus.HEALTHY
                    msg = f"{available_mb}MB available ({used_pct}% used)"
                
                self._add_result(
                    "Memory",
                    status,
                    msg,
                    {"used_percent": used_pct, "available_mb": available_mb}
                )
            else:
                self._add_result("Memory", HealthStatus.UNKNOWN, "Could not parse meminfo", {})
        except Exception as e:
            self._add_result("Memory", HealthStatus.UNKNOWN, f"Check failed: {str(e)}", {})
    
    def _check_cpu_temperature(self):
        """Check CPU temperature."""
        try:
            # Try Raspberry Pi thermal zone
            temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_file.exists():
                temp_millicelsius = int(temp_file.read_text().strip())
                temp_celsius = temp_millicelsius / 1000
                
                if temp_celsius > 80:
                    status = HealthStatus.CRITICAL
                    msg = f"CPU too hot: {temp_celsius:.1f}°C"
                elif temp_celsius > 70:
                    status = HealthStatus.WARNING
                    msg = f"CPU warm: {temp_celsius:.1f}°C"
                else:
                    status = HealthStatus.HEALTHY
                    msg = f"CPU temp: {temp_celsius:.1f}°C"
                
                self._add_result(
                    "CPU Temperature",
                    status,
                    msg,
                    {"temp_celsius": temp_celsius}
                )
            else:
                self._add_result("CPU Temperature", HealthStatus.UNKNOWN, "Sensor not available", {})
        except Exception as e:
            self._add_result("CPU Temperature", HealthStatus.UNKNOWN, f"Check failed: {str(e)}", {})
    
    def _check_network_connectivity(self):
        """Check network connectivity."""
        try:
            # Ping Google DNS (quick check)
            result = subprocess.run(['ping', '-c', '1', '-W', '2', '8.8.8.8'],
                                   capture_output=True, timeout=3)
            if result.returncode == 0:
                self._add_result(
                    "Network",
                    HealthStatus.HEALTHY,
                    "Connected",
                    {}
                )
            else:
                self._add_result(
                    "Network",
                    HealthStatus.WARNING,
                    "No internet connectivity",
                    {}
                )
        except Exception as e:
            self._add_result(
                "Network",
                HealthStatus.WARNING,
                "Connectivity check failed",
                {}
            )
    
    def _check_directory_structure(self):
        """Check required directories exist."""
        required_dirs = [
            "model",
            "rlds/episodes",
            "trainer/logs",
        ]
        
        missing = []
        for dir_name in required_dirs:
            if not (self.config_dir / dir_name).exists():
                missing.append(dir_name)
        
        if not missing:
            self._add_result(
                "Directory Structure",
                HealthStatus.HEALTHY,
                "All required directories exist",
                {}
            )
        else:
            self._add_result(
                "Directory Structure",
                HealthStatus.WARNING,
                f"Missing: {', '.join(missing)}",
                {"missing": missing}
            )
    
    def _check_permissions(self):
        """Check file permissions are correct."""
        import os
        
        # Check if we can write to config dir
        try:
            test_file = self.config_dir / ".health_check_test"
            test_file.write_text("test")
            test_file.unlink()
            
            self._add_result(
                "Permissions",
                HealthStatus.HEALTHY,
                "Write access to config directory",
                {}
            )
        except PermissionError:
            self._add_result(
                "Permissions",
                HealthStatus.CRITICAL,
                f"No write access to {self.config_dir}",
                {}
            )
        except Exception as e:
            self._add_result(
                "Permissions",
                HealthStatus.WARNING,
                f"Check failed: {str(e)}",
                {}
            )
    
    def _check_mission_statement(self):
        """Verify the mission statement guardrail is present for decision-making."""

        mission_path = Path(__file__).resolve().parent / "MISSION_STATEMENT.md"

        if not mission_path.exists():
            self._add_result(
                "Mission Statement",
                HealthStatus.CRITICAL,
                "Mission statement missing; cannot enforce humanity-first guardrail",
                {"path": str(mission_path), "present": False},
            )
            return

        try:
            mission_text = mission_path.read_text(encoding="utf-8").strip()
            if not mission_text:
                self._add_result(
                    "Mission Statement",
                    HealthStatus.CRITICAL,
                    "Mission statement file is empty; guardrail unavailable",
                    {"path": str(mission_path), "present": True, "headline": ""},
                )
                return

            headline = mission_text.splitlines()[0].strip()
            details = {
                "path": str(mission_path),
                "present": True,
                "headline": headline,
            }
            self._add_result(
                "Mission Statement",
                HealthStatus.HEALTHY,
                "Mission statement loaded; align autonomy to humanity-first, collective intelligence goals",
                details,
            )
        except Exception as exc:  # noqa: BLE001
            self._add_result(
                "Mission Statement",
                HealthStatus.WARNING,
                f"Could not read mission statement: {exc}",
                {"path": str(mission_path), "present": True},
            )

    def _check_safety_config(self):
        """Check safety configuration is present."""
        safety_manifest = self.config_dir / "model" / "manifest.pi5.safety.example.json"
        
        if safety_manifest.exists():
            try:
                with open(safety_manifest) as f:
                    config = json.load(f)
                
                has_safety = "safety" in config or "safety_head" in config
                
                if has_safety:
                    self._add_result(
                        "Safety Config",
                        HealthStatus.HEALTHY,
                        "Safety configuration present",
                        {}
                    )
                else:
                    self._add_result(
                        "Safety Config",
                        HealthStatus.WARNING,
                        "No safety configuration in manifest",
                        {}
                    )
            except Exception as e:
                self._add_result(
                    "Safety Config",
                    HealthStatus.WARNING,
                    f"Could not parse manifest: {str(e)}",
                    {}
                )
        else:
            self._add_result(
                "Safety Config",
                HealthStatus.WARNING,
                "No safety manifest found",
                {}
            )
    
    def _compute_overall_status(self) -> HealthStatus:
        """Compute overall system status from individual checks."""
        has_critical = any(r.status == HealthStatus.CRITICAL for r in self.results)
        has_warning = any(r.status == HealthStatus.WARNING for r in self.results)
        
        if has_critical:
            return HealthStatus.CRITICAL
        elif has_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _print_summary(self, overall_status: HealthStatus):
        """Print health check summary."""
        duration_ms = (self.end_time_ns - self.start_time_ns) / 1e6
        
        print("=" * 60)
        print("📋 Health Check Summary")
        print("=" * 60)
        
        # Count by status
        counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0,
        }
        
        for result in self.results:
            counts[result.status] += 1
        
        print(f"Total Checks: {len(self.results)}")
        print(f"  ✅ Healthy: {counts[HealthStatus.HEALTHY]}")
        print(f"  ⚠️  Warning: {counts[HealthStatus.WARNING]}")
        print(f"  ❌ Critical: {counts[HealthStatus.CRITICAL]}")
        print(f"  ❓ Unknown: {counts[HealthStatus.UNKNOWN]}")
        print()
        
        # Overall status
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️ ",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓",
        }[overall_status]
        
        print(f"Overall Status: {status_emoji} {overall_status.value.upper()}")
        print(f"Check Duration: {duration_ms:.1f}ms")
        print("=" * 60)
        print()
        
        # Action items
        if overall_status == HealthStatus.CRITICAL:
            print("⚠️  CRITICAL ISSUES DETECTED - System may not function properly!")
            print("\nAction required:")
            for result in self.results:
                if result.status == HealthStatus.CRITICAL:
                    print(f"  • Fix {result.component}: {result.message}")
            print()
        elif overall_status == HealthStatus.WARNING:
            print("⚠️  Warnings detected - System functional but degraded")
            print()
    
    def save_report(self, output_path: str = "/tmp/continuonbrain_health.json"):
        """Save health check report to JSON file."""
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "timestamp_ns": self.start_time_ns,
            "duration_ms": (self.end_time_ns - self.start_time_ns) / 1e6,
            "overall_status": self._compute_overall_status().value,
            "checks": [
                {
                    "component": r.component,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Health report saved to: {output_file}")
        return output_file


def main():
    """Run system health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ContinuonBrain System Health Check")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check (skip expensive tests)"
    )
    parser.add_argument(
        "--config-dir",
        default="/opt/continuonos/brain",
        help="ContinuonBrain config directory (default: /opt/continuonos/brain)"
    )
    parser.add_argument(
        "--save-report",
        help="Save report to file (default: /tmp/continuonbrain_health.json)"
    )
    
    args = parser.parse_args()
    
    # Run health checks
    checker = SystemHealthChecker(config_dir=args.config_dir)
    overall_status, results = checker.run_all_checks(quick_mode=args.quick)
    
    # Save report if requested
    if args.save_report:
        checker.save_report(args.save_report)
    
    # Exit with appropriate code
    if overall_status == HealthStatus.CRITICAL:
        exit(2)
    elif overall_status == HealthStatus.WARNING:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
