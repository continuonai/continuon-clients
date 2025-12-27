"""
Tests for StartupManager refactoring and HardwareDetector cross-platform/mock behavior.
"""
import pytest
import os
import platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from continuonbrain.startup_manager import StartupManager, ServiceRegistry, ServiceDefinition
from continuonbrain.sensors.hardware_detector import HardwareDetector, HardwareDevice


def test_service_definition_init():
    """Verify ServiceDefinition defaults."""
    svc = ServiceDefinition(name="Test", module="test.module")
    assert svc.name == "Test"
    assert svc.enabled is True
    assert svc.priority == 100
    assert svc.args == []


def test_service_registry_registration():
    """Verify ServiceRegistry collects and sorts services."""
    registry = ServiceRegistry(Path("/tmp"), Path("/repo"), "python", {{}})
    
    s1 = ServiceDefinition(name="S1", module="m1", priority=20)
    s2 = ServiceDefinition(name="S2", module="m2", priority=10)
    s3 = ServiceDefinition(name="S3", module="m3", enabled=False)
    
    registry.register(s1)
    registry.register(s2)
    registry.register(s3)
    
    enabled = registry.get_enabled_services()
    assert len(enabled) == 2
    assert enabled[0].name == "S2"  # Priority 10
    assert enabled[1].name == "S1"  # Priority 20


@patch("subprocess.Popen")
def test_service_registry_start_service(mock_popen):
    """Verify registry starts a subprocess correctly."""
    mock_popen.return_value = MagicMock(pid=1234)
    
    registry = ServiceRegistry(Path("/tmp"), Path("/repo"), "python", {{"BASE": "1"}})
    svc = ServiceDefinition(name="TestSvc", module="test.mod", args=["--flag"], env_vars={{"EXTRA": "2"}})
    
    proc = registry.start_service(svc)
    
    assert proc is not None
    assert proc.pid == 1234
    
    # Check popen call
    args, kwargs = mock_popen.call_args
    assert args[0] == ["python", "-m", "test.mod", "--flag"]
    assert kwargs["env"]["BASE"] == "1"
    assert kwargs["env"]["EXTRA"] == "2"


def test_hardware_detector_mock_injection():
    """Verify HardwareDetector injects mocks on non-Linux platforms or when forced."""
    detector = HardwareDetector()
    detector.use_mock = True
    
    # Force detection run
    devices = detector.detect_all()
    
    # Check for mock devices
    mock_names = [d.name for d in devices if d.is_mock]
    assert "MOCK OAK-D" in mock_names
    assert "MOCK PCA9685" in mock_names
    
    config = detector.generate_config()
    assert config["primary"]["depth_camera_is_mock"] is True


@patch("continuonbrain.startup_manager.ServiceRegistry.start_service")
def test_startup_manager_starts_services(mock_start):
    """Verify StartupManager uses registry to start services."""
    manager = StartupManager(config_dir="/tmp/test_brain", start_services=True)
    
    # Mock some methods to avoid side effects
    manager._load_boot_protocols = MagicMock()
    manager._check_battery_lvc = MagicMock(return_value=True)
    manager._record_startup = MagicMock()
    manager._log_event = MagicMock()
    manager.launch_ui = MagicMock()
    
    # Mock LANDiscovery and ModeManager
    with patch("continuonbrain.startup_manager.LANDiscoveryService"), \
         patch("continuonbrain.startup_manager.RobotModeManager"):
        
        success = manager.startup()
        
        assert success is True
        # Registry should have been used to start at least Safety Kernel and API Server
        assert mock_start.call_count >= 2
        
        # Check that registry was stored
        assert manager._service_registry is not None
