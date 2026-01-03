"""
Hardware Discovery Module

Scans for connected accessories via:
- USB (using pyudev or /sys/bus/usb)
- I2C (smbus scanning)
- GPIO (configuration-based)
- Bluetooth (optional, via bleak)
- Network (mDNS/Bonjour)
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess
import re

from .accessory_registry import AccessoryRegistry, Accessory, AccessoryType

logger = logging.getLogger(__name__)


def scan_usb() -> List[Tuple[str, str, str]]:
    """
    Scan for USB devices.
    Returns list of (vid, pid, device_path) tuples.
    """
    devices = []
    
    try:
        # Method 1: lsusb parsing
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split('\n'):
            # Bus 001 Device 004: ID 03e7:2485 Intel Corp. Movidius MyriadX
            match = re.search(r'ID ([0-9a-fA-F]{4}):([0-9a-fA-F]{4})', line)
            if match:
                vid, pid = match.groups()
                devices.append((vid, pid, line))
    except Exception as e:
        logger.warning(f"lsusb scan failed: {e}")
    
    try:
        # Method 2: /sys/bus/usb/devices
        usb_path = Path('/sys/bus/usb/devices')
        if usb_path.exists():
            for device_dir in usb_path.iterdir():
                vid_file = device_dir / 'idVendor'
                pid_file = device_dir / 'idProduct'
                
                if vid_file.exists() and pid_file.exists():
                    try:
                        vid = vid_file.read_text().strip()
                        pid = pid_file.read_text().strip()
                        
                        # Check for tty device
                        tty_path = None
                        for subdir in device_dir.glob('**/tty*'):
                            if subdir.is_dir():
                                for tty in subdir.iterdir():
                                    tty_path = f"/dev/{tty.name}"
                                    break
                        
                        # Only add if not already in list
                        if not any(v == vid and p == pid for v, p, _ in devices):
                            devices.append((vid, pid, tty_path or str(device_dir)))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"/sys scan failed: {e}")
    
    return devices


def scan_i2c(bus_number: int = 1) -> List[int]:
    """
    Scan I2C bus for devices.
    Returns list of addresses that responded.
    """
    addresses = []
    
    try:
        import smbus2
        bus = smbus2.SMBus(bus_number)
        
        for address in range(0x03, 0x78):
            try:
                bus.read_byte(address)
                addresses.append(address)
            except Exception:
                pass
        
        bus.close()
    except ImportError:
        logger.warning("smbus2 not installed, trying i2cdetect")
        
        try:
            result = subprocess.run(
                ['i2cdetect', '-y', str(bus_number)],
                capture_output=True, text=True, timeout=5
            )
            # Parse output: addresses shown as hex, -- for no device
            for line in result.stdout.split('\n')[1:]:
                parts = line.split(':')
                if len(parts) == 2:
                    for addr in parts[1].split():
                        if addr != '--' and addr != 'UU':
                            try:
                                addresses.append(int(addr, 16))
                            except ValueError:
                                pass
        except Exception as e:
            logger.warning(f"i2cdetect failed: {e}")
    except Exception as e:
        logger.warning(f"I2C scan failed: {e}")
    
    return addresses


def scan_gpio() -> Dict[int, str]:
    """
    Scan GPIO for configured devices.
    Returns dict of pin -> description.
    """
    gpio_config = {}
    
    # Check for GPIO configuration file
    config_path = Path("/opt/continuonos/brain/config/gpio.json")
    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                gpio_config = json.load(f)
        except Exception as e:
            logger.warning(f"GPIO config load failed: {e}")
    
    # Check for common GPIO configurations
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Common pins to check
        common_pins = [4, 17, 18, 22, 23, 24, 25, 27]
        for pin in common_pins:
            if pin not in gpio_config:
                try:
                    # Just note the pin exists
                    gpio_config[pin] = "available"
                except Exception:
                    pass
    except ImportError:
        logger.debug("RPi.GPIO not available")
    except Exception as e:
        logger.debug(f"GPIO check failed: {e}")
    
    return gpio_config


def scan_pcie() -> List[Tuple[str, str]]:
    """
    Scan for PCIe devices.
    Returns list of (vendor:device, description) tuples.
    """
    devices = []
    
    try:
        result = subprocess.run(['lspci', '-n'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().split('\n'):
            # 0000:01:00.0 1200: 1e60:2864 (rev 01)
            match = re.search(r'([0-9a-fA-F]{4}):([0-9a-fA-F]{4})', line)
            if match:
                vid, did = match.groups()
                devices.append((f"{vid}:{did}", line))
    except Exception as e:
        logger.warning(f"lspci scan failed: {e}")
    
    return devices


def scan_bluetooth() -> List[Dict]:
    """
    Scan for Bluetooth devices (if available).
    Returns list of device info dicts.
    """
    devices = []
    
    try:
        # Check if Bluetooth is available
        result = subprocess.run(['hciconfig'], capture_output=True, text=True, timeout=5)
        if 'UP' not in result.stdout:
            logger.debug("Bluetooth not enabled")
            return devices
        
        # Quick scan
        result = subprocess.run(
            ['timeout', '5', 'hcitool', 'scan'],
            capture_output=True, text=True, timeout=10
        )
        
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                devices.append({
                    'address': parts[0],
                    'name': parts[1] if len(parts) > 1 else 'Unknown',
                })
    except Exception as e:
        logger.debug(f"Bluetooth scan failed: {e}")
    
    return devices


def discover_accessories(
    scan_usb_devices: bool = True,
    scan_i2c_bus: bool = True,
    scan_gpio_pins: bool = False,
    scan_pcie_devices: bool = True,
    scan_bluetooth_devices: bool = False,
) -> List[Accessory]:
    """
    Discover all connected accessories.
    Returns list of discovered Accessory objects.
    """
    registry = AccessoryRegistry()
    discovered = []
    
    logger.info("Starting accessory discovery...")
    
    # USB scan
    if scan_usb_devices:
        logger.debug("Scanning USB devices...")
        usb_devices = scan_usb()
        
        for vid, pid, path in usb_devices:
            known_key = registry.match_usb_device(vid, pid)
            if known_key:
                accessory = registry.create_from_known(known_key, 'usb', path)
                if accessory:
                    registry.register(accessory)
                    discovered.append(accessory)
                    logger.info(f"Discovered USB: {accessory.name} at {path}")
    
    # I2C scan
    if scan_i2c_bus:
        logger.debug("Scanning I2C bus...")
        for bus in [1, 0]:  # Try bus 1 first (common on Pi)
            try:
                i2c_devices = scan_i2c(bus)
                
                for address in i2c_devices:
                    known_keys = registry.match_i2c_address(address)
                    for known_key in known_keys:
                        accessory = registry.create_from_known(
                            known_key, 'i2c', f"{bus}:0x{address:02x}"
                        )
                        if accessory:
                            registry.register(accessory)
                            discovered.append(accessory)
                            logger.info(f"Discovered I2C: {accessory.name} at 0x{address:02x}")
                break  # Found a working bus
            except Exception:
                continue
    
    # PCIe scan (for Hailo, etc.)
    if scan_pcie_devices:
        logger.debug("Scanning PCIe devices...")
        pcie_devices = scan_pcie()
        
        for dev_id, desc in pcie_devices:
            # Check for Hailo
            if '1e60:2864' in dev_id:
                accessory = registry.create_from_known('hailo-8', 'pcie', dev_id)
                if accessory:
                    registry.register(accessory)
                    discovered.append(accessory)
                    logger.info(f"Discovered PCIe: {accessory.name}")
    
    # Bluetooth scan (optional, slow)
    if scan_bluetooth_devices:
        logger.debug("Scanning Bluetooth devices...")
        bt_devices = scan_bluetooth()
        
        for bt_dev in bt_devices:
            # Create generic Bluetooth accessory
            accessory = Accessory(
                id=f"bt_{bt_dev['address'].replace(':', '')}",
                name=bt_dev.get('name', 'Bluetooth Device'),
                type=AccessoryType.UNKNOWN,
                bus='bluetooth',
                address=bt_dev['address'],
                connected=True,
            )
            registry.register(accessory)
            discovered.append(accessory)
            logger.info(f"Discovered Bluetooth: {bt_dev.get('name')}")
    
    logger.info(f"Discovery complete: {len(discovered)} accessories found")
    return discovered


def get_system_info() -> Dict:
    """Get system hardware info."""
    info = {
        'platform': 'unknown',
        'model': 'unknown',
        'memory_gb': 0,
        'cpu_cores': 0,
    }
    
    try:
        # Platform
        with open('/proc/device-tree/model', 'r') as f:
            info['model'] = f.read().strip().rstrip('\x00')
            if 'Raspberry Pi' in info['model']:
                info['platform'] = 'raspberry_pi'
            elif 'Jetson' in info['model']:
                info['platform'] = 'jetson'
    except Exception:
        pass
    
    try:
        # Memory
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    info['memory_gb'] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    
    try:
        # CPU cores
        info['cpu_cores'] = os.cpu_count() or 1
    except Exception:
        pass
    
    return info

