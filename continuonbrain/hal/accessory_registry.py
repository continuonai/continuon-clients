"""
Accessory Registry - Central database of known hardware accessories

Supports:
- SO-ARM100 robotic arm
- OAK-D cameras
- Hailo NPU
- Generic sensors (IMU, lidar, etc.)
- Custom accessories via plugin
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class AccessoryType(Enum):
    """Categories of accessories."""
    ROBOTIC_ARM = auto()
    CAMERA = auto()
    NPU = auto()
    SENSOR = auto()
    ACTUATOR = auto()
    DISPLAY = auto()
    AUDIO = auto()
    NETWORK = auto()
    STORAGE = auto()
    POWER = auto()
    UNKNOWN = auto()


@dataclass
class AccessoryCapability:
    """What an accessory can do."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class Accessory:
    """Represents a discovered or known accessory."""
    id: str
    name: str
    type: AccessoryType
    vendor: str = "unknown"
    model: str = "unknown"
    version: str = "1.0"
    
    # Connection info
    bus: str = ""  # usb, i2c, gpio, bluetooth, network
    address: str = ""  # /dev/ttyUSB0, 0x40, 192.168.1.x, etc.
    
    # Capabilities
    capabilities: List[AccessoryCapability] = field(default_factory=list)
    
    # State
    connected: bool = False
    initialized: bool = False
    last_seen: float = 0.0
    
    # Driver
    driver_module: Optional[str] = None
    driver_instance: Any = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.name,
            'vendor': self.vendor,
            'model': self.model,
            'version': self.version,
            'bus': self.bus,
            'address': self.address,
            'connected': self.connected,
            'capabilities': [
                {'name': c.name, 'description': c.description}
                for c in self.capabilities
            ],
        }


# ============= Known Accessory Database =============

KNOWN_ACCESSORIES = {
    # SO-ARM100 Robotic Arm (via USB serial)
    'so-arm100': {
        'name': 'SO-ARM100 Robotic Arm',
        'type': AccessoryType.ROBOTIC_ARM,
        'vendor': 'SO-Robotics',
        'model': 'ARM100',
        'usb_vid_pid': [('1a86', '7523'), ('0403', '6001')],  # CH340, FTDI
        'capabilities': [
            AccessoryCapability('move_joint', 'Move individual joint', {'joint': 'int', 'angle': 'float'}),
            AccessoryCapability('move_cartesian', 'Move to XYZ position', {'x': 'float', 'y': 'float', 'z': 'float'}),
            AccessoryCapability('grip', 'Control gripper', {'force': 'float'}),
            AccessoryCapability('home', 'Return to home position', {}),
            AccessoryCapability('get_state', 'Get current joint angles', {}),
        ],
        'driver_module': 'continuonbrain.hal.drivers.so_arm100',
    },
    
    # OAK-D Camera
    'oak-d': {
        'name': 'OAK-D Spatial AI Camera',
        'type': AccessoryType.CAMERA,
        'vendor': 'Luxonis',
        'model': 'OAK-D',
        'usb_vid_pid': [('03e7', '2485')],
        'capabilities': [
            AccessoryCapability('rgb_stream', 'RGB video stream', {'resolution': 'str', 'fps': 'int'}),
            AccessoryCapability('depth_stream', 'Depth map stream', {'resolution': 'str'}),
            AccessoryCapability('object_detection', 'On-device object detection', {'model': 'str'}),
            AccessoryCapability('spatial_detection', '3D object localization', {}),
        ],
        'driver_module': 'continuonbrain.hal.drivers.oak_d',
    },
    
    # Hailo-8 NPU
    'hailo-8': {
        'name': 'Hailo-8 NPU',
        'type': AccessoryType.NPU,
        'vendor': 'Hailo',
        'model': 'Hailo-8',
        'usb_vid_pid': [],
        'pcie_device': '1e60:2864',
        'capabilities': [
            AccessoryCapability('inference', 'Run neural network inference', {'model': 'str'}),
            AccessoryCapability('batch_inference', 'Batched inference', {'batch_size': 'int'}),
        ],
        'driver_module': 'continuonbrain.hal.drivers.hailo',
    },
    
    # Generic IMU (MPU6050, etc.)
    'imu-mpu6050': {
        'name': 'MPU6050 IMU',
        'type': AccessoryType.SENSOR,
        'vendor': 'InvenSense',
        'model': 'MPU6050',
        'i2c_address': [0x68, 0x69],
        'capabilities': [
            AccessoryCapability('accelerometer', 'Read acceleration XYZ', {}),
            AccessoryCapability('gyroscope', 'Read rotation rate XYZ', {}),
            AccessoryCapability('temperature', 'Read temperature', {}),
        ],
        'driver_module': 'continuonbrain.hal.drivers.mpu6050',
    },
    
    # Generic Servo Controller
    'pca9685': {
        'name': 'PCA9685 PWM Controller',
        'type': AccessoryType.ACTUATOR,
        'vendor': 'NXP',
        'model': 'PCA9685',
        'i2c_address': [0x40, 0x41, 0x42, 0x43],
        'capabilities': [
            AccessoryCapability('set_pwm', 'Set PWM for channel', {'channel': 'int', 'value': 'int'}),
            AccessoryCapability('set_servo', 'Set servo angle', {'channel': 'int', 'angle': 'float'}),
        ],
        'driver_module': 'continuonbrain.hal.drivers.pca9685',
    },
}


class AccessoryRegistry:
    """
    Central registry for all accessories.
    Supports discovery, registration, and capability queries.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._accessories: Dict[str, Accessory] = {}
        self._discovery_callbacks: List[Callable[[Accessory], None]] = []
        self._disconnect_callbacks: List[Callable[[str], None]] = []
        
        # Load known accessories database
        self._known_db = KNOWN_ACCESSORIES.copy()
        
        # Load custom accessories from config
        self._load_custom_accessories()
        
        self._initialized = True
        logger.info("AccessoryRegistry initialized")
    
    def _load_custom_accessories(self):
        """Load custom accessory definitions from config."""
        config_path = Path("/opt/continuonos/brain/config/accessories.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    custom = json.load(f)
                    for key, value in custom.items():
                        self._known_db[key] = value
                        logger.info(f"Loaded custom accessory: {key}")
            except Exception as e:
                logger.error(f"Failed to load custom accessories: {e}")
    
    def register(self, accessory: Accessory) -> None:
        """Register an accessory."""
        self._accessories[accessory.id] = accessory
        logger.info(f"Registered accessory: {accessory.name} ({accessory.id})")
        
        # Notify callbacks
        for callback in self._discovery_callbacks:
            try:
                callback(accessory)
            except Exception as e:
                logger.error(f"Discovery callback error: {e}")
    
    def unregister(self, accessory_id: str) -> None:
        """Unregister an accessory."""
        if accessory_id in self._accessories:
            del self._accessories[accessory_id]
            logger.info(f"Unregistered accessory: {accessory_id}")
            
            for callback in self._disconnect_callbacks:
                try:
                    callback(accessory_id)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")
    
    def get(self, accessory_id: str) -> Optional[Accessory]:
        """Get accessory by ID."""
        return self._accessories.get(accessory_id)
    
    def get_all(self) -> List[Accessory]:
        """Get all registered accessories."""
        return list(self._accessories.values())
    
    def get_by_type(self, accessory_type: AccessoryType) -> List[Accessory]:
        """Get accessories of a specific type."""
        return [a for a in self._accessories.values() if a.type == accessory_type]
    
    def get_connected(self) -> List[Accessory]:
        """Get all connected accessories."""
        return [a for a in self._accessories.values() if a.connected]
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all available capabilities grouped by accessory."""
        return {
            a.id: [c.name for c in a.capabilities]
            for a in self._accessories.values()
            if a.connected
        }
    
    def match_usb_device(self, vid: str, pid: str) -> Optional[str]:
        """Match USB VID:PID to known accessory."""
        for key, info in self._known_db.items():
            if 'usb_vid_pid' in info:
                for known_vid, known_pid in info['usb_vid_pid']:
                    if vid.lower() == known_vid.lower() and pid.lower() == known_pid.lower():
                        return key
        return None
    
    def match_i2c_address(self, address: int) -> List[str]:
        """Match I2C address to known accessories."""
        matches = []
        for key, info in self._known_db.items():
            if 'i2c_address' in info:
                if address in info['i2c_address']:
                    matches.append(key)
        return matches
    
    def create_from_known(self, known_key: str, bus: str, address: str) -> Optional[Accessory]:
        """Create an Accessory instance from the known database."""
        if known_key not in self._known_db:
            return None
            
        info = self._known_db[known_key]
        
        # Parse capabilities
        caps = []
        if 'capabilities' in info:
            for cap in info['capabilities']:
                if isinstance(cap, AccessoryCapability):
                    caps.append(cap)
                elif isinstance(cap, dict):
                    caps.append(AccessoryCapability(
                        name=cap.get('name', ''),
                        description=cap.get('description', ''),
                        parameters=cap.get('parameters', {})
                    ))
        
        accessory = Accessory(
            id=f"{known_key}_{address.replace('/', '_').replace(':', '_')}",
            name=info.get('name', known_key),
            type=info.get('type', AccessoryType.UNKNOWN),
            vendor=info.get('vendor', 'unknown'),
            model=info.get('model', 'unknown'),
            bus=bus,
            address=address,
            capabilities=caps,
            driver_module=info.get('driver_module'),
            connected=True,
        )
        
        return accessory
    
    def on_discovery(self, callback: Callable[[Accessory], None]) -> None:
        """Register callback for accessory discovery."""
        self._discovery_callbacks.append(callback)
    
    def on_disconnect(self, callback: Callable[[str], None]) -> None:
        """Register callback for accessory disconnect."""
        self._disconnect_callbacks.append(callback)
    
    def to_dict(self) -> Dict:
        """Export registry state."""
        return {
            'accessories': [a.to_dict() for a in self._accessories.values()],
            'known_types': [t.name for t in AccessoryType],
        }

