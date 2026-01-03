"""
Robot Builder - Autonomous Robot Construction

Enables a robot to build new robots from available parts.

Workflow:
1. Owner provides inventory of available parts
2. Robot analyzes parts and proposes build plans
3. Owner approves a build plan
4. Robot executes construction:
   - Assembles chassis (Lego, 3D printed, etc.)
   - Installs compute (Pi 5, Jetson, etc.)
   - Clones seed image
   - Wires sensors/actuators
   - Tests and calibrates
5. New robot is paired to owner

Safety:
- All construction requires owner authorization
- Parts must be owned by the owner
- Construction in designated safe workspace only
- Human supervision recommended for first builds
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum, auto
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class PartCategory(Enum):
    """Categories of robot parts."""
    # Compute
    COMPUTE_SBC = "compute_sbc"         # Single-board computers (Pi, Jetson)
    COMPUTE_MCU = "compute_mcu"         # Microcontrollers (Arduino, ESP32)
    NPU = "npu"                         # Neural processing units (Hailo, Coral)
    
    # Power
    POWER_BATTERY = "power_battery"
    POWER_SUPPLY = "power_supply"
    POWER_REGULATOR = "power_regulator"
    
    # Actuation
    ACTUATOR_SERVO = "actuator_servo"
    ACTUATOR_MOTOR = "actuator_motor"
    ACTUATOR_STEPPER = "actuator_stepper"
    ACTUATOR_LINEAR = "actuator_linear"
    
    # Sensing
    SENSOR_CAMERA = "sensor_camera"
    SENSOR_DEPTH = "sensor_depth"
    SENSOR_LIDAR = "sensor_lidar"
    SENSOR_IMU = "sensor_imu"
    SENSOR_TOUCH = "sensor_touch"
    SENSOR_AUDIO = "sensor_audio"
    
    # Structure
    STRUCTURE_LEGO = "structure_lego"
    STRUCTURE_3DPRINT = "structure_3dprint"
    STRUCTURE_ALUMINUM = "structure_aluminum"
    STRUCTURE_ACRYLIC = "structure_acrylic"
    
    # Connectivity
    CONNECTIVITY_WIFI = "connectivity_wifi"
    CONNECTIVITY_BT = "connectivity_bluetooth"
    CONNECTIVITY_LORA = "connectivity_lora"
    
    # Storage
    STORAGE_SD = "storage_sd"
    STORAGE_SSD = "storage_ssd"
    STORAGE_EMMC = "storage_emmc"
    
    # Misc
    WIRING = "wiring"
    FASTENERS = "fasteners"
    COOLING = "cooling"


@dataclass
class Part:
    """A single part available for robot construction."""
    part_id: str
    name: str
    category: str
    quantity: int = 1
    specifications: Dict[str, Any] = field(default_factory=dict)
    location: str = ""  # Where the part is stored
    condition: str = "new"  # new, used, refurbished
    compatible_with: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Part':
        return cls(**data)


@dataclass
class PartInventory:
    """Inventory of available parts for robot construction."""
    owner_id: str
    parts: List[Part] = field(default_factory=list)
    last_updated: str = ""
    
    def add_part(self, part: Part):
        """Add a part to inventory."""
        self.parts.append(part)
        self.last_updated = datetime.now().isoformat()
    
    def find_parts(self, category: PartCategory) -> List[Part]:
        """Find all parts in a category."""
        return [p for p in self.parts if p.category == category.value]
    
    def has_minimum_for_robot(self) -> Tuple[bool, List[str]]:
        """Check if inventory has minimum parts for a robot."""
        missing = []
        
        # Minimum requirements
        if not self.find_parts(PartCategory.COMPUTE_SBC):
            missing.append("Single-board computer (e.g., Raspberry Pi 5)")
        if not self.find_parts(PartCategory.STORAGE_SD) and not self.find_parts(PartCategory.STORAGE_SSD):
            missing.append("Storage (SD card or SSD)")
        if not self.find_parts(PartCategory.POWER_SUPPLY) and not self.find_parts(PartCategory.POWER_BATTERY):
            missing.append("Power source")
        
        return len(missing) == 0, missing
    
    def to_dict(self) -> Dict:
        return {
            'owner_id': self.owner_id,
            'parts': [p.to_dict() for p in self.parts],
            'last_updated': self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PartInventory':
        inv = cls(owner_id=data['owner_id'])
        inv.parts = [Part.from_dict(p) for p in data.get('parts', [])]
        inv.last_updated = data.get('last_updated', '')
        return inv


class RobotArchetype(Enum):
    """Pre-defined robot archetypes."""
    STATIONARY_ASSISTANT = "stationary_assistant"  # Desk robot, no mobility
    WHEELED_ROVER = "wheeled_rover"                # Mobile platform on wheels
    TRACKED_ROVER = "tracked_rover"                # Tank-style tracks
    ARM_MANIPULATOR = "arm_manipulator"            # Fixed arm (like SO-ARM100)
    MOBILE_MANIPULATOR = "mobile_manipulator"      # Arm on mobile base
    DRONE = "drone"                                # Flying robot
    CUSTOM = "custom"                              # User-defined


@dataclass
class BuildStep:
    """A single step in the construction process."""
    step_id: str
    order: int
    description: str
    action_type: str  # "assemble", "connect", "install", "configure", "test"
    parts_needed: List[str]  # Part IDs
    estimated_time_minutes: int = 5
    requires_human: bool = False
    safety_notes: str = ""
    verification: str = ""  # How to verify step completed correctly


@dataclass
class BuildPlan:
    """Complete plan for building a robot."""
    plan_id: str
    name: str
    archetype: str
    description: str
    
    # Parts
    required_parts: List[str]  # Part IDs from inventory
    optional_parts: List[str] = field(default_factory=list)
    
    # Steps
    steps: List[BuildStep] = field(default_factory=list)
    
    # Estimates
    estimated_total_time_minutes: int = 60
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    
    # Status
    status: str = "draft"  # draft, approved, in_progress, completed, failed
    created_at: str = ""
    approved_at: str = ""
    approved_by: str = ""
    
    # Result
    resulting_robot_id: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['steps'] = [asdict(s) for s in self.steps]
        return d


class RobotBuilder:
    """
    Autonomous robot construction system.
    
    Enables a robot to build new robots from available parts.
    """
    
    # Known compute platforms and their capabilities
    KNOWN_COMPUTE = {
        "raspberry_pi_5": {
            "ram_gb": 8,
            "has_gpio": True,
            "has_pcie": True,
            "supports_hailo": True,
            "power_watts": 5,
        },
        "raspberry_pi_4": {
            "ram_gb": 4,
            "has_gpio": True,
            "has_pcie": False,
            "supports_hailo": False,
            "power_watts": 3,
        },
        "jetson_nano": {
            "ram_gb": 4,
            "has_gpio": True,
            "has_gpu": True,
            "power_watts": 10,
        },
        "jetson_orin_nano": {
            "ram_gb": 8,
            "has_gpio": True,
            "has_gpu": True,
            "power_watts": 15,
        },
    }
    
    def __init__(self, data_dir: Path = Path("/opt/continuonos/brain/swarm/builds")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._inventories: Dict[str, PartInventory] = {}
        self._plans: Dict[str, BuildPlan] = {}
        self._load_data()
    
    def _load_data(self):
        """Load saved inventories and plans."""
        inv_file = self.data_dir / "inventories.json"
        if inv_file.exists():
            try:
                with open(inv_file) as f:
                    data = json.load(f)
                    for owner_id, inv_data in data.items():
                        self._inventories[owner_id] = PartInventory.from_dict(inv_data)
            except Exception as e:
                logger.error(f"Failed to load inventories: {e}")
        
        plans_file = self.data_dir / "plans.json"
        if plans_file.exists():
            try:
                with open(plans_file) as f:
                    data = json.load(f)
                    for plan_id, plan_data in data.items():
                        self._plans[plan_id] = BuildPlan(**{
                            k: v for k, v in plan_data.items() if k != 'steps'
                        })
                        self._plans[plan_id].steps = [
                            BuildStep(**s) for s in plan_data.get('steps', [])
                        ]
            except Exception as e:
                logger.error(f"Failed to load plans: {e}")
    
    def _save_data(self):
        """Save inventories and plans."""
        inv_file = self.data_dir / "inventories.json"
        with open(inv_file, 'w') as f:
            json.dump({
                owner_id: inv.to_dict() 
                for owner_id, inv in self._inventories.items()
            }, f, indent=2)
        
        plans_file = self.data_dir / "plans.json"
        with open(plans_file, 'w') as f:
            json.dump({
                plan_id: plan.to_dict() 
                for plan_id, plan in self._plans.items()
            }, f, indent=2)
    
    def register_inventory(self, owner_id: str, parts: List[Part]) -> PartInventory:
        """Register an owner's parts inventory."""
        inv = PartInventory(owner_id=owner_id, parts=parts)
        inv.last_updated = datetime.now().isoformat()
        self._inventories[owner_id] = inv
        self._save_data()
        logger.info(f"Registered inventory for {owner_id} with {len(parts)} parts")
        return inv
    
    def get_inventory(self, owner_id: str) -> Optional[PartInventory]:
        """Get an owner's parts inventory."""
        return self._inventories.get(owner_id)
    
    def analyze_parts(self, owner_id: str) -> Dict[str, Any]:
        """
        Analyze available parts and suggest possible robot builds.
        
        Returns analysis with:
        - Can build a robot? (yes/no)
        - What's missing (if any)
        - Suggested archetypes
        - Capabilities of resulting robot
        """
        inv = self._inventories.get(owner_id)
        if not inv:
            return {'error': 'No inventory registered for owner'}
        
        can_build, missing = inv.has_minimum_for_robot()
        
        analysis = {
            'owner_id': owner_id,
            'total_parts': len(inv.parts),
            'can_build_robot': can_build,
            'missing_parts': missing,
            'suggested_archetypes': [],
            'capabilities': [],
        }
        
        if not can_build:
            return analysis
        
        # Analyze capabilities
        compute_parts = inv.find_parts(PartCategory.COMPUTE_SBC)
        camera_parts = inv.find_parts(PartCategory.SENSOR_CAMERA)
        motor_parts = inv.find_parts(PartCategory.ACTUATOR_MOTOR) + inv.find_parts(PartCategory.ACTUATOR_SERVO)
        arm_parts = inv.find_parts(PartCategory.ACTUATOR_SERVO)
        
        if camera_parts:
            analysis['capabilities'].append('vision')
        if motor_parts:
            analysis['capabilities'].append('mobility')
        if len(arm_parts) >= 4:  # Enough servos for an arm
            analysis['capabilities'].append('manipulation')
        if inv.find_parts(PartCategory.NPU):
            analysis['capabilities'].append('edge_ai')
        if inv.find_parts(PartCategory.SENSOR_AUDIO):
            analysis['capabilities'].append('audio')
        
        # Suggest archetypes based on parts
        if 'mobility' in analysis['capabilities'] and 'manipulation' in analysis['capabilities']:
            analysis['suggested_archetypes'].append(RobotArchetype.MOBILE_MANIPULATOR.value)
        elif 'mobility' in analysis['capabilities']:
            analysis['suggested_archetypes'].append(RobotArchetype.WHEELED_ROVER.value)
        elif 'manipulation' in analysis['capabilities']:
            analysis['suggested_archetypes'].append(RobotArchetype.ARM_MANIPULATOR.value)
        else:
            analysis['suggested_archetypes'].append(RobotArchetype.STATIONARY_ASSISTANT.value)
        
        return analysis
    
    def generate_build_plan(
        self,
        owner_id: str,
        archetype: RobotArchetype,
        name: str = "New Robot",
    ) -> Optional[BuildPlan]:
        """
        Generate a build plan for a robot.
        
        Returns a draft plan that must be approved by owner.
        """
        inv = self._inventories.get(owner_id)
        if not inv:
            logger.error(f"No inventory for owner {owner_id}")
            return None
        
        # Find suitable parts
        compute = inv.find_parts(PartCategory.COMPUTE_SBC)
        storage = inv.find_parts(PartCategory.STORAGE_SD) or inv.find_parts(PartCategory.STORAGE_SSD)
        power = inv.find_parts(PartCategory.POWER_SUPPLY) or inv.find_parts(PartCategory.POWER_BATTERY)
        structure = (
            inv.find_parts(PartCategory.STRUCTURE_LEGO) or
            inv.find_parts(PartCategory.STRUCTURE_3DPRINT) or
            inv.find_parts(PartCategory.STRUCTURE_ALUMINUM)
        )
        
        if not (compute and storage and power):
            logger.error("Missing essential parts")
            return None
        
        # Create build plan
        plan = BuildPlan(
            plan_id=str(uuid.uuid4()),
            name=name,
            archetype=archetype.value,
            description=f"Build a {archetype.value} robot using available parts",
            required_parts=[compute[0].part_id, storage[0].part_id, power[0].part_id],
            created_at=datetime.now().isoformat(),
            status="draft",
        )
        
        # Generate steps based on archetype
        steps = self._generate_steps(archetype, inv)
        plan.steps = steps
        plan.estimated_total_time_minutes = sum(s.estimated_time_minutes for s in steps)
        
        self._plans[plan.plan_id] = plan
        self._save_data()
        
        logger.info(f"Generated build plan {plan.plan_id} for {name}")
        return plan
    
    def _generate_steps(self, archetype: RobotArchetype, inv: PartInventory) -> List[BuildStep]:
        """Generate construction steps for an archetype."""
        steps = []
        order = 1
        
        # Step 1: Prepare workspace
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Clear and prepare workspace. Ensure good lighting and ventilation.",
            action_type="prepare",
            parts_needed=[],
            estimated_time_minutes=5,
            requires_human=True,
            safety_notes="Ensure workspace is clear of obstacles and hazards",
        ))
        order += 1
        
        # Step 2: Clone SD card with seed image
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Clone ContinuonBrain seed image to storage device",
            action_type="install",
            parts_needed=[p.part_id for p in inv.find_parts(PartCategory.STORAGE_SD)[:1]],
            estimated_time_minutes=15,
            requires_human=False,
            safety_notes="Ensure source image is verified and up-to-date",
            verification="Boot test: LED blinks green 3 times",
        ))
        order += 1
        
        # Step 3: Assemble base structure
        structure_parts = (
            inv.find_parts(PartCategory.STRUCTURE_LEGO) or
            inv.find_parts(PartCategory.STRUCTURE_3DPRINT)
        )
        if structure_parts:
            steps.append(BuildStep(
                step_id=str(uuid.uuid4()),
                order=order,
                description="Assemble base chassis/structure",
                action_type="assemble",
                parts_needed=[p.part_id for p in structure_parts],
                estimated_time_minutes=20,
                requires_human=False,
                safety_notes="Ensure structure is stable before adding components",
                verification="Structure supports weight of compute unit",
            ))
            order += 1
        
        # Step 4: Mount compute unit
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Mount single-board computer to chassis",
            action_type="assemble",
            parts_needed=[p.part_id for p in inv.find_parts(PartCategory.COMPUTE_SBC)[:1]],
            estimated_time_minutes=10,
            requires_human=False,
            safety_notes="Use standoffs to prevent shorts. Handle by edges.",
            verification="Compute unit is securely mounted",
        ))
        order += 1
        
        # Step 5: Install storage
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Insert cloned storage device into compute unit",
            action_type="install",
            parts_needed=[],
            estimated_time_minutes=2,
            requires_human=False,
            verification="Storage clicks into place",
        ))
        order += 1
        
        # Step 6: Connect power
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Connect power supply/battery to compute unit",
            action_type="connect",
            parts_needed=[p.part_id for p in (inv.find_parts(PartCategory.POWER_SUPPLY) or inv.find_parts(PartCategory.POWER_BATTERY))[:1]],
            estimated_time_minutes=5,
            requires_human=True,  # Power connections need human verification
            safety_notes="Double-check polarity before connecting!",
            verification="Power LED illuminates",
        ))
        order += 1
        
        # Archetype-specific steps
        if archetype in [RobotArchetype.WHEELED_ROVER, RobotArchetype.MOBILE_MANIPULATOR]:
            motors = inv.find_parts(PartCategory.ACTUATOR_MOTOR)
            if motors:
                steps.append(BuildStep(
                    step_id=str(uuid.uuid4()),
                    order=order,
                    description="Mount and wire drive motors",
                    action_type="connect",
                    parts_needed=[p.part_id for p in motors],
                    estimated_time_minutes=15,
                    requires_human=False,
                    verification="Motors respond to test commands",
                ))
                order += 1
        
        if archetype in [RobotArchetype.ARM_MANIPULATOR, RobotArchetype.MOBILE_MANIPULATOR]:
            servos = inv.find_parts(PartCategory.ACTUATOR_SERVO)
            if servos:
                steps.append(BuildStep(
                    step_id=str(uuid.uuid4()),
                    order=order,
                    description="Assemble and mount robot arm",
                    action_type="assemble",
                    parts_needed=[p.part_id for p in servos],
                    estimated_time_minutes=30,
                    requires_human=False,
                    safety_notes="Keep fingers clear of pinch points during assembly",
                    verification="All joints move through full range",
                ))
                order += 1
        
        # Camera if available
        cameras = inv.find_parts(PartCategory.SENSOR_CAMERA)
        if cameras:
            steps.append(BuildStep(
                step_id=str(uuid.uuid4()),
                order=order,
                description="Mount and connect camera",
                action_type="connect",
                parts_needed=[p.part_id for p in cameras[:1]],
                estimated_time_minutes=10,
                requires_human=False,
                verification="Camera image visible in test mode",
            ))
            order += 1
        
        # NPU if available
        npus = inv.find_parts(PartCategory.NPU)
        if npus:
            steps.append(BuildStep(
                step_id=str(uuid.uuid4()),
                order=order,
                description="Install neural processing unit (NPU)",
                action_type="install",
                parts_needed=[p.part_id for p in npus[:1]],
                estimated_time_minutes=10,
                requires_human=False,
                verification="NPU detected by system",
            ))
            order += 1
        
        # Final steps: Boot and pair
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Power on and perform initial boot",
            action_type="test",
            parts_needed=[],
            estimated_time_minutes=5,
            requires_human=True,
            verification="System boots to HOPE agent manager",
        ))
        order += 1
        
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Pair new robot to owner using QR code",
            action_type="configure",
            parts_needed=[],
            estimated_time_minutes=5,
            requires_human=True,
            verification="Owner displayed in robot's owner list",
        ))
        order += 1
        
        steps.append(BuildStep(
            step_id=str(uuid.uuid4()),
            order=order,
            description="Run self-test and calibration",
            action_type="test",
            parts_needed=[],
            estimated_time_minutes=10,
            requires_human=False,
            verification="All self-tests pass",
        ))
        
        return steps
    
    def approve_plan(
        self,
        plan_id: str,
        approver_id: str,
        approver_role: str,
    ) -> Tuple[bool, str]:
        """
        Approve a build plan.
        
        Only owner/creator can approve.
        """
        if plan_id not in self._plans:
            return False, "Plan not found"
        
        plan = self._plans[plan_id]
        
        if approver_role not in ["creator", "owner"]:
            return False, f"Role '{approver_role}' cannot approve build plans"
        
        plan.status = "approved"
        plan.approved_at = datetime.now().isoformat()
        plan.approved_by = approver_id
        
        self._save_data()
        logger.info(f"Build plan {plan_id} approved by {approver_id}")
        
        return True, "Plan approved"
    
    def get_plan(self, plan_id: str) -> Optional[BuildPlan]:
        """Get a build plan by ID."""
        return self._plans.get(plan_id)
    
    def list_plans(self, owner_id: Optional[str] = None) -> List[BuildPlan]:
        """List all build plans, optionally filtered by owner."""
        plans = list(self._plans.values())
        # In a real implementation, we'd filter by owner
        return plans


# Example inventory for testing
def create_example_inventory() -> List[Part]:
    """Create an example parts inventory."""
    return [
        Part(
            part_id="pi5_001",
            name="Raspberry Pi 5 8GB",
            category=PartCategory.COMPUTE_SBC.value,
            specifications={"ram_gb": 8, "cpu": "BCM2712"},
        ),
        Part(
            part_id="sd_001",
            name="Samsung EVO 64GB microSD",
            category=PartCategory.STORAGE_SD.value,
            specifications={"capacity_gb": 64, "speed_class": "A2"},
        ),
        Part(
            part_id="psu_001",
            name="Official Pi 5 27W USB-C PSU",
            category=PartCategory.POWER_SUPPLY.value,
            specifications={"watts": 27, "voltage": 5},
        ),
        Part(
            part_id="hailo8_001",
            name="Hailo-8 M.2 AI Accelerator",
            category=PartCategory.NPU.value,
            specifications={"tops": 26},
        ),
        Part(
            part_id="oakd_001",
            name="OAK-D Lite Camera",
            category=PartCategory.SENSOR_CAMERA.value,
            specifications={"depth": True, "resolution": "4K"},
        ),
        Part(
            part_id="lego_001",
            name="Lego Technic Parts Box",
            category=PartCategory.STRUCTURE_LEGO.value,
            quantity=200,
        ),
        Part(
            part_id="servo_001",
            name="STS3215 Serial Servo",
            category=PartCategory.ACTUATOR_SERVO.value,
            quantity=6,
            specifications={"torque_kg_cm": 35},
        ),
    ]

