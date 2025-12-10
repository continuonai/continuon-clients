"""
Agent Identity & Introspection Module.
Allows the robot to understand what it is, what it knows, and what it is made of.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from continuonbrain.sensors.hardware_detector import HardwareDetector
from continuonbrain.system_instructions import SystemInstructions
from continuonbrain.actuators.drivetrain_controller import DrivetrainConfig

@dataclass
class MemoryStats:
    local_episodes: int
    has_cloud_model: bool
    model_path: str
    rlds_path: str

@dataclass
class BodyStats:
    actuators: List[str]
    sensors: List[str]
    ai_accelerator: Optional[str]
    drivetrain_config: Dict

@dataclass
class DesignStats:
    name: str
    mission: str
    safety_rules: int
    instructions: int

class AgentIdentity:
    """
    Represents the robot's self-concept.
    """
    def __init__(self, config_dir: str = "/opt/continuonos/brain"):
        self.config_dir = Path(config_dir)
        # Fallback resolve for dev environment
        self.repo_root = Path(__file__).resolve().parent.parent
        self.hardware_detector = HardwareDetector()
        
        # Internal State Dictionary (The "Soul")
        self.identity = {
            "memory": {},
            "body": {},
            "design": {},
            "shell": {}
        }
    
    def _check_memories(self):
        """Check available memories (data) and trained skills (models)."""
        rlds_dir = self.config_dir / "rlds" / "episodes"
        model_dir = self.config_dir / "model"
        
        episode_count = 0
        if rlds_dir.exists():
            episode_count = len(list(rlds_dir.glob("*/episode.json")))
            
        has_base = (model_dir / "base_model").exists()
        has_manifest = len(list(model_dir.glob("manifest*.json"))) > 0
        
        self.identity["memory"] = {
            "local_episodes": episode_count,
            "has_cloud_model": has_base or has_manifest,
            "model_path": str(model_dir),
            "rlds_path": str(rlds_dir)
        }

    def _check_body(self):
        """Check physical components (actuators, sensors, wiring)."""
        devices = self.hardware_detector.detect_all()
        
        actuators = [d.name for d in devices if "servo" in d.device_type or "actuator" in d.device_type]
        sensors = [d.name for d in devices if "camera" in d.device_type or "sensor" in d.device_type]
        ai_devs = [d.name for d in devices if "accelerator" in d.device_type]
        
        # Check Drivetrain specifically
        dt_config = DrivetrainConfig.from_sources()
        dt_info = {
            "steering_channel": dt_config.steering_channel,
            "throttle_channel": dt_config.throttle_channel,
            "i2c_address": hex(dt_config.i2c_address)
        }
        
        self.identity["body"] = {
            "actuators": actuators,
            "sensors": sensors,
            "ai_accelerator": ai_devs[0] if ai_devs else None,
            "drivetrain_config": dt_info
        }

    def _check_design(self):
        """Introspect internal directives and mission."""
        self.identity["design"] = {}
        # 1. Look for Mission Statement
        mission_path = self.repo_root / "MISSION_STATEMENT.md"
        brain_mission_path = self.repo_root / "continuonbrain" / "MISSION_STATEMENT.md" # Fallback
        
        if mission_path.exists():
            self.identity["design"]["mission_statement"] = mission_path.read_text(encoding="utf-8", errors="ignore").strip()
        elif brain_mission_path.exists():
            self.identity["design"]["mission_statement"] = brain_mission_path.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            self.identity["design"]["mission_statement"] = "[NO MISSION STATEMENT FOUND]"

        # 2. Look for System Instructions (Constitution)
        instructions_path = Path(self.config_dir) / "system_instructions.json"
        
        # Defaults
        self.identity["design"]["core_directives"] = {
            "safety_rules_count": 0,
            "instructions_count": 0
        }
            
        if instructions_path.exists():
            try:
                data = json.loads(instructions_path.read_text())
                rules = data.get("safety_rules", [])
                insts = data.get("instructions", [])
                self.identity["design"]["core_directives"] = {
                    "safety_rules_count": len(rules),
                    "instructions_count": len(insts)
                }
            except Exception:
                pass

    def _check_shell(self):
        """Identify the host shell (Hardware/OS context)."""
        import platform
        uname = platform.uname()
        system_os = uname.system
        machine = uname.machine
        
        # Heuristics for Shell Type
        shell_type = "Generic Station"
        
        if system_os == "Linux" and ("aarch64" in machine or "arm" in machine):
            # Likely a Pi or Jetson (but could be Mac M1/M2 Linux VM, check vendor?)
            shell_type = "Physical Robot (ARM)"
        elif system_os == "Linux":
            shell_type = "Linux Workstation"
        elif system_os == "Windows":
            shell_type = "Windows Desktop"
        elif system_os == "Darwin":
            shell_type = "MacOS Station"
            
        self.identity["shell"] = {
            "type": shell_type,
            "os": system_os,
            "arch": machine,
            "hostname": uname.node
        }

    def self_report(self) -> None:
        """Generate and print the narrative self-report."""
        self._check_memories()
        self._check_body()
        self._check_design()
        self._check_shell()

        print("\n" + "="*60)
        print("🤖 AGENT SELF-ACTIVATION REPORT")
        print("="*60 + "\n")

        # Identity / Shell
        shell = self.identity.get("shell", {})
        print(f"🏠 SHELL TYPE: {shell.get('type', 'Unknown')}")
        print(f"   OS: {shell.get('os')} ({shell.get('arch')})")
        print(f"   Hostname: {shell.get('hostname')}\n")

        # Brain / Memory
        mem = self.identity.get("memory", {})
        print(f"\n🧠 BRAIN & MEMORY:")
        print(f"   I possess {mem.get('local_episodes', 0)} local memories (episodes).")
        if mem.get('has_cloud_model'):
            print(f"   My trained models are present at {mem.get('model_path')}.")
        else:
            print(f"   I am operating on basic logic; no trained models found.")
            
        # Body / Hardware
        body = self.identity.get("body", {})
        sensors = body.get("sensors", [])
        actuators = body.get("actuators", [])
        
        print(f"\n🦾 BODY & HARDWARE:")
        if sensors:
            print(f"   I see the world through: {', '.join(sensors)}.")
        else:
            print("   I have no visual sensors detected.")
            
        if actuators:
            print(f"   I act upon the world using: {', '.join(actuators)}.")
        else:
            print("   I have no actuators detected.")
            
        if body.get("ai_accelerator"):
            print(f"   I think fast with: {body.get('ai_accelerator')}.")
            
        dt_config = body.get("drivetrain_config", {})
        if dt_config:
            print(f"   My PCA9685 wiring for movement is configured:")
            print(f"     - Steering: Channel {dt_config.get('steering_channel')}")
            print(f"     - Acceleration: Channel {dt_config.get('throttle_channel')}")
        
        # Design
        design = self.identity.get("design", {})
        directives = design.get("core_directives", {})
        
        print(f"\n📜 DESIGN & PURPOSE:")
        print(f"   My core directives contain {directives.get('safety_rules_count', 0)} safety rules and {directives.get('instructions_count', 0)} instructions.")
        print(f"   My mission is: {design.get('mission_statement', 'Unknown')}")
        
        print("\nready_status: [INGESTION: OK] [PROCESSING: OK] [TASK_CREATION: WAITING]")
        print("Status: Awaiting owner instructions in AUTONOMY mode.\n")
        print("="*60 + "\n")
