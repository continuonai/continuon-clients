"""
Hardware Detection for Universal Seed Model

Detects hardware platform and selects optimal configuration
for the seed model to run on any architecture.
"""

import os
import platform
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Architecture(Enum):
    """Supported CPU architectures."""
    ARM64 = "arm64"
    X86_64 = "x86_64"
    RISCV = "riscv64"
    APPLE_SILICON = "apple_silicon"
    UNKNOWN = "unknown"


class Accelerator(Enum):
    """Supported hardware accelerators."""
    CPU = "cpu"
    CUDA = "cuda"
    TPU = "tpu"
    HAILO = "hailo"
    METAL = "metal"
    ANE = "ane"  # Apple Neural Engine
    NPU = "npu"  # Generic NPU
    LOIHI = "loihi"  # Intel neuromorphic
    QPU = "qpu"  # Quantum processor
    UNKNOWN = "unknown"


@dataclass
class HardwareProfile:
    """
    Hardware profile for a robot/device.
    
    Used to select optimal seed model configuration.
    """
    # Architecture
    architecture: Architecture = Architecture.UNKNOWN
    cpu_name: str = ""
    cpu_cores: int = 1
    
    # Memory
    ram_mb: int = 0
    swap_mb: int = 0
    
    # Accelerators
    accelerators: List[Accelerator] = field(default_factory=list)
    accelerator_memory_mb: int = 0
    
    # Device-specific
    device_name: str = ""
    device_class: str = ""  # "edge", "workstation", "cloud", "embedded"
    
    # Capabilities
    has_gpu: bool = False
    has_npu: bool = False
    has_tpu: bool = False
    has_camera: bool = False
    has_depth: bool = False
    has_arm: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_mb": self.ram_mb,
            "accelerators": [a.value for a in self.accelerators],
            "accelerator_memory_mb": self.accelerator_memory_mb,
            "device_name": self.device_name,
            "device_class": self.device_class,
            "has_gpu": self.has_gpu,
            "has_npu": self.has_npu,
            "has_tpu": self.has_tpu,
        }


def detect_architecture() -> Architecture:
    """Detect CPU architecture."""
    machine = platform.machine().lower()
    
    if machine in ('aarch64', 'arm64'):
        # Check for Apple Silicon
        if platform.system() == 'Darwin':
            return Architecture.APPLE_SILICON
        return Architecture.ARM64
    
    if machine in ('x86_64', 'amd64'):
        return Architecture.X86_64
    
    if 'riscv' in machine:
        return Architecture.RISCV
    
    return Architecture.UNKNOWN


def detect_accelerators() -> List[Accelerator]:
    """Detect available hardware accelerators."""
    accelerators = [Accelerator.CPU]  # CPU always available
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            accelerators.append(Accelerator.CUDA)
            logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    
    # Check for TPU
    try:
        import jax
        devices = jax.devices()
        if any('tpu' in str(d).lower() for d in devices):
            accelerators.append(Accelerator.TPU)
            logger.info("TPU detected")
    except Exception:
        pass
    
    # Check for Hailo
    try:
        from hailo_platform import HailoRuntimeException
        accelerators.append(Accelerator.HAILO)
        logger.info("Hailo NPU detected")
    except ImportError:
        pass
    
    # Check for Apple Metal/ANE
    if platform.system() == 'Darwin':
        try:
            import jax
            devices = jax.devices()
            if any('metal' in str(d).lower() for d in devices):
                accelerators.append(Accelerator.METAL)
        except Exception:
            pass
    
    return accelerators


def detect_memory() -> tuple:
    """Detect system memory (RAM, swap)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return mem.total // (1024 * 1024), swap.total // (1024 * 1024)
    except ImportError:
        # Fallback to /proc/meminfo on Linux
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
                mem_total = 0
                swap_total = 0
                for line in lines:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1]) // 1024
                    if line.startswith('SwapTotal:'):
                        swap_total = int(line.split()[1]) // 1024
                return mem_total, swap_total
        except Exception:
            return 0, 0


def detect_device_class(arch: Architecture, ram_mb: int, accelerators: List[Accelerator]) -> str:
    """Classify device based on capabilities."""
    if Accelerator.TPU in accelerators:
        return "cloud"
    
    if Accelerator.CUDA in accelerators and ram_mb > 16000:
        return "workstation"
    
    if arch == Architecture.ARM64:
        if ram_mb < 2000:
            return "embedded"
        return "edge"
    
    if ram_mb > 32000:
        return "workstation"
    
    return "edge"


def detect_hardware() -> HardwareProfile:
    """
    Detect full hardware profile of current system.
    
    Returns:
        HardwareProfile with detected capabilities
    """
    arch = detect_architecture()
    accelerators = detect_accelerators()
    ram_mb, swap_mb = detect_memory()
    device_class = detect_device_class(arch, ram_mb, accelerators)
    
    # Determine device name
    device_name = os.environ.get('CONTINUON_DEVICE_NAME', '')
    if not device_name:
        if arch == Architecture.ARM64:
            # Check for Pi
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip()
                    if 'raspberry pi' in model.lower():
                        device_name = model
            except Exception:
                device_name = "ARM64 Device"
        else:
            device_name = platform.node() or "Unknown Device"
    
    profile = HardwareProfile(
        architecture=arch,
        cpu_name=platform.processor() or platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        ram_mb=ram_mb,
        swap_mb=swap_mb,
        accelerators=accelerators,
        device_name=device_name,
        device_class=device_class,
        has_gpu=Accelerator.CUDA in accelerators or Accelerator.METAL in accelerators,
        has_npu=Accelerator.HAILO in accelerators or Accelerator.NPU in accelerators,
        has_tpu=Accelerator.TPU in accelerators,
    )
    
    logger.info(f"Detected hardware: {profile.device_name} ({arch.value}, {ram_mb}MB RAM)")
    logger.info(f"Accelerators: {[a.value for a in accelerators]}")
    logger.info(f"Device class: {device_class}")
    
    return profile


# Pre-defined profiles for common hardware
PROFILES = {
    "pi5": HardwareProfile(
        architecture=Architecture.ARM64,
        cpu_name="Cortex-A76",
        cpu_cores=4,
        ram_mb=8192,
        accelerators=[Accelerator.CPU, Accelerator.HAILO],
        device_name="Raspberry Pi 5",
        device_class="edge",
        has_npu=True,
    ),
    "jetson_orin": HardwareProfile(
        architecture=Architecture.ARM64,
        cpu_name="Cortex-A78AE",
        cpu_cores=12,
        ram_mb=32768,
        accelerators=[Accelerator.CPU, Accelerator.CUDA],
        accelerator_memory_mb=16384,
        device_name="Jetson Orin",
        device_class="edge",
        has_gpu=True,
    ),
    "cloud_tpu": HardwareProfile(
        architecture=Architecture.X86_64,
        cpu_name="Xeon",
        cpu_cores=96,
        ram_mb=340000,
        accelerators=[Accelerator.CPU, Accelerator.TPU],
        device_name="Cloud TPU v4",
        device_class="cloud",
        has_tpu=True,
    ),
    "workstation": HardwareProfile(
        architecture=Architecture.X86_64,
        cpu_name="AMD Ryzen / Intel Core",
        cpu_cores=16,
        ram_mb=65536,
        accelerators=[Accelerator.CPU, Accelerator.CUDA],
        accelerator_memory_mb=24576,
        device_name="Workstation",
        device_class="workstation",
        has_gpu=True,
    ),
}


def get_profile(target: Optional[str] = None) -> HardwareProfile:
    """
    Get hardware profile for target or current system.
    
    Args:
        target: Optional target name ('pi5', 'jetson', 'cloud', etc.)
                If None, auto-detects current hardware.
    
    Returns:
        HardwareProfile for the target/current system
    """
    if target is None:
        return detect_hardware()
    
    target_lower = target.lower()
    
    if target_lower in PROFILES:
        return PROFILES[target_lower]
    
    # Fuzzy match
    for name, profile in PROFILES.items():
        if target_lower in name or name in target_lower:
            return profile
    
    logger.warning(f"Unknown target '{target}', auto-detecting hardware")
    return detect_hardware()

