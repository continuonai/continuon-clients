"""
Runtime Context - Inference & Training Mode Management

Manages the robot's operational context:
- Primary Mode: INFERENCE vs TRAINING
- Sub-Mode: MANUAL vs AUTONOMOUS
- Hardware Detection: Hailo, SAM3, OAK-D, etc.

This integrates with the existing RobotMode system but adds
awareness of the inference/training context.
"""

import os
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class PrimaryMode(Enum):
    """Primary operational mode."""
    INFERENCE = "inference"  # Live perception and action
    TRAINING = "training"    # Learning from data
    HYBRID = "hybrid"        # Both (inference + background training)


class SubMode(Enum):
    """Sub-mode for control type."""
    MANUAL = "manual"        # Human-controlled
    AUTONOMOUS = "autonomous"  # AI-controlled


class InferenceBackend(Enum):
    """Available inference backends."""
    HAILO = "hailo"          # Hailo-8/8L NPU
    OAK_VPU = "oak_vpu"      # OAK-D Myriad X VPU
    CUDA = "cuda"            # NVIDIA GPU
    CPU = "cpu"              # CPU fallback
    JAX_TPU = "jax_tpu"      # JAX on TPU


@dataclass
class SemanticSearchCapabilities:
    """Semantic search and memory capabilities."""
    encoder_available: bool = False
    encoder_model: str = ""  # e.g., "all-MiniLM-L6-v2"
    memories_count: int = 0
    validated_count: int = 0
    category_hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_available": self.encoder_available,
            "encoder_model": self.encoder_model,
            "memories_count": self.memories_count,
            "validated_count": self.validated_count,
            "category_hints": self.category_hints,
        }


@dataclass
class WorldModelCapabilities:
    """World model / physics simulation capabilities."""
    jax_available: bool = False
    mamba_available: bool = False
    world_model_type: str = ""  # "jax_core", "mamba", "stub"
    checkpoint_path: str = ""
    can_predict: bool = False
    can_plan: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jax_available": self.jax_available,
            "mamba_available": self.mamba_available,
            "world_model_type": self.world_model_type,
            "checkpoint_path": self.checkpoint_path,
            "can_predict": self.can_predict,
            "can_plan": self.can_plan,
        }


@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities for inference."""
    hailo_available: bool = False
    hailo_model: str = ""  # "Hailo-8" or "Hailo-8L"
    hailo_tops: float = 0.0
    hailo_hef_count: int = 0
    
    oak_available: bool = False
    oak_model: str = ""  # "OAK-D", "OAK-D Lite", etc.
    oak_has_depth: bool = False
    oak_has_vpu: bool = False
    
    sam3_available: bool = False
    sam_models: List[str] = field(default_factory=list)  # ["sam3", "sam2", ...]
    
    cuda_available: bool = False
    cuda_device: str = ""
    cuda_vram_gb: float = 0.0
    
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hailo": {
                "available": self.hailo_available,
                "model": self.hailo_model,
                "tops": self.hailo_tops,
                "hef_count": self.hailo_hef_count,
            },
            "oak_camera": {
                "available": self.oak_available,
                "model": self.oak_model,
                "has_depth": self.oak_has_depth,
                "has_vpu": self.oak_has_vpu,
            },
            "sam": {
                "available": self.sam3_available,
                "models": self.sam_models,
            },
            "cuda": {
                "available": self.cuda_available,
                "device": self.cuda_device,
                "vram_gb": self.cuda_vram_gb,
            },
            "system": {
                "cpu_cores": self.cpu_cores,
                "ram_total_gb": self.ram_total_gb,
                "ram_available_gb": self.ram_available_gb,
            },
        }


@dataclass 
class RuntimeContext:
    """
    Current runtime context of the robot.
    
    Tracks what mode we're in and what hardware is available.
    """
    primary_mode: PrimaryMode = PrimaryMode.HYBRID
    sub_mode: SubMode = SubMode.AUTONOMOUS
    inference_backend: InferenceBackend = InferenceBackend.CPU
    
    # Hardware capabilities (populated at startup)
    hardware: HardwareCapabilities = field(default_factory=HardwareCapabilities)
    
    # State tracking
    inference_active: bool = False
    training_active: bool = False
    last_inference_ms: float = 0.0
    last_training_step: int = 0
    
    # Timestamps
    context_start_time: float = field(default_factory=time.time)
    last_mode_change: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_mode": self.primary_mode.value,
            "sub_mode": self.sub_mode.value,
            "inference_backend": self.inference_backend.value,
            "hardware": self.hardware.to_dict(),
            "state": {
                "inference_active": self.inference_active,
                "training_active": self.training_active,
                "last_inference_ms": self.last_inference_ms,
                "last_training_step": self.last_training_step,
            },
            "uptime_seconds": time.time() - self.context_start_time,
        }


class RuntimeContextManager:
    """
    Manages runtime context detection and mode transitions.
    
    Integrates with:
    - HardwareDetector for device discovery
    - RobotModeManager for mode transitions
    - API server for status reporting
    """
    
    def __init__(self, config_dir: str = "/opt/continuonos/brain"):
        self.config_dir = Path(config_dir)
        self.context = RuntimeContext()
        self._detected = False
    
    def detect_hardware(self) -> HardwareCapabilities:
        """
        Detect all available inference hardware.
        
        This should be called at startup to populate capabilities.
        """
        caps = HardwareCapabilities()
        
        # System resources
        try:
            import psutil
            caps.cpu_cores = psutil.cpu_count()
            mem = psutil.virtual_memory()
            caps.ram_total_gb = round(mem.total / (1024**3), 1)
            caps.ram_available_gb = round(mem.available / (1024**3), 1)
        except Exception:
            pass
        
        # Hailo detection
        try:
            from continuonbrain.sensors.hardware_detector import HardwareDetector
            detector = HardwareDetector()
            detector.detect_all()
            
            for device in detector.detected_devices:
                if device.device_type == "ai_accelerator" and device.vendor == "Hailo":
                    caps.hailo_available = True
                    caps.hailo_model = device.name
                    caps.hailo_tops = device.config.get("tops", 0)
                
                if device.device_type == "depth_camera" and device.vendor == "Luxonis":
                    caps.oak_available = True
                    caps.oak_model = device.name
                    caps.oak_has_depth = "depth" in device.capabilities
                    caps.oak_has_vpu = device.config.get("onboard_vpu") is not None
                
                if device.device_type == "gpu" and device.vendor == "NVIDIA":
                    caps.cuda_available = True
                    caps.cuda_device = device.name
                    caps.cuda_vram_gb = device.config.get("vram_gb", 0)
        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
        
        # Count Hailo HEF files
        hef_paths = [
            Path("/opt/continuonos/brain/models/hailo"),
            Path("/usr/share/hailo-models"),
        ]
        for hef_dir in hef_paths:
            if hef_dir.exists():
                caps.hailo_hef_count += len(list(hef_dir.glob("*.hef")))
        
        # SAM detection
        try:
            from continuonbrain.services.sam3_vision import create_sam_service
            sam = create_sam_service()
            if sam.is_available():
                caps.sam3_available = True
                caps.sam_models = sam.get_available_models()
        except Exception:
            pass
        
        self.context.hardware = caps
        self._detected = True
        
        # Determine best inference backend
        if caps.hailo_available:
            self.context.inference_backend = InferenceBackend.HAILO
        elif caps.cuda_available:
            self.context.inference_backend = InferenceBackend.CUDA
        elif caps.oak_has_vpu:
            self.context.inference_backend = InferenceBackend.OAK_VPU
        else:
            self.context.inference_backend = InferenceBackend.CPU
        
        return caps
    
    def set_mode(
        self,
        primary: Optional[PrimaryMode] = None,
        sub: Optional[SubMode] = None,
    ) -> RuntimeContext:
        """
        Set the runtime mode.
        
        Args:
            primary: INFERENCE, TRAINING, or HYBRID
            sub: MANUAL or AUTONOMOUS
            
        Returns:
            Updated RuntimeContext
        """
        changed = False
        
        if primary is not None and primary != self.context.primary_mode:
            self.context.primary_mode = primary
            changed = True
            logger.info(f"Primary mode changed to: {primary.value}")
        
        if sub is not None and sub != self.context.sub_mode:
            self.context.sub_mode = sub
            changed = True
            logger.info(f"Sub-mode changed to: {sub.value}")
        
        if changed:
            self.context.last_mode_change = time.time()
            
            # Update active states based on mode
            if self.context.primary_mode == PrimaryMode.INFERENCE:
                self.context.inference_active = True
                self.context.training_active = False
            elif self.context.primary_mode == PrimaryMode.TRAINING:
                self.context.inference_active = False
                self.context.training_active = True
            else:  # HYBRID
                self.context.inference_active = True
                self.context.training_active = True
        
        return self.context
    
    def get_context(self) -> RuntimeContext:
        """Get current runtime context."""
        if not self._detected:
            self.detect_hardware()
        return self.context
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status for API."""
        ctx = self.get_context()
        status = ctx.to_dict()
        
        # Add recommended actions based on mode
        status["recommendations"] = self._get_recommendations()
        
        return status
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on current context."""
        recs = []
        hw = self.context.hardware
        
        if hw.hailo_available and hw.hailo_hef_count == 0:
            recs.append("No Hailo HEF models found. Download models to /opt/continuonos/brain/models/hailo/")
        
        if not hw.oak_available:
            recs.append("No OAK-D camera detected. Connect USB depth camera for vision.")
        
        if not hw.sam3_available:
            recs.append("SAM segmentation unavailable. Install: pip install transformers torch")
        
        if self.context.sub_mode == SubMode.AUTONOMOUS and not hw.hailo_available:
            recs.append("Autonomous mode without hardware acceleration may be slow.")
        
        return recs
    
    def print_summary(self):
        """Print a human-readable summary."""
        ctx = self.get_context()
        hw = ctx.hardware
        
        print("\n" + "=" * 60)
        print("🤖 RUNTIME CONTEXT")
        print("=" * 60)
        
        print(f"\nMode: {ctx.primary_mode.value.upper()} / {ctx.sub_mode.value.upper()}")
        print(f"Backend: {ctx.inference_backend.value.upper()}")
        
        print("\n📦 Hardware:")
        if hw.hailo_available:
            print(f"   ✅ {hw.hailo_model} ({hw.hailo_tops} TOPS, {hw.hailo_hef_count} HEFs)")
        else:
            print("   ❌ No Hailo accelerator")
        
        if hw.oak_available:
            vpu = " + VPU" if hw.oak_has_vpu else ""
            depth = " + Depth" if hw.oak_has_depth else ""
            print(f"   ✅ {hw.oak_model}{depth}{vpu}")
        else:
            print("   ❌ No OAK camera")
        
        if hw.sam3_available:
            print(f"   ✅ SAM segmentation ({', '.join(hw.sam_models[:3])})")
        else:
            print("   ❌ No SAM available")
        
        if hw.cuda_available:
            print(f"   ✅ CUDA: {hw.cuda_device} ({hw.cuda_vram_gb}GB)")
        
        print(f"\n💾 System: {hw.cpu_cores} cores, {hw.ram_available_gb}/{hw.ram_total_gb} GB RAM")
        
        recs = self._get_recommendations()
        if recs:
            print("\n💡 Recommendations:")
            for rec in recs:
                print(f"   • {rec}")
        
        print("=" * 60 + "\n")


# Singleton instance
_runtime_context_manager: Optional[RuntimeContextManager] = None


def get_runtime_context_manager(config_dir: str = "/opt/continuonos/brain") -> RuntimeContextManager:
    """Get or create the runtime context manager singleton."""
    global _runtime_context_manager
    if _runtime_context_manager is None:
        _runtime_context_manager = RuntimeContextManager(config_dir)
    return _runtime_context_manager


def initialize_runtime_context(config_dir: str = "/opt/continuonos/brain") -> RuntimeContext:
    """
    Initialize runtime context at startup.
    
    Detects hardware and sets initial mode.
    Should be called early in startup sequence.
    """
    manager = get_runtime_context_manager(config_dir)
    manager.detect_hardware()
    
    # Default to HYBRID/AUTONOMOUS for production
    manager.set_mode(
        primary=PrimaryMode.HYBRID,
        sub=SubMode.AUTONOMOUS,
    )
    
    return manager.get_context()

