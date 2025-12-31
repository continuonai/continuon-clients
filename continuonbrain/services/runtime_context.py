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
    adapter_ready: bool = False
    checkpoint_path: str = ""
    can_predict: bool = False
    can_plan: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jax_available": self.jax_available,
            "mamba_available": self.mamba_available,
            "world_model_type": self.world_model_type,
            "adapter_ready": self.adapter_ready,
            "checkpoint_path": self.checkpoint_path,
            "can_predict": self.can_predict,
            "can_plan": self.can_plan,
        }


@dataclass
class HOPEAgentCapabilities:
    """HOPE Agent Manager integration status."""
    hope_brain_available: bool = False
    integrated_world_model: bool = False
    integrated_semantic_search: bool = False
    integrated_vision: bool = False
    confidence_threshold: float = 0.6
    active_columns: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hope_brain_available": self.hope_brain_available,
            "integrated_world_model": self.integrated_world_model,
            "integrated_semantic_search": self.integrated_semantic_search,
            "integrated_vision": self.integrated_vision,
            "confidence_threshold": self.confidence_threshold,
            "active_columns": self.active_columns,
            "fully_integrated": (
                self.hope_brain_available and 
                self.integrated_world_model and 
                self.integrated_semantic_search
            ),
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
    
    # Cognitive capabilities
    semantic_search: SemanticSearchCapabilities = field(default_factory=SemanticSearchCapabilities)
    world_model: WorldModelCapabilities = field(default_factory=WorldModelCapabilities)
    hope_agent: HOPEAgentCapabilities = field(default_factory=HOPEAgentCapabilities)
    
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
            "semantic_search": self.semantic_search.to_dict(),
            "world_model": self.world_model.to_dict(),
            "hope_agent": self.hope_agent.to_dict(),
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
        
        # Semantic Search detection
        caps.semantic_search = self._detect_semantic_search()
        
        # World Model detection
        caps.world_model = self._detect_world_model()
        
        # HOPE Agent integration detection
        caps.hope_agent = self._detect_hope_agent(caps)
        
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
    
    def _detect_semantic_search(self) -> SemanticSearchCapabilities:
        """Detect semantic search / memory retrieval capabilities."""
        caps = SemanticSearchCapabilities()
        
        # Check for sentence-transformers encoder
        try:
            from sentence_transformers import SentenceTransformer
            caps.encoder_available = True
            caps.encoder_model = "all-MiniLM-L6-v2"  # Default model
            logger.info("✅ Semantic search encoder available (sentence-transformers)")
        except ImportError:
            logger.info("⚠️ Semantic search encoder not available (sentence-transformers not installed)")
        
        # Check for existing memories
        try:
            experiences_dir = self.config_dir / "experiences"
            conversations_file = experiences_dir / "learned_conversations.jsonl"
            if conversations_file.exists():
                with open(conversations_file, 'r') as f:
                    caps.memories_count = sum(1 for _ in f)
            
            # Check feedback database for validated count
            feedback_db = experiences_dir / "feedback.db"
            if feedback_db.exists():
                import sqlite3
                conn = sqlite3.connect(str(feedback_db))
                try:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM feedback WHERE validated = 1"
                    )
                    caps.validated_count = cursor.fetchone()[0]
                except Exception:
                    pass
                finally:
                    conn.close()
        except Exception as e:
            logger.debug(f"Memory count check failed: {e}")
        
        # Category hints for topic classification
        caps.category_hints = ["Safety", "Motion", "Vision", "Identity", "Knowledge"]
        
        return caps
    
    def _detect_world_model(self) -> WorldModelCapabilities:
        """Detect world model / physics simulation capabilities."""
        caps = WorldModelCapabilities()
        
        # Check JAX availability
        try:
            import jax
            caps.jax_available = True
            logger.info("✅ JAX available for world model")
        except ImportError:
            logger.info("⚠️ JAX not available for world model")
        
        # Check Mamba availability
        try:
            # Import mamba_brain (aliased from 03_mamba_brain)
            from continuonbrain.mamba_brain import world_model as mamba_wm
            # Try to build a world model and check if mamba is available
            wm = mamba_wm.build_world_model(prefer_mamba=True)
            # Check if it's using real Mamba or stub
            from continuonbrain.mamba_brain.world_model import MambaWorldModel
            if isinstance(wm, MambaWorldModel) and wm._stub is None:
                caps.mamba_available = True
                logger.info("✅ Mamba SSM available for world model")
        except Exception:
            pass
        
        # Determine world model type
        if caps.jax_available:
            caps.world_model_type = "jax_core"
            caps.can_predict = True
            caps.can_plan = True
        elif caps.mamba_available:
            caps.world_model_type = "mamba"
            caps.can_predict = True
            caps.can_plan = True
        else:
            caps.world_model_type = "stub"
            caps.can_predict = True  # Stub still works, just not learned
            caps.can_plan = False
        
        # Check for trained checkpoint
        checkpoint_paths = [
            self.config_dir / "model" / "adapters" / "current" / "world_model.pt",
            self.config_dir / "model" / "base_model" / "world_model.pt",
            Path("/opt/continuonos/brain/model/world_model.pt"),
        ]
        for cp in checkpoint_paths:
            if cp.exists():
                caps.checkpoint_path = str(cp)
                break
        
        return caps
    
    def _detect_hope_agent(self, hardware_caps: HardwareCapabilities) -> HOPEAgentCapabilities:
        """Detect HOPE Agent Manager integration status."""
        caps = HOPEAgentCapabilities()
        
        # Check if HOPE brain is available
        try:
            from continuonbrain.hope_impl.config import HOPEConfig
            from continuonbrain.hope_impl.brain import HOPEBrain
            caps.hope_brain_available = True
            logger.info("✅ HOPE brain available")
            
            # Try to get column count
            try:
                config = HOPEConfig()
                brain = HOPEBrain(config)
                caps.active_columns = len(brain.columns) if hasattr(brain, 'columns') else 0
            except Exception:
                pass
        except ImportError as e:
            logger.info(f"⚠️ HOPE brain not available: {e}")
        
        # Check integration with other systems
        caps.integrated_world_model = hardware_caps.world_model.adapter_ready
        caps.integrated_semantic_search = hardware_caps.semantic_search.encoder_available
        caps.integrated_vision = hardware_caps.sam3_available or hardware_caps.hailo_available
        
        # Log integration status
        if caps.hope_brain_available:
            integrations = []
            if caps.integrated_world_model:
                integrations.append("world_model")
            if caps.integrated_semantic_search:
                integrations.append("semantic_search")
            if caps.integrated_vision:
                integrations.append("vision")
            
            if integrations:
                logger.info(f"✅ HOPE Agent integrated with: {', '.join(integrations)}")
            else:
                logger.info("⚠️ HOPE Agent has no cognitive integrations")
        
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

    def mark_world_model_ready(
        self,
        world_model_type: Optional[str] = None,
        can_plan: Optional[bool] = None,
    ) -> WorldModelCapabilities:
        """Mark the runtime context as having an initialized world model adapter."""
        wm = self.context.hardware.world_model
        wm.adapter_ready = True
        if world_model_type:
            wm.world_model_type = world_model_type
        wm.can_predict = True
        if can_plan is not None:
            wm.can_plan = bool(can_plan)
        self.context.hardware.hope_agent.integrated_world_model = True
        return wm
    
    def get_context(self) -> RuntimeContext:
        """Get current runtime context."""
        if not self._detected:
            self.detect_hardware()
        return self.context
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status for API."""
        ctx = self.get_context()
        status = ctx.to_dict()
        status["world_model_ready"] = bool(ctx.hardware.world_model.adapter_ready)
        
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
        
        # Semantic search recommendations
        ss = hw.semantic_search
        if not ss.encoder_available:
            recs.append("Semantic memory unavailable. Install: pip install sentence-transformers")
        elif ss.memories_count == 0:
            recs.append("No learned memories yet. Chat with the robot to build its memory.")
        elif ss.validated_count == 0 and ss.memories_count > 10:
            recs.append(f"Have {ss.memories_count} memories but none validated. Use feedback to confirm correct responses.")
        
        # World model recommendations
        wm = hw.world_model
        if (wm.jax_available or wm.mamba_available) and not wm.adapter_ready:
            recs.append("World model runtime not initialized; start the adapter or verify JAX/Mamba setup.")
        if not wm.jax_available and not wm.mamba_available:
            recs.append("World model limited to stub. Install JAX for physics prediction: pip install jax jaxlib")
        elif not wm.checkpoint_path:
            recs.append("World model untrained. Run training to learn physics from experience.")
        elif not wm.can_plan:
            recs.append("Planning unavailable. Enable JAX or Mamba for motion planning.")
        
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
        
        # Cognitive capabilities
        print("\n🧠 Cognitive Capabilities:")
        
        # Semantic search
        ss = hw.semantic_search
        if ss.encoder_available:
            mem_info = f"{ss.memories_count} memories" if ss.memories_count > 0 else "no memories yet"
            validated = f", {ss.validated_count} validated" if ss.validated_count > 0 else ""
            print(f"   ✅ Semantic Search ({ss.encoder_model}) - {mem_info}{validated}")
        else:
            print("   ❌ Semantic Search unavailable (install sentence-transformers)")
        
        # World model
        wm = hw.world_model
        if wm.can_predict:
            checkpoint = " + checkpoint" if wm.checkpoint_path else " (untrained)"
            planning = " + planning" if wm.can_plan else ""
            readiness = " + adapter_ready" if wm.adapter_ready else " (adapter not initialized)"
            print(f"   ✅ World Model ({wm.world_model_type}){checkpoint}{planning}{readiness}")
        else:
            print("   ❌ World Model unavailable")
        
        # HOPE Agent integration
        ha = hw.hope_agent
        if ha.hope_brain_available:
            integrations = []
            if ha.integrated_world_model:
                integrations.append("physics")
            if ha.integrated_semantic_search:
                integrations.append("memory")
            if ha.integrated_vision:
                integrations.append("vision")
            
            if integrations:
                print(f"   ✅ HOPE Agent Manager (integrated: {', '.join(integrations)})")
            else:
                print(f"   ⚠️ HOPE Agent Manager (no integrations)")
        else:
            print("   ❌ HOPE Agent Manager unavailable")
        
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
