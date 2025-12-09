"""
JAX Hardware Detection

Detects available JAX backends (TPU, GPU, CPU) and AI accelerators (Hailo).
Integrates with existing hardware detector.
"""

import platform
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import jaxlib
    JAXLIB_AVAILABLE = True
except ImportError:
    JAXLIB_AVAILABLE = False


@dataclass
class JAXBackendInfo:
    """Information about a JAX backend."""
    backend_type: str  # "tpu", "gpu", "cpu", "hailo"
    device_count: int
    device_name: Optional[str] = None
    available: bool = False
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class JAXHardwareDetector:
    """
    Detects available JAX backends and accelerators.
    
    Checks for:
    - Google Cloud TPU
    - CUDA GPU
    - CPU (fallback)
    - Hailo AI HAT+ (for inference)
    """
    
    def __init__(self):
        self.backends: Dict[str, JAXBackendInfo] = {}
        self.primary_backend: Optional[str] = None
    
    def detect_all(self) -> Dict[str, JAXBackendInfo]:
        """
        Detect all available JAX backends.
        
        Returns:
            Dictionary mapping backend type to backend info
        """
        self.backends = {}
        
        if not JAX_AVAILABLE:
            return self.backends
        
        # Detect TPU
        tpu_info = self._detect_tpu()
        if tpu_info.available:
            self.backends['tpu'] = tpu_info
            self.primary_backend = 'tpu'
        
        # Detect GPU
        gpu_info = self._detect_gpu()
        if gpu_info.available:
            self.backends['gpu'] = gpu_info
            if self.primary_backend is None:
                self.primary_backend = 'gpu'
        
        # Detect CPU (always available)
        cpu_info = self._detect_cpu()
        self.backends['cpu'] = cpu_info
        if self.primary_backend is None:
            self.primary_backend = 'cpu'
        
        # Detect Hailo AI HAT+ (for inference, not training)
        hailo_info = self._detect_hailo()
        if hailo_info.available:
            self.backends['hailo'] = hailo_info
        
        return self.backends
    
    def _detect_tpu(self) -> JAXBackendInfo:
        """Detect Google Cloud TPU."""
        info = JAXBackendInfo(
            backend_type='tpu',
            device_count=0,
            available=False,
        )
        
        if not JAX_AVAILABLE:
            return info
        
        try:
            # Check for TPU via JAX
            devices = jax.devices()
            tpu_devices = [d for d in devices if d.device_kind == 'tpu']
            
            if tpu_devices:
                info.available = True
                info.device_count = len(tpu_devices)
                info.device_name = f"TPU v{getattr(tpu_devices[0], 'platform_version', 'unknown')}"
                info.capabilities = ['training', 'inference', 'xla']
                
                # Detect TPU topology
                try:
                    from jax.experimental import mesh_utils
                    mesh = mesh_utils.create_device_mesh((len(tpu_devices),))
                    info.capabilities.append(f'mesh_{len(tpu_devices)}x1')
                except Exception:
                    pass
        except Exception as e:
            # TPU not available or not configured
            pass
        
        return info
    
    def _detect_gpu(self) -> JAXBackendInfo:
        """Detect CUDA GPU."""
        info = JAXBackendInfo(
            backend_type='gpu',
            device_count=0,
            available=False,
        )
        
        if not JAX_AVAILABLE:
            return info
        
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.device_kind == 'gpu']
            
            if gpu_devices:
                info.available = True
                info.device_count = len(gpu_devices)
                
                # Get GPU name
                try:
                    import jaxlib.xla_extension as xla_ext
                    gpu_name = gpu_devices[0].device_kind
                    info.device_name = gpu_name
                except Exception:
                    info.device_name = "GPU"
                
                info.capabilities = ['training', 'inference', 'cuda']
        except Exception as e:
            # GPU not available
            pass
        
        return info
    
    def _detect_cpu(self) -> JAXBackendInfo:
        """Detect CPU backend."""
        info = JAXBackendInfo(
            backend_type='cpu',
            device_count=1,
            available=True,
            device_name=platform.processor() or "CPU",
        )
        
        if JAX_AVAILABLE:
            try:
                devices = jax.devices()
                cpu_devices = [d for d in devices if d.device_kind == 'cpu']
                info.device_count = len(cpu_devices)
                info.capabilities = ['training', 'inference']
            except Exception:
                pass
        
        return info
    
    def _detect_hailo(self) -> JAXBackendInfo:
        """
        Detect Hailo AI HAT+ accelerator.
        
        Note: Hailo is not a JAX backend, but we detect it here for
        inference routing purposes.
        """
        info = JAXBackendInfo(
            backend_type='hailo',
            device_count=0,
            available=False,
        )
        
        # Check for Hailo via PCIe (common on Raspberry Pi with AI HAT+)
        try:
            import subprocess
            result = subprocess.run(
                ['lspci'],
                capture_output=True,
                text=True,
                timeout=2,
            )
            
            if 'hailo' in result.stdout.lower() or '1e3e' in result.stdout.lower():
                info.available = True
                info.device_count = 1
                info.device_name = "Hailo-8L AI HAT+"
                info.capabilities = ['inference']  # Hailo is inference-only
        except Exception:
            # lspci not available or Hailo not present
            pass
        
        # Alternative: Check for Hailo device file
        try:
            from pathlib import Path
            hailo_devices = list(Path('/dev').glob('hailo*'))
            if hailo_devices:
                info.available = True
                info.device_count = len(hailo_devices)
                info.device_name = "Hailo-8L AI HAT+"
                info.capabilities = ['inference']
        except Exception:
            pass
        
        return info
    
    def get_best_backend(self, for_training: bool = True) -> Optional[str]:
        """
        Get the best available backend for the given use case.
        
        Args:
            for_training: If True, prefer training-capable backends
        
        Returns:
            Backend type string, or None if no suitable backend
        """
        if not self.backends:
            self.detect_all()
        
        if for_training:
            # Prefer TPU > GPU > CPU for training
            for backend_type in ['tpu', 'gpu', 'cpu']:
                if backend_type in self.backends:
                    backend = self.backends[backend_type]
                    if backend.available and 'training' in backend.capabilities:
                        return backend_type
        else:
            # For inference: Hailo > TPU > GPU > CPU
            for backend_type in ['hailo', 'tpu', 'gpu', 'cpu']:
                if backend_type in self.backends:
                    backend = self.backends[backend_type]
                    if backend.available and 'inference' in backend.capabilities:
                        return backend_type
        
        return None
    
    def is_tpu_available(self) -> bool:
        """Check if TPU is available."""
        if 'tpu' not in self.backends:
            self.detect_all()
        return self.backends.get('tpu', JAXBackendInfo('tpu', 0)).available
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        if 'gpu' not in self.backends:
            self.detect_all()
        return self.backends.get('gpu', JAXBackendInfo('gpu', 0)).available
    
    def is_hailo_available(self) -> bool:
        """Check if Hailo accelerator is available."""
        if 'hailo' not in self.backends:
            self.detect_all()
        return self.backends.get('hailo', JAXBackendInfo('hailo', 0)).available
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get summary information about detected backends.
        
        Returns:
            Dictionary with backend information
        """
        if not self.backends:
            self.detect_all()
        
        return {
            'backends': {k: {
                'type': v.backend_type,
                'available': v.available,
                'device_count': v.device_count,
                'device_name': v.device_name,
                'capabilities': v.capabilities,
            } for k, v in self.backends.items()},
            'primary_backend': self.primary_backend,
            'best_for_training': self.get_best_backend(for_training=True),
            'best_for_inference': self.get_best_backend(for_training=False),
        }


def detect_jax_backends() -> Dict[str, Any]:
    """
    Convenience function to detect JAX backends.
    
    Returns:
        Dictionary with backend information
    """
    detector = JAXHardwareDetector()
    return detector.get_backend_info()


if __name__ == "__main__":
    """CLI entry point for JAX backend detection."""
    detector = JAXHardwareDetector()
    info = detector.get_backend_info()
    
    print("JAX Backend Detection:")
    print("=" * 50)
    
    for backend_type, backend_info in info['backends'].items():
        status = "✅" if backend_info['available'] else "❌"
        print(f"{status} {backend_type.upper()}:")
        print(f"   Available: {backend_info['available']}")
        print(f"   Devices: {backend_info['device_count']}")
        if backend_info['device_name']:
            print(f"   Name: {backend_info['device_name']}")
        if backend_info['capabilities']:
            print(f"   Capabilities: {', '.join(backend_info['capabilities'])}")
        print()
    
    print(f"Primary backend: {info['primary_backend']}")
    print(f"Best for training: {info['best_for_training']}")
    print(f"Best for inference: {info['best_for_inference']}")

