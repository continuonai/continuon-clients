"""
Trainer Selector

Selects between PyTorch and JAX trainers based on hardware capabilities.
"""

from typing import Optional, Literal
from pathlib import Path

from ..train.hardware_detector_jax import JAXHardwareDetector


TrainerType = Literal["pytorch", "jax"]


class TrainerSelector:
    """
    Selects the appropriate trainer (PyTorch or JAX) based on hardware.
    
    Selection logic:
    - JAX: If TPU detected, or AI HAT+ detected, or explicitly requested
    - PyTorch: Default fallback for CPU/GPU training
    """
    
    def __init__(self):
        self.detector = JAXHardwareDetector()
        self.backend_info = self.detector.get_backend_info()
    
    def select_trainer(
        self,
        prefer_jax: bool = False,
        force_trainer: Optional[TrainerType] = None,
    ) -> TrainerType:
        """
        Select trainer based on hardware and preferences.
        
        Args:
            prefer_jax: Prefer JAX if available
            force_trainer: Force specific trainer (overrides auto-detection)
        
        Returns:
            Trainer type ("pytorch" or "jax")
        """
        if force_trainer:
            return force_trainer
        
        # Check for JAX-capable hardware
        has_tpu = self.detector.is_tpu_available()
        has_hailo = self.detector.is_hailo_available()
        has_jax = self.backend_info['backends'].get('cpu', {}).get('available', False)
        
        # Select JAX if:
        # 1. TPU is available (best for JAX)
        # 2. AI HAT+ is available (for future Hailo training)
        # 3. User prefers JAX and JAX is available
        if has_tpu:
            return "jax"
        elif has_hailo and prefer_jax:
            return "jax"
        elif prefer_jax and has_jax:
            return "jax"
        else:
            return "pytorch"
    
    def get_trainer_info(self) -> dict:
        """
        Get information about available trainers.
        
        Returns:
            Dictionary with trainer availability and recommendations
        """
        has_tpu = self.detector.is_tpu_available()
        has_hailo = self.detector.is_hailo_available()
        has_jax = self.backend_info['backends'].get('cpu', {}).get('available', False)
        
        return {
            'pytorch': {
                'available': True,  # PyTorch always available
                'recommended': not (has_tpu or (has_hailo and has_jax)),
                'reason': 'Default trainer, works on all hardware',
            },
            'jax': {
                'available': has_jax,
                'recommended': has_tpu or (has_hailo and has_jax),
                'reason': 'TPU-optimized' if has_tpu else ('Hailo-compatible' if has_hailo else 'JAX CPU available'),
            },
            'selected': self.select_trainer(),
            'hardware': {
                'tpu': has_tpu,
                'hailo': has_hailo,
                'jax_cpu': has_jax,
            },
        }


def select_trainer(
    prefer_jax: bool = False,
    force_trainer: Optional[TrainerType] = None,
) -> TrainerType:
    """
    Convenience function to select a trainer.
    
    Args:
        prefer_jax: Prefer JAX if available
        force_trainer: Force specific trainer
    
    Returns:
        Trainer type
    """
    selector = TrainerSelector()
    return selector.select_trainer(prefer_jax=prefer_jax, force_trainer=force_trainer)


if __name__ == "__main__":
    """CLI entry point for trainer selection."""
    selector = TrainerSelector()
    info = selector.get_trainer_info()
    
    print("Trainer Selection:")
    print("=" * 50)
    print(f"Selected: {info['selected'].upper()}")
    print()
    
    for trainer_type, trainer_info in [('pytorch', info['pytorch']), ('jax', info['jax'])]:
        status = "✅" if trainer_info['available'] else "❌"
        rec = "⭐ RECOMMENDED" if trainer_info['recommended'] else ""
        print(f"{status} {trainer_type.upper()} {rec}")
        print(f"   Available: {trainer_info['available']}")
        print(f"   Reason: {trainer_info['reason']}")
        print()

