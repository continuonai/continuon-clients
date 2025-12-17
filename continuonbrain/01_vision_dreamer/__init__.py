"""Entry points for the Vision Dreamer prototypes.

These modules remain lightweight so the Continuon Brain runtime can
import them on devices without optional vision hardware.
"""

from .model_vqgan import VQGANModel, export_hailo_vqgan
from .inference_oak import OakVisionAdapter

__all__ = ["VQGANModel", "OakVisionAdapter", "export_hailo_vqgan"]
