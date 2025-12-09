"""
JAX CPU Export

Export trained model parameters for JAX CPU inference on Pi.
Uses Orbax checkpoint format.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..core_model import CoreModel, CoreModelConfig, make_core_model
from .checkpointing import CheckpointManager


def export_for_inference(
    checkpoint_path: str,
    output_path: str,
    config: CoreModelConfig,
    obs_dim: int,
    action_dim: int,
    output_dim: int,
    quantization: Optional[str] = None,  # "int8" or "fp16"
) -> Path:
    """
    Export model for JAX CPU inference.
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Output directory for inference model
        config: Model configuration
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_dim: Output dimension
        quantization: Optional quantization ("int8" or "fp16")
    
    Returns:
        Path to exported model
    """
    if not ORBAX_AVAILABLE:
        raise ImportError("Orbax is required for export")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    manager = CheckpointManager(checkpoint_path, use_gcs=checkpoint_path.startswith("gs://"))
    checkpoint = manager.load()
    
    params = checkpoint['params']
    
    # Apply quantization if requested
    if quantization == "int8":
        print("Applying INT8 quantization...")
        params = quantize_params_int8(params)
    elif quantization == "fp16":
        print("Applying FP16 quantization...")
        params = quantize_params_fp16(params)
    
    # Save inference checkpoint
    inference_manager = CheckpointManager(str(output_dir))
    inference_manager.save(
        step=checkpoint.get('step', 0),
        params=params,
        opt_state=None,  # No optimizer state needed for inference
        metadata={
            'config': {
                'd_s': config.d_s,
                'd_w': config.d_w,
                'd_p': config.d_p,
                'd_e': config.d_e,
                'd_k': config.d_k,
                'd_c': config.d_c,
                'num_levels': config.num_levels,
                'cms_sizes': config.cms_sizes,
                'cms_dims': config.cms_dims,
            },
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'output_dim': output_dim,
            'quantization': quantization,
        }
    )
    
    # Save model manifest
    manifest = {
        'model_type': 'jax_core_model',
        'checkpoint_path': str(output_dir),
        'config': {
            'd_s': config.d_s,
            'd_w': config.d_w,
            'd_p': config.d_p,
            'd_e': config.d_e,
            'd_k': config.d_k,
            'd_c': config.d_c,
            'num_levels': config.num_levels,
            'cms_sizes': config.cms_sizes,
            'cms_dims': config.cms_dims,
        },
        'input_dims': {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
        },
        'output_dim': output_dim,
        'quantization': quantization,
        'backend': 'jax_cpu',
    }
    
    manifest_path = output_dir / "model_manifest.json"
    with manifest_path.open('w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Model exported to {output_dir}")
    print(f"   Manifest: {manifest_path}")
    
    return output_dir


def quantize_params_int8(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quantize parameters to INT8.
    
    Args:
        params: Model parameters
    
    Returns:
        Quantized parameters
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for quantization")
    
    quantized = {}
    for key, value in params.items():
        if isinstance(value, dict):
            quantized[key] = quantize_params_int8(value)
        elif isinstance(value, jnp.ndarray):
            # Scale to [-128, 127] range
            scale = jnp.abs(value).max() / 127.0
            quantized[key] = jnp.round(value / scale).astype(jnp.int8)
            # Store scale for dequantization
            quantized[f"{key}_scale"] = scale
        else:
            quantized[key] = value
    
    return quantized


def quantize_params_fp16(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quantize parameters to FP16.
    
    Args:
        params: Model parameters
    
    Returns:
        Quantized parameters
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for quantization")
    
    quantized = {}
    for key, value in params.items():
        if isinstance(value, dict):
            quantized[key] = quantize_params_fp16(value)
        elif isinstance(value, jnp.ndarray):
            quantized[key] = value.astype(jnp.float16)
        else:
            quantized[key] = value
    
    return quantized


def load_inference_model(
    model_path: str,
) -> tuple[CoreModel, Dict[str, Any], Dict[str, Any]]:
    """
    Load model for inference.
    
    Args:
        model_path: Path to exported model directory
    
    Returns:
        (model, params, manifest)
    """
    if not ORBAX_AVAILABLE:
        raise ImportError("Orbax is required for loading")
    
    model_dir = Path(model_path)
    
    # Load manifest
    manifest_path = model_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with manifest_path.open('r') as f:
        manifest = json.load(f)
    
    # Reconstruct config
    config_dict = manifest['config']
    config = CoreModelConfig(
        d_s=config_dict['d_s'],
        d_w=config_dict['d_w'],
        d_p=config_dict['d_p'],
        d_e=config_dict['d_e'],
        d_k=config_dict['d_k'],
        d_c=config_dict['d_c'],
        num_levels=config_dict['num_levels'],
        cms_sizes=config_dict['cms_sizes'],
        cms_dims=config_dict['cms_dims'],
    )
    
    # Create model
    rng_key = jax.random.PRNGKey(0)
    model, _ = make_core_model(
        rng_key,
        manifest['input_dims']['obs_dim'],
        manifest['input_dims']['action_dim'],
        manifest['output_dim'],
        config,
    )
    
    # Load parameters
    manager = CheckpointManager(str(model_dir))
    checkpoint = manager.load()
    params = checkpoint['params']
    
    return model, params, manifest

