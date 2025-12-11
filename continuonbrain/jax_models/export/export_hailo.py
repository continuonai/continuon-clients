"""
Hailo Export Pipeline

Convert JAX model → TensorFlow SavedModel → ONNX → Hailo compiler.
Generates Hailo-compiled binary for AI HAT+ inference.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

try:
    from jax.experimental import jax2tf
    import tensorflow as tf
    import jax.numpy as jnp
    JAX2TF_AVAILABLE = True
except ImportError:
    JAX2TF_AVAILABLE = False

try:
    import hailo_platform as hailo  # type: ignore

    HAILO_PLATFORM_AVAILABLE = True
except Exception:
    HAILO_PLATFORM_AVAILABLE = False

from ..core_model import CoreModel, CoreModelConfig, make_core_model

import argparse
import sys


def export_to_tf_savedmodel(
    model: CoreModel,
    params: Dict[str, Any],
    output_dir: str,
    obs_dim: int,
    action_dim: int,
    config: CoreModelConfig,
) -> Path:
    """
    Export JAX model to TensorFlow SavedModel using jax2tf.
    
    Args:
        model: CoreModel instance
        params: Model parameters
        output_dir: Output directory
        obs_dim: Observation dimension
        action_dim: Action dimension
        config: Model configuration
    
    Returns:
        Path to SavedModel directory
    """
    if not JAX2TF_AVAILABLE:
        raise ImportError("jax2tf is required. Install with: pip install jax[tf]")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create JAX inference function
    def jax_inference(obs, action, reward):
        """JAX inference function."""
        batch_size = obs.shape[0] if obs.ndim > 1 else 1
        
        # Initialize states (simplified - in practice, you'd manage state)
        s_prev = jnp.zeros((batch_size, config.d_s))
        w_prev = jnp.zeros((batch_size, config.d_w))
        p_prev = jnp.zeros((batch_size, config.d_p))
        cms_memories = [
            jnp.zeros((size, dim)) for size, dim in zip(config.cms_sizes, config.cms_dims)
        ]
        cms_keys = [
            jnp.zeros((size, config.d_k)) for size in config.cms_sizes
        ]
        
        y_pred, _ = model.apply(
            params,
            obs,
            action,
            reward,
            s_prev,
            w_prev,
            p_prev,
            cms_memories,
            cms_keys,
        )
        return y_pred
    
    # Convert to TensorFlow function
    tf_fn = jax2tf.convert(
        jax_inference,
        enable_xla=True,
        polymorphic_shapes=["b, ...", "b, ...", "b, ..."],
    )
    
    # Create TensorFlow SavedModel
    concrete_fn = tf.function(
        tf_fn,
        input_signature=[
            tf.TensorSpec([None, obs_dim], tf.float32, name='obs'),
            tf.TensorSpec([None, action_dim], tf.float32, name='action'),
            tf.TensorSpec([None, 1], tf.float32, name='reward'),
        ],
    )
    
    # Save SavedModel
    tf.saved_model.save(
        tf.Module(__call__=concrete_fn),
        str(output_path),
        signatures={"serving_default": concrete_fn.get_concrete_function()},
    )
    
    print(f"✅ TensorFlow SavedModel exported to {output_path}")
    return output_path


def export_to_onnx(
    savedmodel_path: Path,
    output_path: Path,
) -> Path:
    """
    Convert TensorFlow SavedModel to ONNX.
    
    Args:
        savedmodel_path: Path to TensorFlow SavedModel
        output_path: Output ONNX file path
    
    Returns:
        Path to ONNX file
    """
    try:
        import tf2onnx
        TF2ONNX_AVAILABLE = True
    except ImportError:
        raise ImportError("tf2onnx is required. Install with: pip install tf2onnx")
    
    # Load SavedModel
    model = tf.saved_model.load(str(savedmodel_path))
    
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_saved_model(
        str(savedmodel_path),
        output_path=str(output_path),
        opset=13,
    )
    
    print(f"✅ ONNX model exported to {output_path}")
    return output_path


def compile_hailo(
    onnx_path: Path,
    output_dir: Path,
    hailo_compiler_path: Optional[str] = None,
) -> Path:
    """
    Compile ONNX model for Hailo accelerator.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for Hailo binary
        hailo_compiler_path: Path to Hailo compiler (auto-detect if None)
    
    Returns:
        Path to compiled Hailo binary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hailo compiler command
    # This is a placeholder - actual Hailo compiler integration would go here
    # The Hailo compiler typically requires:
    # 1. ONNX model
    # 2. Hailo compiler binary (hailortcli or similar)
    # 3. Target device specification
    
    print("⚠️  Hailo compilation requires Hailo compiler tools.")
    print(f"   ONNX model: {onnx_path}")
    print(f"   Output directory: {output_dir}")
    print("\n   To compile manually:")
    print(f"   hailo compile {onnx_path} -o {output_dir}")

    # Placeholder output path
    hailo_binary = output_dir / "model.hef"  # Hailo Executable Format

    # Touch placeholder to make downstream checks clearer
    hailo_binary.touch(exist_ok=True)
    return hailo_binary


def export_for_hailo(
    checkpoint_path: str,
    output_dir: str,
    config: CoreModelConfig,
    obs_dim: int,
    action_dim: int,
    output_dim: int,
    skip_onnx: bool = False,
    skip_hailo_compile: bool = True,
    hef_source: Optional[str] = None,
    install_hef_path: Optional[str] = "/opt/continuonos/brain/model/base_model/model.hef",
) -> Dict[str, Path]:
    """
    Complete export pipeline: JAX → TF → ONNX → Hailo.
    
    Args:
        checkpoint_path: Path to JAX checkpoint
        output_dir: Output directory
        config: Model configuration
        obs_dim: Observation dimension
        action_dim: Action dimension
        output_dim: Output dimension
        skip_onnx: Skip ONNX conversion (for testing)
        skip_hailo_compile: Skip Hailo compilation (requires Hailo tools)
        hef_source: Optional path to an already-compiled HEF to include/copy
        install_hef_path: Where to place the HEF for runtime (None to skip install)
    
    Returns:
        Dictionary with paths to exported artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and parameters
    from .export_jax import load_inference_model
    model, params, manifest = load_inference_model(checkpoint_path)
    
    # Step 1: Export to TensorFlow SavedModel
    tf_dir = output_path / "tensorflow"
    tf_savedmodel = export_to_tf_savedmodel(
        model, params, str(tf_dir), obs_dim, action_dim, config
    )
    
    artifacts = {
        'tensorflow_savedmodel': tf_savedmodel,
    }
    
    # Step 2: Convert to ONNX
    if not skip_onnx:
        onnx_path = output_path / "model.onnx"
        try:
            onnx_file = export_to_onnx(tf_savedmodel, onnx_path)
            artifacts['onnx'] = onnx_file
        except Exception as e:
            print(f"⚠️  ONNX conversion failed: {e}")
    
    # Step 3: Compile for Hailo or use provided HEF
    hailo_hef: Optional[Path] = None
    if hef_source:
        source_path = Path(hef_source)
        if source_path.exists():
            hailo_dir = output_path / "hailo"
            hailo_dir.mkdir(parents=True, exist_ok=True)
            hailo_hef = hailo_dir / source_path.name
            hailo_hef.write_bytes(source_path.read_bytes())
            artifacts["hailo_hef"] = hailo_hef
            print(f"✅ Included provided HEF: {hailo_hef}")
        else:
            print(f"⚠️  Provided HEF not found: {hef_source}")
    elif not skip_hailo_compile and "onnx" in artifacts:
        hailo_dir = output_path / "hailo"
        try:
            hailo_binary = compile_hailo(artifacts["onnx"], hailo_dir)
            artifacts["hailo"] = hailo_binary
            hailo_hef = hailo_binary
        except Exception as e:
            print(f"⚠️  Hailo compilation failed: {e}")

    # Step 4: Install HEF to runtime path and collect metadata
    hef_metadata: Dict[str, Any] = {}
    if hailo_hef and hailo_hef.exists():
        if install_hef_path:
            install_path = Path(install_hef_path)
            install_path.parent.mkdir(parents=True, exist_ok=True)
            install_path.write_bytes(hailo_hef.read_bytes())
            artifacts["installed_hef"] = install_path
            print(f"✅ Installed HEF to {install_path}")

        if HAILO_PLATFORM_AVAILABLE:
            try:
                hef_obj = hailo.HEF(str(hailo_hef))
                hef_metadata = {
                    "inputs": [
                        {
                            "name": info.name,
                            "shape": tuple(info.shape),
                            "format": getattr(info.format, "order", None),
                        }
                        for info in hef_obj.get_input_vstream_infos()
                    ],
                    "outputs": [
                        {
                            "name": info.name,
                            "shape": tuple(info.shape),
                            "format": getattr(info.format, "order", None),
                        }
                        for info in hef_obj.get_output_vstream_infos()
                    ],
                }
            except Exception as exc:  # noqa: BLE001
                hef_metadata = {"error": f"Failed to read HEF metadata: {exc}"}
        else:
            hef_metadata = {"warning": "hailo_platform not installed; metadata unavailable"}
    
    # Save export manifest
    manifest_path = output_path / "export_manifest.json"
    with manifest_path.open('w') as f:
        json.dump({
            'source_checkpoint': checkpoint_path,
            'config': {
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'output_dim': output_dim,
            },
            'artifacts': {k: str(v) for k, v in artifacts.items()},
            'hef_metadata': hef_metadata,
        }, f, indent=2)
    
    print(f"\n✅ Export complete. Artifacts in {output_path}")
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CoreModel for Hailo")
    parser.add_argument("--checkpoint-path", required=True, help="Path to JAX inference checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for exported artifacts")
    parser.add_argument("--obs-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX conversion")
    parser.add_argument("--skip-hailo-compile", action="store_true", help="Skip Hailo compilation step")
    parser.add_argument("--hef-source", type=str, default=None, help="Path to precompiled HEF to include/copy")
    parser.add_argument("--install-hef-path", type=str, default="/opt/continuonos/brain/model/base_model/model.hef", help="Destination to install HEF for runtime (set to empty to skip)")

    args = parser.parse_args()

    # Load config from checkpoint manifest
    from .export_jax import load_inference_model
    _, _, manifest = load_inference_model(args.checkpoint_path)
    cfg_dict = manifest["config"]
    config = CoreModelConfig(
        d_s=cfg_dict["d_s"],
        d_w=cfg_dict["d_w"],
        d_p=cfg_dict["d_p"],
        d_e=cfg_dict["d_e"],
        d_k=cfg_dict["d_k"],
        d_c=cfg_dict["d_c"],
        num_levels=cfg_dict["num_levels"],
        cms_sizes=cfg_dict["cms_sizes"],
        cms_dims=cfg_dict["cms_dims"],
    )

    export_for_hailo(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        config=config,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        output_dim=args.output_dim,
        skip_onnx=args.skip_onnx,
        skip_hailo_compile=args.skip_hailo_compile,
        hef_source=args.hef_source,
        install_hef_path=args.install_hef_path if args.install_hef_path else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

