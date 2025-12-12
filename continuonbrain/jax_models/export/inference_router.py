"""
Inference Router (Pi SSM seed edge bundle)

Hardware-aware inference routing with fallback chain:
Hailo → TPU → JAX CPU
"""

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from ..train.hardware_detector_jax import JAXHardwareDetector
from ..core_model import CoreModel, CoreModelConfig
from .export_jax import load_inference_model


class InferenceRouter:
    """
    Routes inference calls to the best available backend.
    
    Fallback chain: Hailo → TPU → JAX CPU
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[CoreModelConfig] = None,
    ):
        """
        Initialize inference router.
        
        Args:
            model_path: Path to exported model
            config: Model configuration (loaded from manifest if None)
        """
        self.model_path = model_path
        self.config = config
        
        # Detect available backends
        self.detector = JAXHardwareDetector()
        self.backend_info = self.detector.get_backend_info()
        
        # Load model for JAX backends
        self.jax_model = None
        self.jax_params = None
        self.hailo_model = None
        
        # Initialize backends
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize available inference backends."""
        # Load JAX model (for TPU/CPU)
        try:
            self.jax_model, self.jax_params, manifest = load_inference_model(self.model_path)
            if self.config is None:
                # Reconstruct config from manifest
                from ..core_model import CoreModelConfig
                cfg_dict = manifest['config']
                self.config = CoreModelConfig(
                    d_s=cfg_dict['d_s'],
                    d_w=cfg_dict['d_w'],
                    d_p=cfg_dict['d_p'],
                    d_e=cfg_dict['d_e'],
                    d_k=cfg_dict['d_k'],
                    d_c=cfg_dict['d_c'],
                    num_levels=cfg_dict['num_levels'],
                    cms_sizes=cfg_dict['cms_sizes'],
                    cms_dims=cfg_dict['cms_dims'],
                )
        except Exception as e:
            print(f"Warning: Failed to load JAX model: {e}")
        
        # Try to load Hailo model
        if self.detector.is_hailo_available():
            try:
                self._load_hailo_model()
            except Exception as e:
                print(f"Warning: Failed to load Hailo model: {e}")
    
    def _load_hailo_model(self):
        """Load Hailo-compiled model (metadata only until tensor I/O is wired)."""
        candidates = [
            Path(self.model_path) / "hailo" / "model.hef",
            Path("/opt/continuonos/brain/model/base_model/hailo/model.hef"),
            Path("/opt/continuonos/brain/model/base_model/model.hef"),
        ]
        hailo_path = next((p for p in candidates if p.exists()), None)
        if not hailo_path:
            print("⚠️  Hailo .hef not found; skipping Hailo backend.")
            return

        self.hailo_model = {"hef": str(hailo_path)}

        try:
            import hailo_platform as hailo  # type: ignore  # noqa: WPS433
        except ImportError:
            print("⚠️  hailo_platform not installed; Hailo runtime not available.")
            return

        try:
            vdevice = hailo.VDevice()
            infer_model = vdevice.create_infer_model(str(hailo_path))
            input_meta = [
                {"name": inp.name, "shape": inp.shape, "format": getattr(inp.format, "order", None)}
                for inp in getattr(infer_model, "inputs", [])
            ]
            output_meta = [
                {"name": out.name, "shape": out.shape, "format": getattr(out.format, "order", None)}
                for out in getattr(infer_model, "outputs", [])
            ]
            self.hailo_model.update(
                {
                    "vdevice": vdevice,
                    "infer_model": infer_model,
                    "configured": None,
                    "hailo_ready": True,
                    "inputs": input_meta,
                    "outputs": output_meta,
                }
            )
            print(f"✅ Hailo .hef loaded ({hailo_path}), inputs={len(input_meta)}, outputs={len(output_meta)}")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Hailo runtime init failed: {exc}")
    
    def infer(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        backend: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Run inference using the best available backend.
        
        Args:
            obs: Observation array
            action: Action array
            reward: Reward array
            backend: Force specific backend (None = auto-select)
        
        Returns:
            (output, info) where info contains backend used and metadata
        """
        # Select backend
        if backend is None:
            backend = self._select_backend()
        
        info = {
            'backend': backend,
            'backend_available': self.backend_info['backends'].get(backend, {}).get('available', False),
        }
        
        # Route to appropriate backend
        if backend == 'hailo' and self.hailo_model is not None:
            return self._infer_hailo(obs, action, reward), info
        elif backend in ('tpu', 'gpu') and self.jax_model is not None:
            return self._infer_jax(obs, action, reward), info
        elif backend == 'cpu' and self.jax_model is not None:
            return self._infer_jax(obs, action, reward), info

        # Fallback to CPU JAX if chosen backend unavailable
        if self.jax_model is not None:
            info['backend'] = 'cpu'
            return self._infer_jax(obs, action, reward), info

        raise RuntimeError(f"Backend {backend} not available")
    
    def _select_backend(self) -> str:
        """Select best available backend for inference."""
        # Prefer Hailo > TPU > CPU
        if self.detector.is_hailo_available() and self.hailo_model is not None:
            return 'hailo'
        if self.detector.is_tpu_available():
            return 'tpu'
        if self.detector.is_gpu_available():
            return 'gpu'
        return 'cpu'
    
    def _infer_hailo(self, obs: jnp.ndarray, action: jnp.ndarray, reward: jnp.ndarray) -> jnp.ndarray:
        """Run inference on Hailo accelerator."""
        if not self.hailo_model or not self.hailo_model.get("hailo_ready"):
            raise RuntimeError("Hailo model not ready; ensure .hef exists and hailo_platform is installed.")

        try:
            import numpy as np
            import hailo_platform as hailo  # type: ignore  # noqa: WPS433
        except ImportError:
            raise RuntimeError("hailo_platform not installed; cannot run Hailo inference.")

        infer_model = self.hailo_model.get("infer_model")
        vdevice = self.hailo_model.get("vdevice")
        if infer_model is None or vdevice is None:
            raise RuntimeError("Hailo infer model missing.")

        # Configure once and cache
        configured = self.hailo_model.get("configured")
        if configured is None:
            configured = infer_model.configure()
            self.hailo_model["configured"] = configured

        input_specs = self.hailo_model.get("inputs") or []
        output_specs = self.hailo_model.get("outputs") or []
        if not input_specs or not output_specs:
            raise RuntimeError("Hailo HEF missing input/output metadata.")

        # For now, map the first input to obs (as contiguous numpy, batch-aware).
        target_input = input_specs[0]
        target_name = target_input["name"]
        target_shape = tuple(target_input["shape"])

        obs_np = np.asarray(obs)
        # If obs is batched, try to reshape last dimensions to match target_shape.
        if obs_np.size != np.prod(target_shape):
            # If batch dimension present, flatten batch and then reshape per-frame
            if obs_np.ndim > len(target_shape):
                obs_np = obs_np.reshape(-1, *target_shape)
            else:
                try:
                    obs_np = obs_np.reshape(target_shape)
                except Exception:
                    raise RuntimeError(
                        f"Input shape mismatch for Hailo: got {obs_np.shape}, expected {target_shape} (name={target_name})"
                    )

        # Create bindings and populate buffers
        bindings = configured.create_bindings()
        input_stream = bindings.input(target_name)
        input_buffer = np.ascontiguousarray(obs_np, dtype=np.float32)
        input_stream.set_buffer(input_buffer)

        outputs: dict = {}
        for spec in output_specs:
            name = spec["name"]
            shape = tuple(spec["shape"])
            out_buf = np.empty(shape, dtype=np.float32)
            bindings.output(name).set_buffer(out_buf)
            outputs[name] = out_buf

        # Run synchronously
        configured.run([bindings], timeout=5000)

        # Collect outputs; use tf_format=None to avoid transformations
        results = {}
        for name, buf in outputs.items():
            results[name] = bindings.output(name).get_buffer(tf_format=None)

        return jnp.array(list(results.values()))
    
    def _infer_jax(self, obs: jnp.ndarray, action: jnp.ndarray, reward: jnp.ndarray) -> jnp.ndarray:
        """Run inference on JAX backend (CPU/GPU/TPU)."""
        if self.jax_model is None or self.jax_params is None:
            raise RuntimeError("JAX model not loaded")
        
        batch_size = obs.shape[0] if obs.ndim > 1 else 1
        
        # Initialize states
        s_prev = jnp.zeros((batch_size, self.config.d_s))
        w_prev = jnp.zeros((batch_size, self.config.d_w))
        p_prev = jnp.zeros((batch_size, self.config.d_p))
        cms_memories = [
            jnp.zeros((size, dim)) for size, dim in zip(self.config.cms_sizes, self.config.cms_dims)
        ]
        cms_keys = [
            jnp.zeros((size, self.config.d_k)) for size in self.config.cms_sizes
        ]
        
        # Forward pass
        y_pred, info = self.jax_model.apply(
            self.jax_params,
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


def create_inference_router(model_path: str) -> InferenceRouter:
    """
    Convenience function to create an inference router.
    
    Args:
        model_path: Path to exported model
    
    Returns:
        InferenceRouter instance
    """
    return InferenceRouter(model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference router smoke test")
    parser.add_argument("--model-path", type=str, required=True, help="Path to exported model directory")
    parser.add_argument("--backend", type=str, choices=["auto", "hailo", "tpu", "gpu", "cpu"], default="auto")
    parser.add_argument("--obs-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=32)
    args = parser.parse_args()

    router = InferenceRouter(args.model_path)
    obs = jnp.zeros((1, args.obs_dim), dtype=jnp.float32)
    act = jnp.zeros((1, args.action_dim), dtype=jnp.float32)
    rew = jnp.zeros((1, 1), dtype=jnp.float32)

    backend = None if args.backend == "auto" else args.backend
    try:
        out, info = router.infer(obs, act, rew, backend=backend)
        print(f"✅ Inference succeeded on backend={info['backend']}, shape={out.shape}")
        return 0
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

