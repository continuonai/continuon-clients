"""
Unified trainer entrypoint.

Selects between the existing PyTorch LoRA trainer and the new JAX/Flax trainer
based on CLI flags or automatic hardware detection.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.jax_models.utils.trainer_selector import select_trainer
from continuonbrain.jax_models.train.local_sanity_check import run_sanity_check
from continuonbrain.jax_models.train.cloud.tpu_train import run_tpu_training
from continuonbrain.jax_models.config import CoreModelConfig
from continuonbrain.trainer.local_lora_trainer import (
    LocalTrainerJobConfig,
    SafetyGateConfig,
    build_stub_hooks,
    maybe_run_local_training,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trainer")


def _load_config_from_args(args) -> Optional[CoreModelConfig]:
    cfg = None
    if getattr(args, "config_preset", None) == "pi5":
        cfg = CoreModelConfig.pi5_optimized()
    elif getattr(args, "config_preset", None) == "dev":
        cfg = CoreModelConfig.development()
    elif getattr(args, "config_preset", None) == "tpu":
        cfg = CoreModelConfig.tpu_optimized()

    if getattr(args, "config_json", None):
        import json
        raw = json.loads(Path(args.config_json).read_text())
        base = cfg or CoreModelConfig()
        cfg = CoreModelConfig(
            d_s=raw.get("d_s", base.d_s),
            d_w=raw.get("d_w", base.d_w),
            d_p=raw.get("d_p", base.d_p),
            d_e=raw.get("d_e", base.d_e),
            d_k=raw.get("d_k", base.d_k),
            d_c=raw.get("d_c", base.d_c),
            num_levels=raw.get("num_levels", base.num_levels),
            cms_sizes=raw.get("cms_sizes", base.cms_sizes),
            cms_dims=raw.get("cms_dims", base.cms_dims),
            cms_decays=raw.get("cms_decays", base.cms_decays),
            learning_rate=raw.get("learning_rate", base.learning_rate),
            gradient_clip=raw.get("gradient_clip", base.gradient_clip),
            obs_type=raw.get("obs_type", base.obs_type),
            output_type=raw.get("output_type", base.output_type),
        )
    return cfg


def _run_pytorch_local(
    cfg_path: Optional[Path],
    rlds_dir: Optional[Path],
    hooks_module: Optional[str],
) -> int:
    """
    Run the PyTorch LoRA trainer with stub hooks.
    """
    cfg_path = cfg_path or Path("/tmp/continuonbrain/trainer_config.json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        # Create a minimal default config pointing to local RLDS if available
        default_rlds = rlds_dir or Path("continuonbrain/rlds/episodes")
        cfg = LocalTrainerJobConfig(rlds_dir=default_rlds)
        cfg.to_json(cfg_path)
        logger.info(f"Created default trainer config at {cfg_path}")

    cfg = LocalTrainerJobConfig.from_json(cfg_path)

    # Load hooks
    if hooks_module:
        import importlib

        module = importlib.import_module(hooks_module)
        if not hasattr(module, "build_hooks"):
            raise SystemExit(f"{hooks_module} must define build_hooks() returning ModelHooks")
        hooks = module.build_hooks()
        logger.info("Loaded hooks from %s", hooks_module)
    else:
        hooks = build_stub_hooks()
        logger.warning("Using stub hooks (no real training). Pass --hooks-module to use real model hooks.")
    safety = SafetyGateConfig()

    result = maybe_run_local_training(
        cfg=cfg,
        hooks=hooks,
        safety_cfg=safety,
        gating=None,
    )

    if result.status == "ok":
        logger.info(
            "PyTorch trainer finished: steps=%s avg_loss=%.4f",
            result.steps,
            result.avg_loss,
        )
        return 0
    logger.warning("PyTorch trainer skipped/failed: %s", result.reason)
    return 1


def _run_jax_local(args) -> int:
    """
    Run the JAX local sanity check training.
    """
    cfg = _load_config_from_args(args)

    results = run_sanity_check(
        rlds_dir=args.rlds_dir,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        output_dim=args.output_dim,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_synthetic_data=args.rlds_dir is None,
        config=cfg,
    )
    return 0 if results["status"] == "ok" else 1


def _run_jax_tpu(args) -> int:
    """
    Run the JAX TPU training loop.
    """
    cfg = _load_config_from_args(args)

    results = run_tpu_training(
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        gcs_checkpoint_dir=args.gcs_checkpoint_dir,
        metrics_path=str(args.metrics_path) if args.metrics_path else None,
        resume=not args.no_resume,
        config=cfg,
    )
    return 0 if results["status"] == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified trainer launcher")
    prefer_jax_default = os.getenv("CONTINUON_PREFER_JAX", "1").lower() in {"1", "true", "yes", "on"}
    parser.add_argument(
        "--trainer",
        choices=["auto", "pytorch", "jax"],
        default="auto",
        help="Which trainer to run (auto = hardware-based selection)",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "tpu"],
        default="local",
        help="Training mode: local (Pi CPU) or tpu (cloud training)",
    )
    parser.add_argument("--rlds-dir", type=Path, help="Path to RLDS TFRecord directory")
    parser.add_argument("--data-path", type=Path, help="TFRecord path (for TPU mode)")
    parser.add_argument("--output-dir", type=Path, help="Output dir (for TPU mode)")
    parser.add_argument("--obs-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--config", type=Path, help="Trainer config path (PyTorch)")
    parser.add_argument(
        "--prefer-jax",
        action="store_true",
        default=prefer_jax_default,
        help=f"Prefer JAX if available (env CONTINUON_PREFER_JAX, default={prefer_jax_default})",
    )
    parser.add_argument("--hooks-module", type=str, help="Python module path providing build_hooks() for PyTorch trainer")
    parser.add_argument("--config-preset", choices=["pi5", "dev", "tpu"], help="Preset CoreModelConfig for JAX paths")
    parser.add_argument("--config-json", type=Path, help="Path to JSON with CoreModelConfig overrides (JAX)")
    parser.add_argument("--metrics-path", type=Path, help="Path to write metrics CSV/JSON (JAX)")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume (JAX TPU)")

    args = parser.parse_args()

    # Determine trainer
    trainer = (
        select_trainer(prefer_jax=args.prefer_jax)
        if args.trainer == "auto"
        else args.trainer
    )

    logger.info("Selected trainer: %s (mode=%s)", trainer, args.mode)

    if trainer == "jax":
        if args.mode == "tpu":
            if not args.data_path or not args.output_dir:
                parser.error("--data-path and --output-dir are required for TPU mode")
            return _run_jax_tpu(args)
        return _run_jax_local(args)

    # PyTorch path
    return _run_pytorch_local(cfg_path=args.config, rlds_dir=args.rlds_dir, hooks_module=args.hooks_module)


if __name__ == "__main__":
    sys.exit(main())
