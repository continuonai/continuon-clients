from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import pickle

from continuonbrain.jax_models.train.local_sanity_check import run_sanity_check
from continuonbrain.jax_models.export.export_jax import export_for_inference
from continuonbrain.jax_models.core_model import CoreModelConfig


@dataclass
class WavecoreLoopConfig:
    """Per-loop settings for the JAX seed trainer."""

    name: str
    rlds_dir: Optional[Path] = None
    use_synthetic: bool = False
    max_steps: int = 16
    batch_size: int = 4
    learning_rate: float = 1e-3
    obs_dim: int = 128
    action_dim: int = 32
    output_dim: int = 32
    disable_jit: bool = True
    metrics_path: Optional[Path] = None
    arch_preset: Optional[str] = None
    sparsity_lambda: float = 0.0


class WavecoreTrainer:
    """
    Runs fast/mid/slow loops using the JAX CoreModel seed.

    This wraps the local sanity check trainer so we can exercise the CoreModel
    across three loop speeds with configurable hyperparameters and data sources.
    """

    def __init__(
        self,
        default_rlds_dir: Path = Path("/opt/continuonos/brain/rlds/episodes"),
        log_dir: Path = Path("/opt/continuonos/brain/trainer/logs"),
        checkpoint_dir: Path = Path("/opt/continuonos/brain/trainer/checkpoints/core_model_seed"),
        export_dir: Path = Path("/opt/continuonos/brain/model/adapters/candidate/core_model_seed"),
    ) -> None:
        self.default_rlds_dir = default_rlds_dir
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = Path("/opt/continuonos/brain/trainer/status.json")
        self.checkpoint_dir = checkpoint_dir
        self.export_dir = export_dir
        self.questions_path = Path(__file__).resolve().parent.parent / "eval" / "hope_eval_questions.json"
        self.facts_questions_path = Path(__file__).resolve().parent.parent / "eval" / "facts_eval_questions.json"

    async def run_loops(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run fast/mid/slow loops in a worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, payload)

    def _run_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        env_backup = {k: os.environ.get(k) for k in ("CONTINUON_PREFER_JAX", "JAX_DISABLE_JIT")}
        try:
            os.environ["CONTINUON_PREFER_JAX"] = "1"

            # Allow overrides for checkpoint/export
            self.checkpoint_dir = Path(payload.get("checkpoint_dir", self.checkpoint_dir))
            self.export_dir = Path(payload.get("export_dir", self.export_dir))
            quantization = payload.get("quantization")

            # Pi5-friendly defaults: small steps and disable_jit=True unless explicitly overridden.
            # Users can still override per-loop in payload.
            fast_cfg = self._parse_loop(payload.get("fast"), default_lr=1e-3, default_steps=12, name="fast")
            mid_cfg = self._parse_loop(payload.get("mid"), default_lr=5e-4, default_steps=24, name="mid")
            slow_cfg = self._parse_loop(payload.get("slow"), default_lr=2e-4, default_steps=32, name="slow")

            results = {
                "fast": self._run_loop(fast_cfg),
                "mid": self._run_loop(mid_cfg),
                "slow": self._run_loop(slow_cfg, checkpoint_dir=self.checkpoint_dir),
            }
            if payload.get("run_hope_eval"):
                from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log

                service = payload.get("service")
                if service is None or getattr(service, "chat_adapter", None) is None:
                    results["hope_eval"] = {"status": "skipped", "reason": "chat_adapter unavailable"}
                else:
                    eval_res = asyncio.run(
                        run_hope_eval_and_log(
                            service=service,
                            questions_path=Path(payload.get("questions_path") or self.questions_path),
                            rlds_dir=Path(payload.get("eval_rlds_dir") or self.default_rlds_dir),
                            use_fallback=bool(payload.get("eval_use_fallback", True)),
                            fallback_order=payload.get("eval_fallback_order") or ["google/gemma-370m", "google/gemma-3n-2b"],
                            episode_prefix="hope_eval",
                            model_label="hope-agent",
                        )
                    )
                    results["hope_eval"] = eval_res

            if payload.get("run_facts_eval"):
                from continuonbrain.eval.hope_eval_runner import run_hope_eval_and_log

                service = payload.get("service")
                if service is None or getattr(service, "chat_adapter", None) is None:
                    results["facts_eval"] = {"status": "skipped", "reason": "chat_adapter unavailable"}
                else:
                    facts_res = asyncio.run(
                        run_hope_eval_and_log(
                            service=service,
                            questions_path=Path(payload.get("facts_questions_path") or self.facts_questions_path),
                            rlds_dir=Path(payload.get("eval_rlds_dir") or self.default_rlds_dir),
                            use_fallback=bool(payload.get("facts_use_fallback", True)),
                            fallback_order=payload.get("facts_fallback_order") or ["google/gemma-370m", "google/gemma-3n-2b"],
                            episode_prefix="facts_eval",
                            model_label="facts-lite",
                        )
                    )
                    results["facts_eval"] = facts_res
            export_info = None
            slow_result = results.get("slow", {})
            ckpt_file = slow_result.get("checkpoint_dir")
            compact = bool(payload.get("compact_export", False))
            export_dir = self.export_dir if not compact else self.export_dir.parent / (self.export_dir.name + "_compact")
            if ckpt_file:
                try:
                    export_dir.mkdir(parents=True, exist_ok=True)
                    cfg: CoreModelConfig = slow_result["result"]["config"]
                    checkpoint_format = slow_result.get("checkpoint_format")
                    if checkpoint_format == "pickle":
                        ckpt_src = Path(ckpt_file)
                        ckpt_dst = export_dir / ckpt_src.name
                        ckpt_dst.write_bytes(ckpt_src.read_bytes())
                        manifest = {
                            "model_type": "jax_core_model",
                            "checkpoint_path": str(ckpt_dst),
                            # Avoid json serialization issues (CoreModelConfig isn't a dataclass).
                            "config": getattr(cfg, "__dict__", str(cfg)),
                            "input_dims": {
                                "obs_dim": slow_result["request"]["obs_dim"],
                                "action_dim": slow_result["request"]["action_dim"],
                            },
                            "output_dim": slow_result["request"]["output_dim"],
                            "quantization": quantization,
                            "format": "pickle",
                            "arch_preset": slow_result["request"].get("arch_preset"),
                            "sparsity_lambda": slow_result["request"].get("sparsity_lambda"),
                            "compact": compact,
                        }
                        manifest_path = export_dir / "model_manifest.json"
                        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
                        export_info = {
                            "export_dir": str(export_dir),
                            "format": "pickle",
                            "manifest": str(manifest_path),
                            "arch_preset": slow_result["request"].get("arch_preset"),
                            "sparsity_lambda": slow_result["request"].get("sparsity_lambda"),
                            "compact": compact,
                        }
                    else:
                        export_path = export_for_inference(
                            checkpoint_path=str(Path(ckpt_file).parent),
                            output_path=str(export_dir),
                            config=cfg,
                            obs_dim=slow_result["request"]["obs_dim"],
                            action_dim=slow_result["request"]["action_dim"],
                            output_dim=slow_result["request"]["output_dim"],
                            quantization=quantization,
                        )
                        export_info = {"export_dir": str(export_path), "quantization": quantization, "compact": compact}
                except Exception as exc:
                    export_info = {"error": str(exc)}

            results["status"] = "ok"
            results["export"] = export_info
            self._write_status(results)
            return results
        except Exception as exc:  # pragma: no cover - defensive
            payload_out = {"status": "error", "message": str(exc)}
            self._write_status(payload_out)
            return payload_out
        finally:
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def _run_loop(self, cfg: WavecoreLoopConfig, checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
        if cfg.disable_jit:
            os.environ["JAX_DISABLE_JIT"] = "1"
        else:
            os.environ.pop("JAX_DISABLE_JIT", None)

        rlds_dir = cfg.rlds_dir or self.default_rlds_dir
        metrics_path = cfg.metrics_path or (self.log_dir / f"wavecore_{cfg.name}_metrics.json")

        result = run_sanity_check(
            rlds_dir=rlds_dir,
            arch_preset=cfg.arch_preset,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            output_dim=cfg.output_dim,
            max_steps=cfg.max_steps,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            # If explicitly requested OR if rlds_dir is empty/missing, fall back to synthetic.
            use_synthetic_data=cfg.use_synthetic or not (rlds_dir and Path(rlds_dir).exists() and any(Path(rlds_dir).glob("*.json"))),
            metrics_path=metrics_path,
            checkpoint_dir=checkpoint_dir if cfg.name == "slow" else None,
            sparsity_lambda=cfg.sparsity_lambda,
        )
        return {
            "loop": cfg.name,
            "request": asdict(cfg),
            "result": result,
            "metrics_path": str(metrics_path),
            "rlds_dir": str(rlds_dir),
            "checkpoint_dir": result.get("checkpoint_dir"),
            "checkpoint_step": result.get("checkpoint_step"),
            "checkpoint_format": result.get("checkpoint_format"),
        }

    def _parse_loop(self, data: Optional[Dict[str, Any]], default_lr: float, default_steps: int, name: str) -> WavecoreLoopConfig:
        data = data or {}
        rlds_dir = Path(data["rlds_dir"]) if data.get("rlds_dir") else None
        metrics_path = Path(data["metrics_path"]) if data.get("metrics_path") else None
        return WavecoreLoopConfig(
            name=name,
            rlds_dir=rlds_dir,
            use_synthetic=bool(data.get("use_synthetic", False)),
            max_steps=int(data.get("max_steps", default_steps)),
            batch_size=int(data.get("batch_size", 4)),
            learning_rate=float(data.get("learning_rate", default_lr)),
            obs_dim=int(data.get("obs_dim", 128)),
            action_dim=int(data.get("action_dim", 32)),
            output_dim=int(data.get("output_dim", 32)),
            disable_jit=bool(data.get("disable_jit", True)),
            metrics_path=metrics_path,
            # Default to pi5 preset for on-device runs.
            arch_preset=data.get("arch_preset", "pi5"),
            sparsity_lambda=float(data.get("sparsity_lambda", 0.0)),
        )

    def _write_status(self, payload: Dict[str, Any]) -> None:
        try:
            self.status_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass
