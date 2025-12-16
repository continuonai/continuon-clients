from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.jax_models.train.local_sanity_check import run_sanity_check


@dataclass
class ManualTrainerRequest:
    """Configurable knobs for the manual JAX trainer surface."""

    rlds_dir: Optional[Path] = None
    use_synthetic: bool = False
    max_steps: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    obs_dim: int = 128
    action_dim: int = 32
    output_dim: int = 32
    disable_jit: bool = True
    metrics_path: Optional[Path] = None


class ManualTrainer:
    """Runs the JAX sanity trainer with JSON/TFRecord or synthetic data."""

    def __init__(
        self,
        default_rlds_dir: Path = Path("/opt/continuonos/brain/rlds/episodes"),
        log_dir: Path = Path("/opt/continuonos/brain/trainer/logs"),
    ) -> None:
        self.default_rlds_dir = default_rlds_dir
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Share the existing status location used by the web server
        self.status_path = Path("/opt/continuonos/brain/trainer/status.json")

    async def run(self, request: ManualTrainerRequest) -> Dict[str, Any]:
        """Run trainer in a worker thread to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, request)

    def _run_sync(self, request: ManualTrainerRequest) -> Dict[str, Any]:
        # Guard against missing JAX
        try:
            import jax
        except ImportError:
             return {"status": "error", "message": "JAX not installed. Manual training requires JAX."}

        env_backup = {k: os.environ.get(k) for k in ("CONTINUON_PREFER_JAX", "JAX_DISABLE_JIT")}
        try:
            os.environ["CONTINUON_PREFER_JAX"] = "1"
            if request.disable_jit:
                os.environ["JAX_DISABLE_JIT"] = "1"
            elif "JAX_DISABLE_JIT" in os.environ:
                os.environ.pop("JAX_DISABLE_JIT")

            rlds_dir = request.rlds_dir or self.default_rlds_dir
            metrics_path = request.metrics_path or (self.log_dir / "manual_trainer_metrics.json")

            result = run_sanity_check(
                rlds_dir=rlds_dir,
                obs_dim=request.obs_dim,
                action_dim=request.action_dim,
                output_dim=request.output_dim,
                max_steps=request.max_steps,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                use_synthetic_data=request.use_synthetic,
                metrics_path=metrics_path,
            )

            payload = {
                "status": "ok",
                "request": asdict(request),
                "result": result,
                "metrics_path": str(metrics_path),
                "rlds_dir": str(rlds_dir),
            }
            self._write_status(payload)
            return payload
        except Exception as exc:  # pragma: no cover - defensive
            payload = {
                "status": "error",
                "error": str(exc),
                "request": asdict(request),
            }
            self._write_status(payload)
            return payload
        finally:
            for key, val in env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def _write_status(self, payload: Dict[str, Any]) -> None:
        try:
            self.status_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass
