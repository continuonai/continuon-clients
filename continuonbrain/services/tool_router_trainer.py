from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.jax_models.train.tool_router_train import ToolRouterTrainConfig, train


@dataclass
class ToolRouterTrainRequest:
    episodes_root: Optional[Path] = None
    include_dirs_prefix: str = "toolchat_hf"
    max_episodes_scan: int = 20000
    top_k_tools: int = 128
    features_dim: int = 4096
    batch_size: int = 64
    max_steps: int = 600
    learning_rate: float = 3e-3
    seed: int = 0


class ToolRouterTrainer:
    """Runs a lightweight JAX tool-router trainer and writes into trainer/status.json."""

    def __init__(
        self,
        status_path: Path = Path("/opt/continuonos/brain/trainer/status.json"),
    ) -> None:
        self.status_path = status_path

    async def run(self, request: ToolRouterTrainRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, request)

    def _run_sync(self, request: ToolRouterTrainRequest) -> Dict[str, Any]:
        cfg = ToolRouterTrainConfig(
            episodes_root=request.episodes_root or Path("/opt/continuonos/brain/rlds/episodes"),
            include_dirs_prefix=request.include_dirs_prefix,
            max_episodes_scan=request.max_episodes_scan,
            top_k_tools=request.top_k_tools,
            features_dim=request.features_dim,
            batch_size=request.batch_size,
            max_steps=request.max_steps,
            learning_rate=request.learning_rate,
            seed=request.seed,
        )
        self._write_status({"status": "running", "tool_router": {"config": cfg.__dict__}})
        try:
            res = train(cfg)
            self._write_status({"status": "ok", "tool_router": res})
            return res
        except Exception as exc:  # noqa: BLE001
            self._write_status({"status": "error", "tool_router": {"message": str(exc)}})
            raise

    def _write_status(self, payload: Dict[str, Any]) -> None:
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass


