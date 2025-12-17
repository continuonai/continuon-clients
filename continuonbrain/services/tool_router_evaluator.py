from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.jax_models.eval.tool_router_eval import ToolRouterEvalConfig, evaluate


@dataclass
class ToolRouterEvalRequest:
    episodes_root: Optional[Path] = None
    include_dirs_prefix: str = "toolchat_hf"
    export_dir: Optional[Path] = None
    max_episodes_scan: int = 30000
    eval_mod: int = 10
    eval_bucket: int = 0
    k: int = 5


class ToolRouterEvaluator:
    """Runs tool-router heldout eval and writes a compact result into trainer/status.json."""

    def __init__(
        self,
        status_path: Path = Path("/opt/continuonos/brain/trainer/status.json"),
    ) -> None:
        self.status_path = status_path

    async def run(self, request: ToolRouterEvalRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_sync, request)

    def _run_sync(self, request: ToolRouterEvalRequest) -> Dict[str, Any]:
        cfg = ToolRouterEvalConfig(
            episodes_root=request.episodes_root or Path("/opt/continuonos/brain/rlds/episodes"),
            include_dirs_prefix=request.include_dirs_prefix,
            export_dir=request.export_dir or Path("/opt/continuonos/brain/model/adapters/candidate/tool_router_seed"),
            max_episodes_scan=request.max_episodes_scan,
            eval_mod=request.eval_mod,
            eval_bucket=request.eval_bucket,
            k=request.k,
        )
        self._write_status({"status": "running", "tool_router_eval": {"config": cfg.__dict__}})
        try:
            res = evaluate(cfg)
            self._write_status({"status": "ok", "tool_router_eval": res})
            return res
        except Exception as exc:  # noqa: BLE001
            self._write_status({"status": "error", "tool_router_eval": {"message": str(exc)}})
            raise

    def _write_status(self, payload: Dict[str, Any]) -> None:
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            pass


