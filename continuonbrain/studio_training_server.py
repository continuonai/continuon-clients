"""
Studio training server (hardware-free).

Goal: provide a lightweight server that can be used from Continuon Brain Studio UI
to trigger WaveCore JAX loops and write artifacts into canonical runtime paths:

  /opt/continuonos/brain/trainer/checkpoints/core_model_seed
  /opt/continuonos/brain/model/adapters/candidate/core_model_seed

This server intentionally avoids importing depth cameras / servo controllers so it
can run on laptops (WSL2 GPU) as well as Pi.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

from continuonbrain.server.routes import SimpleJSONServer
from continuonbrain.services.wavecore_trainer import WavecoreTrainer


class TrainingOnlyService:
    def __init__(self, *, config_dir: str, runtime_root: str = "/opt/continuonos/brain") -> None:
        self.config_dir = config_dir
        self.runtime_root = Path(runtime_root)
        self.wavecore_trainer = WavecoreTrainer(
            default_rlds_dir=self.runtime_root / "rlds" / "episodes",
            log_dir=self.runtime_root / "trainer" / "logs",
            checkpoint_dir=self.runtime_root / "trainer" / "checkpoints" / "core_model_seed",
            export_dir=self.runtime_root / "model" / "adapters" / "candidate" / "core_model_seed",
        )
        self._mode = "idle"
        self._allow_motion = False

    async def GetRobotStatus(self) -> dict:
        return {
            "success": True,
            "status": {
                "mode": self._mode,
                "allow_motion": self._allow_motion,
                "hardware_mode": "training_only",
                "capabilities": {"training": True, "wavecore": True},
            },
        }

    async def GetGates(self) -> dict:
        return {"success": True, "gates": {"allow_motion": self._allow_motion, "mode": self._mode}, "mode": self._mode}

    async def GetLoopHealth(self) -> dict:
        return {"success": True, "metrics": {}, "gates": {"allow_motion": self._allow_motion, "mode": self._mode}}

    async def SetRobotMode(self, mode: str) -> dict:
        self._mode = (mode or "idle").strip()
        if self._mode not in ("idle", "emergency_stop"):
            # Keep training server safe by default.
            self._mode = "idle"
        return {"success": True, "mode": self._mode}

    async def ResetSafetyGates(self) -> dict:
        self._mode = "idle"
        self._allow_motion = False
        return {"success": True, "mode": self._mode, "gates": {"allow_motion": self._allow_motion}}

    async def RunWavecoreLoops(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure canonical dirs exist
        (self.runtime_root / "trainer" / "logs").mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "trainer" / "checkpoints" / "core_model_seed").mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "model" / "adapters" / "candidate" / "core_model_seed").mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "rlds" / "episodes").mkdir(parents=True, exist_ok=True)

        # Force JAX preference for the process
        os.environ["CONTINUON_PREFER_JAX"] = "1"

        return await self.wavecore_trainer.run_loops(payload or {})

    # Optional no-op stubs for templates/pages that might call tasks/skills
    async def ListTasks(self, include_ineligible: bool = False) -> dict:
        return {"success": True, "tasks": [], "selected_task_id": None, "source": "training_only"}

    async def ListSkills(self, include_ineligible: bool = False) -> dict:
        return {"success": True, "skills": [], "selected_skill_id": None, "source": "training_only"}

    async def GetTaskSummary(self, task_id: str) -> dict:
        return {"success": False, "message": "not supported"}

    async def GetSkillSummary(self, skill_id: str) -> dict:
        return {"success": False, "message": "not supported"}

    async def SelectTask(self, task_id: str, reason: Optional[str] = None) -> dict:
        return {"success": False, "accepted": False, "message": "not supported"}

    async def SendCommand(self, payload: Dict[str, Any]) -> dict:
        return {"success": False, "message": "not supported"}

    async def Drive(self, steering: Any, throttle: Any) -> dict:
        return {"success": False, "message": "not supported"}

    async def ChatWithGemma(self, message: str, history: list) -> dict:
        return {"success": False, "message": "chat not available in training_only server"}


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Continuon Studio Training Server (hardware-free)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--config-dir", default="/tmp/continuonbrain_demo")
    parser.add_argument("--runtime-root", default="/opt/continuonos/brain")
    args = parser.parse_args()

    svc = TrainingOnlyService(config_dir=args.config_dir, runtime_root=args.runtime_root)
    server = SimpleJSONServer(svc)
    await server.start(host=args.host, port=args.port)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()


