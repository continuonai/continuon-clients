from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .local_lora_trainer import (
    GatingSensors,
    LocalTrainerJobConfig,
    ModelHooks,
    SafetyGateConfig,
    TrainerResult,
    list_local_episodes,
    maybe_run_local_training,
    should_run_training,
)


class SidecarTrainer:
    """Thin helper to run the local trainer from a sidecar process.

    - Reuses the RLDS directory defined in the job config.
    - Skips work if the robot is busy or battery/thermals fail gating sensors.
    - Only promotes adapters after the safety gate inside ``maybe_run_local_training`` passes.
    """

    def __init__(
        self,
        cfg_path: Path,
        hooks: ModelHooks,
        safety_cfg: SafetyGateConfig,
        gating: Optional[GatingSensors] = None,
    ) -> None:
        self.cfg_path = cfg_path
        self.hooks = hooks
        self.safety_cfg = safety_cfg
        self.gating = gating
        self._last_episode_mtime: Optional[float] = None
        self._last_episode_count: int = 0

    def _latest_episode_time(self, episodes_dir: Path) -> float:
        episodes = list_local_episodes(episodes_dir)
        if not episodes:
            return 0.0
        return max(p.stat().st_mtime for p in episodes)

    def train_if_new_data(self) -> TrainerResult:
        cfg = LocalTrainerJobConfig.from_json(self.cfg_path)

        ok, reason = should_run_training(self.gating)
        if not ok:
            return TrainerResult(
                status="skipped",
                reason=reason,
                steps=0,
                avg_loss=0.0,
                adapter_path=None,
                log_path=None,
                wall_time_s=0.0,
                log=[reason] if reason else [],
            )

        if not cfg.rlds_dir.exists():
            message = f"RLDS directory missing: {cfg.rlds_dir}"
            return TrainerResult(
                status="skipped",
                reason=message,
                steps=0,
                avg_loss=0.0,
                adapter_path=None,
                log_path=None,
                wall_time_s=0.0,
                log=[message],
            )

        episodes = list_local_episodes(cfg.rlds_dir)
        if len(episodes) < cfg.min_episodes:
            message = f"Not enough episodes in {cfg.rlds_dir} (found {len(episodes)}, need {cfg.min_episodes})"
            return TrainerResult(
                status="skipped",
                reason=message,
                steps=0,
                avg_loss=0.0,
                adapter_path=None,
                log_path=None,
                wall_time_s=0.0,
                log=[message],
            )

        latest_mtime = self._latest_episode_time(cfg.rlds_dir)
        if (
            self._last_episode_mtime is not None
            and latest_mtime <= self._last_episode_mtime
            and len(episodes) == self._last_episode_count
        ):
            return TrainerResult(
                status="skipped",
                reason="No new RLDS episodes since last promotion",
                steps=0,
                avg_loss=0.0,
                adapter_path=None,
                log_path=None,
                wall_time_s=0.0,
                log=["No new RLDS episodes since last promotion"],
            )

        start = time.time()
        result = maybe_run_local_training(
            cfg=cfg,
            hooks=self.hooks,
            safety_cfg=self.safety_cfg,
            gating=self.gating,
        )

        if result.status == "ok":
            self._last_episode_mtime = latest_mtime
            self._last_episode_count = len(episodes)
        else:
            # Preserve timing for skipped/failed runs
            result.wall_time_s = max(result.wall_time_s, time.time() - start)
        return result

    def as_dict(self) -> dict:
        return {
            "cfg_path": str(self.cfg_path),
            "last_episode_mtime": self._last_episode_mtime,
            "last_episode_count": self._last_episode_count,
            "hooks": type(self.hooks).__name__,
            "safety_cfg": asdict(self.safety_cfg),
        }
