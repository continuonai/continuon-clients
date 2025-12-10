"""
WaveCore-style multi-speed loops for nested learning (fast/mid/slow).

Fast loop: reactive, low-latency updates (e.g., on-device small adapters).
Mid loop: bounded background learning (e.g., LoRA refresh from latest RLDS).
Slow loop: consolidation/aggregation for cloud/JAX/TPU seed updates.

This is a sketch scaffold; plug in your real hooks/adapters/exporters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .local_lora_trainer import (
    LocalTrainerJobConfig,
    SafetyGateConfig,
    maybe_run_local_training,
    list_local_episodes,
)
from .hooks_numpy import build_numpy_hooks
from .budget_guard import BudgetGuard


@dataclass
class LoopConfig:
    """Configuration for each loop speed."""

    name: str
    job_config: LocalTrainerJobConfig
    safety_cfg: SafetyGateConfig
    run_condition: Optional[Callable[[], bool]] = None  # e.g., idle/battery/temp checks


def run_fast_loop(cfg: LoopConfig, guard: BudgetGuard | None = None):
    """
    Fast, reactive loop: can run frequently with small budgets.
    Example: quick adapter refresh using lightweight hooks (numpy).
    """
    if guard:
        guard.check_limits()
    hooks = build_numpy_hooks(lr=5e-2)
    return maybe_run_local_training(cfg.job_config, hooks, safety_cfg=cfg.safety_cfg)


def run_mid_loop(cfg: LoopConfig, guard: BudgetGuard | None = None):
    """
    Mid loop: uses the same pipeline but may include heavier hooks or more data.
    """
    if guard:
        guard.check_limits()
    hooks = build_numpy_hooks(lr=1e-2)
    return maybe_run_local_training(cfg.job_config, hooks, safety_cfg=cfg.safety_cfg)


def run_slow_loop(cfg: LoopConfig):
    """
    Slow loop: consolidation/export placeholder.
    In a real implementation, this would:
      - Export adapters/weights
      - Package RLDS for cloud/TPU
      - Trigger JAX/TPU training jobs
    """
    episodes = list_local_episodes(cfg.job_config.rlds_dir)
    return {
        "status": "pending_export",
        "episodes_available": len(episodes),
        "export_hint": "Package RLDS + adapters for cloud JAX/TPU run.",
    }


def run_wavecore_loops(
    fast_cfg: LoopConfig,
    mid_cfg: LoopConfig,
    slow_cfg: LoopConfig,
    guard: BudgetGuard | None = None,
):
    """
    Execute the three loops sequentially. Replace with schedulers/cron/systemd for real use.
    WaveCore/HOPE framing:
      - Fast: reactive updates (on-device, small adapters)
      - Mid: bounded background learning
      - Slow: consolidation/export for cloud/TPU
    """
    if guard is None:
        guard = BudgetGuard()
    results = {
        "fast": run_fast_loop(fast_cfg, guard),
        "mid": run_mid_loop(mid_cfg, guard),
        "slow": run_slow_loop(slow_cfg),
        "budget": guard.snapshot(),
    }
    return results


