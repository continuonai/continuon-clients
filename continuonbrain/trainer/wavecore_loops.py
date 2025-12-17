"""
WaveCore-style multi-speed loops for nested learning (fast/mid/slow).

- Fast loop: reactive, low-latency updates (~ms–100 ms reflex) for Pi SSM seed adapters.
- Mid loop: bounded background learning (short RLDS windows, LoRA refresh).
- Slow loop: consolidation/aggregation for cloud/JAX/TPU seed exports.

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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .budget_guard import BudgetGuard
from .export_jax import export_adapters_to_npz
from .local_lora_trainer import ModelHooks
from .hooks_numpy import build_numpy_hooks


@dataclass
class LoopConfig:
    """Configuration for each loop speed."""

    name: str
    job_config: LocalTrainerJobConfig
    safety_cfg: SafetyGateConfig
    run_condition: Optional[Callable[[], bool]] = None  # e.g., idle/battery/temp checks


def run_fast_loop(cfg: LoopConfig, guard: BudgetGuard | None = None):
    """
    Fast, reactive loop: prefers richer torch LoRA hooks; falls back to numpy if torch fails.
    """
    if guard:
        guard.check_limits()
    try:
        hooks = _build_torch_lora_hooks(rank=8, hidden=64, lr=5e-3, weight_decay=0.0)
    except Exception:
        hooks = build_numpy_hooks(lr=5e-2)
    return maybe_run_local_training(cfg.job_config, hooks, safety_cfg=cfg.safety_cfg)


def run_mid_loop(cfg: LoopConfig, guard: BudgetGuard | None = None):
    """
    Mid loop: same pipeline with slightly smaller LR.
    """
    if guard:
        guard.check_limits()
    try:
        hooks = _build_torch_lora_hooks(rank=8, hidden=64, lr=2e-3, weight_decay=0.0)
    except Exception:
        hooks = build_numpy_hooks(lr=1e-2)
    return maybe_run_local_training(cfg.job_config, hooks, safety_cfg=cfg.safety_cfg)


def run_slow_loop(cfg: LoopConfig):
    """
    Slow loop: consolidation/export.
      - Export adapters to JAX-friendly npz
      - Report episodes available for cloud/TPU
    """
    episodes = list_local_episodes(cfg.job_config.rlds_dir)
    export_info = None
    candidate = cfg.job_config.adapters_out_dir / cfg.job_config.adapter_filename
    if candidate.exists():
        out_npz = candidate.with_suffix(".jax.npz")
        try:
            export_info = export_adapters_to_npz(candidate, out_npz)
        except Exception as exc:
            export_info = {"error": str(exc)}
    return {
        "status": "exported" if export_info else "pending_export",
        "episodes_available": len(episodes),
        "export": export_info,
    }


def _build_torch_lora_hooks(rank: int, hidden: int, lr: float, weight_decay: float) -> ModelHooks:
    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
            super().__init__()
            self.base = base
            self.rank = rank
            self.alpha = alpha
            in_features = base.in_features
            out_features = base.out_features
            self.A = nn.Parameter(torch.zeros(in_features, rank))
            self.B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            for p in self.base.parameters():
                p.requires_grad = False

        def forward(self, x):
            base_out = self.base(x)
            lora_out = x @ self.A @ self.B * (self.alpha / self.rank)
            return base_out + lora_out

    class TinyPolicy(nn.Module):
        def __init__(self, obs_dim: int = 2, hidden_dim: int = 64, action_dim: int = 2):
            super().__init__()
            self.q_proj = nn.Linear(obs_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.o_proj = nn.Linear(hidden_dim, action_dim)

        def forward(self, obs_batch):
            rows = []
            for obs in obs_batch:
                if isinstance(obs, dict):
                    rows.append([float(v) for v in obs.values() if isinstance(v, (int, float))])
                elif isinstance(obs, (list, tuple)):
                    rows.append([float(v) for v in obs])
                elif isinstance(obs, (int, float)):
                    rows.append([float(obs)])
                else:
                    rows.append([0.0])
            max_len = max(len(r) for r in rows) if rows else 1
            rows = [r + [0.0] * (max_len - len(r)) for r in rows]
            x = torch.tensor(rows, dtype=torch.float32)
            x = F.relu(self.q_proj(x))
            x = F.relu(self.k_proj(x))
            x = F.relu(self.v_proj(x))
            return self.o_proj(x)

    def loss_fn(pred, target):
        target_rows = []
        for act in target:
            if isinstance(act, dict):
                target_rows.append([float(act.get("steering", 0.0)), float(act.get("throttle", 0.0))])
            elif isinstance(act, (list, tuple)):
                target_rows.append([float(act[0]), float(act[1] if len(act) > 1 else 0.0)])
            elif isinstance(act, (int, float)):
                target_rows.append([float(act), 0.0])
            else:
                target_rows.append([0.0, 0.0])
        tgt = torch.tensor(target_rows, dtype=torch.float32)
        if pred.shape[1] != tgt.shape[1]:
            min_dim = min(pred.shape[1], tgt.shape[1])
            pred = pred[:, :min_dim]
            tgt = tgt[:, :min_dim]
        return F.mse_loss(pred, tgt)

    def build_model():
        m = TinyPolicy(hidden_dim=hidden)
        for p in m.parameters():
            p.requires_grad = False
        return m

    def attach(model, layer_names):
        params = []
        for name in layer_names:
            if hasattr(model, name):
                base = getattr(model, name)
                if isinstance(base, nn.Linear):
                    lora = LoRALinear(base, rank=rank, alpha=rank)
                    setattr(model, name, lora)
                    params.extend([p for p in lora.parameters() if p.requires_grad])
        for p in params:
            p.requires_grad = True
        return params

    def make_opt(params, lr, wd):
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["obs"])
        l = loss_fn(pred, batch["action"])
        l.backward()
        optimizer.step()
        model.eval()
        return float(l.detach().cpu().item())

    def save(model, path: Path):
        lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "A" in k or "B" in k}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state, path)

    def load(model, path: Path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    def eval_forward(model, obs):
        with torch.no_grad():
            return model([obs])

    return ModelHooks(
        build_model=build_model,
        attach_lora_adapters=attach,
        make_optimizer=make_opt,
        train_step=step,
        save_adapters=save,
        load_adapters=load,
        eval_forward=eval_forward,
    )


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


