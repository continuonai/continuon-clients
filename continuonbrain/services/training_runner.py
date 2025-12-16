import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class TrainingRunner:
    """Run the local LoRA trainer in a background thread."""

    def __init__(
        self,
        config_path: str = "continuonbrain/configs/pi5-donkey.json",
        config_dir: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.config_dir = config_dir

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._run_sync)

    def _base_dir(self) -> Path:
        """Resolve the offline-first runtime directory for trainer artifacts."""

        if self.config_dir:
            return Path(self.config_dir)
        return Path("/opt/continuonos/brain")

    def _safe_write_status(self, status_path: Path, payload: dict[str, Any]) -> None:
        """Persist trainer status without raising if the path is unavailable."""

        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"[TrainingRunner] Failed to write status to {status_path}: {exc}")

    def _run_sync(self) -> None:
        # Lazy imports to avoid heavy startup cost at module import time
        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from continuonbrain.trainer.local_lora_trainer import (
            LocalTrainerJobConfig,
            run_local_lora_training_job,
            default_episode_loader,
        )
        from continuonbrain.trainer.local_lora_trainer import ModelHooks

        base_dir = self._base_dir()
        trainer_dir = base_dir / "trainer"
        status_path = trainer_dir / "status.json"
        logs_dir = trainer_dir / "logs"
        log_path: Optional[Path] = None
        log_dir_ready = False

        logger = logging.getLogger("continuonbrain.training_runner")
        logger.setLevel(logging.INFO)

        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            log_path = logs_dir / f"trainer_{timestamp}.log"
            handler: logging.Handler = logging.FileHandler(log_path)
            log_dir_ready = True
        except Exception as exc:  # noqa: BLE001
            print(f"[TrainingRunner] Falling back to stdout logging: {exc}")
            handler = logging.StreamHandler()

        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

        start_time = datetime.utcnow().isoformat()

        def write_status(
            state: str,
            steps: int = 0,
            avg_loss: Optional[float] = None,
            adapter_path: Optional[Path] = None,
            message: Optional[str] = None,
            completed_at: Optional[str] = None,
        ) -> None:
            payload = {
                "state": state,
                "steps": steps,
                "avg_loss": avg_loss,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "log_path": str(log_path) if log_path else None,
                "started_at": start_time,
                "updated_at": datetime.utcnow().isoformat(),
            }
            if completed_at:
                payload["completed_at"] = completed_at
            if message:
                payload["message"] = message
            self._safe_write_status(status_path, payload)

        write_status("starting", steps=0, avg_loss=None, adapter_path=None)
        logger.info("Starting LoRA training job using config %s", self.config_path)

        config_path = Path(self.config_path)
        cfg = LocalTrainerJobConfig.from_json(config_path)
        if log_dir_ready:
            cfg.log_dir = logs_dir
        # Keep artifacts within the resolved base directory when possible.
        if not cfg.adapters_out_dir.is_absolute() or str(cfg.adapters_out_dir).startswith("/opt/continuonos/brain"):
            cfg.adapters_out_dir = base_dir / "model" / "adapters" / "candidate"
        cfg.adapters_out_dir.mkdir(parents=True, exist_ok=True)

        steps_seen = 0
        total_loss = 0.0
        batches_seen = 0

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
                for param in self.base.parameters():
                    param.requires_grad = False

            def forward(self, x):
                base_out = self.base(x)
                lora_out = x @ self.A @ self.B * (self.alpha / self.rank)
                return base_out + lora_out

        OBS_DIM = 2
        HIDDEN = 64

        class TinyPolicy(nn.Module):
            def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN, action_dim: int = 2):
                super().__init__()
                self.q_proj = nn.Linear(obs_dim, hidden)
                self.k_proj = nn.Linear(hidden, hidden)
                self.v_proj = nn.Linear(hidden, hidden)
                self.o_proj = nn.Linear(hidden, action_dim)

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
                max_len = max(len(row) for row in rows) if rows else 1
                rows = [row + [0.0] * (max_len - len(row)) for row in rows]
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

        def make_hooks():
            trainable_box = {}

            def build_model():
                model = TinyPolicy()
                for param in model.parameters():
                    param.requires_grad = False
                return model

            def attach(model, layer_names):
                params = []
                for name in layer_names:
                    if hasattr(model, name):
                        base = getattr(model, name)
                        if isinstance(base, nn.Linear):
                            lora = LoRALinear(base, rank=8, alpha=8.0)
                            setattr(model, name, lora)
                            params.extend([param for param in lora.parameters() if param.requires_grad])
                            trainable_box[name] = lora
                return trainable_box, params

            def make_opt(params):
                return torch.optim.Adam(params, lr=1e-3)

            def step(model, batch, optimizer):
                optimizer.zero_grad()
                pred = model(batch["obs"])
                loss = loss_fn(pred, batch["action"])
                loss.backward()
                optimizer.step()
                model.eval()
                nonlocal steps_seen, total_loss, batches_seen
                loss_value = float(loss.detach().cpu().item())
                steps_seen += 1
                total_loss += loss_value
                batches_seen += 1
                avg_loss = total_loss / max(1, batches_seen)
                write_status("running", steps=steps_seen, avg_loss=avg_loss, adapter_path=None)
                logger.info("Step %s avg_loss=%.6f", steps_seen, avg_loss)
                return loss_value

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

        hooks = make_hooks()
        try:
            result = run_local_lora_training_job(cfg, hooks, episode_loader=default_episode_loader)
            completed_at = datetime.utcnow().isoformat()
            write_status(
                result.status,
                steps=result.steps,
                avg_loss=result.avg_loss,
                adapter_path=result.adapter_path,
                completed_at=completed_at,
            )
            logger.info(
                "RunTraining completed: status=%s steps=%s avg_loss=%.6f adapter=%s",
                result.status,
                result.steps,
                result.avg_loss,
                result.adapter_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training run failed: %s", exc)
            write_status(
                "error",
                steps=steps_seen,
                avg_loss=(total_loss / max(1, batches_seen)) if batches_seen else None,
                adapter_path=None,
                message=str(exc),
                completed_at=datetime.utcnow().isoformat(),
            )
        finally:
            logger.removeHandler(handler)
            handler.close()
