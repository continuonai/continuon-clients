import asyncio
from pathlib import Path


class TrainingRunner:
    """Run the local LoRA trainer in a background thread."""

    def __init__(self, config_path: str = "continuonbrain/configs/pi5-donkey.json") -> None:
        self.config_path = config_path

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._run_sync)

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
                return float(loss.detach().cpu().item())

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

        config_path = Path(self.config_path)
        cfg = LocalTrainerJobConfig.from_json(config_path)
        hooks = make_hooks()
        result = run_local_lora_training_job(cfg, hooks, episode_loader=default_episode_loader)
        print("RunTraining completed:", result.status, "steps", result.steps, "avg_loss", result.avg_loss)

