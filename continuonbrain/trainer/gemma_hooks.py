"""
Gemma-focused LoRA hooks for Pi 5.

This module provides a torch-based LoRA injector that mutates target Linear
layers in-place, keeping compatibility with the existing trainer interface
(`attach_lora_adapters` mutates the model instance instead of wrapping it).

Use when you have a Gemma model loader that returns a torch.nn.Module already
placed on CPU (or a delegated device).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence

from .local_lora_trainer import ModelHooks, ensure_dir


def build_gemma_lora_hooks(
    base_model_loader: Callable[[], Any],
    loss_fn: Callable[[Any, Any], Any],
    *,
    target_linear_names: Sequence[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    action_distance_fn: Callable[[Any, Any], float] | None = None,
    safety_fn: Callable[[Any], bool] | None = None,
    grad_clip_norm: float = 1.0,
) -> ModelHooks:
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
            super().__init__()
            self.base = base
            self.rank = rank
            self.alpha = alpha
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            in_features = base.in_features
            out_features = base.out_features
            self.A = nn.Parameter(torch.zeros(in_features, rank))
            self.B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.kaiming_uniform_(self.A, a=(5**0.5))
            nn.init.zeros_(self.B)
            for p in self.base.parameters():
                p.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_dropped = self.dropout(x)
            lora_out = x_dropped @ self.A @ self.B * (self.alpha / self.rank)
            return self.base(x) + lora_out

    trainable_box: Dict[str, Sequence[torch.nn.Parameter]] = {}

    def build_model() -> Any:
        model = base_model_loader()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def attach_lora_adapters(model: Any, layers: Sequence[str]) -> Sequence[Any]:
        target_names = set(layers or target_linear_names)
        replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name in target_names:
                lora = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                _replace_submodule(model, name, lora)
                replaced += 1
        params = [p for p in model.parameters() if p.requires_grad]
        trainable_box["params"] = params
        if replaced == 0:
            raise RuntimeError(f"No target Linear layers matched: {target_names}")
        return params

    def make_optimizer(trainable_params: Sequence[Any], lr: float, weight_decay: float) -> Any:
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    def train_step(model: Any, optimizer: Any, batch: Dict[str, Any]) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["obs"])
        loss = loss_fn(pred, batch["action"])
        loss.backward()
        if grad_clip_norm and "params" in trainable_box:
            torch.nn.utils.clip_grad_norm_(trainable_box["params"], grad_clip_norm)
        optimizer.step()
        model.eval()
        return float(loss.detach().cpu().item())

    def save_adapters(model: Any, path) -> None:
        ensure_dir(path.parent)
        lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
        torch.save(lora_state, path)

    def load_adapters(model: Any, path) -> Any:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    def eval_forward(model: Any, obs: Any) -> Any:
        with torch.no_grad():
            return model(obs)

    return ModelHooks(
        build_model=build_model,
        attach_lora_adapters=attach_lora_adapters,
        make_optimizer=make_optimizer,
        train_step=train_step,
        save_adapters=save_adapters,
        load_adapters=load_adapters,
        eval_forward=eval_forward,
        action_distance=action_distance_fn,
        violates_safety=safety_fn,
    )


def _replace_submodule(model: Any, name: str, new_module: Any) -> None:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)
