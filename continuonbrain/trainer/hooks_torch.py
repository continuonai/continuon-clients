"""
Torch-based ModelHooks builder.

This wires LocalTrainerJobConfig into a usable torch training loop with LoRA
adapters, assuming the caller supplies:
  - base_model_loader(): returns a model with base weights loaded and frozen.
  - lora_injector(model, layers): adds LoRA modules and returns trainable params.
  - loss_fn(pred, target): standard torch loss.
Optionally:
  - action_distance_fn(new_action, old_action)
  - safety_fn(action) -> bool
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence

from .local_lora_trainer import ModelHooks, ensure_dir

if TYPE_CHECKING:
    import torch


def build_torch_hooks(
    base_model_loader: Callable[[], "torch.nn.Module"],
    lora_injector: Callable[[Any, Sequence[str]], Sequence["torch.nn.Parameter"]],
    loss_fn: Callable[[Any, Any], "torch.Tensor"],
    *,
    action_distance_fn: Callable[[Any, Any], float] | None = None,
    safety_fn: Callable[[Any], bool] | None = None,
    grad_clip_norm: float = 1.0,
) -> ModelHooks:
    import torch

    trainable_params_box: Dict[str, Sequence[torch.nn.Parameter]] = {}

    def build_model() -> torch.nn.Module:
        model = base_model_loader()
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

    def attach_lora_adapters(model: torch.nn.Module, layers: Sequence[str]) -> Sequence[torch.nn.Parameter]:
        params = lora_injector(model, layers)
        for p in params:
            p.requires_grad = True
        trainable_params_box["params"] = params
        return params

    def make_optimizer(trainable_params: Sequence[torch.nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Dict[str, Any]) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        obs = batch["obs"]
        action = batch["action"]
        pred = model(obs)
        loss = loss_fn(pred, action)
        loss.backward()
        if grad_clip_norm and "params" in trainable_params_box:
            torch.nn.utils.clip_grad_norm_(trainable_params_box["params"], grad_clip_norm)
        optimizer.step()
        model.eval()
        return float(loss.detach().cpu().item())

    def save_adapters(model: torch.nn.Module, path: Path) -> None:
        ensure_dir(path.parent)
        lora_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
        torch.save(lora_state, path)

    def load_adapters(model: torch.nn.Module, path: Path) -> torch.nn.Module:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    def eval_forward(model: torch.nn.Module, obs: Any) -> Any:
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
