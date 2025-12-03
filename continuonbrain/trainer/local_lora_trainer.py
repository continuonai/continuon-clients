"""
Local LoRA trainer scaffold for Pi-class devices (offline-first).

This module keeps the base model frozen and only updates adapters under strict
wall-time and step budgets. It operates exclusively on local RLDS episodes and
uses a candidate/current/history adapter rotation with a simple safety gate.

The hooks are intentionally pluggable so you can drop in Torch/ONNX/TFLite
implementations without changing the control flow.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence


# ------------------------------- Data classes ------------------------------- #


@dataclass
class LocalTrainerJobConfig:
    rlds_dir: Path
    min_episodes: int = 16
    max_episodes: int = 256
    max_steps: int = 500
    max_wall_time_s: int = 300
    batch_size: int = 32
    lora_layers: Sequence[str] = ()
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adapters_out_dir: Path = Path("/opt/continuonos/brain/model/adapters/candidate")
    adapter_filename: str = "lora_adapters.pt"
    log_dir: Path = Path("/opt/continuonos/brain/trainer/logs")
    shuffle_buffer_multiplier: int = 4
    log_every_steps: int = 10
    base_model_path: Optional[Path] = None
    base_model_url: Optional[str] = None

    @staticmethod
    def from_json(path: Path) -> "LocalTrainerJobConfig":
        data = json.loads(path.read_text())
        # Allow rlds_dir to be passed as string
        data["rlds_dir"] = Path(data["rlds_dir"])
        data["adapters_out_dir"] = Path(data.get("adapters_out_dir", LocalTrainerJobConfig.adapters_out_dir))
        data["log_dir"] = Path(data.get("log_dir", LocalTrainerJobConfig.log_dir))
        if "base_model_path" in data and data["base_model_path"] is not None:
            data["base_model_path"] = Path(data["base_model_path"])
        if "base_model_url" not in data:
            data["base_model_url"] = None
        return LocalTrainerJobConfig(**data)

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        payload["rlds_dir"] = str(self.rlds_dir)
        payload["adapters_out_dir"] = str(self.adapters_out_dir)
        payload["log_dir"] = str(self.log_dir)
        if self.base_model_path is not None:
            payload["base_model_path"] = str(self.base_model_path)
        if self.base_model_url is None:
            payload.pop("base_model_url", None)
        path.write_text(json.dumps(payload, indent=2))

    @property
    def adapter_path(self) -> Path:
        return self.adapters_out_dir / self.adapter_filename


@dataclass
class TrainerResult:
    status: str
    steps: int
    avg_loss: float
    adapter_path: Optional[Path]
    log_path: Optional[Path]
    wall_time_s: float
    reason: Optional[str] = None
    log: Optional[List[str]] = None


@dataclass
class SafetyGateConfig:
    avg_action_delta_threshold: float = 0.25
    max_eval_steps: int = 1024
    eval_tail_episodes: int = 8


@dataclass
class GatingSensors:
    robot_idle: Callable[[], bool]
    battery_level: Callable[[], float]
    cpu_temp_c: Callable[[], float]
    teleop_active: Callable[[], bool]
    min_battery: float = 0.4
    max_temp_c: float = 75.0


@dataclass
class ModelHooks:
    """
    Hook surface for integrating a concrete model/optimizer implementation.

    Required:
      - build_model(): returns a model with base weights loaded and frozen.
      - attach_lora_adapters(model, layers): mutates model, returns trainable params.
      - make_optimizer(trainable_params, lr, weight_decay): returns optimizer object.
      - train_step(model, optimizer, batch): performs forward/backward/update, returns float loss.
      - save_adapters(model, path): persists adapter state only.
      - load_adapters(model, path): loads adapter state into model (used for eval gate).

    Optional:
      - eval_forward(model, obs): inference used during safety gate (defaults to train forward).
      - action_distance(new_action, old_action): returns float delta for safety gate metric.
      - violates_safety(action): returns True if action should be rejected.
    """

    build_model: Callable[[], Any]
    attach_lora_adapters: Callable[[Any, Sequence[str]], Sequence[Any]]
    make_optimizer: Callable[[Sequence[Any], float, float], Any]
    train_step: Callable[[Any, Any, Dict[str, Any]], float]
    save_adapters: Callable[[Any, Path], None]
    load_adapters: Callable[[Any, Path], Any]
    eval_forward: Optional[Callable[[Any, Any], Any]] = None
    action_distance: Optional[Callable[[Any, Any], float]] = None
    violates_safety: Optional[Callable[[Any], bool]] = None


# ------------------------------- RLDS helpers ------------------------------- #


def list_local_episodes(rlds_dir: Path) -> List[Path]:
    if not rlds_dir.exists():
        return []
    episodes = sorted(
        [p for p in rlds_dir.iterdir() if p.is_file() and p.suffix in {".tfrecord", ".jsonl", ".json"}]
    )
    return episodes


def default_episode_loader(episode_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Default lightweight loader:
    - For *.jsonl: each line is a step with keys "obs" and "action".
    - For *.json: expects {"steps": [{obs:..., action:...}, ...]}.
    - TFRecord is intentionally unsupported here to avoid heavy deps; supply a custom loader.
    """
    if episode_path.suffix == ".jsonl":
        with episode_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "obs" in sample and "action" in sample:
                    yield sample
    elif episode_path.suffix == ".json":
        payload = json.loads(episode_path.read_text())
        for step in payload.get("steps", []):
            if "obs" in step and "action" in step:
                yield step
    else:
        raise ValueError(f"No loader for {episode_path.suffix}; provide a custom episode_loader.")


def make_batch_iterator(
    episode_files: Sequence[Path],
    batch_size: int,
    episode_loader: Callable[[Path], Iterable[Dict[str, Any]]],
    buffer_multiplier: int = 4,
    drop_last: bool = True,
) -> Iterator[Dict[str, Any]]:
    buffer: List[Dict[str, Any]] = []
    target_buffer = max(batch_size * buffer_multiplier, batch_size)

    def flush_buffer(allow_partial: bool = False) -> Iterator[Dict[str, Any]]:
        nonlocal buffer
        if not buffer:
            return iter(())

        random.shuffle(buffer)
        while len(buffer) >= batch_size:
            batch = buffer[:batch_size]
            buffer = buffer[batch_size:]
            yield collate_batch(batch)
        if allow_partial and buffer:
            yield collate_batch(buffer)
            buffer = []

    for episode_path in episode_files:
        for sample in episode_loader(episode_path):
            buffer.append(sample)
            if len(buffer) >= target_buffer:
                yield from flush_buffer()
    if buffer:
        yield from flush_buffer(allow_partial=not drop_last)


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs = [item["obs"] for item in batch]
    action = [item["action"] for item in batch]
    return {"obs": obs, "action": action}


# ------------------------------- Logging utils ------------------------------ #


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotate_logs(log_dir: Path, keep: int = 20) -> None:
    ensure_dir(log_dir)
    logs = sorted(log_dir.glob("trainer_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in logs[keep:]:
        stale.unlink(missing_ok=True)


def write_trainer_log(log_dir: Path, lines: List[str]) -> Path:
    ensure_dir(log_dir)
    ts = time.strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"trainer_{ts}.log"
    log_path.write_text("\n".join(lines))
    rotate_logs(log_dir)
    return log_path


# ------------------------------- Trainer core ------------------------------- #


def run_local_lora_training_job(
    cfg: LocalTrainerJobConfig,
    hooks: ModelHooks,
    episode_loader: Callable[[Path], Iterable[Dict[str, Any]]] = default_episode_loader,
) -> TrainerResult:
    start_time = time.time()
    log: List[str] = []

    episode_files = list_local_episodes(cfg.rlds_dir)
    if len(episode_files) < cfg.min_episodes:
        reason = f"Not enough episodes ({len(episode_files)} < {cfg.min_episodes})"
        log.append(reason)
        log_path = write_trainer_log(cfg.log_dir, log)
        return TrainerResult(
            status="skipped",
            reason=reason,
            steps=0,
            avg_loss=0.0,
            adapter_path=None,
            log_path=log_path,
            wall_time_s=0.0,
            log=log,
        )

    # Keep only the most recent max_episodes to bound work
    episode_files = episode_files[-cfg.max_episodes :]
    log.append(f"Using {len(episode_files)} local episodes from {cfg.rlds_dir}")

    model = hooks.build_model()
    trainable_params = hooks.attach_lora_adapters(model, cfg.lora_layers)
    optimizer = hooks.make_optimizer(trainable_params, cfg.learning_rate, cfg.weight_decay)

    steps = 0
    total_loss = 0.0
    batches_seen = 0

    for batch in make_batch_iterator(
        episode_files,
        cfg.batch_size,
        episode_loader=episode_loader,
        buffer_multiplier=cfg.shuffle_buffer_multiplier,
    ):
        now = time.time()
        if now - start_time > cfg.max_wall_time_s:
            log.append("Time budget reached; stopping training.")
            break
        if steps >= cfg.max_steps:
            log.append("Step budget reached; stopping training.")
            break

        loss_value = hooks.train_step(model, optimizer, batch)
        total_loss += float(loss_value)
        batches_seen += 1
        steps += 1

        if steps % cfg.log_every_steps == 0:
            avg_loss = total_loss / max(1, batches_seen)
            log.append(f"Step {steps}: avg_loss={avg_loss:.4f}")

    if batches_seen == 0:
        reason = "No training batches produced; skipping adapter save."
        log.append(reason)
        wall_time_s = time.time() - start_time
        log_path = write_trainer_log(cfg.log_dir, log)
        return TrainerResult(
            status="no_data",
            reason=reason,
            steps=steps,
            avg_loss=0.0,
            adapter_path=None,
            log_path=log_path,
            wall_time_s=wall_time_s,
            log=log,
        )

    ensure_dir(cfg.adapters_out_dir)
    adapter_path = cfg.adapter_path
    hooks.save_adapters(model, adapter_path)

    avg_loss = total_loss / max(1, batches_seen)
    log.append(f"Training finished at step={steps}, avg_loss={avg_loss:.4f}")
    log.append(f"Saved candidate adapters to {adapter_path}")

    wall_time_s = time.time() - start_time
    log_path = write_trainer_log(cfg.log_dir, log)

    return TrainerResult(
        status="ok",
        steps=steps,
        avg_loss=avg_loss,
        adapter_path=adapter_path,
        log_path=log_path,
        wall_time_s=wall_time_s,
        log=log,
    )


# ------------------------------- Safety gate -------------------------------- #


def evaluate_candidate_adapters(
    candidate_path: Path,
    hooks: ModelHooks,
    safety_cfg: SafetyGateConfig,
    episode_loader: Callable[[Path], Iterable[Dict[str, Any]]] = default_episode_loader,
    eval_files: Optional[Sequence[Path]] = None,
) -> bool:
    if not candidate_path.exists():
        return False

    model = hooks.build_model()
    hooks.load_adapters(model, candidate_path)
    forward = hooks.eval_forward or (lambda m, obs: m(obs))
    action_delta = hooks.action_distance or default_action_distance
    violates_safety = hooks.violates_safety or (lambda action: False)

    metrics = {"safety_violations": 0, "avg_action_delta": 0.0, "steps": 0}

    if eval_files is None or len(eval_files) == 0:
        return True  # Nothing to evaluate against; allow promotion

    steps_budget = safety_cfg.max_eval_steps
    for episode_path in eval_files:
        for sample in episode_loader(episode_path):
            obs = sample.get("obs")
            old_action = sample.get("action")
            new_action = forward(model, obs)
            delta = action_delta(new_action, old_action)
            metrics["avg_action_delta"] += delta
            metrics["steps"] += 1
            if violates_safety(new_action):
                metrics["safety_violations"] += 1
            if metrics["steps"] >= steps_budget:
                break
        if metrics["steps"] >= steps_budget:
            break

    if metrics["steps"] > 0:
        metrics["avg_action_delta"] /= metrics["steps"]

    if metrics["safety_violations"] > 0:
        return False
    if metrics["avg_action_delta"] > safety_cfg.avg_action_delta_threshold:
        return False
    return True


def promote_candidate_adapters_if_safe(
    candidate_path: Path,
    current_dir: Path,
    history_dir: Path,
    hooks: ModelHooks,
    safety_cfg: SafetyGateConfig,
    eval_files: Sequence[Path],
    episode_loader: Callable[[Path], Iterable[Dict[str, Any]]] = default_episode_loader,
) -> bool:
    if not candidate_path.exists():
        return False

    if not evaluate_candidate_adapters(
        candidate_path=candidate_path,
        hooks=hooks,
        safety_cfg=safety_cfg,
        episode_loader=episode_loader,
        eval_files=eval_files,
    ):
        candidate_path.unlink(missing_ok=True)
        return False

    ensure_dir(current_dir)
    ensure_dir(history_dir)

    ts = time.strftime("%Y%m%dT%H%M%S")
    current_adapter = current_dir / candidate_path.name
    if current_adapter.exists():
        archived = history_dir / f"{candidate_path.stem}_{ts}{candidate_path.suffix}"
        shutil.move(str(current_adapter), str(archived))

    shutil.move(str(candidate_path), str(current_adapter))
    return True


def default_action_distance(new_action: Any, old_action: Any) -> float:
    try:
        return abs(float(new_action) - float(old_action))
    except Exception:
        return 0.0


# ------------------------------- Scheduler hook ----------------------------- #


def should_run_training(gating: Optional[GatingSensors]) -> (bool, Optional[str]):
    if gating is None:
        return True, None
    if not gating.robot_idle():
        return False, "robot not idle"
    if gating.teleop_active():
        return False, "teleop active"
    if gating.battery_level() < gating.min_battery:
        return False, "battery too low"
    if gating.cpu_temp_c() > gating.max_temp_c:
        return False, "cpu temperature too high"
    return True, None


def maybe_run_local_training(
    cfg: LocalTrainerJobConfig,
    hooks: ModelHooks,
    safety_cfg: SafetyGateConfig,
    gating: Optional[GatingSensors] = None,
    episode_loader: Callable[[Path], Iterable[Dict[str, Any]]] = default_episode_loader,
) -> TrainerResult:
    ok, reason = should_run_training(gating)
    if not ok:
        return TrainerResult(
            status="skipped",
            reason=reason,
            steps=0,
            avg_loss=0.0,
            adapter_path=None,
            log_path=None,
            wall_time_s=0.0,
            log=[f"Skipped: {reason}"] if reason else [],
        )

    result = run_local_lora_training_job(cfg, hooks, episode_loader=episode_loader)
    if result.status != "ok" or result.adapter_path is None:
        return result

    # Use most recent eval_tail_episodes for shadow test
    episodes = list_local_episodes(cfg.rlds_dir)
    eval_files = episodes[-safety_cfg.eval_tail_episodes :]

    promoted = promote_candidate_adapters_if_safe(
        candidate_path=result.adapter_path,
        current_dir=cfg.adapters_out_dir.parent / "current",
        history_dir=cfg.adapters_out_dir.parent / "history",
        hooks=hooks,
        safety_cfg=safety_cfg,
        eval_files=eval_files,
        episode_loader=episode_loader,
    )
    if promoted:
        result.log = (result.log or []) + ["Promoted candidate adapters to current/"]
    else:
        result.log = (result.log or []) + ["Candidate adapters rejected by safety gate"]
    return result


# ------------------------------- CLI helper --------------------------------- #


def build_stub_hooks() -> ModelHooks:
    """
    Minimal stub hooks that make the pipeline executable without external deps.
    They do not perform real training; adapters are stored as JSON metadata.
    """
    class _DummyModel(dict):
        def __call__(self, obs: Any) -> Any:
            return 0.0

    def build_model() -> Any:
        return _DummyModel()

    def attach_lora_adapters(model: Any, layers: Sequence[str]) -> Sequence[Any]:
        model["lora_layers"] = list(layers)
        return [model]  # placeholder trainable params

    def make_optimizer(trainable_params: Sequence[Any], lr: float, weight_decay: float) -> Any:
        return {"params": len(trainable_params), "lr": lr, "weight_decay": weight_decay}

    def train_step(model: Any, optimizer: Any, batch: Dict[str, Any]) -> float:
        # Pseudo-loss: scales with batch size for visibility
        return float(len(batch.get("obs", [])))

    def save_adapters(model: Any, path: Path) -> None:
        ensure_dir(path.parent)
        payload = {"adapter_layers": model.get("lora_layers", []), "timestamp": time.time()}
        path.write_text(json.dumps(payload, indent=2))

    def load_adapters(model: Any, path: Path) -> Any:
        if path.exists():
            payload = json.loads(path.read_text())
            model["lora_layers"] = payload.get("adapter_layers", [])
        return model

    def eval_forward(model: Any, obs: Any) -> Any:
        return 0.0

    return ModelHooks(
        build_model=build_model,
        attach_lora_adapters=attach_lora_adapters,
        make_optimizer=make_optimizer,
        train_step=train_step,
        save_adapters=save_adapters,
        load_adapters=load_adapters,
        eval_forward=eval_forward,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local LoRA trainer (offline-first).")
    parser.add_argument("--config", type=Path, required=True, help="Path to job config JSON.")
    parser.add_argument(
        "--use-stub-hooks",
        action="store_true",
        help="Run with built-in stub hooks for dry-runs without framework deps.",
    )
    args = parser.parse_args()

    cfg = LocalTrainerJobConfig.from_json(args.config)
    hooks = build_stub_hooks() if args.use_stub_hooks else None
    if hooks is None:
        raise SystemExit("Provide concrete ModelHooks or run with --use-stub-hooks.")

    result = maybe_run_local_training(
        cfg=cfg,
        hooks=hooks,
        safety_cfg=SafetyGateConfig(),
        gating=None,
    )
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
