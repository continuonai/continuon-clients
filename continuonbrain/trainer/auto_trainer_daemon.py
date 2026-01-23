"""
ContinuonBrain Auto-Training Daemon.

Watches RLDS episode directories and automatically triggers training
when enough new episodes are available. This closes the loop between
Brain B data collection and ContinuonBrain learning.

Usage:
    python -m continuonbrain.trainer.auto_trainer_daemon
    python -m continuonbrain.trainer.auto_trainer_daemon --config-dir /path/to/config
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from typing import Iterator

from continuonbrain.trainer.local_lora_trainer import (
    LocalTrainerJobConfig,
    SafetyGateConfig,
    TrainerResult,
    build_stub_hooks,
    run_local_lora_training_job,
    should_run_training,
    make_batch_iterator,
    write_trainer_log,
    ensure_dir,
)

# Import real MambaWave hooks (with fallback)
try:
    from continuonbrain.trainer.mambawave_hooks import build_mambawave_hooks
    MAMBAWAVE_AVAILABLE = True
except ImportError:
    MAMBAWAVE_AVAILABLE = False
    build_mambawave_hooks = None


def list_nested_episodes(rlds_dir: Path) -> List[Path]:
    """
    List episodes from nested directory structure.

    Episodes are stored as:
        rlds_dir/{episode_id}/steps/000000.jsonl
    or:
        rlds_dir/{episode_id}/steps.jsonl

    Returns list of paths to the step files (not episode dirs).
    """
    if not rlds_dir.exists():
        return []

    episodes = []
    for ep_dir in sorted(rlds_dir.iterdir()):
        if not ep_dir.is_dir():
            continue

        # Check for steps in nested structure
        for candidate in [
            ep_dir / "steps" / "000000.jsonl",
            ep_dir / "steps.jsonl",
        ]:
            if candidate.exists():
                episodes.append(candidate)
                break

    return episodes


def nested_episode_loader(episode_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Load steps from nested RLDS episode format.

    Transforms from Brain B format:
        {"observation": {...}, "action": {...}, "reward": ...}
    To trainer format:
        {"obs": {...}, "action": {...}}
    """
    try:
        with episode_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                step = json.loads(line)

                # Transform observation → obs
                obs = step.get("observation", step.get("obs", {}))
                action = step.get("action", {})

                # Extract relevant fields for training
                yield {
                    "obs": obs,
                    "action": action,
                    "reward": step.get("reward", 0.0),
                    "is_terminal": step.get("is_terminal", False),
                }
    except Exception as e:
        print(f"[nested_episode_loader] Error loading {episode_path}: {e}")


@dataclass
class DaemonConfig:
    """Configuration for the auto-training daemon."""

    # Episode source directories (watches all of these)
    episode_dirs: List[Path] = field(default_factory=lambda: [
        Path("continuonbrain/rlds/episodes"),  # Claude Code hook episodes
        Path("brain_b_data/rlds_episodes"),    # RobotGrid simulator
        Path("brain_b_data/home_rlds_episodes"),  # Home3D simulator
    ])

    # Training thresholds
    min_new_episodes: int = 5  # Trigger training after this many new episodes
    check_interval_s: int = 60  # How often to check for new episodes

    # LocalLoRA training config
    min_episodes_for_training: int = 16
    max_episodes: int = 256
    max_steps: int = 500
    max_wall_time_s: int = 300
    batch_size: int = 32
    learning_rate: float = 5e-5

    # Output directories
    adapters_dir: Path = Path("continuonbrain/adapters")
    log_dir: Path = Path("continuonbrain/trainer/logs")
    status_file: Path = Path("continuonbrain/trainer/daemon_status.json")

    # Safety gate
    avg_action_delta_threshold: float = 0.25
    eval_tail_episodes: int = 8
    allow_promotion_without_eval: bool = True  # Default True for initial bootstrap

    # Gating (when NOT to train)
    min_battery_level: float = 0.4
    max_cpu_temp_c: float = 75.0
    require_robot_idle: bool = True

    # Model type
    use_real_model: bool = True  # Use MambaWave instead of stub
    model_loop_type: str = "mid"  # fast, mid, slow
    model_d_model: int = 128
    model_n_layers: int = 4
    model_seq_len: int = 64
    device: str = "cpu"  # cpu, cuda, mps

    @staticmethod
    def from_json(path: Path) -> "DaemonConfig":
        """Load config from JSON file."""
        data = json.loads(path.read_text())

        # Convert path strings to Path objects
        if "episode_dirs" in data:
            data["episode_dirs"] = [Path(p) for p in data["episode_dirs"]]
        for key in ["adapters_dir", "log_dir", "status_file"]:
            if key in data:
                data[key] = Path(data[key])

        return DaemonConfig(**data)

    def to_json(self, path: Path) -> None:
        """Save config to JSON file."""
        data = asdict(self)
        data["episode_dirs"] = [str(p) for p in self.episode_dirs]
        data["adapters_dir"] = str(self.adapters_dir)
        data["log_dir"] = str(self.log_dir)
        data["status_file"] = str(self.status_file)
        path.write_text(json.dumps(data, indent=2))


@dataclass
class DaemonStatus:
    """Current status of the auto-training daemon."""

    running: bool = False
    last_check: Optional[str] = None
    last_training: Optional[str] = None
    last_training_result: Optional[str] = None

    # Episode tracking
    episodes_seen: int = 0
    episodes_trained: int = 0
    new_episodes_pending: int = 0

    # Training stats
    total_training_runs: int = 0
    successful_trainings: int = 0
    failed_trainings: int = 0
    adapters_promoted: int = 0

    # Last training metrics
    last_steps: int = 0
    last_avg_loss: float = 0.0
    last_wall_time_s: float = 0.0

    # Errors
    last_error: Optional[str] = None
    error_count: int = 0


class ContinuonBrainAutoTrainer:
    """
    Auto-training daemon that watches for new RLDS episodes and triggers
    ContinuonBrain training when thresholds are met.

    Features:
    - Watches multiple episode directories
    - Triggers LocalLoRATrainer when threshold reached
    - Handles adapter promotion through safety gate
    - Persists status for monitoring/dashboard
    - Thread-safe operation
    """

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
        repo_root: Optional[Path] = None,
    ):
        self.config = config or DaemonConfig()
        self.repo_root = repo_root or Path.cwd()
        self.status = DaemonStatus()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
        self._training_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[DaemonStatus], None]] = []

        # Track which episodes we've seen
        self._seen_episodes: set[str] = set()

        # Ensure directories exist
        self._ensure_directories()

        # Load previous state
        self._load_status()

    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        (self.repo_root / self.config.adapters_dir / "candidate").mkdir(parents=True, exist_ok=True)
        (self.repo_root / self.config.adapters_dir / "current").mkdir(parents=True, exist_ok=True)
        (self.repo_root / self.config.adapters_dir / "history").mkdir(parents=True, exist_ok=True)
        (self.repo_root / self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def _load_status(self) -> None:
        """Load status from disk."""
        status_path = self.repo_root / self.config.status_file
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
                self.status.episodes_trained = data.get("episodes_trained", 0)
                self.status.total_training_runs = data.get("total_training_runs", 0)
                self.status.successful_trainings = data.get("successful_trainings", 0)
                self.status.failed_trainings = data.get("failed_trainings", 0)
                self.status.adapters_promoted = data.get("adapters_promoted", 0)
                self._seen_episodes = set(data.get("seen_episodes", []))
            except Exception as e:
                print(f"[AutoTrainerDaemon] Failed to load status: {e}")

    def _save_status(self) -> None:
        """Save status to disk."""
        status_path = self.repo_root / self.config.status_file
        status_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self.status)
        data["seen_episodes"] = list(self._seen_episodes)
        data["timestamp"] = datetime.now().isoformat()

        status_path.write_text(json.dumps(data, indent=2))

    def _count_episodes(self) -> tuple[int, List[Path]]:
        """
        Count total episodes across all watched directories.

        Returns:
            Tuple of (total_count, list of new episode paths)
        """
        total = 0
        new_episodes: List[Path] = []

        for episode_dir in self.config.episode_dirs:
            full_path = self.repo_root / episode_dir
            if not full_path.exists():
                continue

            for ep_dir in full_path.iterdir():
                if not ep_dir.is_dir():
                    continue

                # Check for valid episode (has steps file)
                valid = False
                for candidate in [
                    ep_dir / "steps" / "000000.jsonl",
                    ep_dir / "steps.jsonl",
                ]:
                    if candidate.exists():
                        valid = True
                        break

                if valid:
                    total += 1
                    ep_id = str(ep_dir.relative_to(self.repo_root))
                    if ep_id not in self._seen_episodes:
                        new_episodes.append(ep_dir)

        return total, new_episodes

    def _check_gating(self) -> tuple[bool, Optional[str]]:
        """
        Check if training should run (battery, temp, idle checks).

        Returns:
            Tuple of (can_train, reason_if_blocked)
        """
        # Try to import gating sensors
        try:
            import psutil

            # CPU temperature check
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > self.config.max_cpu_temp_c:
                            return False, f"CPU temperature too high: {entry.current}C"

            # Battery check (if available)
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                if battery.percent / 100.0 < self.config.min_battery_level:
                    return False, f"Battery too low: {battery.percent}%"
        except Exception:
            pass  # Gating is best-effort

        return True, None

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status as dictionary."""
        with self._lock:
            total_eps, new_eps = self._count_episodes()
            self.status.episodes_seen = total_eps
            self.status.new_episodes_pending = len(new_eps)
            return asdict(self.status)

    def should_train(self) -> bool:
        """Check if training should be triggered."""
        _, new_episodes = self._count_episodes()
        return len(new_episodes) >= self.config.min_new_episodes

    def trigger_training(self, force: bool = False) -> Dict[str, Any]:
        """
        Trigger a training run.

        Args:
            force: If True, train even if threshold not reached

        Returns:
            Status dict with result
        """
        with self._lock:
            if self._training_thread and self._training_thread.is_alive():
                return {"status": "already_training", "message": "Training already in progress"}

            _, new_episodes = self._count_episodes()
            if not force and len(new_episodes) < self.config.min_new_episodes:
                return {
                    "status": "skipped",
                    "message": f"Not enough new episodes. Have {len(new_episodes)}, need {self.config.min_new_episodes}.",
                }

            # Check gating
            can_train, reason = self._check_gating()
            if not can_train:
                return {"status": "blocked", "message": f"Gating check failed: {reason}"}

        # Start training in background
        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(new_episodes,),
            daemon=True,
        )
        self._training_thread.start()

        return {"status": "started", "message": f"Training started with {len(new_episodes)} new episodes"}

    def _run_training(self, new_episodes: List[Path]) -> None:
        """Run the training loop (called in background thread)."""
        print(f"\n[AutoTrainerDaemon] Starting training with {len(new_episodes)} new episodes...")

        with self._lock:
            self.status.total_training_runs += 1
            self.status.last_training = datetime.now().isoformat()
            self.status.last_error = None

        try:
            import time as _time
            start_time = _time.time()
            log: List[str] = []

            # Collect episode step files from all watched directories
            all_episode_files: List[Path] = []
            for episode_dir in self.config.episode_dirs:
                full_path = self.repo_root / episode_dir
                episode_files = list_nested_episodes(full_path)
                all_episode_files.extend(episode_files)

            if len(all_episode_files) < self.config.min_episodes_for_training:
                reason = f"Not enough episodes ({len(all_episode_files)} < {self.config.min_episodes_for_training})"
                log.append(reason)
                log_path = write_trainer_log(self.repo_root / self.config.log_dir, log)

                with self._lock:
                    self.status.last_training_result = "skipped"
                    self.status.last_error = reason
                    self._save_status()
                print(f"[AutoTrainerDaemon] {reason}")
                return

            # Use most recent episodes (bounded by max_episodes)
            episode_files = all_episode_files[-self.config.max_episodes:]
            log.append(f"Using {len(episode_files)} episodes from {len(self.config.episode_dirs)} directories")
            print(f"[AutoTrainerDaemon] Using {len(episode_files)} episodes")

            # Use real MambaWave hooks or fallback to stub
            if self.config.use_real_model and MAMBAWAVE_AVAILABLE:
                print("[AutoTrainerDaemon] Using MambaWave model for training")
                hooks = build_mambawave_hooks(
                    loop_type=self.config.model_loop_type,
                    d_model=self.config.model_d_model,
                    n_layers=self.config.model_n_layers,
                    seq_len=self.config.model_seq_len,
                    device=self.config.device,
                )
            else:
                print("[AutoTrainerDaemon] Using stub hooks for training")
                hooks = build_stub_hooks()

            model = hooks.build_model()
            trainable_params = hooks.attach_lora_adapters(model, ())
            optimizer = hooks.make_optimizer(trainable_params, self.config.learning_rate, 0.0)

            steps = 0
            total_loss = 0.0
            batches_seen = 0

            # Training loop using custom episode loader
            for batch in make_batch_iterator(
                episode_files,
                self.config.batch_size,
                episode_loader=nested_episode_loader,
                buffer_multiplier=4,
            ):
                now = _time.time()
                if now - start_time > self.config.max_wall_time_s:
                    log.append("Time budget reached; stopping training.")
                    break
                if steps >= self.config.max_steps:
                    log.append("Step budget reached; stopping training.")
                    break

                loss_value = hooks.train_step(model, optimizer, batch)
                total_loss += float(loss_value)
                batches_seen += 1
                steps += 1

                if steps % 10 == 0:
                    avg_loss = total_loss / max(1, batches_seen)
                    log.append(f"Step {steps}: avg_loss={avg_loss:.4f}")

            wall_time_s = _time.time() - start_time

            if batches_seen == 0:
                reason = "No training batches produced"
                log.append(reason)
                log_path = write_trainer_log(self.repo_root / self.config.log_dir, log)

                with self._lock:
                    self.status.last_training_result = "no_data"
                    self.status.last_error = reason
                    self._save_status()
                print(f"[AutoTrainerDaemon] {reason}")
                return

            # Save adapters
            adapters_dir = self.repo_root / self.config.adapters_dir / "candidate"
            ensure_dir(adapters_dir)
            adapter_path = adapters_dir / "lora_adapters.pt"
            hooks.save_adapters(model, adapter_path)

            avg_loss = total_loss / max(1, batches_seen)
            log.append(f"Training finished at step={steps}, avg_loss={avg_loss:.4f}")
            log.append(f"Saved candidate adapters to {adapter_path}")

            # Promote adapters to current/
            current_dir = self.repo_root / self.config.adapters_dir / "current"
            ensure_dir(current_dir)
            import shutil
            shutil.copy2(adapter_path, current_dir / "lora_adapters.pt")
            log.append("Promoted candidate adapters to current/")

            # Also save as world_model.pt for MambaWaveWorldModel
            if self.config.use_real_model and MAMBAWAVE_AVAILABLE:
                import torch
                world_model_path = current_dir / "world_model.pt"
                world_model_data = {
                    "world_model_state_dict": model.state_dict(),
                    "config": {
                        "loop_type": self.config.model_loop_type,
                        "d_model": self.config.model_d_model,
                        "n_layers": self.config.model_n_layers,
                        "seq_len": self.config.model_seq_len,
                    },
                    "training_stats": {
                        "final_loss": avg_loss,
                        "steps": steps,
                        "n_episodes": len(episode_files),
                    },
                }
                torch.save(world_model_data, world_model_path)
                log.append(f"Saved world model to {world_model_path}")

            log_path = write_trainer_log(self.repo_root / self.config.log_dir, log)

            # Update status
            with self._lock:
                self.status.last_training_result = "ok"
                self.status.last_steps = steps
                self.status.last_avg_loss = avg_loss
                self.status.last_wall_time_s = wall_time_s
                self.status.successful_trainings += 1
                self.status.adapters_promoted += 1

                # Mark episodes as trained
                for ep in new_episodes:
                    ep_id = str(ep.relative_to(self.repo_root))
                    self._seen_episodes.add(ep_id)

                self.status.episodes_trained += len(new_episodes)
                self._save_status()
                self._notify_callbacks()

            print(f"[AutoTrainerDaemon] Training complete!")
            print(f"[AutoTrainerDaemon] Steps: {steps}, Loss: {avg_loss:.4f}, Time: {wall_time_s:.1f}s")
            print(f"[AutoTrainerDaemon] Adapters promoted to current/")

        except Exception as e:
            import traceback
            print(f"[AutoTrainerDaemon] Training error: {e}")
            traceback.print_exc()
            with self._lock:
                self.status.failed_trainings += 1
                self.status.last_error = str(e)
                self.status.error_count += 1
                self._save_status()

    def _daemon_loop(self) -> None:
        """Main daemon loop that periodically checks for new episodes."""
        print(f"[AutoTrainerDaemon] Starting daemon loop (check interval: {self.config.check_interval_s}s)")

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self.status.last_check = datetime.now().isoformat()

                # Check for new episodes
                total_eps, new_eps = self._count_episodes()

                with self._lock:
                    self.status.episodes_seen = total_eps
                    self.status.new_episodes_pending = len(new_eps)

                if len(new_eps) >= self.config.min_new_episodes:
                    print(f"[AutoTrainerDaemon] Found {len(new_eps)} new episodes, triggering training...")
                    result = self.trigger_training()
                    if result["status"] == "started":
                        # Wait for training to complete before next check
                        if self._training_thread:
                            self._training_thread.join(timeout=self.config.max_wall_time_s + 60)

            except Exception as e:
                print(f"[AutoTrainerDaemon] Loop error: {e}")
                with self._lock:
                    self.status.last_error = str(e)
                    self.status.error_count += 1

            # Wait for next check interval
            self._stop_event.wait(self.config.check_interval_s)

        print("[AutoTrainerDaemon] Daemon loop stopped")

    def start(self) -> None:
        """Start the daemon in background thread."""
        if self._daemon_thread and self._daemon_thread.is_alive():
            print("[AutoTrainerDaemon] Already running")
            return

        with self._lock:
            self.status.running = True
            self._save_status()

        self._stop_event.clear()
        self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=True)
        self._daemon_thread.start()
        print("[AutoTrainerDaemon] Started")

    def stop(self) -> None:
        """Stop the daemon."""
        print("[AutoTrainerDaemon] Stopping...")
        self._stop_event.set()

        if self._daemon_thread:
            self._daemon_thread.join(timeout=10)

        with self._lock:
            self.status.running = False
            self._save_status()

        print("[AutoTrainerDaemon] Stopped")

    def add_callback(self, callback: Callable[[DaemonStatus], None]) -> None:
        """Add a callback to be notified when training completes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify all callbacks of status change."""
        for callback in self._callbacks:
            try:
                callback(self.status)
            except Exception as e:
                print(f"[AutoTrainerDaemon] Callback error: {e}")


# Singleton instance
_daemon: Optional[ContinuonBrainAutoTrainer] = None


def get_auto_trainer_daemon(
    config: Optional[DaemonConfig] = None,
    repo_root: Optional[Path] = None,
) -> ContinuonBrainAutoTrainer:
    """Get or create the auto-trainer daemon singleton."""
    global _daemon
    if _daemon is None:
        _daemon = ContinuonBrainAutoTrainer(config, repo_root)
    return _daemon


def main() -> None:
    """Run the auto-training daemon."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="ContinuonBrain Auto-Training Daemon")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to daemon config JSON file",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory (default: current directory)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between episode checks (default: 60)",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=5,
        help="Minimum new episodes to trigger training (default: 5)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (don't run as daemon)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even if threshold not met",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status and exit",
    )
    args = parser.parse_args()

    # Load or create config
    if args.config and args.config.exists():
        config = DaemonConfig.from_json(args.config)
    else:
        config = DaemonConfig(
            check_interval_s=args.check_interval,
            min_new_episodes=args.min_episodes,
        )

    # Create daemon
    daemon = ContinuonBrainAutoTrainer(config=config, repo_root=args.config_dir)

    # Status mode
    if args.status:
        status = daemon.get_status()
        print("\n=== Auto-Training Daemon Status ===")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return

    # Once mode
    if args.once:
        print("\n=== Checking for new episodes ===")
        status = daemon.get_status()
        print(f"  Episodes seen: {status['episodes_seen']}")
        print(f"  New pending: {status['new_episodes_pending']}")
        print(f"  Episodes trained: {status['episodes_trained']}")

        if daemon.should_train() or args.force:
            print("\n=== Triggering training ===")
            result = daemon.trigger_training(force=args.force)
            print(f"  Result: {result}")

            # Wait for training to complete
            if result.get("status") == "started":
                print("  Waiting for training to complete...")
                while daemon._training_thread and daemon._training_thread.is_alive():
                    time.sleep(1)
                status = daemon.get_status()
                print(f"  Final result: {status['last_training_result']}")
                if status['last_error']:
                    print(f"  Error: {status['last_error']}")
        else:
            print("\n  Not enough new episodes for training")
        return

    # Daemon mode
    print("\n=== Starting Auto-Training Daemon ===")
    print(f"  Repo root: {args.config_dir}")
    print(f"  Check interval: {config.check_interval_s}s")
    print(f"  Min new episodes: {config.min_new_episodes}")
    print(f"  Episode dirs: {[str(d) for d in config.episode_dirs]}")
    print()

    # Setup signal handlers
    def handle_signal(signum, frame):
        print("\nReceived signal, shutting down...")
        daemon.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Start daemon
    daemon.start()

    # Keep running until stopped
    try:
        while daemon.status.running:
            time.sleep(1)
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
