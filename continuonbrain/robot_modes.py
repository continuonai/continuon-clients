"""
Robot operational modes for ContinuonBrain.
Manages transitions between training, autonomous, and sleep modes.
"""
import json
import math
import subprocess
import time
import urllib.request
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from continuonbrain.trainer.local_lora_trainer import LocalTrainerJobConfig
from continuonbrain.system_context import SystemContext
from continuonbrain.system_instructions import SystemInstructions


class RobotMode(Enum):
    """Robot operational modes."""
    MANUAL_CONTROL = "manual_control"  # Direct human control (teleoperation, monitoring)
    MANUAL_TRAINING = "manual_training"  # Human teleop for training data collection
    AUTONOMOUS = "autonomous"  # VLA policy control
    SLEEP_LEARNING = "sleep_learning"  # Self-training on saved memories
    AUTO_CHARGING = "auto_charging"  # Autonomous docking and charging
    IDLE = "idle"  # Awake but not active
    EMERGENCY_STOP = "emergency_stop"  # Safety stop


@dataclass
class ModeConfig:
    """Configuration for each mode."""
    mode: RobotMode
    timestamp: float
    allow_motion: bool
    record_episodes: bool
    run_inference: bool
    self_train: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BandwidthLimiter:
    """Tracks download usage against a fixed ceiling."""

    max_bytes: int
    log_dir: Path
    bytes_used: int = 0
    limit_hit: bool = False
    events: List[Dict[str, Any]] = None

    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.events is None:
            self.events = []

    def consume(self, byte_count: int, note: str = "") -> bool:
        self.bytes_used += byte_count
        self.events.append(
            {
                "timestamp": time.time(),
                "note": note,
                "bytes_used": self.bytes_used,
            }
        )
        if self.bytes_used > self.max_bytes:
            self.limit_hit = True
        return not self.limit_hit

    def remaining(self) -> int:
        return max(0, self.max_bytes - self.bytes_used)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "bytes_used": self.bytes_used,
            "limit_hit": self.limit_hit,
            "events": self.events,
        }


class RobotModeManager:
    """
    Manages robot operational modes and transitions.
    Handles manual training, autonomous operation, and sleep learning.
    """
    
    def __init__(
        self,
        config_dir: str = "/opt/continuonos/brain",
        *,
        system_instructions: Optional[SystemInstructions] = None,
    ):
        self.config_dir = Path(config_dir)
        self.state_file = self.config_dir / ".robot_mode"
        self.current_mode: RobotMode = RobotMode.IDLE
        self.mode_start_time: float = 0
        self.trainer_log_dir = self.config_dir / "trainer" / "logs"
        self.system_instructions: Optional[SystemInstructions] = system_instructions or SystemContext.get_instructions()

        if system_instructions and SystemContext.get_instructions() is None:
            # Register for other components in this process.
            SystemContext.register_instructions(system_instructions)
    
    def get_mode_config(self, mode: RobotMode) -> ModeConfig:
        """Get configuration for a specific mode."""
        configs = {
            RobotMode.MANUAL_CONTROL: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=True,
                record_episodes=False,  # Just control, no training
                run_inference=False,
                self_train=False,
                metadata={"control_source": "human_teleop", "show_live_feed": True}
            ),
            RobotMode.MANUAL_TRAINING: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=True,
                record_episodes=True,
                run_inference=False,
                self_train=False,
                metadata={"control_source": "human_teleop"}
            ),
            RobotMode.AUTONOMOUS: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=True,
                record_episodes=True,  # Record for continuous learning
                run_inference=True,
                self_train=False,
                metadata={"control_source": "vla_policy"}
            ),
            RobotMode.SLEEP_LEARNING: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=False,
                record_episodes=False,
                run_inference=False,
                self_train=True,
                metadata={
                    "training_type": "offline_replay",
                    "use_gemma": True  # Use Gemma-3 for knowledge
                }
            ),
            RobotMode.IDLE: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=False,
                record_episodes=False,
                run_inference=False,
                self_train=False,
                metadata={}
            ),
            RobotMode.EMERGENCY_STOP: ModeConfig(
                mode=mode,
                timestamp=time.time(),
                allow_motion=False,
                record_episodes=False,
                run_inference=False,
                self_train=False,
                metadata={"reason": "emergency"}
            ),
        }
        
        return configs[mode]
    
    def set_mode(self, new_mode: RobotMode, metadata: Optional[Dict] = None) -> bool:
        """
        Change robot mode.
        
        Args:
            new_mode: Target mode
            metadata: Optional metadata for the mode
        
        Returns:
            True if mode changed successfully
        """
        old_mode = self.current_mode
        
        # Validate transition
        if not self._validate_transition(old_mode, new_mode):
            print(f"  Invalid transition: {old_mode.value} -> {new_mode.value}")
            return False
        
        # Get mode config
        config = self.get_mode_config(new_mode)
        if metadata:
            config.metadata.update(metadata)
        
        # Execute mode change
        self.current_mode = new_mode
        self.mode_start_time = time.time()
        
        # Save state
        self._save_state(config)
        
        # Print status
        print("=" * 60)
        print(f" Mode Change: {old_mode.value} -> {new_mode.value}")
        print("=" * 60)
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Motion: {'Enabled' if config.allow_motion else 'Disabled'}")
        print(f"Recording: {'ON' if config.record_episodes else 'OFF'}")
        print(f"Inference: {'ON' if config.run_inference else 'OFF'}")
        print(f"Self-Training: {'ON' if config.self_train else 'OFF'}")
        
        if config.metadata:
            print("\nMode Configuration:")
            for key, value in config.metadata.items():
                print(f"  {key}: {value}")
        
        print("=" * 60)
        print()
        
        return True
    
    def _validate_transition(self, from_mode: RobotMode, to_mode: RobotMode) -> bool:
        """Validate mode transition is allowed."""
        # Emergency stop can always be triggered
        if to_mode == RobotMode.EMERGENCY_STOP:
            return True
        
        # Can't leave emergency stop without going through idle
        if from_mode == RobotMode.EMERGENCY_STOP and to_mode != RobotMode.IDLE:
            return False
        
        # Sleep learning can only start from idle
        if to_mode == RobotMode.SLEEP_LEARNING and from_mode != RobotMode.IDLE:
            print("  Must be idle before sleep learning")
            return False
        
        # All other transitions allowed
        return True
    
    def _save_state(self, config: ModeConfig):
        """Save mode state to file."""
        state = {
            "mode": config.mode.value,  # Convert enum to string
            "timestamp": config.timestamp,
            "config": {
                "mode": config.mode.value,  # Convert enum to string
                "timestamp": config.timestamp,
                "allow_motion": config.allow_motion,
                "record_episodes": config.record_episodes,
                "run_inference": config.run_inference,
                "self_train": config.self_train,
                "metadata": config.metadata
            }
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _write_watchdog_log(self, payload: Dict[str, Any]) -> Path:
        """Persist watchdog/log data to the trainer log directory."""
        self.trainer_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.trainer_log_dir / f"sleep_watchdog_{int(time.time())}.json"
        log_path.write_text(json.dumps(payload, indent=2))
        return log_path

    def _resolve_training_config(self) -> Optional[Path]:
        """Find a training config for sleep learning."""
        candidates = [
            self.config_dir / "configs" / "sleep_learning.json",
            self.config_dir / "configs" / "pi5-donkey.json",
            Path(__file__).parent / "configs" / "pi5-donkey.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _estimate_remote_size(self, url: str) -> Optional[int]:
        """Attempt to read Content-Length for a remote resource."""
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                header_value = response.getheader("Content-Length")
                if header_value:
                    return int(header_value)
        except Exception:
            return None
        return None

    def _download_with_budget(self, url: str, dest: Path, limiter: BandwidthLimiter) -> bool:
        """Stream a download while enforcing the bandwidth ceiling."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(request, timeout=30) as response, dest.open("wb") as out:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    if not limiter.consume(len(chunk), note=f"download:{url}"):
                        limiter.limit_hit = True
                        return False
                    out.write(chunk)
        except Exception as exc:
            self._write_watchdog_log(
                {
                    "status": "download_failed",
                    "url": url,
                    "error": str(exc),
                    "bytes_used": limiter.bytes_used,
                }
            )
            return False
        return True

    def _maybe_fetch_base_model(self, cfg: LocalTrainerJobConfig, limiter: BandwidthLimiter) -> bool:
        """Ensure the base model exists or is downloaded within budget."""
        if cfg.base_model_path and cfg.base_model_path.exists():
            return True
        if cfg.base_model_url is None:
            self._write_watchdog_log(
                {
                    "status": "skipped_download",
                    "reason": "base model missing and no URL provided",
                    "bytes_used": limiter.bytes_used,
                }
            )
            return False

        estimated_size = self._estimate_remote_size(cfg.base_model_url)
        if estimated_size is not None and estimated_size > limiter.remaining():
            self._write_watchdog_log(
                {
                    "status": "download_budget_exceeded",
                    "required_bytes": estimated_size,
                    "remaining_bytes": limiter.remaining(),
                    "url": cfg.base_model_url,
                }
            )
            limiter.limit_hit = True
            return False

        destination = cfg.base_model_path or (self.config_dir / "model" / "base_model.bin")
        succeeded = self._download_with_budget(cfg.base_model_url, destination, limiter)
        if not succeeded:
            return False
        self._write_watchdog_log(
            {
                "status": "download_complete",
                "url": cfg.base_model_url,
                "bytes_used": limiter.bytes_used,
            }
        )
        return True

    def _launch_sleep_training_process(self, config_path: Path) -> Optional[subprocess.Popen]:
        """Start the trainer subprocess with stub hooks to minimize deps."""
        import sys

        self.trainer_log_dir.mkdir(parents=True, exist_ok=True)
        trainer_output = self.trainer_log_dir / "sleep_training_output.log"
        cmd = [
            sys.executable,
            "-m",
            "continuonbrain.trainer.local_lora_trainer",
            "--config",
            str(config_path),
            "--use-stub-hooks",
        ]
        try:
            log_handle = trainer_output.open("a", encoding="utf-8")
            process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            log_handle.close()
            return process
        except Exception as exc:
            self._write_watchdog_log(
                {
                    "status": "trainer_launch_failed",
                    "config": str(config_path),
                    "error": str(exc),
                }
            )
            return None

    def _run_sleep_training_with_watchdog(
        self,
        *,
        max_sleep_training_hours: float,
        max_download_bytes: int,
        config_path: Optional[Path] = None,
    ) -> None:
        """Run sleep training with wall-time and download ceilings."""
        config = config_path or self._resolve_training_config()
        if config is None:
            self._write_watchdog_log({"status": "skipped", "reason": "no training config found"})
            self.return_to_idle()
            return

        limiter = BandwidthLimiter(max_bytes=max_download_bytes, log_dir=self.trainer_log_dir)
        try:
            cfg = LocalTrainerJobConfig.from_json(config)
        except Exception as exc:
            self._write_watchdog_log(
                {
                    "status": "skipped",
                    "reason": "invalid training config",
                    "config": str(config),
                    "error": str(exc),
                }
            )
            self.return_to_idle()
            return

        if cfg.base_model_path and not cfg.base_model_path.exists():
            ok = self._maybe_fetch_base_model(cfg, limiter)
            if not ok:
                self.return_to_idle()
                return

        time_limit_s = max_sleep_training_hours * 3600
        start_ts = time.time()
        process = self._launch_sleep_training_process(config)
        if process is None:
            self.return_to_idle()
            return

        stop_reason: Optional[str] = None
        while process.poll() is None:
            elapsed = time.time() - start_ts
            if elapsed >= time_limit_s:
                stop_reason = "time_limit_reached"
                process.terminate()
                break
            if limiter.limit_hit:
                stop_reason = "download_budget_exhausted"
                process.terminate()
                break
            time.sleep(5)

        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()

        end_ts = time.time()
        log_payload = {
            "status": stop_reason or "trainer_complete",
            "config": str(config),
            "elapsed_seconds": end_ts - start_ts,
            "max_sleep_training_hours": max_sleep_training_hours,
            "max_download_bytes": max_download_bytes,
            "bandwidth": limiter.snapshot(),
        }
        # Attach latest trainer status/promote audit if present.
        try:
            status_path = Path("/opt/continuonos/brain/trainer/status.json")
            if status_path.exists():
                log_payload["trainer_status"] = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        self._write_watchdog_log(log_payload)

        if stop_reason:
            self.return_to_idle()
    
    def load_state(self) -> Optional[RobotMode]:
        """Load last mode from file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            mode_str = state.get("mode")
            if mode_str:
                return RobotMode(mode_str)
        
        except Exception as e:
            print(f"Error loading mode state: {e}")
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current mode status."""
        config = self.get_mode_config(self.current_mode)
        duration = time.time() - self.mode_start_time

        instructions = self.system_instructions or SystemContext.get_instructions()

        return {
            "mode": self.current_mode.value,
            "duration_seconds": duration,
            "config": asdict(config),
            "timestamp": time.time(),
            "system_instructions": instructions.as_dict() if instructions else None,
        }

    def get_gate_snapshot(self) -> Dict[str, Any]:
        """Summarize gate decisions for UI and telemetry."""
        config = self.get_mode_config(self.current_mode)
        uptime = max(0.0, time.time() - self.mode_start_time)

        return {
            "mode": self.current_mode.value,
            "allow_motion": config.allow_motion,
            "record_episodes": config.record_episodes,
            "run_inference": config.run_inference,
            "self_train": config.self_train,
            "mode_uptime_seconds": uptime,
            "transition_gate": uptime > 1.0 and config.allow_motion,
            "recording_gate": config.record_episodes and config.allow_motion,
            "last_transition_timestamp": self.mode_start_time,
        }

    def get_loop_metrics(self, limit: int = 60) -> Dict[str, Any]:
        """Simulate HOPE/CMS loop activity for dashboards and mock mode."""
        import random

        elapsed = max(0.0, time.time() - self.mode_start_time)
        # Create a stable but lively wave/particle mix without needing hardware signals.
        wave_particle_balance = 0.5 + 0.45 * math.sin(elapsed / 4.0)

        is_active = self.get_mode_config(self.current_mode).allow_motion
        fast_loop_hz = 12.0 if is_active else 4.0
        mid_loop_hz = fast_loop_hz / 2
        slow_loop_hz = max(0.5, fast_loop_hz / 6)

        heartbeat_period = 1.0 if fast_loop_hz > 0 else 2.0
        beats = int(elapsed / heartbeat_period)
        last_beat = self.mode_start_time + beats * heartbeat_period

        cms_ratio = 0.55 + 0.1 * math.sin(elapsed / 9.0)
        maintenance_ratio = 1.0 - cms_ratio

        # Generate simulated control loop period metrics (in ms)
        target_ms = 1000.0 / fast_loop_hz
        points = []
        for i in range(limit):
            # Simulated jitter: target +/- 5% with occasional spikes
            jitter = random.gauss(0, target_ms * 0.02)
            spike = target_ms * 0.5 if random.random() > 0.98 else 0
            points.append(round(target_ms + jitter + spike, 2))

        return {
            "hope_loops": {
                "fast": {"hz": round(fast_loop_hz, 2), "latency_ms": round(1000 / fast_loop_hz, 1)},
                "mid": {"hz": round(mid_loop_hz, 2), "latency_ms": round(1000 / mid_loop_hz, 1)},
                "slow": {"hz": round(slow_loop_hz, 2), "latency_ms": round(1000 / slow_loop_hz, 1)},
            },
            "cms": {
                "policy_ratio": round(cms_ratio, 2),
                "maintenance_ratio": round(maintenance_ratio, 2),
                "buffer_fill": round(0.5 + 0.2 * math.sin(elapsed / 7.0), 2),
            },
            "wave_particle_balance": round(max(0.0, min(1.0, wave_particle_balance)), 2),
            "heartbeat": {
                "last_beat": last_beat,
                "period_seconds": heartbeat_period,
                "ok": (time.time() - last_beat) <= (heartbeat_period * 1.5),
            },
            "period_ms": {
                "points": points
            }
        }
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Trigger emergency stop."""
        print(f" EMERGENCY STOP: {reason}")
        self.set_mode(RobotMode.EMERGENCY_STOP, {"reason": reason})
    
    def start_manual_control(self):
        """Enter manual control mode for teleoperation."""
        print(" Starting manual control mode...")
        print("   Direct human control - no training data recorded")
        print("   Live feed and full system status available")
        self.set_mode(RobotMode.MANUAL_CONTROL)
    
    def start_manual_training(self):
        """Enter manual training mode for human teleop."""
        print(" Starting manual training mode...")
        print("   Use Flutter app or web UI to control robot")
        print("   All actions will be recorded for training")
        self.set_mode(RobotMode.MANUAL_TRAINING)
    
    def start_autonomous(self):
        """Enter autonomous mode with VLA policy."""
        print(" Starting autonomous mode...")
        print("   Robot will use VLA policy for control")
        print("   Actions still recorded for continuous learning")
        self.set_mode(RobotMode.AUTONOMOUS)
    
    def start_sleep_learning(
        self,
        episodes_to_train: Optional[int] = None,
        *,
        max_sleep_training_hours: float = 6.0,
        max_download_bytes: int = 1024 * 1024 * 1024,
        training_config: Optional[Path] = None,
    ):
        """
        Enter sleep learning mode.
        Robot self-trains on saved episodes and uses Gemma for knowledge.

        Args:
            episodes_to_train: Optional count of episodes to emphasize.
            max_sleep_training_hours: Wall-clock ceiling for training before halting.
            max_download_bytes: Maximum bytes allowed for model/assets downloads.
            training_config: Optional explicit path to the trainer config JSON.
        """
        print(" Entering sleep learning mode...")
        print("   Robot will self-train on saved memories")
        print("   Using Gemma-3 model for knowledge extraction")
        print("   Motion disabled during learning")

        metadata = {
            "training_type": "offline_replay",
            "use_gemma": True,
            "max_sleep_training_hours": max_sleep_training_hours,
            "max_download_bytes": max_download_bytes,
        }
        if episodes_to_train:
            metadata["episodes_to_train"] = episodes_to_train

        self.set_mode(RobotMode.SLEEP_LEARNING, metadata)
        self._run_sleep_training_with_watchdog(
            max_sleep_training_hours=max_sleep_training_hours,
            max_download_bytes=max_download_bytes,
            config_path=training_config,
        )
    
    def return_to_idle(self):
        """Return to idle mode."""
        print("  Returning to idle...")
        self.set_mode(RobotMode.IDLE)


def main():
    """Test mode manager."""
    manager = RobotModeManager(config_dir="/tmp/mode_test")
    
    print("Testing robot mode transitions...\n")
    
    # Idle → Manual Training
    manager.start_manual_training()
    time.sleep(2)
    
    # Manual Training → Idle
    manager.return_to_idle()
    time.sleep(1)
    
    # Idle → Autonomous
    manager.start_autonomous()
    time.sleep(2)
    
    # Autonomous → Idle
    manager.return_to_idle()
    time.sleep(1)
    
    # Idle → Sleep Learning
    manager.start_sleep_learning(episodes_to_train=16)
    time.sleep(2)
    
    # Emergency stop (can trigger from any mode)
    manager.emergency_stop("Test trigger")
    time.sleep(1)
    
    # Emergency → Idle (required)
    manager.return_to_idle()
    
    print("\n Mode transition test complete!")


if __name__ == "__main__":
    main()
