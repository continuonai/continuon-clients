"""
Orchestrator Configuration

Centralized configuration for the orchestrator system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator system."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path("/opt/continuonos/brain"))
    rlds_dir: Path = field(default_factory=lambda: Path("/opt/continuonos/brain/rlds/episodes"))
    model_dir: Path = field(default_factory=lambda: Path("/opt/continuonos/brain/model"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("/opt/continuonos/brain/checkpoints"))
    logs_dir: Path = field(default_factory=lambda: Path("/opt/continuonos/brain/logs"))

    # Worker settings
    max_workers: int = 4
    worker_timeout_sec: int = 300

    # Task queue settings
    max_queue_size: int = 1000
    task_timeout_sec: int = 600

    # Training defaults
    default_batch_size: int = 4
    default_max_steps: int = 100
    default_learning_rate: float = 1e-3
    default_config_preset: str = "pi5"

    # Inference settings
    inference_port: int = 8080
    inference_host: str = "0.0.0.0"
    use_jit: bool = True

    # Monitoring
    health_check_interval_sec: int = 30
    metrics_retention_hours: int = 24

    # Event system
    event_buffer_size: int = 10000

    # State persistence
    state_file: Path = field(default_factory=lambda: Path("/opt/continuonos/brain/orchestrator_state.json"))
    auto_save_interval_sec: int = 60

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        if isinstance(self.rlds_dir, str):
            self.rlds_dir = Path(self.rlds_dir)
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.logs_dir, str):
            self.logs_dir = Path(self.logs_dir)
        if isinstance(self.state_file, str):
            self.state_file = Path(self.state_file)

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Create config from environment variables."""
        return cls(
            base_dir=Path(os.environ.get("CONTINUON_BASE_DIR", "/opt/continuonos/brain")),
            rlds_dir=Path(os.environ.get("CONTINUON_RLDS_DIR", "/opt/continuonos/brain/rlds/episodes")),
            model_dir=Path(os.environ.get("CONTINUON_MODEL_DIR", "/opt/continuonos/brain/model")),
            max_workers=int(os.environ.get("CONTINUON_MAX_WORKERS", "4")),
            inference_port=int(os.environ.get("CONTINUON_INFERENCE_PORT", "8080")),
        )

    @classmethod
    def from_file(cls, path: Path) -> "OrchestratorConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: Path) -> None:
        """Save config to JSON file."""
        data = {
            "base_dir": str(self.base_dir),
            "rlds_dir": str(self.rlds_dir),
            "model_dir": str(self.model_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "logs_dir": str(self.logs_dir),
            "max_workers": self.max_workers,
            "worker_timeout_sec": self.worker_timeout_sec,
            "max_queue_size": self.max_queue_size,
            "task_timeout_sec": self.task_timeout_sec,
            "default_batch_size": self.default_batch_size,
            "default_max_steps": self.default_max_steps,
            "default_learning_rate": self.default_learning_rate,
            "default_config_preset": self.default_config_preset,
            "inference_port": self.inference_port,
            "inference_host": self.inference_host,
            "use_jit": self.use_jit,
            "health_check_interval_sec": self.health_check_interval_sec,
            "metrics_retention_hours": self.metrics_retention_hours,
            "event_buffer_size": self.event_buffer_size,
            "state_file": str(self.state_file),
            "auto_save_interval_sec": self.auto_save_interval_sec,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def development(cls) -> "OrchestratorConfig":
        """Create development config with local paths."""
        base = Path(__file__).parent.parent
        return cls(
            base_dir=base,
            rlds_dir=base / "rlds" / "episodes",
            model_dir=base / "model",
            checkpoint_dir=Path("/tmp/continuon_checkpoints"),
            logs_dir=Path("/tmp/continuon_logs"),
            state_file=Path("/tmp/continuon_state.json"),
            max_workers=2,
        )
