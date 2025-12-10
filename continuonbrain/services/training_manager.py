"""
TrainingManager orchestrates the end-to-end Pi5 → Cloud → Pi OTA training plan.

It keeps execution safe by default (dry-run). Individual steps are only executed
when explicitly requested via CLI flags. This avoids accidental long-running
jobs on constrained devices.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from continuonbrain.system_health import SystemHealthChecker


StepStatus = str  # "ok" | "warn" | "fail" | "skipped"


@dataclass
class StepReport:
    name: str
    status: StepStatus
    details: Dict[str, object] = field(default_factory=dict)
    commands: List[List[str]] = field(default_factory=list)


class TrainingManager:
    """Lightweight orchestrator for the training plan."""

    def __init__(
        self,
        trainer_config: Path,
        episodes_dir: Path,
        export_dir: Path,
        run_health: bool = False,
        run_local_trainer: bool = False,
        run_export: bool = False,
        run_post_validate: bool = False,
        real_hardware: bool = False,
    ):
        self.trainer_config = trainer_config
        self.episodes_dir = episodes_dir
        self.export_dir = export_dir
        self.run_health = run_health
        self.run_local_trainer = run_local_trainer
        self.run_export = run_export
        self.run_post_validate = run_post_validate
        self.real_hardware = real_hardware

        self.config_payload = self._load_trainer_config(trainer_config)

    def run(self) -> List[StepReport]:
        reports: List[StepReport] = []

        reports.append(self._step_health())
        reports.append(self._step_episode_inventory())
        reports.append(self._step_local_training())
        reports.append(self._step_export())
        reports.append(self._step_post_validate())

        return reports

    def _step_health(self) -> StepReport:
        if not self.run_health:
            return StepReport(
                name="health_check",
                status="skipped",
                details={"reason": "run with --health to execute system health checks"},
            )

        checker = SystemHealthChecker()
        status, results = checker.run_all_checks(quick_mode=True)
        report_path = Path("/tmp") / f"health_{int(time.time())}.json"
        checker.save_report(str(report_path))

        return StepReport(
            name="health_check",
            status="ok" if status.value in {"healthy", "warning"} else "fail",
            details={
                "overall": status.value,
                "report_path": str(report_path),
                "checks": [r.component for r in results],
            },
        )

    def _step_episode_inventory(self) -> StepReport:
        episodes = self._list_episodes()
        min_eps = int(self.config_payload.get("min_episodes", 0))
        status: StepStatus = "ok" if len(episodes) >= min_eps else "warn"
        return StepReport(
            name="episode_inventory",
            status=status,
            details={
                "episodes_found": len(episodes),
                "min_required": min_eps,
                "episodes_dir": str(self.episodes_dir),
            },
        )

    def _step_local_training(self) -> StepReport:
        if not self.run_local_trainer:
            return StepReport(
                name="local_training",
                status="skipped",
                details={"reason": "run with --train-local to launch trainer"},
                commands=[self._trainer_cmd()],
            )

        cmd = self._trainer_cmd()
        proc = self._run_cmd(cmd)
        status: StepStatus = "ok" if proc.returncode == 0 else "fail"
        return StepReport(
            name="local_training",
            status=status,
            details={
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            commands=[cmd],
        )

    def _step_export(self) -> StepReport:
        if not self.run_export:
            return StepReport(
                name="export_rlds",
                status="skipped",
                details={"reason": "run with --export to anonymize/validate episodes"},
                commands=[self._export_cmd()],
            )

        cmd = self._export_cmd()
        proc = self._run_cmd(cmd)
        status: StepStatus = "ok" if proc.returncode == 0 else "fail"
        return StepReport(
            name="export_rlds",
            status=status,
            details={
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            commands=[cmd],
        )

    def _step_post_validate(self) -> StepReport:
        if not self.run_post_validate:
            return StepReport(
                name="post_deploy_validation",
                status="skipped",
                details={"reason": "run with --post-validate to execute integration test"},
                commands=[self._post_validate_cmd()],
            )

        cmd = self._post_validate_cmd()
        proc = self._run_cmd(cmd)
        status: StepStatus = "ok" if proc.returncode == 0 else "fail"
        return StepReport(
            name="post_deploy_validation",
            status=status,
            details={
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            commands=[cmd],
        )

    def _trainer_cmd(self) -> List[str]:
        return [
            sys.executable,
            "-m",
            "continuonbrain.run_trainer",
            "--trainer",
            "auto",
            "--mode",
            "local",
            "--config",
            str(self.trainer_config),
        ]

    def _export_cmd(self) -> List[str]:
        # Use export_pipeline entrypoint to anonymize and validate
        return [
            sys.executable,
            "-m",
            "continuonbrain.rlds.export_pipeline",
            "--episodes-dir",
            str(self.episodes_dir),
            "--output-dir",
            str(self.export_dir),
        ]

    def _post_validate_cmd(self) -> List[str]:
        args = ["--detect-only"] if not self.real_hardware else ["--real-hardware"]
        return [
            sys.executable,
            "-m",
            "continuonbrain.tests.integration_test",
            *args,
        ]

    def _list_episodes(self) -> List[Path]:
        if not self.episodes_dir.exists():
            return []
        # Count episode-level JSON files (episode.json) or top-level json episodes
        episode_files = list(self.episodes_dir.glob("**/episode.json"))
        if not episode_files:
            episode_files = list(self.episodes_dir.glob("*.json"))
        return episode_files

    def _run_cmd(self, cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True)

    @staticmethod
    def _load_trainer_config(path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training plan orchestrator (dry-run by default)")
    parser.add_argument(
        "--trainer-config",
        type=Path,
        default=Path("continuonbrain/configs/pi5-donkey.json"),
        help="Trainer config JSON (default: pi5-donkey)",
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path("/opt/continuonos/brain/rlds/episodes"),
        help="RLDS episodes directory",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("/opt/continuonos/brain/rlds/export"),
        help="Output directory for anonymized export",
    )
    parser.add_argument("--health", action="store_true", help="Run quick system health check")
    parser.add_argument("--train-local", action="store_true", help="Run local trainer")
    parser.add_argument("--export", action="store_true", help="Anonymize/validate episodes for cloud export")
    parser.add_argument("--post-validate", action="store_true", help="Run integration test after deployment")
    parser.add_argument("--real-hardware", action="store_true", help="Use real hardware path for post validation")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manager = TrainingManager(
        trainer_config=args.trainer_config,
        episodes_dir=args.episodes_dir,
        export_dir=args.export_dir,
        run_health=args.health,
        run_local_trainer=args.train_local,
        run_export=args.export,
        run_post_validate=args.post_validate,
        real_hardware=args.real_hardware,
    )
    reports = manager.run()
    for report in reports:
        print(f"[{report.status.upper():7}] {report.name}")
        if report.details:
            print(f"  details: {report.details}")
        if report.commands:
            print(f"  commands: {report.commands}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

