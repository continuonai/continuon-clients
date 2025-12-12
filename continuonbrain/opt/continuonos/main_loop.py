"""Lightweight entry point for developer kit loops on Pi 5.

This script keeps imports minimal so it can run before optional
hardware dependencies (Hailo SDK, camera drivers) are installed.
Use it to wire experiments in the numbered folders back into the
Continuon Brain runtime that lives under `/opt/continuonos/brain/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

DEVKIT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = Path("/opt/continuonos/brain")


def _build_layout() -> Dict[str, Path]:
    return {
        "00_data": DEVKIT_ROOT / "00_data",
        "01_vision_dreamer": DEVKIT_ROOT / "01_vision_dreamer",
        "02_liquid_reflex": DEVKIT_ROOT / "02_liquid_reflex",
        "03_mamba_brain": DEVKIT_ROOT / "03_mamba_brain",
    }


def ensure_layout() -> None:
    """Create the numbered dev kit directories if they are missing."""
    for path in _build_layout().values():
        path.mkdir(parents=True, exist_ok=True)


def describe_layout(check_runtime: bool = False) -> str:
    """Return a human-readable description of the dev kit layout.

    Args:
        check_runtime: When True, append Continuon Brain runtime and Hailo
            placeholder paths so operators can confirm the Pi 5 is ready.
    """

    lines = [f"Developer kit root: {DEVKIT_ROOT}"]
    for name, path in _build_layout().items():
        status = "present" if path.exists() else "missing"
        lines.append(f"- {name}: {path} ({status})")

    if check_runtime:
        runtime_paths = {
            "runtime_root": RUNTIME_ROOT,
            "rlds": RUNTIME_ROOT / "rlds/episodes",
            "current_adapter": RUNTIME_ROOT / "model/adapters/current",
            "candidate_adapter": RUNTIME_ROOT / "model/adapters/candidate",
            "hailo_hef": RUNTIME_ROOT / "model/base_model/model.hef",
        }
        lines.append("Runtime alignment checks:")
        for label, path in runtime_paths.items():
            status = "present" if path.exists() else "missing"
            lines.append(f"  - {label}: {path} ({status})")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ContinuonOS developer loop stub")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the dev kit layout and optional runtime paths",
    )
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Include Continuon Brain runtime/Hailo path checks in the description",
    )
    args = parser.parse_args()

    ensure_layout()

    if args.describe:
        print(describe_layout(check_runtime=args.check_runtime))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
