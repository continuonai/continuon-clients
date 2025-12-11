#!/usr/bin/env python
"""
Compact RLDS episodes by keeping the most recent N JSON episodes,
archiving older ones, and writing a small summary.

Usage:
  python scripts/compact_rlds.py \
    --episodes-dir /opt/continuonos/brain/rlds/episodes \
    --archive-dir /opt/continuonos/brain/rlds/archive \
    --summary-dir /opt/continuonos/brain/rlds/compact \
    --keep-latest 20
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact RLDS episodes (keep latest N, archive the rest).")
    p.add_argument("--episodes-dir", type=Path, default=Path("/opt/continuonos/brain/rlds/episodes"))
    p.add_argument("--archive-dir", type=Path, default=Path("/opt/continuonos/brain/rlds/archive"))
    p.add_argument("--summary-dir", type=Path, default=Path("/opt/continuonos/brain/rlds/compact"))
    p.add_argument("--keep-latest", type=int, default=20, help="Number of newest JSON episodes to retain")
    return p.parse_args()


def list_json_files(ep_dir: Path) -> List[Path]:
    return sorted([p for p in ep_dir.glob("*.json") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)


def move_older(files: List[Path], keep: int, archive_dir: Path) -> Tuple[List[Path], List[Path]]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    keep_files = files[:keep]
    archived = []
    for p in files[keep:]:
        dest = archive_dir / p.name
        try:
            shutil.move(str(p), str(dest))
            archived.append(dest)
        except Exception:
            continue
    return keep_files, archived


def summarize(files: List[Path]) -> dict:
    total_steps = 0
    for p in files:
        try:
            data = json.loads(p.read_text())
            steps = data.get("steps", [])
            total_steps += len(steps)
        except Exception:
            continue
    return {"episodes": len(files), "total_steps": total_steps}


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    files = list_json_files(args.episodes_dir)
    keep_files, archived = move_older(files, args.keep_latest, args.archive_dir)
    summary = {
        "kept": summarize(keep_files),
        "archived_count": len(archived),
        "kept_files": [p.name for p in keep_files],
        "archived_files": [p.name for p in archived],
    }
    summary_path = args.summary_dir / "compact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
