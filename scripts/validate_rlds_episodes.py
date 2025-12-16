#!/usr/bin/env python3
"""
Validate RLDS episodes with variant-aware rules.

This repo contains multiple episode shapes (see docs/rlds-variants.md):
- canonical episode_dir: <episode>/metadata.json + <episode>/steps/000000.jsonl
- legacy studio/mock episode.json: {"metadata":..., "steps":[...]}

This script auto-detects the path type and applies the appropriate validator.
"""
from __future__ import annotations

import sys
from pathlib import Path

from continuonbrain.rlds.variant_validators import validate_path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: validate_rlds_episodes.py <episode_dir|episode.json> [<path2> ...]")
        return 1

    failures = 0
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"{path}: missing")
            failures += 1
            continue
        result = validate_path(path)
        if result.errors:
            failures += 1
            print(f"{path}: FAIL")
            for err in result.errors:
                print(f"  - {err}")
        else:
            print(f"{path}: OK")
        if result.warnings:
            print(f"{path}: WARN")
            for warning in result.warnings:
                print(f"  - {warning}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

