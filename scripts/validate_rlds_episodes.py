#!/usr/bin/env python3
"""
Validate RLDS episode JSON files against continuonbrain.rlds.validators.
Intended for quick pre-commit/CI checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from continuonbrain.rlds.validators import validate_episode


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: validate_rlds_episodes.py <episode.json> [<episode2.json> ...]")
        return 1

    failures = 0
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"{path}: missing")
            failures += 1
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"{path}: failed to read/parse JSON: {e}")
            failures += 1
            continue
        result = validate_episode(data)
        if result.errors:
            failures += 1
            print(f"{path}: FAIL")
            for err in result.errors:
                print(f"  - {err}")
        else:
            print(f"{path}: OK")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

