#!/usr/bin/env python3
"""
Main entry point for running the orchestrator as a module.

Usage:
    python -m orchestrator --dev health
    python -m orchestrator --dev start
"""
import sys
from pathlib import Path

# Add parent directory to path for package resolution
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from orchestrator.cli import main

if __name__ == "__main__":
    sys.exit(main())
