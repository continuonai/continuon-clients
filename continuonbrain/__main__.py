"""
ContinuonBrain Package Entry Point

This redirects to the startup_manager, which is the canonical way to start
all ContinuonBrain services.

Usage:
    python -m continuonbrain --config-dir /path/to/config --port 8081

For production, use systemd:
    systemctl --user start continuonbrain.service

Or the shell script:
    ./scripts/start_services.sh start --mode desktop|rpi
"""

import sys
from continuonbrain.startup_manager import main

if __name__ == "__main__":
    # Redirect to startup_manager as the canonical entry point
    print("=" * 70)
    print("ContinuonBrain Package Entry Point")
    print("=" * 70)
    print("")
    print("Starting via startup_manager (the recommended entry point)")
    print("")
    main()

