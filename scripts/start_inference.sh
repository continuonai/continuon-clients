#!/usr/bin/env bash
set -e

# Wrapper to start services in INFERENCE mode (No Training Loops)
export CONTINUON_RUN_WAVECORE=0
export CONTINUON_ENABLE_BACKGROUND_TRAINER=0
export CONTINUON_FORCE_MOCK_HARDWARE=1
export CONTINUON_HEADLESS=1

echo "Starting Continuon Services in INFERENCE MODE..."
echo " - WaveCore: DISABLED"
echo " - Background Trainer: DISABLED"
echo " - Hardware: MOCK"

./scripts/start_services.sh start --mode desktop
