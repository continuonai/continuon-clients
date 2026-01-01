#!/usr/bin/env bash
set -e

# ContinuonXR Swap Expander
# Usage: sudo ./scripts/expand_swap.sh [size_mb]

TARGET_SWAP_MB=${1:-4096} # Default to 4GB

echo "==========================================="
echo "ContinuonXR Swap Expander"
echo "Target Size: ${TARGET_SWAP_MB}MB"
echo "==========================================="

if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root."
   exit 1
fi

# Detect Raspberry Pi OS with dphys-swapfile
if command -v dphys-swapfile >/dev/null; then
    echo "Detected dphys-swapfile (Raspberry Pi OS)."
    CONF_FILE="/etc/dphys-swapfile"
    
    # Backup
    cp $CONF_FILE "${CONF_FILE}.bak"
    
    # Update configuration
    echo "Updating ${CONF_FILE}..."
    sed -i "s/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=${TARGET_SWAP_MB}/" $CONF_FILE
    
    # Apply
    echo "Applying new swap size..."
    dphys-swapfile stop
    dphys-swapfile setup
    dphys-swapfile start
    
    echo "✅ Success! Swap updated."
    free -h
    exit 0
fi

# Fallback for generic Linux (Swapfile)
SWAP_FILE="/swapfile"
echo "dphys-swapfile not found. Attempting generic swapfile at ${SWAP_FILE}..."

if [[ -f "$SWAP_FILE" ]]; then
    echo "Existing swapfile found. Disabling..."
    swapoff $SWAP_FILE
    rm $SWAP_FILE
fi

echo "Allocating ${TARGET_SWAP_MB}MB..."
fallocate -l ${TARGET_SWAP_MB}M $SWAP_FILE
chmod 600 $SWAP_FILE
mkswap $SWAP_FILE
swapon $SWAP_FILE

# Persist in fstab if not present
if ! grep -q "$SWAP_FILE" /etc/fstab; then
    echo "${SWAP_FILE} none swap sw 0 0" >> /etc/fstab
    echo "Added to /etc/fstab for persistence."
fi

echo "✅ Success! Swap created."
free -h
