#!/bin/bash
# setup_remote.sh - Automate Pi 5 remote discovery and SSH setup

ROBOT_IP=$1

if [ -z "$ROBOT_IP" ]; then
    echo "Usage: ./scripts/setup_remote.sh <robot_ip>"
    exit 1
fi

echo "🛡️  Setting up remote conductor for robot at $ROBOT_IP..."

# 1. Deploy SSH keys
python scripts/remote_conductor.py --setup-keys

# 2. Install Avahi and setup service
echo "📡 Configuring Avahi/mDNS on robot..."
ssh pi@$ROBOT_IP "sudo apt-get update && sudo apt-get install -y avahi-daemon"
scp continuonbrain/systemd/continuon.service pi@$ROBOT_IP:/tmp/
ssh pi@$ROBOT_IP "sudo mv /tmp/continuon.service /etc/avahi/services/ && sudo systemctl restart avahi-daemon"

# 3. Cache the IP
echo $ROBOT_IP > .robot_ip

echo "✅ Remote setup complete. You can now use './cb' commands."