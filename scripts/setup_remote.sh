#!/bin/bash
# Setup script for Continuon Robot (Pi 5) Remote Access
# Run this on the Raspberry Pi.

set -e

echo "🤖 Continuon Robot Remote Setup"
echo "================================"

# 1. Install Avahi
echo "📦 Checking Avahi..."
if ! command -v avahi-daemon >/dev/null; then
    echo "   Installing avahi-daemon..."
    sudo apt-get update
    sudo apt-get install -y avahi-daemon
fi

# 2. Configure Avahi Service
echo "📢 Configuring mDNS (Avahi)..."
SERVICE_FILE="/etc/avahi/services/continuon.service"
if [ ! -f "$SERVICE_FILE" ]; then
    echo "   Creating $SERVICE_FILE..."
    sudo bash -c 'cat > /etc/avahi/services/continuon.service <<EOF
<?xml version="1.0" standalone="no"?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">Continuon Robot on %h</name>
  <service>
    <type>_continuon._tcp</type>
    <port>8080</port>
    <txt-record>version=1.0</txt-record>
    <txt-record>model=Pi5</txt-record>
    <txt-record>http_port=8080</txt-record>
  </service>
  <service>
    <type>_ssh._tcp</type>
    <port>22</port>
  </service>
</service-group>
EOF'
    sudo systemctl restart avahi-daemon
    echo "   ✅ mDNS service configured."
else
    echo "   ✅ Avahi service already exists."
fi

# 3. Setup ContinuonBrain User/Permissions
echo "👤 Checking user permissions..."
if groups $USER | grep &>/dev/null 'dialout'; then
    echo "   ✅ User $USER in dialout group."
else
    echo "   Adding $USER to dialout/gpio/i2c..."
    sudo usermod -aG dialout,gpio,i2c $USER || true
fi

# 4. SSH Setup
echo "🔒 SSH Setup..."
if [ ! -d ~/.ssh ]; then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
fi

echo "   Ensuring SSH is enabled..."
sudo systemctl enable ssh
sudo systemctl start ssh

echo ""
echo "🎉 Setup Complete!"
echo "   Your robot should now be discoverable as '$HOSTNAME.local' or via 'python scripts/find_robot.py'."
echo "   Next steps:"
echo "   1. Run 'python scripts/setup_ssh.py $HOSTNAME' from your dev machine to deploy keys."
echo "   2. Run 'python scripts/sync_robot.py --host $HOSTNAME' to push code."
