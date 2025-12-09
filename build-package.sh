#!/bin/bash
# Build ContinuonBrain Debian package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/packaging/continuonbrain"
BUILD_DIR="$SCRIPT_DIR/build"
VERSION="1.0.0"
ARCH="arm64"

echo "🏗️  Building ContinuonBrain v${VERSION} for ${ARCH}"

# Clean previous builds
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy application code
echo "📦 Copying application files..."
mkdir -p "$PACKAGE_DIR/opt/continuonbrain/app"
cp -r continuonbrain/* "$PACKAGE_DIR/opt/continuonbrain/app/"

# Copy requirements.txt
cp continuonbrain/requirements.txt "$PACKAGE_DIR/opt/continuonbrain/app/"

# Create default config
echo "⚙️  Creating default configuration..."
cat > "$PACKAGE_DIR/etc/continuonbrain/config.yaml" << EOF
# ContinuonBrain Configuration
version: 1.0.0
robot_name: "ContinuonBot"
server_port: 8080
kiosk_mode: true
auto_update: true
developer_mode: false

# Hardware
hardware:
  auto_detect: true
  camera_enabled: true
  arm_enabled: true
  drivetrain_enabled: true

# AI Models
models:
  chat_model: "google/gemma-3n-E2B-it"
  auto_download: true

# Logging
logging:
  level: "INFO"
  max_size_mb: 100
  retention_days: 30
EOF

# Set correct permissions
echo "🔒 Setting permissions..."
find "$PACKAGE_DIR" -type d -exec chmod 755 {} \;
find "$PACKAGE_DIR" -type f -exec chmod 644 {} \;
chmod 755 "$PACKAGE_DIR/DEBIAN"/{postinst,prerm,postrm}
chmod 755 "$PACKAGE_DIR/usr/bin"/*

# Build package
echo "📦 Building .deb package..."
dpkg-deb --build "$PACKAGE_DIR" "$BUILD_DIR/continuonbrain_${VERSION}_${ARCH}.deb"

# Verify package
echo "✅ Verifying package..."
dpkg-deb --info "$BUILD_DIR/continuonbrain_${VERSION}_${ARCH}.deb"
dpkg-deb --contents "$BUILD_DIR/continuonbrain_${VERSION}_${ARCH}.deb" | head -20

echo ""
echo "✅ Package built successfully!"
echo "📦 Output: $BUILD_DIR/continuonbrain_${VERSION}_${ARCH}.deb"
echo ""
echo "To install:"
echo "  sudo apt install ./build/continuonbrain_${VERSION}_${ARCH}.deb"
echo ""
