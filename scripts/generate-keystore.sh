#!/bin/bash
# Generate a release signing keystore for ContinuonXR
# Run once, then add the .jks to apps/continuonxr/

set -e

KEYSTORE_DIR="$(dirname "$0")/../apps/continuonxr"
KEYSTORE_FILE="$KEYSTORE_DIR/upload-keystore.jks"
ALIAS="upload"
VALIDITY=10000

if [ -f "$KEYSTORE_FILE" ]; then
  echo "Keystore already exists at $KEYSTORE_FILE"
  echo "Delete it first if you want to regenerate."
  exit 1
fi

echo "Generating release keystore..."
echo "You will be prompted for passwords and certificate info."
echo ""

keytool -genkeypair \
  -v \
  -keystore "$KEYSTORE_FILE" \
  -keyalg RSA \
  -keysize 2048 \
  -validity $VALIDITY \
  -alias "$ALIAS" \
  -storetype JKS

echo ""
echo "Keystore created at: $KEYSTORE_FILE"
echo ""
echo "IMPORTANT: Add the following to your local.properties or CI secrets:"
echo "  KEYSTORE_PASSWORD=<your store password>"
echo "  KEY_PASSWORD=<your key password>"
echo ""
echo "DO NOT commit the keystore or passwords to git!"
