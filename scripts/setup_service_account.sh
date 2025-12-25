#!/usr/bin/env bash
set -euo pipefail

# Setup script for Firebase service account credentials
# This script helps download and configure the service account JSON file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Determine config directory (same logic as start_services.sh)
DEFAULT_CONFIG_DIR="${REPO_ROOT}/brain_config"
if [[ -d "${DEFAULT_CONFIG_DIR}" ]]; then
    CONFIG_DIR="${CONFIG_DIR:-${DEFAULT_CONFIG_DIR}}"
elif [[ -d "/opt/continuonos/brain" ]]; then
    CONFIG_DIR="${CONFIG_DIR:-/opt/continuonos/brain}"
else
    CONFIG_DIR="${CONFIG_DIR:-${HOME}/.continuonbrain}"
fi

TARGET_FILE="${CONFIG_DIR}/service-account.json"
TEMPLATE_FILE="${REPO_ROOT}/service-account.template.json"

echo "=== Firebase Service Account Setup ==="
echo ""
echo "Config directory: ${CONFIG_DIR}"
echo "Target file: ${TARGET_FILE}"
echo ""

# Check if service account already exists
if [[ -f "${TARGET_FILE}" ]]; then
    echo "⚠️  Service account file already exists at: ${TARGET_FILE}"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Create config directory if it doesn't exist
mkdir -p "${CONFIG_DIR}"

echo "To download your service account key:"
echo ""
echo "1. Go to Firebase Console: https://console.firebase.google.com/project/continuonai/settings/serviceaccounts/adminsdk"
echo "2. Click 'Generate new private key'"
echo "3. Save the downloaded JSON file"
echo ""
read -p "Enter the path to the downloaded service account JSON file: " downloaded_file

if [[ ! -f "${downloaded_file}" ]]; then
    echo "❌ Error: File not found: ${downloaded_file}"
    exit 1
fi

# Validate JSON structure
if ! python3 -m json.tool "${downloaded_file}" > /dev/null 2>&1; then
    echo "❌ Error: File is not valid JSON"
    exit 1
fi

# Check required fields
required_fields=("type" "project_id" "private_key_id" "private_key" "client_email" "client_id")
missing_fields=()

for field in "${required_fields[@]}"; do
    if ! python3 -c "import json, sys; data = json.load(open('${downloaded_file}')); sys.exit(0 if '${field}' in data else 1)" 2>/dev/null; then
        missing_fields+=("${field}")
    fi
done

if [[ ${#missing_fields[@]} -gt 0 ]]; then
    echo "❌ Error: Missing required fields: ${missing_fields[*]}"
    exit 1
fi

# Copy the file to the target location
cp "${downloaded_file}" "${TARGET_FILE}"

# Set appropriate permissions (readable only by owner)
chmod 600 "${TARGET_FILE}"

echo ""
echo "✅ Service account file installed successfully!"
echo "   Location: ${TARGET_FILE}"
echo ""
echo "The file has been set with restrictive permissions (600 - owner read/write only)."
echo ""
echo "To verify the setup, you can check that CloudRelay can load it:"
echo "  python3 -c \"from firebase_admin import credentials; cred = credentials.Certificate('${TARGET_FILE}'); print('✅ Credentials valid')\""

