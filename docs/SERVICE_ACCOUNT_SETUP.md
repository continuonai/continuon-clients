# Firebase Service Account Setup

This guide explains how to set up Firebase service account credentials for the Continuon Brain runtime.

## Overview

The service account credentials are used by `CloudRelay` to authenticate with Firebase Firestore and enable cloud-based remote control of the robot.

The service account file should be placed at:
```
{config_dir}/service-account.json
```

Where `config_dir` is:
- `brain_config/` (default for desktop/dev mode)
- `/opt/continuonos/brain` (for device/rpi mode)
- `~/.continuonbrain` (fallback)

## Quick Setup

### Option 1: Automated Setup Script

Run the setup script to guide you through the process:

```bash
./scripts/setup_service_account.sh
```

This script will:
1. Determine your config directory
2. Guide you to download the service account key from Firebase Console
3. Validate and install the key file
4. Set appropriate file permissions

### Option 2: Manual Setup

1. **Download the Service Account Key**
   - Go to [Firebase Console](https://console.firebase.google.com/project/continuonai/settings/serviceaccounts/adminsdk)
   - Click "Generate new private key"
   - Save the downloaded JSON file

2. **Place the File**
   ```bash
   # Determine your config directory (see above)
   # Then copy the downloaded file:
   cp ~/Downloads/continuonai-*.json {config_dir}/service-account.json
   
   # Set restrictive permissions
   chmod 600 {config_dir}/service-account.json
   ```

3. **Validate the Setup**
   ```bash
   python3 scripts/validate_service_account.py {config_dir}/service-account.json
   ```

## Service Account Details

- **Project ID**: `continuonai`
- **Service Account Email**: `firebase-adminsdk-generic@continuonai.iam.gserviceaccount.com`

## Verification

To verify that the service account is working:

```python
from firebase_admin import credentials
cred = credentials.Certificate('path/to/service-account.json')
print("✅ Credentials valid")
```

Or check the CloudRelay logs when the brain service starts - it should log:
```
CloudRelay: Firebase initialized.
```

## Security Notes

- The service account file contains sensitive credentials
- File permissions should be set to `600` (owner read/write only)
- Do not commit the service account file to version control
- The template file (`service-account.template.json`) is safe to commit as it contains no secrets

## Troubleshooting

**Error: "No credentials found"**
- Ensure the file is at the correct path: `{config_dir}/service-account.json`
- Check that the config directory is correct for your environment

**Error: "Failed to initialize Firebase"**
- Validate the JSON file: `python3 scripts/validate_service_account.py {config_dir}/service-account.json`
- Ensure `firebase-admin` is installed: `pip install firebase-admin`
- Check file permissions: `ls -l {config_dir}/service-account.json`

**CloudRelay disabled**
- Check logs for the reason (missing credentials, import errors, etc.)
- The system will continue to work without CloudRelay - remote control will just be unavailable

