#!/usr/bin/env python3
"""
Validate a Firebase service account JSON file.
"""

import json
import sys
from pathlib import Path


def validate_service_account(file_path: str):
    """Validate a service account JSON file."""
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    
    required_fields = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
    ]
    
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    
    # Validate type
    if data.get("type") != "service_account":
        return False, f"Expected type='service_account', got '{data.get('type')}'"
    
    # Validate private key format
    private_key = data.get("private_key", "")
    if not private_key.startswith("-----BEGIN"):
        return False, "Private key does not appear to be in PEM format"
    
    # Try to import and validate with firebase_admin if available
    try:
        import firebase_admin
        from firebase_admin import credentials
        
        cred = credentials.Certificate(str(path))
        # If we got here, the credentials are valid
        return True, "✅ Service account file is valid and can be loaded by firebase_admin"
    except ImportError:
        return True, "✅ Service account file structure is valid (firebase_admin not available for full validation)"
    except Exception as e:
        return False, f"Credentials failed to load: {e}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 validate_service_account.py <path-to-service-account.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid, message = validate_service_account(file_path)
    
    if is_valid:
        print(message)
        sys.exit(0)
    else:
        print(f"❌ {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

