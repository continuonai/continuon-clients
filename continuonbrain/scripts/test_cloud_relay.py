#!/usr/bin/env python3
"""
Test Firebase Cloud Relay Setup

This script tests if Firebase is configured correctly for remote access.
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CONFIG_DIR = Path.home() / ".continuonbrain"
CRED_PATH = CONFIG_DIR / "service-account.json"


def test_credentials():
    """Test if credentials file exists and is valid."""
    print("=" * 50)
    print("Firebase Cloud Relay Test")
    print("=" * 50)

    if not CRED_PATH.exists():
        print(f"\n❌ No credentials found at: {CRED_PATH}")
        print("\nTo set up Firebase:")
        print("1. Go to Firebase Console → Project Settings → Service Accounts")
        print("2. Click 'Generate new private key'")
        print("3. Save the file as:", CRED_PATH)
        return False

    print(f"\n✓ Credentials file found: {CRED_PATH}")

    # Validate JSON
    try:
        with open(CRED_PATH) as f:
            creds = json.load(f)

        required_fields = ["type", "project_id", "private_key", "client_email"]
        missing = [f for f in required_fields if f not in creds]

        if missing:
            print(f"❌ Missing required fields: {missing}")
            return False

        print(f"✓ Project ID: {creds.get('project_id')}")
        print(f"✓ Client Email: {creds.get('client_email')}")

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False

    return True


def test_firebase_connection():
    """Test Firebase connectivity."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        print("\n❌ firebase-admin not installed")
        print("Run: pip3 install firebase-admin")
        return False

    print("\n✓ firebase-admin module available")

    try:
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(CRED_PATH))
            firebase_admin.initialize_app(cred)

        print("✓ Firebase initialized")

        # Test Firestore connection
        db = firestore.client()
        print("✓ Firestore client created")

        # Try a simple read
        test_ref = db.collection("_test_connection")
        test_ref.limit(1).get()
        print("✓ Firestore connection successful")

        return True

    except Exception as e:
        print(f"\n❌ Firebase connection failed: {e}")
        return False


def test_device_registration():
    """Test if device is registered in Firestore."""
    try:
        from firebase_admin import firestore
        db = firestore.client()

        # Get device ID
        device_id_path = CONFIG_DIR / "device_id.json"
        if device_id_path.exists():
            with open(device_id_path) as f:
                device_data = json.load(f)
            device_id = device_data.get("device_id", "unknown")
        else:
            device_id = "unknown"

        print(f"\n--- Device Registration ---")
        print(f"Device ID: {device_id}")

        # Check if robot exists in Firestore
        robot_ref = db.collection("robots").document(device_id)
        doc = robot_ref.get()

        if doc.exists:
            print(f"✓ Robot registered in Firestore")
            data = doc.to_dict()
            print(f"  Name: {data.get('name', 'N/A')}")
            print(f"  Last seen: {data.get('last_seen', 'N/A')}")
        else:
            print(f"⚠ Robot not yet registered in Firestore")
            print(f"  Creating registration...")
            robot_ref.set({
                "device_id": device_id,
                "name": "ContinuonBot",
                "status": "online",
                "last_seen": firestore.SERVER_TIMESTAMP,
            })
            print(f"✓ Robot registered")

        return True

    except Exception as e:
        print(f"❌ Device registration check failed: {e}")
        return False


def main():
    """Run all tests."""
    if not test_credentials():
        return 1

    if not test_firebase_connection():
        return 1

    test_device_registration()

    print("\n" + "=" * 50)
    print("✓ Firebase Cloud Relay is ready!")
    print("=" * 50)
    print("\nRemote commands can now be sent via Firestore:")
    print("  Collection: robots/{device_id}/commands")
    print("  Document format: {type: 'set_mode', payload: {mode: 'manual'}, status: 'pending'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
