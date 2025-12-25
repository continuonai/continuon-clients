#!/usr/bin/env python3
"""Test CloudRelay service account integration."""
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from continuonbrain.services.cloud_relay import CloudRelay
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_cloudrelay():
    config_dir = "brain_config"
    device_id = "test-device-001"
    
    print(f"Testing CloudRelay with config_dir={config_dir}, device_id={device_id}")
    print(f"Service account path: {Path(config_dir) / 'service-account.json'}")
    print()
    
    relay = CloudRelay(config_dir, device_id)
    
    # Check if credentials path exists
    cred_path = Path(config_dir) / "service-account.json"
    print(f"Service account file exists: {cred_path.exists()}")
    
    if cred_path.exists():
        print(f"Service account file size: {cred_path.stat().st_size} bytes")
        print(f"Service account file permissions: {oct(cred_path.stat().st_mode)[-3:]}")
    
    # Try to start (this will attempt to initialize Firebase)
    print("\nAttempting to start CloudRelay...")
    try:
        relay.start(None)  # Pass None for brain_service for testing
        print(f"\n✅ CloudRelay initialized successfully!")
        print(f"   Enabled: {relay.enabled}")
        if relay.enabled:
            print(f"   Firestore client: {relay.db is not None}")
    except Exception as e:
        print(f"\n❌ Failed to initialize CloudRelay: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return relay.enabled

if __name__ == "__main__":
    success = test_cloudrelay()
    sys.exit(0 if success else 1)

