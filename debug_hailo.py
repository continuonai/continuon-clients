import sys
from pathlib import Path

# Add project root to path
sys.path.append("/home/craigm26/Downloads/ContinuonXR")

try:
    print("Attempting to import HailoVision...")
    from continuonbrain.services.hailo_vision import HailoVision
    print("✅ Import successful.")
    
    print("Initializing HailoVision...")
    hailo = HailoVision(enabled=True)
    status = hailo.get_state()
    print(f"Hailo Status: {status}")
    
    if not status.get("available"):
        print("\nDIAGNOSIS:")
        print("- Check if 'hailo_platform' is installed in this venv.")
        print("- Check if HEF file exists at configured path.")
        
except ImportError as e:
    print(f"❌ Import Failed: {e}")
except Exception as e:
    print(f"❌ Initialization Failed: {e}")
