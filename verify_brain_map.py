
import sys
import os
from pathlib import Path
import json

# Setup paths
sys.path.insert(0, os.getcwd())

from continuonbrain.services.brain_service import BrainService
from continuonbrain.hope_impl.brain import HOPEBrain
from continuonbrain.hope_impl.config import HOPEConfig
from continuonbrain.api.routes import ui_routes

def verify_brain_map():
    print("🧠 Verifying Brain Map Visualization...")
    
    # 1. Initialize Service (Mock Hardware)
    service = BrainService(
        config_dir="/tmp/brain_map_test",
        prefer_real_hardware=False,
        auto_detect=False,
        allow_mock_fallback=True
    )
    
    # 2. Initialize HOPE Brain manually to ensure structure
    print("Initialize HOPE Brain...")
    config = HOPEConfig()
    service.hope_brain = HOPEBrain(config, obs_dim=10, action_dim=4, output_dim=4)
    service.hope_brain.reset()
    
    # 3. Test Data Endpoint
    print("\n[TEST] get_brain_structure()")
    structure = service.get_brain_structure()
    
    print(json.dumps(structure, indent=2))
    
    # Validation
    if "topology" not in structure:
         print("❌ FAILD: Missing 'topology'")
         return False
    
    if "state" not in structure:
         print("❌ FAILED: Missing 'state'")
         return False
         
    if len(structure["topology"]["columns"]) == 0:
         print("❌ FAILED: No columns found")
         return False
         
    print("\n✅ Data Structure Valid")
    
    # 4. Test HTML Generation
    print("\n[TEST] get_brain_map_html()")
    try:
        html = ui_routes.get_brain_map_html()
        if "<!DOCTYPE html>" in html and "Three.js" in html:
            print(f"✅ HTML Generated ({len(html)} bytes)")
        else:
            print("❌ HTML Content Invalid")
            return False
    except AttributeError:
        print("❌ get_brain_map_html not found in ui_routes")
        return False
        
    return True

if __name__ == "__main__":
    success = verify_brain_map()
    sys.exit(0 if success else 1)
