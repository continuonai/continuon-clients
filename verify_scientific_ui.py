
import sys
import os
from continuonbrain.api.routes import ui_routes

def verify_scientific_ui():
    print("🔬 Verifying Scientific UI Pages...")
    
    pages = {
        "Stability": ui_routes.get_hope_stability_html(),
        "Dynamics": ui_routes.get_hope_dynamics_html(),
        "Memory": ui_routes.get_hope_memory_html(),
        "Performance": ui_routes.get_hope_performance_html(),
        "Brain Map": ui_routes.get_brain_map_html()
    }
    
    all_passed = True
    
    for name, html in pages.items():
        print(f"\n[CHECK] {name} Page")
        
        # Check basic structure
        if "<!DOCTYPE html>" not in html:
            print(f"❌ FAILED: Invalid HTML structure")
            all_passed = False
            continue
            
        # Check for Chart.js (except Memory/Brain Map which use custom/Three.js)
        if name in ["Stability", "Dynamics", "Performance"]:
            if "Chart.js" not in html and "canvas" not in html:
                print(f"❌ FAILED: Missing Chart.js or Canvas")
                all_passed = False
                continue
                
        # Check for Theory Section
        if "Scientific Background" not in html and "Neural Topology" not in html: # Brain map has different HUD
             print(f"❌ FAILED: Missing Theory/Info Section")
             all_passed = False
             continue
             
        print(f"✅ PASSED ({len(html)} bytes)")
        
    return all_passed

if __name__ == "__main__":
    success = verify_scientific_ui()
    if success:
        print("\n✅ All Scientific UI pages verified successfully.")
        sys.exit(0)
    else:
        print("\n❌ Verification Failed.")
        sys.exit(1)
