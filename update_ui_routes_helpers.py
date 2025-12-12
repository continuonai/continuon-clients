import os

file_path = r"c:\Users\CraigM\source\repos\ContinuonXR\continuonbrain\api\routes\ui_routes.py"

with open(file_path, "a", encoding="utf-8") as f:
    f.write("\n\n# --- Helper Functions for server.py ---\n")
    
    f.write("def get_home_html() -> str:\n")
    f.write("    return \"<html><body><h1>Home</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_status_html() -> str:\n")
    f.write("    return \"<html><body><h1>Status</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_dashboard_html() -> str:\n")
    f.write("    return \"<html><body><h1>Dashboard</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_chat_html() -> str:\n")
    f.write("    return \"<html><body><h1>Chat</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_settings_html() -> str:\n")
    f.write("    return \"<html><body><h1>Settings</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_manual_html() -> str:\n")
    f.write("    return \"<html><body><h1>Manual Control</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_tasks_html() -> str:\n")
    f.write("    return \"<html><body><h1>Tasks</h1><p>Placeholder</p></body></html>\"\n\n")
    
    f.write("def get_brain_map_html() -> str:\n")
    f.write("    return \"<html><body><h1>Brain Map</h1><p>Placeholder</p></body></html>\"\n\n")
    
    # Also ensure get_hope_training_html is there (it was in server.py line 448)
    f.write("def get_hope_training_html() -> str:\n")
    f.write("    return \"<html><body><h1>HOPE Training</h1><p>Placeholder</p></body></html>\"\n\n")

print("Successfully added helper functions to ui_routes.py")
