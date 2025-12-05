"""
UI Routes: Serve HTML/JS/CSS for the ContinuonBrain Web Interface.
"""
from typing import Dict
from pathlib import Path

# Placeholder for actual UI templates or static file logic
# In the original monolithic file, this was inline strings.
# We should probably look at moving those strings to separate files or
# keeping them here for now but structured.

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ContinuonBrain</title>
    <style>
        /* Base styles */
        body { font-family: sans-serif; background: #111; color: #eee; margin: 0; display: flex; height: 100vh; }
        
        /* Left Sidebar */
        #sidebar { width: 250px; background: #222; border-right: 1px solid #333; display: flex; flex-direction: column; }
        #sidebar .logo { padding: 20px; font-weight: bold; font-size: 1.2em; border-bottom: 1px solid #333; }
        #sidebar a { padding: 15px 20px; color: #aaa; text-decoration: none; display: block; }
        #sidebar a:hover, #sidebar a.active { background: #333; color: white; }
        
        /* Main Content */
        #content { flex: 1; padding: 20px; overflow-y: auto; }
        
        iframe { width: 100%; height: 100%; border: none; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="logo">🧠 ContinuonBrain</div>
        <a href="/ui/dashboard" target="main_frame" class="active">Dashboard</a>
        <a href="/ui/status" target="main_frame">Brain Status</a>
        <a href="/ui/chat" target="main_frame">Agent Manager</a>
        <a href="/ui/settings" target="main_frame">Settings</a>
    </div>
    <div id="content">
        <iframe name="main_frame" src="/ui/dashboard"></iframe>
    </div>
</body>
</html>
"""

# Brain Status Page
STATUS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Brain Status</title>
    <style>
        body { background: #1a1a1a; color: #ddd; font-family: monospace; padding: 20px; }
        .card { background: #2a2a2a; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
        h2 { margin-top: 0; color: #4CAF50; }
        pre { white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Brain Status introspection</h1>
    
    <div class="card">
        <h2>🏠 Identity (Shell)</h2>
        <div id="identity-report">Loading...</div>
    </div>
    
    <div class="card">
        <h2>🧠 Memory & Learning</h2>
        <div id="memory-stats">Loading...</div>
    </div>

    <script>
        async function loadStatus() {
            try {
                // Fetch introspection data from the API (we need to build this endpoint)
                const res = await fetch('/api/status/introspection');
                const data = await res.json();
                
                document.getElementById('identity-report').textContent = JSON.stringify(data.shell, null, 2);
                document.getElementById('memory-stats').textContent = JSON.stringify(data.memory, null, 2);
            } catch (e) {
                document.getElementById('identity-report').textContent = "Error loading status";
            }
        }
        loadStatus();
        setInterval(loadStatus, 5000);
    </script>
</body>
</html>
"""

def get_home_html() -> str:
    return HOME_HTML

def get_status_html() -> str:
    return STATUS_HTML
