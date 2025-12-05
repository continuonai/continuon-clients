"""
UI Routes: Serve HTML/JS/CSS for the ContinuonBrain Web Interface.
"""
from typing import Dict
from pathlib import Path

# Common CSS for Premium IO Aesthetic
COMMON_CSS = """
    :root {
        --bg-color: #0d0d0d;
        --sidebar-bg: #161616;
        --card-bg: #1e1e1e;
        --accent-color: #00ff88;
        --accent-dim: rgba(0, 255, 136, 0.1);
        --text-color: #e0e0e0;
        --text-dim: #888;
        --border-color: #333;
        --font-main: 'Inter', system-ui, -apple-system, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }
    
    body { font-family: var(--font-main); background: var(--bg-color); color: var(--text-color); margin: 0; display: flex; height: 100vh; overflow: hidden; }
    
    /* Layout */
    #sidebar { width: 260px; background: var(--sidebar-bg); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding-top: 20px; box-shadow: 2px 0 10px rgba(0,0,0,0.5); z-index: 10; }
    #sidebar .logo { padding: 0 24px 24px; font-weight: 800; font-size: 1.4em; letter-spacing: -0.5px; color: #fff; display: flex; align-items: center; gap: 10px; }
    #sidebar .logo span { color: var(--accent-color); }
    
    #content { flex: 1; position: relative; display: flex; flex-direction: column; width: 100%; height: 100%; }
    
    iframe { flex: 1; border: none; width: 100%; height: 100%; }
    
    /* Navigation */
    #sidebar a { padding: 14px 24px; color: var(--text-dim); text-decoration: none; display: flex; align-items: center; gap: 12px; font-size: 0.95em; transition: all 0.2s ease; border-left: 3px solid transparent; }
    #sidebar a:hover { color: #fff; background: rgba(255,255,255,0.03); }
    #sidebar a.active { background: var(--accent-dim); color: var(--accent-color); border-left-color: var(--accent-color); font-weight: 500; }
    
    /* Icons (using unicode for portability or SVG) */
    .icon { width: 20px; text-align: center; }
    
    /* System Status Footer in Sidebar */
    .sidebar-footer { margin-top: auto; padding: 20px; font-size: 0.8em; color: var(--text-dim); border-top: 1px solid var(--border-color); }
    .status-dot { width: 8px; height: 8px; background: var(--accent-color); border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 8px var(--accent-color); }
"""

HOME_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ContinuonBrain</title>
    <style>
        {COMMON_CSS}
    </style>
    <!-- Preload Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    
    <script>
        function selectNav(el) {{
            document.querySelectorAll('#sidebar a').forEach(a => a.classList.remove('active'));
            el.classList.add('active');
        }}
    </script>
</head>
<body>
    <div id="sidebar">
        <div class="logo">
            <span style="font-size: 1.2em;">🧠</span> Continuon<span>OS</span>
        </div>
        
        <div style="flex: 1;">
            <a href="/ui/dashboard" target="main_frame" class="active" onclick="selectNav(this)">
                <span class="icon">📊</span> Dashboard
            </a>
            <a href="/ui/manual" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🎮</span> Manual Control
            </a>
            <a href="/ui/chat" target="main_frame" onclick="selectNav(this)">
                <span class="icon">💬</span> Agent Chat
            </a>
            <a href="/ui/status" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🔍</span> Brain Status
            </a>
            <a href="/ui/settings" target="main_frame" onclick="selectNav(this)">
                <span class="icon">⚙️</span> Settings
            </a>
        </div>
        
        <div class="sidebar-footer">
            <div><span class="status-dot"></span> System Online</div>
            <div style="margin-top: 8px; opacity: 0.6;">v2.0.0-alpha</div>
        </div>
    </div>
    
    <div id="content">
        <iframe name="main_frame" src="/ui/dashboard"></iframe>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dashboard</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; font-weight: 800; letter-spacing: -1px; }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }}
        .card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-2px); border-color: #444; }}
        
        .card h3 {{ margin-top: 0; color: var(--text-dim); font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
        .card .value {{ font-size: 2.5em; font-weight: 700; color: #fff; margin: 10px 0; }}
        .card .meta {{ font-size: 0.9em; color: var(--accent-color); }}
        
        .status-badge {{ display: inline-flex; align-items: center; padding: 6px 12px; background: rgba(0, 255, 136, 0.15); color: var(--accent-color); border-radius: 20px; font-weight: 600; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>Mission Control</h1>
            <p style="color: var(--text-dim);">Real-time monitoring of robotic cortex.</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> AUTONOMOUS MODE
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Active Mode</h3>
            <div class="value">Idle</div>
            <div class="meta">Waiting for instruction</div>
        </div>
        <div class="card">
            <h3>Battery</h3>
            <div class="value">98%</div>
            <div class="meta">~4h 20m remaining</div>
        </div>
        <div class="card">
            <h3>CPU Load</h3>
            <div class="value">12%</div>
            <div class="meta">Temperature: 42°C</div>
        </div>
        <div class="card">
            <h3>Memory Bank</h3>
            <div class="value" id="memory-count">-</div>
            <div class="meta">Local Episodes</div>
        </div>
    </div>
    
    <div style="margin-top: 40px;">
        <h3>Activity Log</h3>
        <div style="background: #111; border-radius: 8px; padding: 20px; font-family: var(--font-mono); font-size: 0.9em; color: #aaa; border: 1px solid var(--border-color);">
            <div>[13:24:55] System booted successfully (Cold Boot)</div>
            <div>[13:24:56] LAN Discovery service started</div>
            <div>[13:24:57] Hardware detected: OAK-D Lite, PCA9685</div>
            <div>[13:24:58] Brain Service initialized</div>
            <div>[13:25:00] Sidecar Trainer process attached</div>
        </div>
    </div>

    <script>
        // Check memory stats
        async function updateStats() {{
            try {{
                const res = await fetch('/api/status/introspection');
                const data = await res.json();
                
                // Update mode (safely)
                const modeEl = document.querySelector('.card:nth-of-type(1) .value');
                if (modeEl) modeEl.innerText = 'Active'; 
                
                if(data.memory) {{
                    const memEl = document.getElementById('memory-count');
                    if (memEl) memEl.innerText = data.memory.local_episodes;
                }}
            }} catch(e) {{ console.log(e); }}
        }}
        setInterval(updateStats, 5000);
        updateStats();
    </script>
</body>
</html>
"""

CHAT_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agent Chat</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 0; display: flex; flex-direction: column; background: #111; }}
        
        #chat-history {{ flex: 1; padding: 40px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; max-width: 900px; margin: 0 auto; width: 100%; box-sizing: border-box; }}
        
        .message {{ display: flex; gap: 16px; max-width: 80%; }}
        .message.user {{ align-self: flex-end; flex-direction: row-reverse; }}
        .message.user .bubble {{ background: #2a2a2a; color: #fff; border: 1px solid #333; }}
        .message.assistant .bubble {{ background: rgba(0, 255, 136, 0.05); color: #e0e0e0; border: 1px solid rgba(0, 255, 136, 0.2); }}
        
        .avatar {{ width: 36px; height: 36px; border-radius: 50%; background: #333; display: flex; align-items: center; justify-content: center; font-size: 1.2em; flex-shrink: 0; }}
        .user .avatar {{ background: #444; }}
        .assistant .avatar {{ background: var(--accent-dim); color: var(--accent-color); }}
        
        .bubble {{ padding: 16px 20px; border-radius: 12px; line-height: 1.5; position: relative; font-size: 0.95em; white-space: pre-wrap; }}
        
        #input-area {{ padding: 30px; background: var(--card-bg); border-top: 1px solid var(--border-color); display: flex; justify-content: center; }}
        .input-container {{ max-width: 900px; width: 100%; position: relative; }}
        
        textarea {{ width: 100%; background: #0d0d0d; border: 1px solid #333; border-radius: 12px; padding: 18px; color: #fff; font-family: var(--font-main); font-size: 1em; resize: none; height: 60px; outline: none; transition: border 0.2s; padding-right: 60px; box-sizing: border-box; }}
        textarea:focus {{ border-color: var(--accent-color); }}
        
        button#send {{ position: absolute; right: 12px; bottom: 12px; background: var(--accent-color); color: #000; border: none; border-radius: 8px; width: 36px; height: 36px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-weight: bold; transition: opacity 0.2s; }}
        button#send:hover {{ opacity: 0.9; }}
        button#send:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        
        .status-pill {{ position: absolute; top: -40px; left: 50%; transform: translateX(-50%); background: #222; padding: 6px 16px; border-radius: 20px; font-size: 0.8em; color: #888; border: 1px solid #333; opacity: 0; transition: opacity 0.3s; }}
        .status-pill.visible {{ opacity: 1; }}

        /* Tool Usage Styling */
        .tool-call {{ font-family: var(--font-mono); font-size: 0.85em; margin-top: 8px; background: #000; padding: 10px; border-radius: 6px; border-left: 3px solid #ff0055; }}

        /* Typing Animation */
        .typing-dot {{
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-color);
            animation: typing 1.4s infinite ease-in-out both;
            margin: 0 2px;
            opacity: 0.7;
        }}
        .typing-dot:nth-child(1) {{ animation-delay: -0.32s; }}
        .typing-dot:nth-child(2) {{ animation-delay: -0.16s; }}
        @keyframes typing {{
            0%, 80%, 100% {{ transform: scale(0); opacity: 0.5; }}
            40% {{ transform: scale(1); opacity: 1; }}
        }}
    </style>
</head>
<body>
    <div id="chat-history">
        <div class="message assistant">
            <div class="avatar">🧠</div>
            <div class="bubble">I am online. Systems nominal. Models loaded.<br><br>How can I assist you today?</div>
        </div>
    </div>
    
    <div id="input-area">
        <!-- Status pill removed in favor of inline bubbles -->
        <div class="input-container">
            <textarea id="msg-input" placeholder="Message the Brain... (Cmd+Enter to send)"></textarea>
            <button id="send">➤</button>
        </div>
    </div>

    <script>
        const input = document.getElementById('msg-input');
        const sendBtn = document.getElementById('send');
        const history = document.getElementById('chat-history');
        
        function appendMessage(role, text) {{
            const div = document.createElement('div');
            div.className = `message ${{role}}`;
            
            // Format tools if visible in text
            let processedText = text;
            if (role === 'assistant' && text.includes('[TOOL:')) {{
             // Simple naive formatting for tool calls
             processedText = processedText.replace(/(\\[TOOL:.*?\\])/g, '<div class="tool-call">$1</div>');
            }}

            div.innerHTML = `
                <div class="avatar">${{role === 'user' ? '👤' : '🧠'}}</div>
                <div class="bubble">${{processedText}}</div>
            `;
            history.appendChild(div);
            history.scrollTop = history.scrollHeight;
        }}

        function showThinking() {{
            const div = document.createElement('div');
            div.id = 'thinking-bubble';
            div.className = 'message assistant';
            div.innerHTML = `
                <div class="avatar">🧠</div>
                <div class="bubble" style="display: flex; align-items: center; height: 24px;">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            `;
            history.appendChild(div);
            history.scrollTop = history.scrollHeight;
            return div;
        }}

        function removeThinking() {{
            const el = document.getElementById('thinking-bubble');
            if (el) el.remove();
        }}
        
        async function sendMessage() {{
            const text = input.value.trim();
            if (!text) return;
            
            // Clear input
            input.value = '';
            input.style.height = '60px'; // Reset height
            
            appendMessage('user', text);
            input.disabled = true;

            // Show Thinking
            showThinking();
            
            try {{
                const res = await fetch('/api/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ message: text }})
                }});
                
                const data = await res.json();
                
                // Remove Thinking before showing response
                removeThinking();
                appendMessage('assistant', data.response);
                
            }} catch(e) {{
                removeThinking();
                appendMessage('assistant', "⚠️ Error: " + e.message);
            }} finally {{
                input.disabled = false;
                input.focus();
            }}
        }}
        
        sendBtn.onclick = sendMessage;
        
        input.onkeydown = (e) => {{
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {{
                sendMessage();
            }}
        }};
    </script>
</body>
</html>
"""

SETTINGS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Settings</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; }}
        
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ border-bottom: 1px solid var(--border-color); padding-bottom: 15px; margin-bottom: 25px; font-size: 1.1em; color: var(--accent-color); }}
        
        .form-group {{ margin-bottom: 20px; }}
        label {{ display: block; margin-bottom: 8px; color: var(--text-dim); font-size: 0.9em; }}
        input[type="text"], select {{ width: 100%; max-width: 400px; padding: 12px; background: #111; border: 1px solid var(--border-color); color: #fff; border-radius: 6px; font-family: var(--font-main); }}
        input[type="checkbox"] {{ transform: scale(1.2); margin-right: 10px; }}
        
        .btn {{ padding: 10px 20px; background: var(--card-bg); border: 1px solid var(--border-color); color: #fff; border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
        .btn:hover {{ background: #333; }}
        .btn.primary {{ background: var(--accent-color); color: #000; border: none; font-weight: bold; }}
        .btn.primary:hover {{ opacity: 0.9; }}
    </style>
</head>
<body>
    <h1>System Settings</h1>
    
    <div class="section">
        <h2>Appearance</h2>
        <div class="form-group">
            <label>Interface Theme</label>
            <select id="theme">
                <option value="dark">Continuon Dark (Default)</option>
                <option value="light" disabled>Light (Coming Soon)</option>
                <option value="cyber">Cyberpunk High Contrast</option>
            </select>
        </div>
    </div>
    
    <div class="section">
        <h2>Brain Configuration</h2>
        <div class="form-group">
            <label>Model Endpoint (OpenAI Compatible)</label>
            <input type="text" id="model_endpoint" value="http://localhost:8000/v1" placeholder="http://localhost:8000/v1">
        </div>
        <div class="form-group">
            <label>
                <input type="checkbox" id="local_learning" checked> Enable Local Learning (HOPE/CMS)
            </label>
        </div>
        <div class="form-group">
            <label>
                <input type="checkbox" id="auto_load" checked> Auto-load last functionality on boot
            </label>
        </div>
    </div>
    
    <div class="section">
        <h2>Hardware Defaults</h2>
        <div class="form-group">
            <label>Default Motion Mode</label>
            <select id="default_mode">
                <option value="idle">Idle / Safe</option>
                <option value="manual_training">Manual Training</option>
                <option value="autonomous">Autonomous</option>
            </select>
        </div>
        <div class="form-group">
            <label>
                <input type="checkbox" id="prefer_real_hardware" checked disabled> Prefer Real Hardware
            </label>
        </div>
        <div class="form-group">
            <button class="btn primary" onclick="scanHardware()">Scan Hardware 🔍</button>
        </div>
    </div>
    
    <div style="margin-top: 40px;">
        <button class="btn primary" onclick="saveSettings()">Save Changes</button>
        <button class="btn" onclick="window.location.reload()" style="margin-left: 10px;">Reset to Defaults</button>
    </div>

    <script>
        async function saveSettings() {{
            const settings = {{
                theme: document.getElementById('theme').value,
                model_endpoint: document.getElementById('model_endpoint').value,
                local_learning: document.getElementById('local_learning').checked,
                auto_load: document.getElementById('auto_load').checked,
                default_mode: document.getElementById('default_mode').value
            }};
            
            try {{
                const res = await fetch('/api/settings', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(settings)
                }});
                
                if (res.ok) {{
                    alert('Settings saved successfully!');
                }} else {{
                    alert('Failed to save settings.');
                }}
            }} catch(e) {{
                alert('Error saving settings: ' + e.message);
            }}
        }}

        async function scanHardware() {{
            const btn = event.target;
            const originalText = btn.innerText;
            btn.innerText = "Scanning...";
            btn.disabled = true;
            
            try {{
                const res = await fetch('/api/hardware/scan', {{ method: 'POST' }});
                const data = await res.json();
                // Format results for display
                let msg = `Found ${{data.device_count}} devices:\\n`;
                if(data.devices.camera) msg += "✅ Camera\\n";
                if(data.devices.arm) msg += "✅ Arm Controller\\n";
                if(data.devices.drivetrain) msg += "✅ Drivetrain\\n";
                if(data.message) msg += `\\n(${{data.message}})`;
                
                alert(msg);
            }} catch(e) {{
                alert('Scan failed: ' + e);
            }} finally {{
                btn.innerText = originalText;
                btn.disabled = false;
            }}
        }}

        // Load existing settings (mock for now if backend doesn't support reading yet)
        // In real impl, we'd fetch from /api/settings/get
    </script>
</body>
</html>
"""

# Brain Status Page (Refined with new CSS)
STATUS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Brain Status</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; margin-bottom: 30px; }}
        pre {{ background: #111; padding: 20px; border-radius: 8px; border: 1px solid var(--border-color); overflow-x: auto; color: #aaa; }}
        
        .card {{ background: var(--card-bg); border-radius: 8px; padding: 25px; margin-bottom: 20px; border: 1px solid var(--border-color); }}
        h2 {{ margin-top: 0; color: var(--accent-color); font-size: 1.1em; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
    </style>
</head>
<body>
    <h1>Brain Introspection</h1>
    
    <div class="grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
        <div class="card">
            <h2>🏠 Identity (Shell)</h2>
            <div id="identity-report">Loading...</div>
        </div>
        
        <div class="card">
            <h2>🧠 Memory & Learning</h2>
            <div id="memory-stats">Loading...</div>
        </div>
        
        <div class="card">
            <h2>🦾 Body & Hardware</h2>
            <div id="body-stats">Loading...</div>
        </div>
        
        <div class="card">
            <h2>📜 Design Directives</h2>
            <div id="design-stats">Loading...</div>
        </div>
    </div>

    <script>
        async function loadStatus() {{
            try {{
                const res = await fetch('/api/status/introspection');
                const data = await res.json();
                
                document.getElementById('identity-report').innerHTML = `<pre>${{JSON.stringify(data.shell, null, 2)}}</pre>`;
                document.getElementById('memory-stats').innerHTML = `<pre>${{JSON.stringify(data.memory, null, 2)}}</pre>`;
                document.getElementById('body-stats').innerHTML = `<pre>${{JSON.stringify(data.body, null, 2)}}</pre>`;
                document.getElementById('design-stats').innerHTML = `<pre>${{JSON.stringify(data.design, null, 2)}}</pre>`;
            }} catch (e) {{
                document.getElementById('identity-report').textContent = "Error loading status";
            }}
        }}
        loadStatus();
        setInterval(loadStatus, 5000);
    </script>
</body>
</html>
"""

# Manual Control Interface
MANUAL_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Manual Control</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 0; overflow: hidden; background: #000; }}
        
        #video-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; display: flex; align-items: center; justify-content: center; }}
        #video-feed {{ width: 100%; height: 100%; object-fit: cover; opacity: 0.7; }}
        
        #controls-layer {{ position: absolute; top: 0; right: 0; width: 320px; height: 100%; background: rgba(16, 22, 38, 0.9); backdrop-filter: blur(10px); z-index: 10; padding: 20px; border-left: 1px solid var(--border-color); display: flex; flex-direction: column; overflow-y: auto; }}
        
        h2 {{ color: #fff; margin-top: 0; font-size: 1.2em; border-bottom: 1px solid #333; padding-bottom: 15px; }}
        
        .control-group {{ margin-bottom: 30px; }}
        .control-group h4 {{ color: var(--accent-color); margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
        
        .slider-row {{ margin-bottom: 15px; }}
        .slider-row label {{ display: flex; justify-content: space-between; font-size: 0.85em; color: #ccc; margin-bottom: 5px; }}
        input[type="range"] {{ width: 100%; accent-color: var(--accent-color); cursor: pointer; }}
        
        .dpad {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; max-width: 180px; margin: 0 auto; }}
        .dpad button {{ aspect-ratio: 1; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff; font-size: 1.2em; cursor: pointer; transition: all 0.1s; }}
        .dpad button:active {{ background: var(--accent-color); color: #000; transform: scale(0.95); }}
        
        .speed-selector {{ display: flex; gap: 5px; margin-top: 15px; }}
        .speed-selector button {{ flex: 1; padding: 8px; border-radius: 6px; border: 1px solid #444; background: #222; color: #aaa; cursor: pointer; font-size: 0.8em; }}
        .speed-selector button.active {{ background: var(--accent-color); color: #000; border-color: var(--accent-color); font-weight: bold; }}
        
        .emergency-stop {{ margin-top: auto; width: 100%; padding: 15px; background: #ff3b30; color: #fff; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; transition: background 0.2s; }}
        .emergency-stop:hover {{ background: #ff5b50; box-shadow: 0 0 15px rgba(255, 59, 48, 0.4); }}
    </style>
</head>
<body>
    <div id="video-layer">
        <img id="video-feed" src="/api/camera/stream" onerror="this.src='data:image/svg+xml;base64,...'">
        <div style="position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.6); padding: 5px 10px; border-radius: 20px; color: #fff; font-size: 0.8em; display: flex; align-items: center; gap: 6px;">
            <span style="width: 8px; height: 8px; background: #00ff88; border-radius: 50%;"></span> LIVE FEED
        </div>
    </div>
    
    <div id="controls-layer">
        <h2>Manual Control</h2>
        
        <div class="control-group">
            <h4>🤖 Arm Manipulator</h4>
            
            <div class="slider-row">
                <label>Base <span id="val-j0">0.0</span></label>
                <input type="range" min="-100" max="100" value="0" oninput="moveJoint(0, this.value)">
            </div>
            <div class="slider-row">
                <label>Shoulder <span id="val-j1">0.0</span></label>
                <input type="range" min="-100" max="100" value="0" oninput="moveJoint(1, this.value)">
            </div>
            <div class="slider-row">
                <label>Elbow <span id="val-j2">0.0</span></label>
                <input type="range" min="-100" max="100" value="0" oninput="moveJoint(2, this.value)">
            </div>
            <div class="slider-row">
                <label>Wrist Pitch <span id="val-j3">0.0</span></label>
                <input type="range" min="-100" max="100" value="0" oninput="moveJoint(3, this.value)">
            </div>
             <div class="slider-row">
                <label>Wrist Roll <span id="val-j4">0.0</span></label>
                <input type="range" min="-100" max="100" value="0" oninput="moveJoint(4, this.value)">
            </div>
            
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <button onclick="moveJoint(5, -100)" style="flex: 1; padding: 10px; background: #333; color: #fff; border: 1px solid #444; border-radius: 6px; cursor: pointer;">Open ✋</button>
                <button onclick="moveJoint(5, 100)" style="flex: 1; padding: 10px; background: #333; color: #fff; border: 1px solid #444; border-radius: 6px; cursor: pointer;">Close ✊</button>
            </div>
        </div>
        
        <div class="control-group">
            <h4>🏎️ Drivetrain</h4>
            <div class="dpad">
                <div></div>
                <button onmousedown="drive('forward')" onmouseup="drive('stop')">⬆️</button>
                <div></div>
                
                <button onmousedown="drive('left')" onmouseup="drive('stop')">⬅️</button>
                <button style="background: #222; cursor: default;">🎯</button>
                <button onmousedown="drive('right')" onmouseup="drive('stop')">➡️</button>
                
                <div></div>
                <button onmousedown="drive('backward')" onmouseup="drive('stop')">⬇️</button>
                <div></div>
            </div>
            
            <div class="speed-selector">
                <button onclick="setSpeed(0.3)" class="active">Slow</button>
                <button onclick="setSpeed(0.6)">Med</button>
                <button onclick="setSpeed(1.0)">Fast</button>
            </div>
        </div>
        
        <button class="emergency-stop" onclick="emergencyStop()">Emergency Stop</button>
    </div>

    <script>
        let currentSpeed = 0.3;
        
        function updateVal(id, val) {{
            document.getElementById('val-'+id).innerText = (val/100).toFixed(2);
        }}
        
        async function moveJoint(idx, val) {{
            if (idx < 5) updateVal('j'+idx, val);
            
            const normalized = val / 100.0;
            // Create 6-float array
            // TODO: Get real current state first to avoid jumping other joints
            // For now we send a specific command payload that backend handles,
            // or we assume we send full state.
            // Let's send a specific endpoint for single joint or full array?
            // The backend SendCommand expects full array. 
            // We should ideally track state locally or fetch it.
            
            // Simplified: Send single joint update object if backend supported it, 
            // but for now let's try to mock the full array structure
            // In a real app we'd fetch /api/robot/status first.
            
            await fetch('/api/robot/joints', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ joint_index: idx, value: normalized }})
            }});
        }}
        
        async function drive(direction) {{
            let steering = 0.0;
            let throttle = 0.0;
            
            if (direction === 'forward') throttle = currentSpeed;
            if (direction === 'backward') throttle = -currentSpeed;
            if (direction === 'left') steering = -1.0;
            if (direction === 'right') steering = 1.0;
            
            await fetch('/api/robot/drive', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ steering, throttle }})
            }});
        }}
        
        function setSpeed(s) {{
            currentSpeed = s;
            document.querySelectorAll('.speed-selector button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
        }}
        
        async function emergencyStop() {{
            await fetch('/api/robot/drive', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ steering: 0, throttle: 0, emergency: true }})
            }});
            alert('EMERGENCY STOP TRIGGERED');
        }}
        
        // Keyboard Support
        document.addEventListener('keydown', (e) => {{
            if(e.repeat) return;
            if(e.key === 'ArrowUp') drive('forward');
            if(e.key === 'ArrowDown') drive('backward');
            if(e.key === 'ArrowLeft') drive('left');
            if(e.key === 'ArrowRight') drive('right');
        }});
        
        document.addEventListener('keyup', (e) => {{
            if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)) drive('stop');
        }});
    </script>
</body>
</html>
"""

def get_home_html() -> str:
    return HOME_HTML

def get_status_html() -> str:
    return STATUS_HTML
    
def get_dashboard_html() -> str:
    return DASHBOARD_HTML

def get_chat_html() -> str:
    return CHAT_HTML
    
def get_settings_html() -> str:
    return SETTINGS_HTML

def get_manual_html() -> str:
    return MANUAL_HTML
