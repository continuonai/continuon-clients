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
        --sidebar-width: 260px;
        --chat-width: 400px;
    }
    
    body { font-family: var(--font-main); background: var(--bg-color); color: var(--text-color); margin: 0; display: flex; height: 100vh; overflow: hidden; }
    
    /* Layout - IDE style with left nav, main content, right chat */
    #sidebar { width: var(--sidebar-width); min-width: var(--sidebar-width); background: var(--sidebar-bg); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; padding-top: 20px; box-shadow: 2px 0 10px rgba(0,0,0,0.5); z-index: 10; transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), min-width 0.3s cubic-bezier(0.4, 0, 0.2, 1); overflow: hidden; }
    #sidebar.collapsed { width: 0; min-width: 0; border-right: none; }
    #sidebar .logo { padding: 0 24px 24px; font-weight: 800; font-size: 1.4em; letter-spacing: -0.5px; color: #fff; display: flex; align-items: center; gap: 10px; white-space: nowrap; }
    #sidebar .logo span { color: var(--accent-color); }
    
    #main-area { flex: 1; display: flex; flex-direction: row; overflow: hidden; }
    #content { flex: 1; position: relative; display: flex; flex-direction: column; overflow: hidden; }
    #chat-sidebar { width: var(--chat-width); min-width: var(--chat-width); background: var(--sidebar-bg); border-left: 1px solid var(--border-color); display: flex; flex-direction: column; transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), min-width 0.3s cubic-bezier(0.4, 0, 0.2, 1); overflow: hidden; }
    #chat-sidebar.collapsed { width: 0; min-width: 0; border-left: none; }
    
    iframe { flex: 1; border: none; width: 100%; height: 100%; }
    
    /* Toggle Buttons */
    .panel-toggle { position: absolute; top: 10px; z-index: 100; background: var(--card-bg); border: 1px solid var(--border-color); color: var(--text-color); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s ease; font-size: 1.2em; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
    .panel-toggle:hover { background: var(--accent-dim); border-color: var(--accent-color); color: var(--accent-color); transform: scale(1.05); }
    .panel-toggle:active { transform: scale(0.95); }
    #toggle-left { left: 10px; }
    #toggle-right { right: 10px; }
    
    /* Navigation */
    #sidebar a { padding: 14px 24px; color: var(--text-dim); text-decoration: none; display: flex; align-items: center; gap: 12px; font-size: 0.95em; transition: all 0.2s ease; border-left: 3px solid transparent; white-space: nowrap; }
    #sidebar a:hover { color: #fff; background: rgba(255,255,255,0.03); }
    #sidebar a.active { background: var(--accent-dim); color: var(--accent-color); border-left-color: var(--accent-color); font-weight: 500; }
    
    /* Icons */
    .icon { width: 20px; text-align: center; }
    
    /* Chat Sidebar */
    .chat-header { padding: 16px 20px; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 1.1em; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
    .chat-messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; min-height: 0; }
    .chat-message { display: flex; flex-direction: column; gap: 8px; }
    .chat-message.user { align-items: flex-end; }
    .chat-message.agent { align-items: flex-start; }
    .chat-message .sender { font-size: 0.85em; color: var(--text-dim); font-weight: 600; }
    .chat-message .bubble { max-width: 85%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; }
    .chat-message.user .bubble { background: var(--accent-color); color: #000; font-weight: 500; }
    .chat-message.agent .bubble { background: var(--card-bg); color: var(--text-color); border: 1px solid var(--border-color); }
    .chat-input-area { padding: 16px; border-top: 1px solid var(--border-color); display: flex; gap: 10px; flex-shrink: 0; z-index: 10; background: var(--sidebar-bg); }
    .chat-input-area textarea { flex: 1; background: var(--card-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 12px; border-radius: 8px; font-family: var(--font-main); font-size: 0.95em; resize: none; min-height: 60px; }
    .chat-input-area textarea:focus { outline: none; border-color: var(--accent-color); }
    .chat-input-area button { padding: 12px 24px; background: var(--accent-color); color: #000; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
    .chat-input-area button:hover { background: #00dd77; }
    .chat-input-area button:disabled { background: #333; color: #666; cursor: not-allowed; }
    
    /* System Status Footer in Sidebar */
    .sidebar-footer { margin-top: auto; padding: 20px; font-size: 0.8em; color: var(--text-dim); border-top: 1px solid var(--border-color); }
    .status-dot { width: 8px; height: 8px; background: var(--accent-color); border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 8px var(--accent-color); }

    /* Dropdown Menu */
    .dropdown { position: relative; display: inline-block; }
    .dropdown-content { display: none; position: absolute; right: 0; top: 100%; background-color: var(--card-bg); min-width: 200px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); z-index: 100; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; margin-top: 8px; }
    .dropdown-content a { color: var(--text-color); padding: 12px 16px; text-decoration: none; display: block; font-size: 0.9em; cursor: pointer; transition: background 0.2s; border-left: 3px solid transparent; }
    .dropdown-content a:hover { background-color: #333; border-left-color: var(--accent-color); }
    .shown { display: block; }
    
    /* Modal */
    .modal { display: none; position: fixed; z-index: 200; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.7); backdrop-filter: blur(2px); }
    .modal-content { background-color: var(--card-bg); margin: 10% auto; padding: 0; border: 1px solid var(--border-color); width: 80%; max-width: 600px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .modal-header { padding: 20px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; }
    .modal-header h2 { margin: 0; font-size: 1.2em; color: var(--accent-color); }
    .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; transition: color 0.2s; }
    .close:hover, .close:focus { color: #fff; text-decoration: none; cursor: pointer; }
    .modal-body { padding: 20px; }
    .info-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #333; }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: var(--text-dim); font-weight: 500; }
    .info-value { color: #fff; font-family: var(--font-mono); font-size: 0.9em; }
    .modal-footer { padding: 15px 20px; border-top: 1px solid var(--border-color); display: flex; justify-content: flex-end; gap: 10px; }
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
        let chatHistory = [];
        
        // Panel toggle functions
        function toggleLeftPanel() {{
            const sidebar = document.getElementById('sidebar');
            const btn = document.getElementById('toggle-left');
            sidebar.classList.toggle('collapsed');
            const isCollapsed = sidebar.classList.contains('collapsed');
            btn.innerHTML = isCollapsed ? '☰' : '◀';
            localStorage.setItem('leftPanelCollapsed', isCollapsed);
        }}
        
        function toggleRightPanel() {{
            const chatSidebar = document.getElementById('chat-sidebar');
            const btn = document.getElementById('toggle-right');
            chatSidebar.classList.toggle('collapsed');
            const isCollapsed = chatSidebar.classList.contains('collapsed');
            btn.innerHTML = isCollapsed ? '💬' : '▶';
            localStorage.setItem('rightPanelCollapsed', isCollapsed);
        }}
        
        // Restore panel states from localStorage
        function restorePanelStates() {{
            const leftCollapsed = localStorage.getItem('leftPanelCollapsed') === 'true';
            const rightCollapsed = localStorage.getItem('rightPanelCollapsed') === 'true';
            
            if (leftCollapsed) {{
                document.getElementById('sidebar').classList.add('collapsed');
                document.getElementById('toggle-left').innerHTML = '☰';
            }}
            
            if (rightCollapsed) {{
                document.getElementById('chat-sidebar').classList.add('collapsed');
                document.getElementById('toggle-right').innerHTML = '💬';
            }}
        }}
        
        function selectNav(el) {{
            document.querySelectorAll('#sidebar a').forEach(a => a.classList.remove('active'));
            el.classList.add('active');
        }}
        
        async function sendMessage() {{
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('user', 'You', message);
            input.value = '';
            input.disabled = true;
            document.getElementById('sendBtn').disabled = true;
            
            // Show agent status
            showAgentStatus('thinking', 'Thinking...');
            
            try {{
                const response = await fetch('/api/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ message, history: chatHistory }})
                }});
                const data = await response.json();
                
                // Hide agent status
                hideAgentStatus();
                
                // Check for intervention needed
                if (data.intervention_needed) {{
                    showInterventionPrompt(data.intervention_question, data.intervention_options);
                }}
                
                // Check for status updates
                if (data.status_updates && data.status_updates.length > 0) {{
                    for (const update of data.status_updates) {{
                        addStatusUpdate(update);
                    }}
                }}
                
                addMessage('agent', 'Agent Manager', data.response || 'No response');
            }} catch(e) {{
                hideAgentStatus();
                addMessage('agent', 'Agent Manager', 'Error: ' + e.message);
            }}
            
            input.disabled = false;
            document.getElementById('sendBtn').disabled = false;
            input.focus();
        }}
        
        function showAgentStatus(state, text) {{
            const statusBar = document.getElementById('agent-status-bar');
            const statusText = document.getElementById('agent-status-text');
            const progress = document.getElementById('agent-progress');
            
            statusBar.style.display = 'block';
            statusText.textContent = '● ' + text;
            
            if (state === 'thinking') {{
                progress.style.display = 'block';
                animateProgress();
            }}
        }}
        
        function hideAgentStatus() {{
            const statusBar = document.getElementById('agent-status-bar');
            const progress = document.getElementById('agent-progress');
            statusBar.style.display = 'none';
            progress.style.display = 'none';
        }}
        
        function animateProgress() {{
            const bar = document.getElementById('progress-bar');
            let width = 0;
            const interval = setInterval(() => {{
                if (width >= 90) {{
                    clearInterval(interval);
                }} else {{
                    width += Math.random() * 10;
                    bar.style.width = Math.min(width, 90) + '%';
                }}
            }}, 200);
        }}
        
        function showInterventionPrompt(question, options) {{
            const prompt = document.getElementById('intervention-prompt');
            const questionEl = document.getElementById('intervention-question');
            const optionsEl = document.getElementById('intervention-options');
            
            questionEl.textContent = question;
            optionsEl.innerHTML = '';
            
            options.forEach(option => {{
                const btn = document.createElement('button');
                btn.textContent = option;
                btn.style.cssText = 'padding: 8px 16px; background: var(--card-bg); border: 1px solid var(--border-color); color: var(--text-color); border-radius: 6px; cursor: pointer; font-size: 0.9em;';
                btn.onmouseover = () => btn.style.background = '#333';
                btn.onmouseout = () => btn.style.background = 'var(--card-bg)';
                btn.onclick = () => {{
                    prompt.style.display = 'none';
                    document.getElementById('chatInput').value = option;
                    sendMessage();
                }};
                optionsEl.appendChild(btn);
            }});
            
            prompt.style.display = 'block';
        }}
        
        function addStatusUpdate(update) {{
            const messagesDiv = document.getElementById('chatMessages');
            const updateDiv = document.createElement('div');
            updateDiv.style.cssText = 'padding: 8px 12px; margin: 8px 0; background: rgba(0,255,136,0.05); border-left: 3px solid var(--accent-color); border-radius: 4px; font-size: 0.85em; color: var(--text-dim);';
            updateDiv.innerHTML = `<span style=\"color: var(--accent-color);\">●</span> ${{escapeHtml(update)}}`;
            messagesDiv.appendChild(updateDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }}
        
        function addMessage(type, sender, text) {{
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${{type}}`;
            messageDiv.innerHTML = `
                <div class="sender">${{sender}}</div>
                <div class="bubble">${{escapeHtml(text)}}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            chatHistory.push({{ role: type === 'user' ? 'user' : 'assistant', content: text }});
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML.replace(/\\n/g, '<br>');
        }}
        
        // Handle Enter key and restore panel states on load
        document.addEventListener('DOMContentLoaded', () => {{
            const input = document.getElementById('chatInput');
            input.addEventListener('keydown', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    sendMessage();
                }}
            }});
            
            // Restore panel states
            restorePanelStates();
            
            // Fetch Agent Info
            fetchAgentInfo();
            
            // Close menu when clicking outside
            window.onclick = function(event) {{
                if (!event.target.matches('.panel-toggle') && !event.target.matches('#menu-btn')) {{
                    var dropdowns = document.getElementsByClassName("dropdown-content");
                    for (var i = 0; i < dropdowns.length; i++) {{
                        var openDropdown = dropdowns[i];
                        if (openDropdown.classList.contains('shown')) {{
                            openDropdown.classList.remove('shown');
                        }}
                    }}
                }}
                
                // Close modal
                if (event.target == document.getElementById('details-modal')) {{
                    closeModal();
                }}
            }}
        }});
        
        async function fetchAgentInfo() {{
            try {{
                const res = await fetch('/api/agent/info');
                const data = await res.json();
                
                // Update Header Title
                if (data.chat_agent && data.chat_agent.model_name) {{
                    const shortName = data.chat_agent.model_name.split('/').pop();
                    document.getElementById('agent-title').innerHTML = `Agent Manager <span style="font-weight: 400; font-size: 0.8em; opacity: 0.7;">(${{shortName}})</span>`;
                }}
                
                // Store data for modal
                window.agentInfo = data;
                
                // Update Learning Toggle Text
                const learningToggle = document.getElementById('learning-toggle');
                if (data.learning_agent && data.learning_agent.enabled) {{
                    const isRunning = data.learning_agent.running;
                    learningToggle.innerText = isRunning ? "⏸ Pause Learning" : "▶ Resume Learning";
                    learningToggle.onclick = isRunning ? pauseLearning : resumeLearning;
                }} else {{
                    learningToggle.innerText = "🚫 Learning Unavailable";
                    learningToggle.style.opacity = "0.5";
                    learningToggle.onclick = null;
                }}
                
            }} catch(e) {{ console.error("Failed to fetch agent info", e); }}
        }}
        
        function toggleMenu() {{
            document.getElementById("dropdown-menu").classList.toggle("shown");
        }}
        
        function showAgentDetails() {{
            const modal = document.getElementById('details-modal');
            modal.style.display = "block";
            toggleMenu(); // Close menu
            
            const data = window.agentInfo;
            if (!data) return;
            
            // Populate Chat Info
            const chatDiv = document.getElementById('chat-agent-info');
            chatDiv.innerHTML = `
                <div class="info-row"><span class="info-label">Model Name</span><span class="info-value">${{data.chat_agent.model_name || 'Unknown'}}</span></div>
                <div class="info-row"><span class="info-label">Device</span><span class="info-value">${{data.chat_agent.device || 'Unknown'}}</span></div>
                <div class="info-row"><span class="info-label">Context Length</span><span class="info-value">${{(data.chat_agent.history_length || 0) * 2}} turns</span></div>
                <div class="info-row"><span class="info-label">HF Token</span><span class="info-value">${{data.chat_agent.has_token ? '✅ Present' : '❌ Missing'}}</span></div>
            `;
            
            // Populate Learning Info
            const learnDiv = document.getElementById('learning-agent-info');
            if (data.learning_agent.enabled) {{
                 const l = data.learning_agent;
                 let stats = '';
                 if (l.total_steps) stats += `<div class="info-row"><span class="info-label">Total Steps</span><span class="info-value">${{l.total_steps}}</span></div>`;
                 if (l.total_episodes) stats += `<div class="info-row"><span class="info-label">Episodes</span><span class="info-value">${{l.total_episodes}}</span></div>`;
                 if (l.curiosity) stats += `<div class="info-row"><span class="info-label">Curiosity</span><span class="info-value">${{(l.curiosity * 100).toFixed(1)}}%</span></div>`;
                 
                 learnDiv.innerHTML = `
                    <div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: ${{l.running ? '#00ff88' : '#ffaa00'}}">${{l.running ? 'Running' : 'Paused'}}</span></div>
                    ${{stats}}
                 `;
            }} else {{
                learnDiv.innerHTML = '<div style="color: var(--text-dim); padding: 10px;">Autonomous learning is disabled or module not found.</div>';
            }}
        }}
        
        function closeModal() {{
            document.getElementById('details-modal').style.display = "none";
        }}
        
        async function clearHistory() {{
            if(!confirm("Are you sure you want to clear the chat history?")) return;
            try {{
                await fetch('/api/chat/history/clear', {{ method: 'POST' }});
                document.getElementById('chatMessages').innerHTML = `
                    <div class="chat-message agent">
                        <div class="sender">Agent Manager</div>
                        <div class="bubble">Memory cleared. Ready for new task.</div>
                    </div>
                `;
                chatHistory = [];
                fetchAgentInfo(); // Refresh context length
            }} catch(e) {{ alert("Failed to clear history"); }}
            toggleMenu();
        }}
        
        async function pauseLearning() {{
            await fetch('/api/learning/pause', {{ method: 'POST' }});
            fetchAgentInfo(); // Refresh status
            toggleMenu();
        }}
        
        async function resumeLearning() {{
            await fetch('/api/learning/resume', {{ method: 'POST' }});
            fetchAgentInfo(); // Refresh status
            toggleMenu();
        }}

        async function saveMemory() {{
            if(!confirm("Save current session as RLDS episode?")) return;
            try {{
                const res = await fetch('/api/memory/save', {{ method: 'POST' }});
                const data = await res.json();
                alert(data.message || "Memory saved!");
            }} catch(e) {{ alert("Failed to save memory: " + e); }}
            toggleMenu();
        }}

    </script>
    <script>
        // --- Web Audio API for Natural UX ---
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        function playTone(freq, type, duration) {{
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            gain.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + duration);
            osc.stop(audioCtx.currentTime + duration);
        }}

        function playStartListening() {{
            if (audioCtx.state === 'suspended') audioCtx.resume();
            playTone(880, 'sine', 0.1); // High beep
             setTimeout(() => playTone(1760, 'sine', 0.1), 150);
        }}

        function playStopListening() {{
            if (audioCtx.state === 'suspended') audioCtx.resume();
            playTone(1760, 'sine', 0.1); 
            setTimeout(() => playTone(880, 'sine', 0.1), 150);
        }}

        let isListening = false;
        function toggleVoice() {{
            const btn = document.getElementById('micBtn');
            const placeholder = document.getElementById('chatInput');
            
            isListening = !isListening;
            
            if (isListening) {{
                btn.classList.add('listening');
                playStartListening();
                placeholder.placeholder = "Listening...";
                // TODO: Wire up actual STT logic here
            }} else {{
                btn.classList.remove('listening');
                playStopListening();
                placeholder.placeholder = "Ask about tasks, control the robot, or chat with sub-agents...";
            }}
        }}
    </script>
</head>
<body>
    <div id="sidebar">
        <div class="logo">
            <span style="font-size: 1.2em;">🧠</span> Continuon<span>Brain</span>
        </div>
        
        <div style="flex: 1;">
            <a href="/ui/dashboard" target="main_frame" class="active" onclick="selectNav(this)">
                <span class="icon">📊</span> Dashboard
            </a>
            <a href="/ui/tasks" target="main_frame" onclick="selectNav(this)">
                <span class="icon">📋</span> Task Library
            </a>
            <a href="/ui/manual" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🎮</span> Manual Control
            </a>
            <a href="/ui/status" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🔍</span> Brain Status
            </a>
            
            <!-- HOPE Monitoring Section -->
            <div style="margin: 20px 0; padding: 10px 24px; font-size: 0.75em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px;">
                HOPE Brain Development
            </div>
            <a href="/ui/hope/training" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🧠</span> Training Dashboard
            </a>
            <a href="/ui/hope/memory" target="main_frame" onclick="selectNav(this)">
                <span class="icon">💾</span> CMS Memory
            </a>
            <a href="/ui/hope/stability" target="main_frame" onclick="selectNav(this)">
                <span class="icon">⚖️</span> Stability Monitor
            </a>
            <a href="/ui/hope/dynamics" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🌊</span> Wave-Particle
            </a>
            <a href="/ui/hope/performance" target="main_frame" onclick="selectNav(this)">
                <span class="icon">⚡</span> Performance
            </a>
            
            <div style="margin: 20px 0; padding: 10px 24px; font-size: 0.75em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px;">
                System
            </div>
            <a href="/ui/settings" target="main_frame" onclick="selectNav(this)">
                <span class="icon">⚙️</span> Settings
            </a>
        </div>
        
        <div class="sidebar-footer">
            <div><span class="status-dot"></span> System Online</div>
            <div style="margin-top: 8px; opacity: 0.6;">v2.0.0-alpha</div>
        </div>
    </div>
    
    <div id="main-area">
        <div id="content">
            <!-- Panel Toggle Buttons -->
            <button id="toggle-left" class="panel-toggle" onclick="toggleLeftPanel()" title="Toggle Navigation Panel">◀</button>
            <button id="toggle-right" class="panel-toggle" onclick="toggleRightPanel()" title="Toggle Chat Panel">▶</button>
            <iframe name="main_frame" src="/ui/dashboard"></iframe>
        </div>
        
        <div id="chat-sidebar">
            <div class="chat-header">
                <span>🤖</span> <span id="agent-title">Agent Manager</span>
                <div style="flex: 1;"></div>
                <!-- Menu Button -->
                <div class="dropdown">
                    <button id="menu-btn" class="panel-toggle" style="position: static; width: 32px; height: 32px; font-size: 1em;" onclick="toggleMenu()">⋮</button>
                    <div id="dropdown-menu" class="dropdown-content">
                        <a onclick="showAgentDetails()">📊 Agent Details</a>
                        <a onclick="toggleLearning()" id="learning-toggle">⏸ Pause Learning</a>
                        <a onclick="saveMemory()">💾 Save Memory (RLDS)</a>
                        <a onclick="clearHistory()" style="color: #ff5555;">🗑 Clear History</a>
                    </div>
                </div>
                
                <!-- Agent Status Bar -->
                <div id="agent-status-bar" style="margin-top: 8px; padding: 6px 10px; background: rgba(0,0,0,0.3); border-radius: 6px; font-size: 0.8em; display: none;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span id="agent-status-text" style="color: var(--accent-color);">● Ready</span>
                        <span id="sub-agents-count" style="color: var(--text-dim); font-size: 0.85em;"></span>
                    </div>
                    <div id="agent-progress" style="margin-top: 4px; height: 2px; background: rgba(0,255,136,0.2); border-radius: 1px; overflow: hidden; display: none;">
                        <div style="height: 100%; background: var(--accent-color); width: 0%; transition: width 0.3s;" id="progress-bar"></div>
                    </div>
                </div>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="chat-message agent">
                    <div class="sender">Agent Manager</div>
                    <div class="bubble">Hello! I'm the lead agent managing tasks and coordinating sub-agents. How can I help you today?</div>
                </div>
            </div>
            <!-- Intervention Prompt (hidden by default) -->
            <div id="intervention-prompt" style="display: none; padding: 16px; background: rgba(255,165,0,0.1); border-top: 2px solid #ffa500; border-bottom: 1px solid var(--border-color); flex-shrink: 0;">
                <div style="font-weight: 600; margin-bottom: 8px; color: #ffa500;">🤷 I need your help deciding</div>
                <div id="intervention-question" style="margin-bottom: 12px; font-size: 0.9em;"></div>
                <div id="intervention-options" style="display: flex; gap: 8px; flex-wrap: wrap;"></div>
            </div>
            <div class="chat-input-area">
                <button id="micBtn" onclick="toggleVoice()" title="Voice Control">🎙️</button>
                <textarea id="chatInput" placeholder="Ask about tasks, control the robot, or chat with sub-agents..."></textarea>
                <button id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
            
            <!-- Agent Details Modal -->
            <div id="details-modal" class="modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2>Agent System Status</h2>
                        <span class="close" onclick="closeModal()">&times;</span>
                    </div>
                    <div class="modal-body">
                        <h3 style="margin-top: 0; font-size: 0.9em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px;">Chat Agent (Active)</h3>
                        <div id="chat-agent-info">Loading...</div>
                        
                        <h3 style="margin-top: 20px; font-size: 0.9em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px;">Learning Sub-Agent (HOPE)</h3>
                        <div id="learning-agent-info">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
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
        <div class="card">
            <h3>System Memory</h3>
            <div class="value" id="sys-mem-percent">--%</div>
            <div class="meta" id="sys-mem-meta">Loading...</div>
            <div style="height: 4px; background: #333; margin-top: 10px; border-radius: 2px; overflow: hidden;">
                <div id="sys-mem-bar" style="height: 100%; width: 0%; background: var(--accent-color); transition: width 0.5s;"></div>
            </div>
        </div>
        <div class="card">
            <h3>Disk Usage</h3>
            <div class="value" id="disk-usage">-- MB</div>
            <div class="meta">Autonomous Checkpoints</div>
            <div class="meta">Autonomous Checkpoints</div>
        </div>
        
        <!-- Hybrid Mode Card -->
        <div class="card" id="hybrid-card">
            <h3>Hybrid Architecture</h3>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 15px;">
                <div>
                    <div class="value" id="hybrid-status" style="font-size: 1.8em;">Standard</div>
                    <div class="meta" id="hybrid-meta">1 Column</div>
                </div>
                <label class="switch" style="position: relative; display: inline-block; width: 60px; height: 34px;">
                    <input type="checkbox" id="hybrid-toggle" onclick="toggleHybridMode()">
                    <span class="slider round" style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #333; transition: .4s; border-radius: 34px;"></span>
                    <span class="slider-knob" style="position: absolute; content: ''; height: 26px; width: 26px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%;"></span>
                </label>
            </div>
            <style>
                input:checked + .slider {{ background-color: var(--accent-color); }}
                input:checked + .slider .slider-knob {{ transform: translateX(26px); }}
            </style>
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
        async function updateStats() {{
            try {{
                // Fetch Introspection Data (Identity)
                const res = await fetch('/api/status/introspection');
                const data = await res.json();
                
                // Update mode (safely)
                const modeEl = document.querySelector('.card:nth-of-type(1) .value');
                if (modeEl) modeEl.innerText = 'Active'; 
                
                if(data.memory) {{
                    const memEl = document.getElementById('memory-count');
                    if (memEl) memEl.innerText = data.memory.local_episodes;
                }}

                // Fetch Resource Data
                const resResources = await fetch('/api/resources');
                if (resResources.ok) {{
                    const rData = await resResources.json();
                    
                    // Update System Memory
                    const memPercentEl = document.getElementById('sys-mem-percent');
                    const memMetaEl = document.getElementById('sys-mem-meta');
                    const memBarEl = document.getElementById('sys-mem-bar');
                    if (memPercentEl) {{
                        memPercentEl.innerText = rData.memory_percent + '%';
                        memMetaEl.innerText = rData.available_mb + ' MB available';
                        memBarEl.style.width = rData.memory_percent + '%';
                        
                        // Color coding
                        if (rData.memory_percent > 90) memBarEl.style.background = '#ff4444';
                        else if (rData.memory_percent > 75) memBarEl.style.background = '#ffbb33';
                        else memBarEl.style.background = 'var(--accent-color)';
                    }}
                    
                    
                    // Update Disk Usage
                    const diskEl = document.getElementById('disk-usage');
                    if (rData.checkpoints_mb !== undefined) {{
                         document.getElementById('disk-usage').textContent = rData.checkpoints_mb.toFixed(0) + ' MB';
                    }}
                
                    // Update Hybrid Status
                    checkHybridStatus();

                }} catch(e) {{ console.error("Stats update failed", e); }}
            }}
            
            async function checkHybridStatus() {{
                // Placeholder
            }}

            async function toggleHybridMode() {{
                const toggle = document.getElementById('hybrid-toggle');
                const statusLabel = document.getElementById('hybrid-status');
                const metaLabel = document.getElementById('hybrid-meta');
                
                // Optimistic UI update
                const isEnabled = toggle.checked;
                statusLabel.textContent = isEnabled ? "Hybrid" : "Standard";
                metaLabel.textContent = isEnabled ? "4 Columns (Voting)" : "1 Column";
                
                try {{
                    const res = await fetch('/api/brain/toggle_hybrid', {{ method: 'POST' }});
                    const data = await res.json();
                    if (data.status !== 'success') {{
                        alert("Failed to toggle mode: " + data.message);
                        toggle.checked = !isEnabled; // Revert
                    }} else {{
                        console.log(data.message);
                    }}
                }} catch(e) {{
                    console.error("Toggle failed", e);
                    alert("Network error toggling mode.");
                    toggle.checked = !isEnabled; // Revert
                }}
            }}
            
            setInterval(updateStats, 2000);
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

        /* Chat Input Area - Natural UX */
        .chat-input-area {{ display: flex; padding: 15px; background-color: var(--card-bg); border-top: 1px solid var(--border-color); align-items: center; gap: 10px; }}
        textarea {{ flex: 1; height: 50px; background-color: rgba(255,255,255,0.05); border: 1px solid var(--border-color); color: #fff; padding: 12px; border-radius: 20px; resize: none; font-family: inherit; font-size: 1em; outline: none; transition: background 0.2s; }}
        textarea:focus {{ background-color: rgba(255,255,255,0.1); border-color: var(--accent-color); }}
        
        /* Mic Button */
        #micBtn {{ width: 50px; height: 50px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid var(--border-color); color: var(--text-dim); font-size: 1.5em; cursor: pointer; transition: all 0.3s ease; display: flex; justify-content: center; align-items: center; margin-right: 0; }}
        #micBtn:hover {{ background: rgba(255,255,255,0.1); color: #fff; transform: scale(1.05); }}
        
        /* Listening State Animation */
        .listening {{ background-color: rgba(255, 69, 58, 0.2) !important; color: #ff453a !important; border-color: #ff453a !important; animation: pulse 1.5s infinite; }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.4); transform: scale(1); }}
            70% {{ box-shadow: 0 0 0 15px rgba(255, 69, 58, 0); transform: scale(1.1); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 69, 58, 0); transform: scale(1); }}
        }}
        
        /* Hide Send Button (Secondary) */
        #sendBtn {{ display: none; }}
        
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
    
    <div class="chat-input-area">
        <button id="micBtn" onclick="toggleVoice()" title="Voice Control">🎙️</button>
        <textarea id="msg-input" placeholder="Message the Brain... (Cmd+Enter to send)"></textarea>
        <!-- Implicit send button for JS reference, hidden via CSS -->
        <button id="sendBtn">Send</button>
    </div>

    <script>
        const input = document.getElementById('msg-input');
        const sendBtn = document.getElementById('sendBtn');
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
        
        // --- Natural UX Audio/Voice Logic ---
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        function playTone(freq, type, duration) {{
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            gain.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + duration);
            osc.stop(audioCtx.currentTime + duration);
        }}

        function playStartListening() {{
            if (audioCtx.state === 'suspended') audioCtx.resume();
            playTone(880, 'sine', 0.1); 
             setTimeout(() => playTone(1760, 'sine', 0.1), 150);
        }}

        function playStopListening() {{
            if (audioCtx.state === 'suspended') audioCtx.resume();
            playTone(1760, 'sine', 0.1); 
            setTimeout(() => playTone(880, 'sine', 0.1), 150);
        }}

        let isListening = false;
        function toggleVoice() {{
            const btn = document.getElementById('micBtn');
            const placeholder = document.getElementById('msg-input');
            
            isListening = !isListening;
            
            if (isListening) {{
                btn.classList.add('listening');
                playStartListening();
                placeholder.placeholder = "Listening...";
            }} else {{
                btn.classList.remove('listening');
                playStopListening();
                placeholder.placeholder = "Message the Brain...";
            }}
        }}
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
        
        /* Chat Input Area - Natural UX */
    .chat-input-area {{ display: flex; padding: 15px; background-color: var(--card-bg); border-top: 1px solid var(--border-color); align-items: center; gap: 10px; }}
    textarea {{ flex: 1; height: 50px; background-color: rgba(255,255,255,0.05); border: 1px solid var(--border-color); color: #fff; padding: 12px; border-radius: 20px; resize: none; font-family: inherit; font-size: 1em; outline: none; transition: background 0.2s; }}
    textarea:focus {{ background-color: rgba(255,255,255,0.1); border-color: var(--accent-color); }}
    
    /* Mic Button */
    #micBtn {{ width: 50px; height: 50px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid var(--border-color); color: var(--text-dim); font-size: 1.5em; cursor: pointer; transition: all 0.3s ease; display: flex; justify-content: center; align-items: center; margin-right: 0; }}
    #micBtn:hover {{ background: rgba(255,255,255,0.1); color: #fff; transform: scale(1.05); }}
    
    /* Listening State Animation */
    .listening {{ background-color: rgba(255, 69, 58, 0.2) !important; color: #ff453a !important; border-color: #ff453a !important; animation: pulse 1.5s infinite; }}
    
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.4); transform: scale(1); }}
        70% {{ box-shadow: 0 0 0 15px rgba(255, 69, 58, 0); transform: scale(1.1); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 69, 58, 0); transform: scale(1); }}
    }}
    
    /* Hide Send Button (Secondary) */
    #sendBtn {{ display: none; }}
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
    
    <div class="section">
        <h2>Agent Manager</h2>
        <div class="form-group">
            <label>
                <input type="checkbox" id="enable_thinking_indicator" checked> Show Thinking Indicator
            </label>
            <small style="color: var(--text-dim); display: block; margin-left: 30px;">Display animated progress bar when agent is processing</small>
        </div>
        <div class="form-group">
            <label>
                <input type="checkbox" id="enable_intervention_prompts" checked> Enable Intervention Prompts
            </label>
            <small style="color: var(--text-dim); display: block; margin-left: 30px;">Ask for human help when agent is uncertain</small>
        </div>
        <div class="form-group">
            <label>Intervention Confidence Threshold</label>
            <input type="number" id="intervention_confidence_threshold" value="0.5" min="0" max="1" step="0.05" style="max-width: 150px;">
            <small style="color: var(--text-dim); display: block; margin-top: 4px;">Trigger intervention when confidence is below this value (0.0 - 1.0)</small>
        </div>
        <div class="form-group">
            <label>
                <input type="checkbox" id="enable_status_updates" checked> Show Status Updates
            </label>
            <small style="color: var(--text-dim); display: block; margin-left: 30px;">Display real-time activity feed in chat</small>
        </div>
        <div class="form-group">
            <label>
                <input type="checkbox" id="enable_autonomous_learning" checked> Enable Autonomous Learning
            </label>
            <small style="color: var(--text-dim); display: block; margin-left: 30px;">Run continuous curiosity-driven learning in background</small>
        </div>
        <div class="form-group">
            <label>Learning Steps Per Cycle</label>
            <input type="number" id="autonomous_learning_steps_per_cycle" value="100" min="10" max="1000" step="10" style="max-width: 150px;">
            <small style="color: var(--text-dim); display: block; margin-top: 4px;">Number of learning steps per cycle (10-1000)</small>
        </div>
        <div class="form-group">
            <label>Checkpoint Interval</label>
            <input type="number" id="autonomous_learning_checkpoint_interval" value="1000" min="100" max="10000" step="100" style="max-width: 150px;">
            <small style="color: var(--text-dim); display: block; margin-top: 4px;">Save brain checkpoint every N steps (100-10000)</small>
        </div>
    </div>
    
    <div style="margin-top: 40px;">
        <button class="btn primary" onclick="saveSettings()">Save Changes</button>
        <button class="btn" onclick="loadSettings()" style="margin-left: 10px;">Reset to Current</button>
    </div>

    <script>
        async function loadSettings() {{
            try {{
                const res = await fetch('/api/settings');
                const data = await response.json();
                
                if (data.success && data.settings) {{
                    const s = data.settings;
                    
                    // Agent Manager settings
                    if (s.agent_manager) {{
                        document.getElementById('enable_thinking_indicator').checked = s.agent_manager.enable_thinking_indicator ?? true;
                        document.getElementById('enable_intervention_prompts').checked = s.agent_manager.enable_intervention_prompts ?? true;
                        document.getElementById('intervention_confidence_threshold').value = s.agent_manager.intervention_confidence_threshold ?? 0.5;
                        document.getElementById('enable_status_updates').checked = s.agent_manager.enable_status_updates ?? true;
                        document.getElementById('enable_autonomous_learning').checked = s.agent_manager.enable_autonomous_learning ?? true;
                        document.getElementById('autonomous_learning_steps_per_cycle').value = s.agent_manager.autonomous_learning_steps_per_cycle ?? 100;
                        document.getElementById('autonomous_learning_checkpoint_interval').value = s.agent_manager.autonomous_learning_checkpoint_interval ?? 1000;
                    }}
                }}
            }} catch(e) {{
                console.error('Error loading settings:', e);
            }}
        }}
        
        async function saveSettings() {{
            const settings = {{
                theme: document.getElementById('theme').value,
                model_endpoint: document.getElementById('model_endpoint').value,
                local_learning: document.getElementById('local_learning').checked,
                auto_load: document.getElementById('auto_load').checked,
                default_mode: document.getElementById('default_mode').value,
                agent_manager: {{
                    enable_thinking_indicator: document.getElementById('enable_thinking_indicator').checked,
                    enable_intervention_prompts: document.getElementById('enable_intervention_prompts').checked,
                    intervention_confidence_threshold: parseFloat(document.getElementById('intervention_confidence_threshold').value),
                    enable_status_updates: document.getElementById('enable_status_updates').checked,
                    enable_autonomous_learning: document.getElementById('enable_autonomous_learning').checked,
                    autonomous_learning_steps_per_cycle: parseInt(document.getElementById('autonomous_learning_steps_per_cycle').value),
                    autonomous_learning_checkpoint_interval: parseInt(document.getElementById('autonomous_learning_checkpoint_interval').value)
                }}
            }};
            
            try {{
                const res = await fetch('/api/settings', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(settings)
                }});
                
                const data = await res.json();
                
                if (data.success) {{
                    alert('Settings saved successfully! Restart the server for changes to take effect.');
                }} else {{
                    alert('Failed to save settings: ' + (data.message || 'Unknown error'));
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

TASKS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Task Library</title>
    <style>
        {COMMON_CSS}
        
        body {{ margin: 0; padding: 24px; background: var(--bg-color); color: var(--text-color); font-family: var(--font-main); overflow-y: auto; }}
        
        h1 {{ font-size: 2em; margin: 0 0 8px; font-weight: 800; }}
        .subtitle {{ color: var(--text-dim); margin-bottom: 32px; }}
        
        .filter-bar {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
        .filter-bar button {{ padding: 8px 16px; background: var(--card-bg); color: var(--text-color); border: 1px solid var(--border-color); border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
        .filter-bar button:hover {{ background: var(--accent-dim); border-color: var(--accent-color); }}
        .filter-bar button.active {{ background: var(--accent-color); color: #000; font-weight: 600; }}
        
        .task-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }}
        
        .task-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; transition: all 0.3s ease; cursor: pointer; position: relative; }}
        .task-card:hover {{ border-color: var(--accent-color); box-shadow: 0 4px 20px rgba(0, 255, 136, 0.15); transform: translateY(-2px); }}
        
        .task-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }}
        .task-icon {{ font-size: 2em; }}
        .task-status {{ padding: 4px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }}
        .status-ready {{ background: rgba(0, 255, 136, 0.2); color: var(--accent-color); }}
        .status-blocked {{ background: rgba(255, 68, 68, 0.2); color: #ff4444; }}
        .status-partial {{ background: rgba(255, 187, 0, 0.2); color: #ffbb00; }}
        
        .task-title {{ font-size: 1.1em; font-weight: 600; margin-bottom: 8px; }}
        .task-description {{ color: var(--text-dim); font-size: 0.9em; line-height: 1.5; margin-bottom: 16px; }}
        
        .task-meta {{ display: flex; gap: 16px; font-size: 0.85em; color: var(--text-dim); }}
        .task-meta span {{ display: flex; align-items: center; gap: 6px; }}
        
        .task-tags {{ display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }}
        .task-tag {{ padding: 4px 10px; background: rgba(255,255,255,0.05); border-radius: 4px; font-size: 0.75em; color: var(--text-dim); }}
        
        .task-details {{ margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color); display: none; }}
        .task-card.expanded .task-details {{ display: block; }}
        
        .task-steps {{ list-style: none; padding: 0; margin: 12px 0; }}
        .task-steps li {{ padding: 8px 0; padding-left: 24px; position: relative; }}
        .task-steps li:before {{ content: "→"; position: absolute; left: 0; color: var(--accent-color); }}
        
        .execute-btn {{ width: 100%; padding: 12px; background: var(--accent-color); color: #000; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 12px; }}
        .execute-btn:hover {{ background: #00dd77; }}
        .execute-btn:disabled {{ background: #333; color: #666; cursor: not-allowed; }}
    </style>
</head>
<body>
    <h1>📋 Task Library</h1>
    <div class="subtitle">Available robot tasks with real-time capability checks</div>
    
    <div class="filter-bar">
        <button class="active" onclick="filterTasks('all')">All Tasks</button>
        <button onclick="filterTasks('ready')">Ready</button>
        <button onclick="filterTasks('Safety')">Safety</button>
        <button onclick="filterTasks('Demo')">Demo</button>
        <button onclick="filterTasks('Training')">Training</button>
    </div>
    
    <div class="task-grid" id="taskGrid">
        <div class="loading">Loading tasks...</div>
    </div>
    
    <script>
        let allTasks = [];
        let currentFilter = 'all';
        
        async function loadTasks() {{
            try {{
                const response = await fetch('/api/tasks/library');
                const data = await response.json();
                allTasks = data.tasks || [];
                renderTasks();
            }} catch(e) {{
                console.error('Failed to load tasks:', e);
                document.getElementById('taskGrid').innerHTML = '<div style="color: #ff4444;">Failed to load tasks. Using demo data.</div>';
                loadDemoTasks();
            }}
        }}
        
        function loadDemoTasks() {{
            // Demo tasks matching BrainService TaskLibrary
            allTasks = [
                {{
                    id: 'workspace-inspection',
                    title: 'Inspect Workspace',
                    description: 'Sweep sensors for loose cables or obstacles before enabling autonomy.',
                    group: 'Safety',
                    tags: ['safety', 'vision', 'preflight'],
                    icon: '🔍',
                    estimated_duration: '45s',
                    eligibility: {{ eligible: true, markers: [] }},
                    required_modalities: ['vision'],
                    steps: [
                        'Spin the camera and depth sensors across the workspace',
                        'Flag blocked envelopes for operator acknowledgement',
                        'Cache a short clip for offline review'
                    ]
                }},
                {{
                    id: 'pick-and-place',
                    title: 'Pick & Place Demo',
                    description: 'Run the default manipulation loop for demos and regressions.',
                    group: 'Demo',
                    tags: ['manipulation', 'vision', 'demo'],
                    icon: '🦾',
                    estimated_duration: '2m',
                    eligibility: {{ eligible: false, markers: [{{ code: 'NO_ARM', label: 'Arm required', blocking: true }}] }},
                    required_modalities: ['vision', 'manipulation'],
                    steps: [
                        'Detect object on table',
                        'Plan grasp approach',
                        'Execute pick motion',
                        'Place at target location'
                    ]
                }},
                {{
                    id: 'self-calibration',
                    title: 'Self-Calibration Routine',
                    description: 'Run joint calibration and depth sensor alignment check.',
                    group: 'System',
                    tags: ['calibration', 'system', 'maintenance'],
                    icon: '⚙️',
                    estimated_duration: '3m',
                    eligibility: {{ eligible: true, markers: [] }},
                    required_modalities: ['manipulation'],
                    steps: [
                        'Move each joint through full range',
                        'Record encoder readings',
                        'Align depth camera reference frame',
                        'Save calibration parameters'
                    ]
                }},
                {{
                    id: 'episode-collection',
                    title: 'Collect Training Episode',
                    description: 'Record a new teleoperation episode for VLA training.',
                    group: 'Training',
                    tags: ['training', 'data', 'teleoperation'],
                    icon: '📹',
                    estimated_duration: '1-5m',
                    eligibility: {{ eligible: true, markers: [] }},
                    required_modalities: ['vision', 'manipulation'],
                    steps: [
                        'Start episode recording',
                        'Accept human teleoperation commands',
                        'Capture RGB-D + proprioception at 10Hz',
                        'Save RLDS episode to local storage'
                    ]
                }}
            ];
            renderTasks();
        }}
        
        function renderTasks() {{
            const grid = document.getElementById('taskGrid');
            let filtered = allTasks;
            
            if (currentFilter !== 'all') {{
                if (currentFilter === 'ready') {{
                    filtered = allTasks.filter(t => t.eligibility.eligible);
                }} else {{
                    filtered = allTasks.filter(t => t.group === currentFilter || t.tags.includes(currentFilter.toLowerCase()));
                }}
            }}
            
            if (filtered.length === 0) {{
                grid.innerHTML = '<div style="color: var(--text-dim);">No tasks match this filter.</div>';
                return;
            }}
            
            grid.innerHTML = filtered.map(task => `
                <div class="task-card" onclick="toggleTask(this, '${{task.id}}')">
                    <div class="task-header">
                        <div class="task-icon">${{task.icon || '📋'}}</div>
                        <div class="task-status ${{getStatusClass(task.eligibility)}}">
                            ${{getStatusLabel(task.eligibility)}}
                        </div>
                    </div>
                    <div class="task-title">${{task.title}}</div>
                    <div class="task-description">${{task.description}}</div>
                    <div class="task-meta">
                        <span>⏱️ ${{task.estimated_duration || 'N/A'}}</span>
                        <span>📦 ${{task.group}}</span>
                    </div>
                    <div class="task-tags">
                        ${{task.tags.map(tag => `<span class="task-tag">${{tag}}</span>`).join('')}}
                    </div>
                    <div class="task-details">
                        <div style="font-weight: 600; margin-bottom: 8px;">Required:</div>
                        <div style="color: var(--text-dim); margin-bottom: 12px;">
                            ${{task.required_modalities.join(', ')}}
                        </div>
                        <div style="font-weight: 600; margin-bottom: 8px;">Steps:</div>
                        <ul class="task-steps">
                            ${{task.steps.map(step => `<li>${{step}}</li>`).join('')}}
                        </ul>
                        ${{renderEligibilityMarkers(task.eligibility)}}
                        <button class="execute-btn" 
                                onclick="executeTask('${{task.id}}', event)" 
                                ${{task.eligibility.eligible ? '' : 'disabled'}}>
                            Execute Task
                        </button>
                    </div>
                </div>
            `).join('');
        }}
        
        function getStatusClass(eligibility) {{
            if (eligibility.eligible) return 'status-ready';
            const blocking = eligibility.markers.some(m => m.blocking);
            return blocking ? 'status-blocked' : 'status-partial';
        }}
        
        function getStatusLabel(eligibility) {{
            if (eligibility.eligible) return 'Ready';
            const blocking = eligibility.markers.some(m => m.blocking);
            return blocking ? 'Blocked' : 'Warning';
        }}
        
        function renderEligibilityMarkers(eligibility) {{
            if (!eligibility.markers || eligibility.markers.length === 0) return '';
            return `
                <div style="margin-top: 12px; padding: 12px; background: rgba(255,68,68,0.1); border-radius: 6px;">
                    <div style="font-weight: 600; margin-bottom: 8px; color: #ff4444;">Issues:</div>
                    ${{eligibility.markers.map(m => `
                        <div style="font-size: 0.85em; margin: 4px 0;">
                            ⚠️ ${{m.label}}${{m.remediation ? ': ' + m.remediation : ''}}
                        </div>
                    `).join('')}}
                </div>
            `;
        }}
        
        function toggleTask(card, taskId) {{
            if (event.target.classList.contains('execute-btn')) return;
            card.classList.toggle('expanded');
        }}
        
        function filterTasks(filter) {{
            currentFilter = filter;
            document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            renderTasks();
        }}
        
        async function executeTask(taskId, event) {{
            event.stopPropagation();
            if (!confirm(`Execute task: ${{taskId}}?`)) return;
            
            try {{
                const response = await fetch(`/api/tasks/${{taskId}}/execute`, {{
                    method: 'POST'
                }});
                const result = await response.json();
                alert(result.message || 'Task started');
            }} catch(e) {{
                alert('Failed to execute task: ' + e.message);
            }}
        }}
        
        // Load on page load
        loadTasks();
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

def get_tasks_html() -> str:
    return TASKS_HTML

# Import HOPE monitoring pages
try:
    from continuonbrain.api.routes.hope_ui_pages import UI_ROUTES_HOPE
    
    def get_hope_training_html() -> str:
        return UI_ROUTES_HOPE["/ui/hope/training"]
    
    def get_hope_memory_html() -> str:
        return UI_ROUTES_HOPE["/ui/hope/memory"]
    
    def get_hope_stability_html() -> str:
        return UI_ROUTES_HOPE["/ui/hope/stability"]
    
    def get_hope_dynamics_html() -> str:
        return UI_ROUTES_HOPE["/ui/hope/dynamics"]
    
    def get_hope_performance_html() -> str:
        return UI_ROUTES_HOPE["/ui/hope/performance"]
except ImportError:
    # HOPE pages not available
    def get_hope_training_html() -> str:
        return "<html><body><h1>HOPE Training Dashboard</h1><p>HOPE implementation not found.</p></body></html>"
    
    def get_hope_memory_html() -> str:
        return "<html><body><h1>CMS Memory Inspector</h1><p>HOPE implementation not found.</p></body></html>"
    
    def get_hope_stability_html() -> str:
        return "<html><body><h1>Stability Monitor</h1><p>HOPE implementation not found.</p></body></html>"
    
    def get_hope_dynamics_html() -> str:
        return "<html><body><h1>Wave-Particle Dynamics</h1><p>HOPE implementation not found.</p></body></html>"
    
    def get_hope_performance_html() -> str:
        return "<html><body><h1>Performance Benchmarks</h1><p>HOPE implementation not found.</p></body></html>"

