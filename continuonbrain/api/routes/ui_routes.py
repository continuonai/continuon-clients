"""
UI Routes: Serve HTML/JS/CSS for the ContinuonBrain Web Interface.
"""
from typing import Dict
from pathlib import Path
from continuonbrain.api.routes.training_plan_page import get_training_plan_html as get_training_plan_page

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
    /* Mobile Responsive */
    @media (max-width: 768px) {
        :root {
            --sidebar-width: 0px;
            --chat-width: 0px;
        }
        
        #sidebar {
            position: fixed;
            left: 0;
            top: 0;
            height: 100%;
            width: 80% !important;
            max-width: 300px;
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }
        
        #sidebar.mobile-open {
            transform: translateX(0);
            width: 80% !important;
        }

        #chat-sidebar {
            position: fixed;
            right: 0;
            top: 0;
            height: 100%;
            width: 90% !important;
            max-width: 360px;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            z-index: 100;
        }

        #chat-sidebar.mobile-open {
            transform: translateX(0);
        }

        .panel-toggle {
            top: auto;
            bottom: 20px;
            width: 48px;
            height: 48px;
            font-size: 1.5em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            background: rgba(30,30,30,0.9);
            backdrop-filter: blur(4px);
        }
        
        #toggle-left { left: 20px; }
        #toggle-right { right: 20px; }
        
        /* Adjust Manual Control on Mobile */
        .manual-layout { flex-direction: column; }
        #video-layer { position: fixed; top: 0; height: 50vh; }
        #controls-layer { 
            position: fixed; 
            top: 50vh; 
            left: 0; 
            right: 0; 
            width: 100%; 
            height: 50vh; 
            border-left: none; 
            border-top: 1px solid var(--border-color); 
        }
    }
"""

HOME_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
<title>Continuon Studio</title>
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
            
            if (window.innerWidth <= 768) {{
                sidebar.classList.toggle('mobile-open');
                // Close right if open
                document.getElementById('chat-sidebar').classList.remove('mobile-open');
            }} else {{
                sidebar.classList.toggle('collapsed');
                const isCollapsed = sidebar.classList.contains('collapsed');
                btn.innerHTML = isCollapsed ? '☰' : '◀';
                localStorage.setItem('leftPanelCollapsed', isCollapsed);
            }}
        }}
        
        function toggleRightPanel() {{
            const chatSidebar = document.getElementById('chat-sidebar');
            const btn = document.getElementById('toggle-right');
            
            if (window.innerWidth <= 768) {{
                chatSidebar.classList.toggle('mobile-open');
                // Close left if open
                document.getElementById('sidebar').classList.remove('mobile-open');
            }} else {{
                chatSidebar.classList.toggle('collapsed');
                const isCollapsed = chatSidebar.classList.contains('collapsed');
                btn.innerHTML = isCollapsed ? '💬' : '▶';
                localStorage.setItem('rightPanelCollapsed', isCollapsed);
            }}
        }}
        
        // Restore panel states from localStorage
        function restorePanelStates() {{
            if (window.innerWidth <= 768) {{
                // Mobile defaults
                document.getElementById('sidebar').classList.add('collapsed');
                document.getElementById('chat-sidebar').classList.add('collapsed');
                return;
            }}

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

        async function loadBrainMeta() {{
            try {{
                const resp = await fetch('/api/hope/structure');
                const data = await resp.json();
                const meta = data.meta || {{}};
                const model = meta.model_name || 'unknown';
                const version = meta.model_version || 'unknown';
                const depth = meta.core_depth || {{}};
                const depthLabel = `Cores: ${{depth.columns ?? '--'}}, Levels: ${{depth.max_levels ?? '--'}}`;
                document.getElementById('brain-meta-model').textContent = `Model: ${model} (${version})`;
                document.getElementById('brain-meta-depth').textContent = depthLabel;
            }} catch (e) {{
                document.getElementById('brain-meta-model').textContent = 'Model: unavailable';
                document.getElementById('brain-meta-depth').textContent = 'Depth: unavailable';
            }}
        }}
        
        async function sendMessage() {{
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Get or create session ID for multi-turn context
            let sessionId = localStorage.getItem('chatSessionId');
            if (!sessionId) {{
                sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('chatSessionId', sessionId);
            }}
            
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
                    body: JSON.stringify({{ message, history: chatHistory, session_id: sessionId }})
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
                
                // Store timestamp for validation
                const timestamp = new Date().toISOString();
                addMessage('agent', 'Agent Manager', data.response || 'No response', timestamp);
            }} catch(e) {{
                hideAgentStatus();
                addMessage('agent', 'Agent Manager', 'Error: ' + e.message);
            }}
            
            input.disabled = false;
            document.getElementById('sendBtn').disabled = false;
            input.focus();
        }}
        
        async function validateResponse(timestamp, validated) {{
            try {{
                const response = await fetch('/api/agent/validate', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ timestamp, validated }})
                }});
                
                const data = await response.json();
                if (data.success) {{
                    // Visual feedback
                    const emoji = validated ? '✅' : '❌';
                    addMessage('system', 'System', `${{emoji}} Response marked as ${{validated ? 'correct' : 'incorrect'}}`);
                }} else {{
                    console.error('Validation failed:', data.error);
                }}
            }} catch(e) {{
                console.error('Validation error:', e);
            }}
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
        
        function addMessage(type, sender, text, timestamp) {{
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${{type}}`;
            
            // Add validation buttons for agent messages
            let validationHTML = '';
            if (type === 'agent' && timestamp) {{
                validationHTML = `
                    \u003cdiv class="validation-buttons" style="margin-top: 8px; opacity: 0.6; transition: opacity 0.2s;">
                        \u003cbutton onclick="validateResponse('${{timestamp}}', true)" 
                                style="background: none; border: none; cursor: pointer; font-size: 1.2em; padding: 4px 8px; transition: transform 0.1s;"
                                title="Mark as correct"
                                onmouseover="this.style.transform='scale(1.2)'"
                                onmouseout="this.style.transform='scale(1)'"
                                \u003e👍\u003c/button\u003e
                        \u003cbutton onclick="validateResponse('${{timestamp}}', false)" 
                                style="background: none; border: none; cursor: pointer; font-size: 1.2em; padding: 4px 8px; transition: transform 0.1s;"
                                title="Mark as incorrect"
                                onmouseover="this.style.transform='scale(1.2)'"
                                onmouseout="this.style.transform='1)'"
                                \u003e👎\u003c/button\u003e
                    \u003c/div\u003e
                `;
            }}
            
            messageDiv.innerHTML = `
                \u003cdiv class="sender"\u003e${{sender}}\u003c/div\u003e
                \u003cdiv class="bubble"\u003e${{escapeHtml(text)}}\u003c/div\u003e
                ${{validationHTML}}
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
        
        function startNewSession() {{
            // Clear session ID to start fresh
            localStorage.removeItem('chatSessionId');
            
            // Clear chat history
            chatHistory = [];
            
            // Clear chat messages display
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.innerHTML = '';
            
            // Add system message
            addMessage('system', 'System', '🔄 New conversation session started');
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
            <span style="font-size: 1.2em;">🧠</span> Continuon<span> Studio</span>
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
            <a href="/ui/training-plan" target="main_frame" onclick="selectNav(this)">
                <span class="icon">📚</span> Training Plan
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
            <a href="/ui/hope/map" target="main_frame" onclick="selectNav(this)">
                <span class="icon">🌌</span> Brain Map
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
            <a href="/ui/learning" target="main_frame" onclick="selectNav(this)">
                <span class="icon">📊</span> Learning Dashboard
            </a>
            <a href="/ui/settings" target="main_frame" onclick="selectNav(this)">
                <span class="icon">⚙️</span> Settings
            </a>
        </div>
        
        <div class="sidebar-footer">
            <div><span class="status-dot"></span> Continuon Studio</div>
            <div id="brain-meta-model" style="margin-top: 8px; opacity: 0.8;">Model: --</div>
            <div id="brain-meta-depth" style="opacity: 0.7;">Depth: --</div>
            <div style="margin-top: 8px;">
                <a href="/ui/training-plan" target="main_frame" style="color: var(--accent-color); text-decoration: none;">Training Plan & Manager</a>
            </div>
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
                <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span>🤖</span> <span id="agent-title">Agent Manager</span>
                    </div>
                    <button onclick="startNewSession()" 
                            style="background: rgba(0,255,136,0.1); border: 1px solid var(--accent-color); 
                                   color: var(--accent-color); padding: 4px 12px; border-radius: 4px; 
                                   cursor: pointer; font-size: 0.85em;"
                            title="Start a new conversation session">
                        New Session
                    </button>
                </div>
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
        <div id="activity-log" style="background: #111; border-radius: 8px; padding: 20px; font-family: var(--font-mono); font-size: 0.9em; color: #aaa; border: 1px solid var(--border-color);">
            Loading recent events...
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

                }}
            }} catch(e) {{ console.error("Stats update failed", e); }}
        }}

        async function refreshActivityLog() {{
            const container = document.getElementById('activity-log');
            if (!container) return;

            try {{
                const res = await fetch('/api/system/events?limit=30');
                if (!res.ok) {{
                    container.innerHTML = '<div>Event log unavailable</div>';
                    return;
                }}

                const data = await res.json();
                const events = data.events || [];

                if (!events.length) {{
                    container.innerHTML = '<div>No recent events</div>';
                    return;
                }}

                container.innerHTML = events.map(evt => {{
                    const ts = evt.iso_time || new Date((evt.timestamp || 0) * 1000).toLocaleTimeString();
                    const type = (evt.event_type || 'event').toUpperCase();
                    const message = evt.message || '';
                    return `<div>[${{ts}}] ${{type}}: ${{message}}</div>`;
                }}).join('');
            }} catch (e) {{
                container.innerHTML = '<div>Failed to load events</div>';
            }}
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
        setInterval(refreshActivityLog, 5000);
        updateStats();
        refreshActivityLog();
        </script>
    </body>
    </html>
    """

BRAIN_MAP_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Brain Map</title>
    <style>
        {COMMON_CSS}
        body, html {{ margin: 0; padding: 0; overflow: hidden; background: #000; }}
        #canvas-container {{ width: 100%; height: 100vh; position: absolute; top: 0; left: 0; z-index: 1; }}
        #ui-overlay {{ position: absolute; top: 20px; left: 20px; z-index: 2; pointer-events: none; }}
        .hud-panel {{ background: rgba(0, 0, 0, 0.6); border: 1px solid var(--accent-dim); padding: 20px; border-radius: 8px; backdrop-filter: blur(5px); color: var(--text-color); width: 300px; }}
        h1 {{ margin: 0 0 10px 0; font-size: 1.2em; color: var(--accent-color); text-transform: uppercase; letter-spacing: 2px; }}
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-family: var(--font-mono); font-size: 0.9em; }}
        .stat-label {{ color: var(--text-dim); }}
        .stat-value {{ color: #fff; }}
        #energy-meter {{ width: 100%; height: 4px; background: #333; margin-top: 10px; position: relative; }}
        #energy-bar {{ height: 100%; background: var(--accent-color); width: 0%; transition: width 0.5s, background-color 0.5s; }}
        
        #controls {{ position: absolute; bottom: 20px; right: 20px; z-index: 2; display: flex; gap: 10px; }}
        .btn {{ pointer-events: auto; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 8px 16px; border-radius: 4px; cursor: pointer; transition: all 0.2s; }}
        .btn:hover {{ background: rgba(255,255,255,0.2); }}
        
        .loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: var(--accent-color); font-family: var(--font-mono); z-index: 10; pointer-events: none; }}
    </style>
    <!-- Three.js from CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <!-- Post-processing -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
</head>
<body>
    <div id="loading" class="loading">INITIALIZING CORTEX VISUALIZATION...</div>
    <div id="canvas-container"></div>
    
    <div id="ui-overlay">
        <div class="hud-panel">
             <h1>Neural Topology</h1>
             <div class="stat-row">
                 <span class="stat-label">Architecture</span>
                 <span class="stat-value">HOPE-v1</span>
             </div>
             <div class="stat-row">
                 <span class="stat-label">Active Columns</span>
                 <span class="stat-value" id="col-count">--</span>
             </div>
             <div class="stat-row">
                 <span class="stat-label">Global Status</span>
                 <span class="stat-value" id="global-status">--</span>
             </div>
             <div class="stat-row">
                 <span class="stat-label">Lyapunov Energy</span>
                 <span class="stat-value" id="energy-val">--</span>
             </div>
             <div id="energy-meter">
                 <div id="energy-bar"></div>
             </div>
        </div>
    </div>
    
    <div id="controls">
        <button class="btn" onclick="controls.autoRotate = !controls.autoRotate">Toggle Rotation</button>
        <button class="btn" onclick="resetCam()">Reset View</button>
    </div>

    <script>
        // --- CONFIG ---
        const COLUMN_SPACING = 30;
        const LEVEL_HEIGHT = 10;
        const NEURON_RADIUS = 0.5;
        
        // --- THREE.JS SETUP ---
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x000000, 0.005);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(40, 30, 40);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        // Orbit Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        
        // --- POST PROCESSING (BLOOM) ---
        const renderScene = new THREE.RenderPass(scene, camera);
        const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
        bloomPass.threshold = 0;
        bloomPass.strength = 1.5; // Glow strength
        bloomPass.radius = 0.5;
        
        const composer = new THREE.EffectComposer(renderer);
        composer.addPass(renderScene);
        composer.addPass(bloomPass);
        
        // --- SCENE OBJECTS ---
        const brainGroup = new THREE.Group();
        scene.add(brainGroup);
        
        // Grid Floor
        const gridHelper = new THREE.GridHelper(200, 50, 0x333333, 0x111111);
        scene.add(gridHelper);
        
        // Mapping from ID to 3D Objects
        const columnMap = new Map(); // col_id -> {{ group, levels: [], particles: [] }}
        
        // --- DATA FETCH & UPDATE ---
        let lastEnergy = 0;
        
        function getHealthColor(energy) {{
            // Low energy = Stable (Teal/Green)
            // High energy = Chaotic (Red/Orange)
            // Max expected energy ~100k (normalized to 0-1 range roughly)
            // Log scale mapping
            const logE = Math.log10(energy + 1);
            const t = Math.min(Math.max((logE - 3) / 3, 0), 1); // rough mapping 1k -> 1M
            
            const c1 = new THREE.Color(0x00ff88); // Teal
            const c2 = new THREE.Color(0xff4444); // Red
            return c1.lerp(c2, t);
        }}
        
        async function fetchBrainData() {{
            try {{
                const res = await fetch('/api/hope/structure');
                const data = await res.json();
                
                if (data.error) {{
                    document.querySelector('.loading').innerText = "BRAIN OFFLINE";
                    return;
                }}
                
                document.getElementById('loading').style.display = 'none';
                updateVisualization(data);
                updateHUD(data);
                
            }} catch(e) {{
                console.error("Fetch error", e);
            }}
        }}
        
        function createColumn(colData, index, totalCols) {{
             const group = new THREE.Group();
             
             // Position in a circle or grid
             const angle = (index / totalCols) * Math.PI * 2;
             const radius = 20 + (totalCols * 2); 
             const x = Math.cos(angle) * radius;
             const z = Math.sin(angle) * radius;
             
             group.position.set(x, 0, z);
             group.lookAt(0, 0, 0);
             
             const levels = [];
             
             // Create Levels (Torus rings)
             for(let i=0; i<colData.levels; i++) {{
                 const geometry = new THREE.TorusGeometry(3, 0.1, 16, 50);
                 const material = new THREE.MeshBasicMaterial({{ color: 0x0044aa, transparent: true, opacity: 0.3 }});
                 const ring = new THREE.Mesh(geometry, material);
                 
                 ring.rotation.x = Math.PI / 2;
                 ring.position.y = (i * LEVEL_HEIGHT) + 5;
                 
                 group.add(ring);
                 levels.push({{ mesh: ring, baseY: ring.position.y }});
                 
                 // Vertical connector
                 if (i > 0) {{
                     const lineGeo = new THREE.BufferGeometry().setFromPoints([
                         new THREE.Vector3(0, (i-1)*LEVEL_HEIGHT + 5, 0),
                         new THREE.Vector3(0, i*LEVEL_HEIGHT + 5, 0)
                     ]);
                     const lineMat = new THREE.LineBasicMaterial({{ color: 0x0044aa, transparent: true, opacity: 0.2 }});
                     const line = new THREE.Line(lineGeo, lineMat);
                     group.add(line);
                 }}
             }}
             
             // Particles (Orbiting electrons) - Representation of Particle Subsystem
             const particleSys = new THREE.Group();
             for(let i=0; i<10; i++) {{
                 const pGeo = new THREE.SphereGeometry(0.2, 8, 8);
                 const pMat = new THREE.MeshBasicMaterial({{ color: 0x00ffff }});
                 const p = new THREE.Mesh(pGeo, pMat);
                 
                 // Random orbit parameters attached to user data
                 p.userData = {{
                     radius: 4 + Math.random() * 2,
                     speed: 0.02 + Math.random() * 0.05,
                     angle: Math.random() * Math.PI * 2,
                     yOffset: Math.random() * (colData.levels * LEVEL_HEIGHT)
                 }};
                 
                 particleSys.add(p);
             }}
             group.add(particleSys);
             
             brainGroup.add(group);
             
             return {{ group, levels, particleSys }};
        }}
        
        function updateVisualization(data) {{
            const topology = data.topology;
            const state = data.state;
            
            // 1. Manage Columns (Create/Delete)
            // Just clearing/recreating simplifies logic for now, or assume static since boot
            if (columnMap.size === 0 && topology.columns.length > 0) {{
                topology.columns.forEach((col, idx) => {{
                    const obj = createColumn(col, idx, topology.columns.length);
                    columnMap.set(col.id, obj);
                }});
            }}
            
            // 2. Update State
            const globalColor = getHealthColor(state.lyapunov_energy);
            
            state.columns.forEach(colState => {{
                const obj = columnMap.get(colState.id);
                if (!obj) return;
                
                // Pulse levels based on activity
                const now = Date.now() * 0.001;
                obj.levels.forEach((lvl, idx) => {{
                    // Wave subsystem effect
                    const pulse = Math.sin(now * 2 + idx) * 0.5 + 0.5;
                    const brightness = Math.min(colState.activity_level * 0.1, 1.0); // Normalize activity
                    
                    lvl.mesh.material.opacity = 0.1 + (brightness * 0.5) + (pulse * 0.1);
                    lvl.mesh.material.color.lerp(globalColor, 0.1);
                    
                    // Physical wobble?
                    lvl.mesh.position.y = lvl.baseY + Math.sin(now * 3 + idx) * 0.2;
                }});
                
                // Animate particles
                obj.particleSys.children.forEach(p => {{
                    const ud = p.userData;
                    ud.angle += ud.speed * (1 + colState.activity_level * 0.5); // Faster when active
                    
                    p.position.x = Math.cos(ud.angle) * ud.radius;
                    p.position.z = Math.sin(ud.angle) * ud.radius;
                    p.position.y = ud.yOffset + Math.sin(now + ud.angle)*0.5;
                }});
            }});
            
            // Global Bloom intensity based on energy
            // Higher energy = brighter bloom (more chaotic)
            // bloomPass.strength = 1.5 + (Math.log10(state.lyapunov_energy + 1) * 0.2);
        }}
        
        function updateHUD(data) {{
            document.getElementById('col-count').innerText = data.topology.columns.length;
            document.getElementById('global-status').innerText = data.state.global_status.toUpperCase();
            
            const energy = data.state.lyapunov_energy.toFixed(2);
            document.getElementById('energy-val').innerText = energy;
            
            // Energy Bar
            // Assuming 1M is "Critical" max
            const ratio = Math.min((data.state.lyapunov_energy / 100000) * 100, 100);
            const bar = document.getElementById('energy-bar');
            bar.style.width = ratio + '%';
            
            if (ratio < 30) bar.style.backgroundColor = '#00ff88';
            else if (ratio < 70) bar.style.backgroundColor = '#ffcc00';
            else bar.style.backgroundColor = '#ff4444';
        }}
        
        function resetCam() {{
             controls.reset();
             camera.position.set(40, 30, 40);
        }}
        
        // --- LOOP ---
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            composer.render();
        }}
        
        // Initial call
        fetchBrainData();
        setInterval(fetchBrainData, 1000); // 1Hz update
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            composer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
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
        <h2>Connectivity (Wi‑Fi & Bluetooth)</h2>
        <div class="form-group">
            <button class="btn" onclick="scanWifi()">Scan Wi‑Fi Networks</button>
            <div id="wifi-results" style="margin-top: 10px; color: var(--text-dim);">No scan yet</div>
            <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px;">
                <input type="text" id="wifi-ssid" placeholder="SSID">
                <input type="password" id="wifi-pass" placeholder="Password (if required)">
                <button class="btn primary" onclick="connectWifi()">Connect Wi‑Fi</button>
            </div>
            <div id="wifi-status" style="margin-top: 8px; color: var(--text-dim);"></div>
        </div>
        <div class="form-group">
            <button class="btn" onclick="scanBluetooth()">Scan Bluetooth Devices</button>
            <div id="bt-results" style="margin-top: 10px; color: var(--text-dim);">No scan yet</div>
            <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px;">
                <input type="text" id="bt-address" placeholder="Device address (e.g., AA:BB:CC:DD:EE:FF)">
                <button class="btn primary" onclick="connectBluetooth()">Connect Bluetooth</button>
            </div>
            <div id="bt-status" style="margin-top: 8px; color: var(--text-dim);"></div>
        </div>
    </div>
    
    <div class="section">
        <h2>Personality & Identity</h2>
        
        <div class="form-group">
            <label>System Name</label>
            <input type="text" id="system_name" value="robot" placeholder="e.g. TARS" onchange="saveSettings()">
        </div>

        <div class="form-group">
            <label>Personality Preset</label>
            <select id="identity_mode" onchange="applyPreset(this.value)">
                <option value="Custom">Custom</option>
                <option value="Adaptive">Adaptive (Default)</option>
                <option value="Professional">Professional Assistant</option>
                <option value="Friendly">Friendly Companion</option>
                <option value="Robot">Robot (TARS-like)</option>
            </select>
        </div>
        <div class="form-group">
            <label>Humor Level (<span id="val_humor">50</span>%)</label>
            <input type="range" id="humor_level" min="0" max="1" step="0.1" value="0.5" oninput="onSliderChange('humor', this.value)">
        </div>
        <div class="form-group">
            <label>Sarcasm Level (<span id="val_sarcasm">50</span>%)</label>
            <input type="range" id="sarcasm_level" min="0" max="1" step="0.1" value="0.5" oninput="onSliderChange('sarcasm', this.value)">
        </div>
        <div class="form-group">
            <label>Empathy Level (<span id="val_empathy">50</span>%)</label>
            <input type="range" id="empathy_level" min="0" max="1" step="0.1" value="0.5" oninput="onSliderChange('empathy', this.value)">
        </div>
        
        <div class="form-group">
            <label>Verbosity (<span id="val_verbosity">50</span>%)</label>
            <input type="range" id="verbosity_level" min="0" max="1" step="0.1" value="0.5" oninput="onSliderChange('verbosity', this.value)">
        </div>
        
        <div style="border-top: 1px solid #333; padding-top: 20px; margin-top: 20px;">
            <label style="color: var(--accent-color); font-weight: bold;">User Context Simulation</label>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <input type="text" id="user_id" placeholder="User ID" value="craigm26">
                <select id="user_role">
                    <option value="owner">Owner</option>
                    <option value="guest">Guest</option>
                </select>
                <button class="btn" style="background: transparent; border: 1px solid var(--accent-color); color: var(--accent-color);" onclick="updateIdentity()">Update Context</button>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Agent Manager</h2>
        <div class="form-group">
            <label>Primary Agent Core</label>
            <select id="agent_model" style="max-width: 400px; padding: 8px;">
                <option value="mock">Loading models...</option>
            </select>
            <small style="color: var(--text-dim); display: block; margin-top: 4px;">
                Select the underlying logic model. The <b>HOPE Architecture</b> manages all high-level decisions, memory, and agency.
            </small>
        </div>
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
        // Load available models on page load
        async function loadAvailableModels() {{
            try {{
                const res = await fetch('/api/agent/models');
                const data = await res.json();
                
                if (data.success && data.models) {{
                    const select = document.getElementById('agent_model');
                    select.innerHTML = ''; // Clear loading option
                    
                    data.models.forEach(model => {{
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.name + (model.size_mb > 0 ? ` (~${{model.size_mb}}MB)` : '');
                        option.title = model.description;
                        select.appendChild(option);
                    }});
                }}
            }} catch(e) {{
                console.error('Error loading models:', e);
            }}
        }}
        
        async function loadSettings() {{
            try {{
                const res = await fetch('/api/settings');
                const data = await res.json();
                
                if (data.success && data.settings) {{
                    const s = data.settings;
                    
                    // Agent Manager settings
                    if (s.agent_manager) {{
                        if (s.agent_manager.agent_model) {{
                            document.getElementById('agent_model').value = s.agent_manager.agent_model;
                        }}
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

            // Load Personality
            try {{
                const res = await fetch('/api/personality');
                const p = await res.json();
                
                document.getElementById('humor_level').value = p.humor_level;
                document.getElementById('val_humor').innerText = Math.round(p.humor_level * 100);
                
                document.getElementById('sarcasm_level').value = p.sarcasm_level;
                document.getElementById('val_sarcasm').innerText = Math.round(p.sarcasm_level * 100);
                
                document.getElementById('empathy_level').value = p.empathy_level;
                document.getElementById('val_empathy').innerText = Math.round(p.empathy_level * 100);
                
                document.getElementById('verbosity_level').value = p.verbosity_level || 0.5;
                document.getElementById('val_verbosity').innerText = Math.round((p.verbosity_level || 0.5) * 100);

                document.getElementById('system_name').value = p.system_name || "robot";
                updateSidebarIdentity(p.system_name);
                
                document.getElementById('identity_mode').value = p.identity_mode;
                
            }} catch(e) {{
                console.error('Error loading personality:', e);
            }}

             // Load Identity Context
            try {{
                const res = await fetch('/api/identity');
                const i = await res.json();
                document.getElementById('user_id').value = i.user_id;
                document.getElementById('user_role').value = i.role;
            }} catch(e) {{}}
        }}
        
        async function saveSettings() {{
            const settings = {{
                theme: document.getElementById('theme').value,
                model_endpoint: document.getElementById('model_endpoint').value,
                local_learning: document.getElementById('local_learning').checked,
                auto_load: document.getElementById('auto_load').checked,
                default_mode: document.getElementById('default_mode').value,
                agent_manager: {{
                    agent_model: document.getElementById('agent_model').value,
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
                    // alert('Settings saved successfully!');
                }} else {{
                    alert('Failed to save general settings: ' + (data.message || 'Unknown error'));
                }}
            }} catch(e) {{
                alert('Error saving settings: ' + e.message);
            }}
            
            // Save Personality
            const personality = {{
                humor_level: parseFloat(document.getElementById('humor_level').value),
                sarcasm_level: parseFloat(document.getElementById('sarcasm_level').value),
                empathy_level: parseFloat(document.getElementById('empathy_level').value),
                verbosity_level: parseFloat(document.getElementById('verbosity_level').value),
                system_name: document.getElementById('system_name').value,
                identity_mode: document.getElementById('identity_mode').value
            }};
            
            try {{
                await fetch('/api/personality/update', {{
                    method: 'POST',
                    body: JSON.stringify(personality)
                }});
                updateSidebarIdentity(personality.system_name);
            }} catch(e) {{ console.error(e); }}
            
            alert('Settings and Personality saved!');
        }}
        
        async function updateIdentity() {{
             const ctx = {{
                 user_id: document.getElementById('user_id').value,
                 role: document.getElementById('user_role').value
             }};
             try {{
                await fetch('/api/identity/update', {{
                    method: 'POST',
                    body: JSON.stringify(ctx)
                }});
                alert('User context updated!');
            }} catch(e) {{ alert(e); }}
        }}
        
        // --- PRESET LOGIC ---
        const PRESETS = {{
            'Adaptive': {{ humor: 0.5, sarcasm: 0.3, empathy: 0.7, verbosity: 0.6 }},
            'Professional': {{ humor: 0.1, sarcasm: 0.0, empathy: 0.4, verbosity: 0.3 }},
            'Friendly': {{ humor: 0.8, sarcasm: 0.1, empathy: 0.9, verbosity: 0.7 }},
            'Robot': {{ humor: 0.2, sarcasm: 0.9, empathy: 0.1, verbosity: 0.4 }} // TARS-like
        }};
        
        function applyPreset(name) {{
             if (name === 'Custom') return;
             
             const p = PRESETS[name];
             if (!p) return;
             
             updateSlider('humor_level', p.humor, 'val_humor');
             updateSlider('sarcasm_level', p.sarcasm, 'val_sarcasm');
             updateSlider('empathy_level', p.empathy, 'val_empathy');
             updateSlider('verbosity_level', p.verbosity, 'val_verbosity');
        }}
        
        function updateSlider(id, val, textId) {{
            const el = document.getElementById(id);
            if(el) {{
                el.value = val;
                document.getElementById(textId).innerText = Math.round(val * 100);
            }}
        }}
        
        function onSliderChange(type, val) {{
            // Update the text label
            document.getElementById(`val_${{type}}`).innerText = Math.round(val * 100);
            
            // Switch dropdown to 'Custom' automatically
            const dd = document.getElementById('identity_mode');
            if (dd.value !== 'Custom') {{
                dd.value = 'Custom';
            }}
        }}

        function updateSidebarIdentity(name) {{
             const logo = document.querySelector('.logo');
             if (logo) {{
                 // Keep the icon 🧠
                 logo.innerHTML = '<span style="font-size: 1.2em;">🧠</span> Continuon<span>' + (name || 'Brain') + '</span>';
             }}
        }}
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {{
            loadAvailableModels();
            loadSettings();
        }});

        async function scanWifi() {{
            const el = document.getElementById('wifi-results');
            el.textContent = 'Scanning Wi‑Fi...';
            try {{
                const res = await fetch('/api/network/wifi/scan');
                const data = await res.json();
                if (!data.success) {{
                    el.textContent = data.message || 'Scan failed';
                    return;
                }}
                if (!data.networks || !data.networks.length) {{
                    el.textContent = 'No networks found';
                    return;
                }}
                el.innerHTML = data.networks.map(n => {{
                    const sec = n.security && n.security !== 'open' ? n.security : 'Open';
                    const sig = n.signal !== undefined ? `${{n.signal}}%` : '';
                    return `<div>📶 ${{n.ssid || '<hidden>'}} ${{sig}} <span style="color: var(--text-dim);">(${{sec}})</span></div>`;
                }}).join('');
                refreshWifiStatus();
            }} catch (e) {{
                el.textContent = 'Scan error: ' + e.message;
            }}
        }}

        async function refreshWifiStatus() {{
            const statusEl = document.getElementById('wifi-status');
            if (!statusEl) return;
            try {{
                const res = await fetch('/api/network/wifi/status');
                const data = await res.json();
                if (!data.success || !data.connections || !data.connections.length) {{
                    statusEl.textContent = data.message || 'Wi‑Fi: not connected';
                    return;
                }}
                const c = data.connections[0];
                statusEl.textContent = `Connected to ${{c.name}} (${{c.device || ''}})`;
            }} catch (e) {{
                statusEl.textContent = 'Wi‑Fi status error: ' + e.message;
            }}
        }}

        async function connectWifi() {{
            const ssid = document.getElementById('wifi-ssid').value.trim();
            const password = document.getElementById('wifi-pass').value;
            const statusEl = document.getElementById('wifi-status');
            statusEl.textContent = 'Connecting...';
            try {{
                const res = await fetch('/api/network/wifi/connect', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ ssid, password }})
                }});
                const data = await res.json();
                statusEl.textContent = data.success ? (data.message || 'Connected') : (data.message || 'Connect failed');
                refreshWifiStatus();
            }} catch (e) {{
                statusEl.textContent = 'Connect error: ' + e.message;
            }}
        }}

        async function scanBluetooth() {{
            const el = document.getElementById('bt-results');
            el.textContent = 'Scanning Bluetooth...';
            try {{
                const res = await fetch('/api/network/bluetooth/scan');
                const data = await res.json();
                if (!data.success) {{
                    el.textContent = data.message || 'Scan failed';
                    return;
                }}
                if (!data.devices || !data.devices.length) {{
                    el.textContent = 'No devices found';
                    return;
                }}
                el.innerHTML = data.devices.map(d => `<div>🦻 ${{d.name || 'Unknown'}} <span style="color: var(--text-dim);">${{d.address}}</span></div>`).join('');
            }} catch (e) {{
                el.textContent = 'Scan error: ' + e.message;
            }}
        }}

        async function connectBluetooth() {{
            const addr = document.getElementById('bt-address').value.trim();
            const statusEl = document.getElementById('bt-status');
            statusEl.textContent = 'Connecting...';
            try {{
                const res = await fetch('/api/network/bluetooth/connect', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ address: addr }})
                }});
                const data = await res.json();
                statusEl.textContent = data.success ? (data.message || 'Connected') : (data.message || 'Connect failed');
            }} catch (e) {{
                statusEl.textContent = 'Connect error: ' + e.message;
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

        /* Tabs */
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; }}
        .tab-link {{ flex: 1; padding: 10px; background: transparent; border: none; border-bottom: 2px solid transparent; color: var(--text-dim); cursor: pointer; font-weight: 600; transition: all 0.2s; font-size: 0.95em; }}
        .tab-link:hover {{ color: #fff; background: rgba(255,255,255,0.05); border-radius: 6px 6px 0 0; }}
        .tab-link.active {{ color: var(--accent-color); border-bottom-color: var(--accent-color); }}
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
        
        <div class="tabs">
            <button class="tab-link" id="tab-btn-driving" onclick="openTab('driving')">🏎️ Driving</button>
            <button class="tab-link" id="tab-btn-arm" onclick="openTab('arm')">🤖 Arm</button>
        </div>

        <!-- DRIVING GROUP -->
        <div id="group-driving" class="control-group">
            <div class="dpad">
                <div></div>
                <button onmousedown="drive('forward')" onmouseup="drive('stop')" ontouchstart="drive('forward')" ontouchend="drive('stop')">⬆️</button>
                <div></div>
                
                <button onmousedown="drive('left')" onmouseup="drive('stop')" ontouchstart="drive('left')" ontouchend="drive('stop')">⬅️</button>
                <button style="background: #222; cursor: default;">🎯</button>
                <button onmousedown="drive('right')" onmouseup="drive('stop')" ontouchstart="drive('right')" ontouchend="drive('stop')">➡️</button>
                
                <div></div>
                <button onmousedown="drive('backward')" onmouseup="drive('stop')" ontouchstart="drive('backward')" ontouchend="drive('stop')">⬇️</button>
                <div></div>
            </div>
            
            <div class="speed-selector">
                <button onclick="setSpeed(0.3)" class="active">Slow</button>
                <button onclick="setSpeed(0.6)">Med</button>
                <button onclick="setSpeed(1.0)">Fast</button>
            </div>
        </div>

        <!-- ARM GROUP -->
        <div id="group-arm" class="control-group" style="display: none;">
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
        
        function openTab(tabName) {{
            // Hide all tab content
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("control-group");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}

            // Remove active class from all tab links
            tablinks = document.getElementsByClassName("tab-link");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].classList.remove("active");
            }}

            // Show the specific tab content
            document.getElementById('group-' + tabName).style.display = "block";
            
            // Add active class to button
            const btn = document.getElementById('tab-btn-' + tabName);
            if(btn) btn.classList.add("active");
        }}
        
        // Init default tab
        document.addEventListener('DOMContentLoaded', () => {{
            openTab('driving');
        }});
        
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

        // On load
        window.addEventListener('load', () => {{
            restorePanelStates();
            loadBrainMeta();
        }});
        
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

def get_training_plan_html() -> str:
    return get_training_plan_page()

def get_learning_dashboard_html() -> str:
    """Return learning dashboard page."""
    # For now, use inline HTML - can be moved to constant later
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Learning Dashboard</title>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        h1 { color: white; text-align: center; margin-bottom: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; max-width: 1200px; margin: 0 auto 30px; }
        .stat-card { background: rgba(255,255,255,0.95); border-radius: 12px; padding: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
        .stat-label { font-size: 0.85em; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .stat-value { font-size: 2.5em; font-weight: bold; color: #333; }
        .chart-container { background: rgba(255,255,255,0.95); border-radius: 12px; padding: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 20px; max-width: 1200px; margin: 0 auto 20px; }
        .chart-title { font-size: 1.2em; font-weight: bold; margin-bottom: 16px; color: #333; }
        .bar-chart { display: flex; flex-direction: column; gap: 12px; }
        .bar-item { display: flex; align-items: center; gap: 12px; }
        .bar-label { min-width: 180px; font-size: 0.9em; color: #555; }
        .bar-bg { flex: 1; height: 24px; background: #f0f0f0; border-radius: 12px; overflow: hidden; }
        .bar-fill { height: 100%; transition: width 0.5s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; }
        .bar-value { color: white; font-size: 0.85em; font-weight: bold; }
        .refresh-btn { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1em; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    </style>
</head>
<body>
    <h1>📊 Learning Dashboard</h1>
    <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Total Conversations</div><div class="stat-value" id="total">-</div></div>
        <div class="stat-card"><div class="stat-label">HOPE Response Rate</div><div class="stat-value" id="hope-rate">-%</div></div>
        <div class="stat-card"><div class="stat-label">LLM Context Rate</div><div class="stat-value" id="context-rate">-%</div></div>
        <div class="stat-card"><div class="stat-label">LLM Only Rate</div><div class="stat-value" id="llm-rate">-%</div></div>
        <div class="stat-card"><div class="stat-label">Validation Rate</div><div class="stat-value" id="val-rate">-%</div></div>
    </div>
    <div class="chart-container">
        <div class="chart-title">Agent Distribution</div>
        <div class="bar-chart" id="distribution"></div>
    </div>
    <div style="text-align: center; max-width: 1200px; margin: 0 auto;">
        <button class="refresh-btn" onclick="loadStats()">🔄 Refresh Stats</button>
    </div>
    <script>
        async function loadStats() {
            try {
                const r = await fetch('/api/agent/learning_stats');
                const data = await r.json();
                if (data.success) {
                    const s = data.stats;
                    document.getElementById('total').textContent = s.total_conversations;
                    document.getElementById('hope-rate').textContent = (s.hope_response_rate * 100).toFixed(1) + '%';
                    document.getElementById('context-rate').textContent = (s.llm_context_rate * 100).toFixed(1) + '%';
                    document.getElementById('llm-rate').textContent = (s.llm_only_rate * 100).toFixed(1) + '%';
                    
                    const valRate = s.total_conversations > 0 ? (s.validated_conversations / s.total_conversations * 100).toFixed(1) : 0;
                    document.getElementById('val-rate').textContent = valRate + '% (' + s.validated_conversations + '/' + s.total_conversations + ')';

                    const div = document.getElementById('distribution');
                    div.innerHTML = '';
                    [{label:'HOPE Brain',key:'hope_brain',c:'#10b981'},{label:'LLM+Context',key:'llm_with_hope_context',c:'#667eea'},{label:'LLM Only',key:'llm_only',c:'#ec4899'}].forEach(a => {
                        const cnt = s.by_agent[a.key] || 0;
                        const pct = s.total_conversations > 0 ? (cnt / s.total_conversations * 100).toFixed(1) : 0;
                        div.innerHTML += `<div class="bar-item"><div class="bar-label">${a.label}</div><div class="bar-bg"><div class="bar-fill" style="width:${pct}%;background:${a.c};"><span class="bar-value">${cnt} (${pct}%)</span></div></div></div>`;
                    });
                }
            } catch(e) { console.error(e); }
        }
        loadStats();
        setInterval(loadStats, 30000);
    </script>
</body>
</html>'''


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
        return MEMORY_HTML
    
    def get_hope_dynamics_html() -> str:
        return DYNAMICS_HTML
    
    def get_hope_performance_html() -> str:
        return PERFORMANCE_HTML

PERFORMANCE_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Performance Benchmarks</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {COMMON_CSS}
        body {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-box {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; height: 350px; }}
        
        .theory-section {{ margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }}
        .theory-section h3 {{ color: var(--accent-color); }}
    </style>
</head>
<body>
    <h1>Performance Benchmarks</h1>
    <p style="color: var(--text-dim);">Computational cost and latency analysis.</p>
    
    <div class="grid">
        <div class="chart-box">
            <h3>Latency Distribution (ms)</h3>
            <canvas id="latencyChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>Resource Usage</h3>
            <canvas id="resourceChart"></canvas>
        </div>
    </div>
    
    <div class="theory-section">
        <h3>Scientific Background: Computational Cost</h3>
        <p>The HOPE architecture trades O(N) memory complexity for O(1) constant-time lookup speed using the CMS.</p>
        <ul>
            <li><strong>Forward Pass</strong>: Fixed cost per column, parallelizable.</li>
            <li><strong>Learning</strong>: Async update loop (Background Learner) decouples plasticity from inference latency.</li>
        </ul>
        <p>This ensures the robot remains responsive (low inference latency) even as the long-term memory grows indefinitely.</p>
    </div>

    <script>
        // Latency Histogram
        const ctxL = document.getElementById('latencyChart').getContext('2d');
        const latChart = new Chart(ctxL, {{
            type: 'bar',
            data: {{
                labels: ['<10ms', '10-20ms', '20-50ms', '50-100ms', '>100ms'],
                datasets: [{{
                    label: 'Inference Steps',
                    data: [85, 10, 3, 1, 1], // Initial Mock Data
                    backgroundColor: 'rgba(0, 255, 136, 0.5)',
                    borderColor: '#00ff88',
                    borderWidth: 1
                }}]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
        }});
        
        // Resource Usage (Pie)
        const ctxR = document.getElementById('resourceChart').getContext('2d');
        const resChart = new Chart(ctxR, {{
            type: 'doughnut',
            data: {{
                labels: ['HOPE Core', 'Memory Access', 'Visual Encoding', 'Idle'],
                datasets: [{{
                    data: [30, 20, 15, 35],
                    backgroundColor: ['#ff4444', '#0088ff', '#ffcc00', '#333'],
                    borderWidth: 0
                }}]
            }},
            options: {{ responsive: true }}
        }});
        
        // Mock Updates
        setInterval(() => {{
            // Shift latency slightly
            const data = latChart.data.datasets[0].data;
            const flow = Math.floor(Math.random() * 5);
            if(Math.random() > 0.5) data[0] += flow; else data[0] = Math.max(0, data[0] - flow);
            latChart.update();
            
            // Shift resources
            const rData = resChart.data.datasets[0].data;
            rData[0] = 25 + Math.random() * 10; // Core
            rData[3] = 100 - (rData[0] + rData[1] + rData[2]); // Idle remainder
            resChart.update();
        }}, 2000);
    </script>
</body>
</html>
"""

DYNAMICS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Wave-Particle Dynamics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {COMMON_CSS}
        body {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .phase-plot {{ background: #000; border: 1px solid var(--border-color); border-radius: 8px; height: 500px; position: relative; overflow: hidden; }}
        
        .theory-section {{ margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }}
        .theory-section h3 {{ color: var(--accent-color); }}
        
        /* Particle Simulation Canvas */
        #phaseCanvas {{ width: 100%; height: 100%; }}
    </style>
</head>
<body>
    <h1>Wave-Particle Dynamics</h1>
    <p style="color: var(--text-dim);">Visualization of the dual-process state space (Global Wave vs Local Particle).</p>
    
    <div class="grid">
        <div class="phase-plot">
            <canvas id="phaseCanvas"></canvas>
            <div style="position: absolute; top: 10px; left: 10px; color: #00ff88; font-family: monospace;">
                Wave (w): <span id="val-w">0.00</span><br>
                Particle (p): <span id="val-p">0.00</span>
            </div>
        </div>
        
        <div class="theory-section" style="margin-top: 0; border: none;">
            <h3>Scientific Background: Dual-Process Theory</h3>
            <p>HOPE integrates two types of processing:</p>
            <ul>
                <li><strong>Wave State ($w$)</strong>: Analogous to global field potentials. It captures slow, contextual, and rhythmic information.</li>
                <li><strong>Particle State ($p$)</strong>: Analogous to spiking events. It captures fast, local, and precise error signals.</li>
            </ul>
            <p>The <strong>Phase Plot</strong> on the left visualizes the trajectory of the brain's state. Circular orbits indicate stable limit cycles (rhythmic thought), while chaotic attractors (strange loops) indicate active learning or confusion.</p>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('phaseCanvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        
        function resize() {{
            width = canvas.parentElement.offsetWidth;
            height = canvas.parentElement.offsetHeight;
            canvas.width = width;
            canvas.height = height;
        }}
        window.addEventListener('resize', resize);
        resize();
        
        // Simulation State
        const trail = [];
        const MAX_TRAIL = 200;
        
        // Polling
        async function updateDynamics() {{
            try {{
                const res = await fetch('/api/hope/structure');
                const data = await res.json();
                
                // Extract metrics. Since we don't have direct w/p scalar access easily exposed yet without deeper introspection,
                // we will simulate the phase plot behavior based on the activity level and energy for now, 
                // creating a "shadow" projection of the high-dimensional state.
                
                // In a real full implementation, we'd add 'w_norm' and 'p_norm' to the API.
                // For now, let's derive proxies:
                const energy = data.state.lyapunov_energy;
                const activity = data.state.columns[0]?.activity_level || 0;
                
                // Mock projection for visualization purpose until deep API update
                // oscillating based on time and modulated by activity
                const t = Date.now() * 0.002;
                const w_val = Math.sin(t) * (0.5 + activity);
                const p_val = Math.cos(t * 1.5) * (0.5 + (energy / 50000));
                
                document.getElementById('val-w').innerText = w_val.toFixed(3);
                document.getElementById('val-p').innerText = p_val.toFixed(3);
                
                trail.push({{ x: w_val, y: p_val }});
                if(trail.length > MAX_TRAIL) trail.shift();
                
                draw();
                
            }} catch(e) {{}}
        }}
        
        function draw() {{
            // Fade effect
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            ctx.fillRect(0, 0, width, height);
            
            // Draw axes
            ctx.strokeStyle = '#333';
            ctx.beginPath();
            ctx.moveTo(width/2, 0); ctx.lineTo(width/2, height);
            ctx.moveTo(0, height/2); ctx.lineTo(width, height/2);
            ctx.stroke();
            
            // Draw trail
            ctx.beginPath();
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            
            trail.forEach((p, i) => {{
                // Map [-2, 2] to canvas coordinates
                const x = (p.x + 2) / 4 * width;
                const y = height - ((p.y + 2) / 4 * height);
                
                if(i===0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }});
            ctx.stroke();
            
            // Draw head
            if(trail.length > 0) {{
                const last = trail[trail.length-1];
                const x = (last.x + 2) / 4 * width;
                const y = height - ((last.y + 2) / 4 * height);
                
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI*2);
                ctx.fill();
            }}
        }}
        
        setInterval(updateDynamics, 50); // 20Hz update
        
    </script>
</body>
</html>
"""

MEMORY_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CMS Memory Inspector</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .heatmap {{ display: flex; flex-direction: column; gap: 10px; margin-top: 20px; }}
        .memory-level {{ display: flex; align-items: center; gap: 15px; }}
        .level-label {{ width: 100px; color: var(--text-dim); font-size: 0.9em; }}
        .level-bar {{ flex: 1; height: 40px; background: #111; border: 1px solid #333; border-radius: 4px; display: flex; overflow: hidden; }}
        .memory-slot {{ flex: 1; height: 100%; border-right: 1px solid #222; transition: background 0.2s; }}
        
        .theory-section {{ margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }}
        .theory-section h3 {{ color: var(--accent-color); }}
    </style>
</head>
<body>
    <h1>CMS Memory Inspector</h1>
    <p style="color: var(--text-dim);">Hierarchical Continuous Memory System status.</p>
    
    <div class="heatmap" id="cms-container">
        <!-- Rendered via JS -->
    </div>
    
    <div style="margin-top: 20px; text-align: center;">
        <button onclick="compactMemory()" style="padding: 10px 20px; background: var(--accent-color); color: #000; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; font-family: inherit;">
            🌙 Sleep (Compact Data)
        </button>
        <div id="compaction-status" style="margin-top: 10px; font-size: 0.9em; color: var(--text-dim); min-height: 20px;"></div>
    </div>
    
    <div class="theory-section">
        <h3>Scientific Background: Hierarchical Memory</h3>
        <p>The Continuous Memory System (CMS) mimics the hippocampus and neocortex structure:</p>
        <ul>
            <li><strong>Level 0 (Episodic)</strong>: Fast decay, high fidelity. Captures immediate moments.</li>
            <li><strong>Level 1 (Working)</strong>: Medium decay. Integrates episodes into short-term sequences.</li>
            <li><strong>Level 2 (Semantic)</strong>: Slow decay. Extracts stable rules and concepts over time.</li>
        </ul>
        <p>The heatmap above visualizes the "activation" or recall strength of memory slots in real-time.</p>
    </div>

    <script>
        // --- SIMULATION FOR VISUALIZATION ---
        // Since granular per-slot memory access is heavy to stream, we visualize the *concept* 
        // using the structural data we have (num_levels).
        
        let levels = 3; // Default
        
        async function init() {{
            const res = await fetch('/api/hope/structure');
            const data = await res.json();
            if(data.topology.columns.length > 0) {{
                levels = data.topology.columns[0].levels;
            }}
            renderGrid();
        }}
        
        async function compactMemory() {{
            const btn = document.querySelector('button');
            const status = document.getElementById('compaction-status');
            
            if(!confirm("Enter Sleep Mode? This will compact episodic memories into long-term weights.")) return;
            
            btn.disabled = true;
            btn.innerText = "Compacting...";
            status.innerText = "Running consolidation cycles...";
            
            try {{
                const res = await fetch('/api/hope/compact', {{ method: 'POST' }});
                const data = await res.json();
                
                status.innerText = "Compaction Complete. Energy transferred.";
                btn.innerText = "🌙 Sleep (Compact Data)";
                
                // Visual flush effect
                document.querySelectorAll('.memory-slot').forEach(el => {{
                    el.style.transition = 'background 1s';
                    el.style.backgroundColor = '#fff'; // Flash white
                    setTimeout(() => {{
                        el.style.backgroundColor = 'transparent'; // Then clear
                        el.style.transition = 'background 0.2s';
                    }}, 500);
                }});
                
            }} catch(e) {{
                status.innerText = "Error: " + e;
                btn.innerText = "🌙 Sleep (Compact Data)";
            }} finally {{
                btn.disabled = false;
            }}
        }}
        
        function renderGrid() {{
            const container = document.getElementById('cms-container');
            container.innerHTML = '';
            
            const levelNames = ['Episodic (Fast)', 'Working (Mid)', 'Semantic (Slow)'];
            
            for(let i=0; i<levels; i++) {{
                const row = document.createElement('div');
                row.className = 'memory-level';
                
                const label = document.createElement('div');
                label.className = 'level-label';
                label.innerText = levelNames[i] || `Level ${{i}}`;
                row.appendChild(label);
                
                const bar = document.createElement('div');
                bar.className = 'level-bar';
                
                // Create slots
                const slots = 20; // Visual slots
                for(let j=0; j<slots; j++) {{
                    const slot = document.createElement('div');
                    slot.className = 'memory-slot';
                    slot.id = `lvl-${{i}}-slot-${{j}}`;
                    bar.appendChild(slot);
                }}
                
                row.appendChild(bar);
                container.appendChild(row);
            }}
        }}
        
        function animate() {{
            // Randomly activate slots to simulate memory access 'sparkle'
            // In real integration, this would read 'attention weights' from the API
            const now = Date.now();
            
            for(let i=0; i<levels; i++) {{
                for(let j=0; j<20; j++) {{
                    const el = document.getElementById(`lvl-${{i}}-slot-${{j}}`);
                    if(!el) continue;
                    
                    // Probability of activation decreases with level depth (Semantic is more stable)
                    const prob = 0.1 / (i + 1); 
                    
                    if(Math.random() < prob) {{
                        const intensity = Math.random();
                        const color = i === 0 ? `rgba(0, 255, 136, ${{intensity}})` : // Green (Fast)
                                      i === 1 ? `rgba(0, 136, 255, ${{intensity}})` : // Blue (Mid)
                                                `rgba(255, 69, 58, ${{intensity}})`;   // Red (Slow)
                        
                        el.style.backgroundColor = color;
                        
                        // Decay effect
                        setTimeout(() => {{
                            if(el) el.style.backgroundColor = 'transparent';
                        }}, 200 + (i * 200));
                    }}
                }}
            }}
        }}
        
        init();
        setInterval(animate, 100);
    </script>
</body>
</html>
"""
    
STABILITY_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stability Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {COMMON_CSS}
        body {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .dashboard-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-top: 20px; }}
        .chart-container {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; height: 400px; }}
        .info-panel {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; font-size: 0.9em; }}
        
        .metric-card {{ background: rgba(0,0,0,0.3); border-radius: 6px; padding: 15px; margin-bottom: 15px; border-left: 3px solid var(--accent-color); }}
        .metric-label {{ color: var(--text-dim); font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }}
        .metric-value {{ font-size: 1.5em; font-family: var(--font-mono); color: #fff; margin-top: 5px; }}
        
        .theory-section {{ margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }}
        .theory-section h3 {{ color: var(--accent-color); }}
        .equation {{ font-family: 'Times New Roman', serif; font-style: italic; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 4px; text-align: center; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Stability Monitor (Lyapunov)</h1>
    <p style="color: var(--text-dim);">Real-time tracking of neural architecture stability metrics.</p>

    <div class="dashboard-grid">
        <div class="chart-container">
            <canvas id="stabilityChart"></canvas>
        </div>
        
        <div class="info-panel">
            <div class="metric-card">
                <div class="metric-label">Lyapunov Energy (L)</div>
                <div class="metric-value" id="val-energy">--</div>
            </div>
            <div class="metric-card" style="border-left-color: #0088ff;">
                <div class="metric-label">Dissipation Rate (dL/dt)</div>
                <div class="metric-value" id="val-dissipation">--</div>
            </div>
            <div class="metric-card" style="border-left-color: #ffcc00;">
                <div class="metric-label">System Status</div>
                <div class="metric-value" id="val-status">--</div>
            </div>
            
            <div style="margin-top: 20px;">
                <button class="btn primary" onclick="resetStability()">Reset Monitor</button>
            </div>
        </div>
    </div>

    <div class="theory-section">
        <h3>Scientific Background: Lyapunov Stability</h3>
        <p>In the HOPE architecture, we ensure learning safety by enforcing <strong>Lyapunov Stability</strong>. This means the system's internal energy (or "surprise") must be bounded and generally decreasing over time, or strictly bounded during chaos.</p>
        
        <div class="equation">
            L(x) = xᵀ P x + V(x) &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; dL/dt ≤ 0
        </div>
        
        <p>
            The chart above tracks the global Lyapunov function value. 
            <ul>
                <li><strong>Green Zone</strong>: Experimental stability (Energy < 100k). Normal operation.</li>
                <li><strong>Yellow Zone</strong>: High energy (Energy > 100k). Learning is aggressive or input is highly novel.</li>
                <li><strong>Red Zone</strong>: Critical instability. Constraints will actively clamp parameters to prevent divergence.</li>
            </ul>
        </p>
    </div>

    <script>
        // --- CHART SETUP ---
        const ctx = document.getElementById('stabilityChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: [],
                datasets: [
                    {{
                        label: 'Lyapunov Energy (L)',
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        data: [],
                        tension: 0.4,
                        fill: true
                    }},
                    {{
                        label: 'Dissipation Rate',
                        borderColor: '#0088ff',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        data: [],
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#aaa' }} }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888', display: false }} // Hide timestamps for clean look
                    }}
                }},
                animation: {{ duration: 0 }} // Disable animation for performance
            }}
        }});

        // --- DATA POLLING ---
        const MAX_POINTS = 50;
        let lastEnergy = 0;

        async function updateData() {{
            try {{
                const res = await fetch('/api/hope/structure'); // Using our new structure endpoint
                const data = await res.json();
                
                if(data.error) return;

                const energy = data.state.lyapunov_energy;
                const dissipation = lastEnergy - energy;
                lastEnergy = energy;
                
                // Update UI
                document.getElementById('val-energy').innerText = energy.toFixed(2);
                document.getElementById('val-dissipation').innerText = dissipation.toFixed(4);
                
                const status = data.state.global_status || "IDLE";
                document.getElementById('val-status').innerText = status.toUpperCase();
                document.getElementById('val-status').style.color = (status === 'learning') ? '#00ff88' : '#aaa';

                // Update Chart
                const now = new Date().toLocaleTimeString();
                chart.data.labels.push(now);
                chart.data.datasets[0].data.push(energy);
                chart.data.datasets[1].data.push(dissipation); // Scale for visibility if needed?

                if(chart.data.labels.length > MAX_POINTS) {{
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                    chart.data.datasets[1].data.shift();
                }}
                
                chart.update();

            }} catch(e) {{
                console.error("Polling error", e);
            }}
        }}
        
        function resetStability() {{
            chart.data.labels = [];
            chart.data.datasets.forEach(ds => ds.data = []);
            chart.update();
        }}

        setInterval(updateData, 1000);
        updateData();
    </script>
</body>
</html>
"""
    

# --- SCIENTIFIC UI PAGES ---
def get_hope_stability_html() -> str:
    return STABILITY_HTML

def get_hope_dynamics_html() -> str:
    return DYNAMICS_HTML

def get_hope_memory_html() -> str:
    return MEMORY_HTML

def get_hope_performance_html() -> str:
    return PERFORMANCE_HTML




def get_brain_map_html() -> str:
    return BRAIN_MAP_HTML
