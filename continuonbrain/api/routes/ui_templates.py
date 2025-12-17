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
"""

PERFORMANCE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Performance Benchmarks</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
""" + COMMON_CSS + """
        body { padding: 20px; max-width: 1200px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-box { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; height: 350px; }
        
        .theory-section { margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }
        .theory-section h3 { color: var(--accent-color); }
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
        const latChart = new Chart(ctxL, {
            type: 'bar',
            data: {
                labels: ['<10ms', '10-20ms', '20-50ms', '50-100ms', '>100ms'],
                datasets: [{
                    label: 'Inference Steps',
                    data: [85, 10, 3, 1, 1], // Initial Mock Data
                    backgroundColor: 'rgba(0, 255, 136, 0.5)',
                    borderColor: '#00ff88',
                    borderWidth: 1
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
        
        // Resource Usage (Pie)
        const ctxR = document.getElementById('resourceChart').getContext('2d');
        const resChart = new Chart(ctxR, {
            type: 'doughnut',
            data: {
                labels: ['HOPE Core', 'Memory Access', 'Visual Encoding', 'Idle'],
                datasets: [{
                    data: [30, 20, 15, 35],
                    backgroundColor: ['#ff4444', '#0088ff', '#ffcc00', '#333'],
                    borderWidth: 0
                }]
            },
            options: { responsive: true }
        });
        
        // Mock Updates
        setInterval(() => {
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
        }, 2000);
    </script>
</body>
</html>
"""

DYNAMICS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>System Dynamics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
""" + COMMON_CSS + """
        body { padding: 20px; max-width: 1200px; margin: 0 auto; }
        .canvas-container { background: #000; border-radius: 8px; overflow: hidden; border: 1px solid #333; margin-top: 20px; }
        canvas { width: 100%; height: 400px; display: block; }
        
        .theory-section { margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }
        .theory-section h3 { color: var(--accent-color); }
    </style>
</head>
<body>
    <h1>System Dynamics (Attractor Landscape)</h1>
    <p style="color: var(--text-dim);">Visualizing the neural state space trajectory.</p>
    
    <div class="canvas-container">
        <canvas id="dynamicsCanvas"></canvas>
    </div>
    
    <div class="theory-section">
        <h3>Scientific Background: Attractor Dynamics</h3>
        <p>The brain's state evolves on a low-dimensional manifold. We visualize this as a trajectory in 2D phase space.</p>
        <ul>
            <li><strong>Stable Point</strong>: System settles into a known state (memory recall).</li>
            <li><strong>Limit Cycle</strong>: Oscillatory behavior (idle/scanning).</li>
            <li><strong>Chaos</strong>: Unpredictable trajectory (learning/confusion).</li>
        </ul>
    </div>

    <script>
        const canvas = document.getElementById('dynamicsCanvas');
        const ctx = canvas.getContext('2d');
        
        // Resize
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        
        // Trail
        const trail = [];
        const MAX_TRAIL = 100;
        
        async function updateDynamics() {
            // Fetch state (mock for now, would be /api/hope/state)
            // We simulate a Lorenz attractor-like behavior
            const t = Date.now() / 1000;
            
            // Mock dynamics
            const x = Math.sin(t) * 100 + Math.sin(t * 2.1) * 50;
            const y = Math.cos(t) * 100 + Math.sin(t * 1.7) * 50;
            
            trail.push({x, y});
            if(trail.length > MAX_TRAIL) trail.shift();
            
            // Render
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'; // Fade effect
            ctx.fillRect(0, 0, width, height);
            
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            if(trail.length > 0) {
                const first = trail[0];
                ctx.moveTo(centerX + first.x, centerY + first.y);
                
                for(let i=1; i<trail.length; i++) {
                    const p = trail[i];
                    ctx.lineTo(centerX + p.x, centerY + p.y);
                }
                ctx.stroke();
                
                // Head
                const last = trail[trail.length - 1];
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(centerX + last.x, centerY + last.y, 4, 0, Math.PI*2);
                ctx.fill();
            }
        }
        
        setInterval(updateDynamics, 50); // 20Hz update
        
    </script>
</body>
</html>
"""

MEMORY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CMS Memory Inspector</title>
    <style>
""" + COMMON_CSS + """
        body { padding: 20px; max-width: 1200px; margin: 0 auto; }
        .heatmap { display: flex; flex-direction: column; gap: 10px; margin-top: 20px; }
        .memory-level { display: flex; align-items: center; gap: 15px; }
        .level-label { width: 100px; color: var(--text-dim); font-size: 0.9em; }
        .level-bar { flex: 1; height: 40px; background: #111; border: 1px solid #333; border-radius: 4px; display: flex; overflow: hidden; }
        .memory-slot { flex: 1; height: 100%; border-right: 1px solid #222; transition: background 0.2s; }
        
        .theory-section { margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }
        .theory-section h3 { color: var(--accent-color); }
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
        
        async function init() {
            const res = await fetch('/api/hope/structure');
            const data = await res.json();
            if(data.topology.columns.length > 0) {
                levels = data.topology.columns[0].levels;
            }
            renderGrid();
        }
        
        async function compactMemory() {
            const btn = document.querySelector('button');
            const status = document.getElementById('compaction-status');
            
            if(!confirm("Enter Sleep Mode? This will compact episodic memories into long-term weights.")) return;
            
            btn.disabled = true;
            btn.innerText = "Compacting...";
            status.innerText = "Running consolidation cycles...";
            
            try {
                const res = await fetch('/api/hope/compact', { method: 'POST' });
                const data = await res.json();
                
                status.innerText = "Compaction Complete. Energy transferred.";
                btn.innerText = "🌙 Sleep (Compact Data)";
                
                // Visual flush effect
                document.querySelectorAll('.memory-slot').forEach(el => {
                    el.style.transition = 'background 1s';
                    el.style.backgroundColor = '#fff'; // Flash white
                    setTimeout(() => {
                        el.style.backgroundColor = 'transparent'; // Then clear
                        el.style.transition = 'background 0.2s';
                    }, 500);
                });
                
            } catch(e) {
                status.innerText = "Error: " + e;
                btn.innerText = "🌙 Sleep (Compact Data)";
            } finally {
                btn.disabled = false;
            }
        }
        
        function renderGrid() {
            const container = document.getElementById('cms-container');
            container.innerHTML = '';
            
            const levelNames = ['Episodic (Fast)', 'Working (Mid)', 'Semantic (Slow)'];
            
            for(let i=0; i<levels; i++) {
                const row = document.createElement('div');
                row.className = 'memory-level';
                
                const label = document.createElement('div');
                label.className = 'level-label';
                label.innerText = levelNames[i] || `Level ${i}`;
                row.appendChild(label);
                
                const bar = document.createElement('div');
                bar.className = 'level-bar';
                
                // Create slots
                const slots = 20; // Visual slots
                for(let j=0; j<slots; j++) {
                    const slot = document.createElement('div');
                    slot.className = 'memory-slot';
                    slot.id = `lvl-${i}-slot-${j}`;
                    bar.appendChild(slot);
                }
                
                row.appendChild(bar);
                container.appendChild(row);
            }
        }
        
        function animate() {
            // Randomly activate slots to simulate memory access 'sparkle'
            // In real integration, this would read 'attention weights' from the API
            const now = Date.now();
            
            for(let i=0; i<levels; i++) {
                for(let j=0; j<20; j++) {
                    const el = document.getElementById(`lvl-${i}-slot-${j}`);
                    if(!el) continue;
                    
                    // Probability of activation decreases with level depth (Semantic is more stable)
                    const prob = 0.1 / (i + 1); 
                    
                    if(Math.random() < prob) {
                        const intensity = Math.random();
                        const color = i === 0 ? `rgba(0, 255, 136, ${intensity})` : // Green (Fast)
                                      i === 1 ? `rgba(0, 136, 255, ${intensity})` : // Blue (Mid)
                                                `rgba(255, 69, 58, ${intensity})`;   // Red (Slow)
                        
                        el.style.backgroundColor = color;
                        
                        // Decay effect
                        setTimeout(() => {
                            if(el) el.style.backgroundColor = 'transparent';
                        }, 200 + (i * 200));
                    }
                }
            }
        }
        
        init();
        setInterval(animate, 100);
    </script>
</body>
</html>
"""

STABILITY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stability Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
""" + COMMON_CSS + """
        body { padding: 20px; max-width: 1200px; margin: 0 auto; }
        .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-top: 20px; }
        .chart-container { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; height: 400px; }
        .info-panel { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 20px; font-size: 0.9em; }
        
        .metric-card { background: rgba(0,0,0,0.3); border-radius: 6px; padding: 15px; margin-bottom: 15px; border-left: 3px solid var(--accent-color); }
        .metric-label { color: var(--text-dim); font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 1.5em; font-family: var(--font-mono); color: #fff; margin-top: 5px; }
        
        .theory-section { margin-top: 30px; border-top: 1px solid var(--border-color); padding-top: 20px; }
        .theory-section h3 { color: var(--accent-color); }
        .equation { font-family: 'Times New Roman', serif; font-style: italic; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 4px; text-align: center; margin: 10px 0; }
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
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Lyapunov Energy (L)',
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        data: [],
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Dissipation Rate',
                        borderColor: '#0088ff',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        data: [],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#aaa' } }
                },
                scales: {
                    y: {
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    },
                    x: {
                        grid: { color: '#333' },
                        ticks: { color: '#888', display: false } // Hide timestamps for clean look
                    }
                },
                animation: { duration: 0 } // Disable animation for performance
            }
        });

        // --- DATA POLLING ---
        const MAX_POINTS = 50;
        let lastEnergy = 0;

        async function updateData() {
            try {
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

                if(chart.data.labels.length > MAX_POINTS) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                    chart.data.datasets[1].data.shift();
                }
                
                chart.update();

            } catch(e) {
                console.error("Polling error", e);
            }
        }
        
        function resetStability() {
            chart.data.labels = [];
            chart.data.datasets.forEach(ds => ds.data = []);
            chart.update();
        }

        setInterval(updateData, 1000);
        updateData();
    </script>
</body>
</html>
"""
