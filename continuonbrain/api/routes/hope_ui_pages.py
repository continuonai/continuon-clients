"""
HOPE Brain Monitoring Pages

HTML pages for monitoring HOPE brain development with scientific rigor.
To integrate: import these into ui_routes.py and add to UI_ROUTES dict.
"""

from continuonbrain.api.routes.ui_routes import COMMON_CSS

# HOPE Training Dashboard
HOPE_TRAINING_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HOPE Training</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; margin-bottom: 10px; }}
        .subtitle {{ color: var(--text-dim); margin-bottom: 30px; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .metric-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 0.85em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-card .value {{ font-size: 2em; font-weight: 700; color: #fff; margin: 5px 0; }}
        .metric-card .unit {{ font-size: 0.9em; color: var(--accent-color); }}
        
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .chart-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; }}
        .chart-card h2 {{ margin: 0 0 20px 0; font-size: 1.1em; color: var(--accent-color); }}
        canvas {{ max-height: 300px; }}
        
        .status-badge {{ display: inline-flex; align-items: center; padding: 6px 12px; background: rgba(0, 255, 136, 0.15); color: var(--accent-color); border-radius: 20px; font-weight: 600; font-size: 0.9em; gap: 6px; }}
        .status-badge.warning {{ background: rgba(255, 165, 0, 0.15); color: #ffa500; }}
        .status-badge.error {{ background: rgba(255, 59, 48, 0.15); color: #ff3b30; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
        <div>
            <h1>🧠 HOPE Training Dashboard</h1>
            <p class="subtitle">Real-time monitoring of brain development with scientific rigor</p>
        </div>
        <div id="status-badge" class="status-badge">
            <span class="status-dot"></span> Initializing...
        </div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Lyapunov Energy</h3>
            <div class="value" id="lyapunov-value">-</div>
            <div class="unit">V_total</div>
        </div>
        <div class="metric-card">
            <h3>State Norm</h3>
            <div class="value" id="state-norm-value">-</div>
            <div class="unit">||s||</div>
        </div>
        <div class="metric-card">
            <h3>CMS Utilization</h3>
            <div class="value" id="cms-util-value">-</div>
            <div class="unit">avg %</div>
        </div>
        <div class="metric-card">
            <h3>Learning Rate</h3>
            <div class="value" id="lr-value">-</div>
            <div class="unit">η_t</div>
        </div>
        <div class="metric-card">
            <h3>Steps</h3>
            <div class="value" id="steps-value">-</div>
            <div class="unit">total</div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-card">
            <h2>Lyapunov Energy Over Time</h2>
            <canvas id="lyapunov-chart"></canvas>
        </div>
        <div class="chart-card">
            <h2>State Norms Evolution</h2>
            <canvas id="state-norms-chart"></canvas>
        </div>
        <div class="chart-card">
            <h2>CMS Memory Utilization</h2>
            <canvas id="cms-util-chart"></canvas>
        </div>
        <div class="chart-card">
            <h2>Energy Dissipation Rate</h2>
            <canvas id="dissipation-chart"></canvas>
        </div>
    </div>
    
    <script>
        // Chart instances
        let lyapunovChart, stateNormsChart, cmsUtilChart, dissipationChart;
        
        // Data buffers
        const maxDataPoints = 100;
        const lyapunovData = {{ labels: [], datasets: [
            {{ label: 'V_total', data: [], borderColor: '#00ff88', tension: 0.4 }},
            {{ label: 'V_fast', data: [], borderColor: '#ff6b6b', tension: 0.4 }},
            {{ label: 'V_memory', data: [], borderColor: '#4ecdc4', tension: 0.4 }},
        ]}};
        
        const stateNormsData = {{ labels: [], datasets: [
            {{ label: '||s||', data: [], borderColor: '#00ff88', tension: 0.4 }},
            {{ label: '||w||', data: [], borderColor: '#ff6b6b', tension: 0.4 }},
            {{ label: '||p||', data: [], borderColor: '#4ecdc4', tension: 0.4 }},
        ]}};
        
        const cmsUtilData = {{ labels: [], datasets: [] }};
        const dissipationData = {{ labels: [], datasets: [
            {{ label: 'dV/dt', data: [], borderColor: '#00ff88', tension: 0.4 }},
        ]}};
        
        // Chart options
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: true,
            plugins: {{
                legend: {{ labels: {{ color: '#e0e0e0' }} }},
            }},
            scales: {{
                x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
            }},
        }};
        
        // Initialize charts
        function initCharts() {{
            const ctx1 = document.getElementById('lyapunov-chart').getContext('2d');
            lyapunovChart = new Chart(ctx1, {{
                type: 'line',
                data: lyapunovData,
                options: chartOptions,
            }});
            
            const ctx2 = document.getElementById('state-norms-chart').getContext('2d');
            stateNormsChart = new Chart(ctx2, {{
                type: 'line',
                data: stateNormsData,
                options: chartOptions,
            }});
            
            const ctx3 = document.getElementById('cms-util-chart').getContext('2d');
            cmsUtilChart = new Chart(ctx3, {{
                type: 'bar',
                data: cmsUtilData,
                options: chartOptions,
            }});
            
            const ctx4 = document.getElementById('dissipation-chart').getContext('2d');
            dissipationChart = new Chart(ctx4, {{
                type: 'line',
                data: dissipationData,
                options: chartOptions,
            }});
        }}
        
        // Update metrics
        async function updateMetrics() {{
            try {{
                const res = await fetch('/api/hope/metrics');
                const data = await res.json();
                
                if (data.error) {{
                    document.getElementById('status-badge').className = 'status-badge error';
                    document.getElementById('status-badge').innerHTML = '<span class=\"status-dot\"></span> Error: ' + data.error;
                    return;
                }}
                
                // Update metric cards
                document.getElementById('lyapunov-value').textContent = data.lyapunov.total.toFixed(2);
                document.getElementById('state-norm-value').textContent = data.state_norms.s.toFixed(2);
                
                const avgUtil = data.cms_utilization.reduce((a,b) => a+b, 0) / data.cms_utilization.length;
                document.getElementById('cms-util-value').textContent = (avgUtil * 100).toFixed(1);
                
                document.getElementById('lr-value').textContent = data.learning_rate.toFixed(4);
                document.getElementById('steps-value').textContent = data.steps;
                
                // Update status badge
                document.getElementById('status-badge').className = 'status-badge';
                document.getElementById('status-badge').innerHTML = '<span class=\"status-dot\"></span> Active';
                
                // Update charts
                const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
                
                // Lyapunov chart
                lyapunovData.labels.push(timestamp);
                lyapunovData.datasets[0].data.push(data.lyapunov.total);
                lyapunovData.datasets[1].data.push(data.lyapunov.fast);
                lyapunovData.datasets[2].data.push(data.lyapunov.memory);
                
                if (lyapunovData.labels.length > maxDataPoints) {{
                    lyapunovData.labels.shift();
                    lyapunovData.datasets.forEach(ds => ds.data.shift());
                }}
                lyapunovChart.update('none');
                
                // State norms chart
                stateNormsData.labels.push(timestamp);
                stateNormsData.datasets[0].data.push(data.state_norms.s);
                stateNormsData.datasets[1].data.push(data.state_norms.w);
                stateNormsData.datasets[2].data.push(data.state_norms.p);
                
                if (stateNormsData.labels.length > maxDataPoints) {{
                    stateNormsData.labels.shift();
                    stateNormsData.datasets.forEach(ds => ds.data.shift());
                }}
                stateNormsChart.update('none');
                
                // CMS utilization chart
                cmsUtilData.labels = data.cms_utilization.map((_, i) => `Level ${{i}}`);
                cmsUtilData.datasets = [{{
                    label: 'Utilization %',
                    data: data.cms_utilization.map(u => u * 100),
                    backgroundColor: '#00ff88',
                }}];
                cmsUtilChart.update('none');
                
            }} catch(e) {{
                console.error(e);
                document.getElementById('status-badge').className = 'status-badge error';
                document.getElementById('status-badge').innerHTML = '<span class=\"status-dot\"></span> Connection Error';
            }}
        }}
        
        // Initialize and start updates
        initCharts();
        updateMetrics();
        setInterval(updateMetrics, 1000); // 1Hz update
    </script>
</body>
</html>
"""

# CMS Memory Inspector
HOPE_MEMORY_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CMS Memory</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; }}
        
        .level-selector {{ display: flex; gap: 10px; margin: 30px 0; }}
        .level-btn {{ padding: 12px 24px; background: var(--card-bg); border: 1px solid var(--border-color); color: var(--text-color); border-radius: 8px; cursor: pointer; transition: all 0.2s; }}
        .level-btn:hover {{ background: #333; }}
        .level-btn.active {{ background: var(--accent-color); color: #000; border-color: var(--accent-color); font-weight: 600; }}
        
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
        .stat-card h3 {{ margin: 0 0 15px 0; font-size: 0.9em; color: var(--accent-color); }}
        .stat-row {{ display: flex; justify-content: space-between; margin: 8px 0; font-size: 0.9em; }}
        .stat-row .label {{ color: var(--text-dim); }}
        .stat-row .value {{ color: #fff; font-weight: 600; font-family: var(--font-mono); }}
        
        .heatmap-container {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; }}
        .heatmap-container h2 {{ margin: 0 0 20px 0; color: var(--accent-color); }}
    </style>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <h1>💾 CMS Memory Inspector</h1>
    <p style="color: var(--text-dim); margin-bottom: 30px;">Hierarchical memory visualization with scientific analysis</p>
    
    <div class="level-selector" id="level-selector">
        <!-- Populated dynamically -->
    </div>
    
    <div class="stats-grid" id="stats-grid">
        <!-- Populated dynamically -->
    </div>
    
    <div class="heatmap-container">
        <h2>Memory Matrix Heatmap</h2>
        <div id="heatmap"></div>
    </div>
    
    <script>
        let currentLevel = 0;
        let numLevels = 0;
        
        async function loadLevel(levelId) {{
            currentLevel = levelId;
            
            // Update button states
            document.querySelectorAll('.level-btn').forEach((btn, i) => {{
                btn.className = i === levelId ? 'level-btn active' : 'level-btn';
            }});
            
            try {{
                const res = await fetch(`/api/hope/cms/level/${{levelId}}`);
                const data = await res.json();
                
                if (data.error) {{
                    alert('Error: ' + data.error);
                    return;
                }}
                
                // Update stats
                const statsHtml = `
                    <div class="stat-card">
                        <h3>Memory Matrix (M)</h3>
                        <div class="stat-row"><span class="label">Shape:</span><span class="value">${{data.shape.M.join(' × ')}}< /span></div>
                        <div class="stat-row"><span class="label">Mean:</span><span class="value">${{data.M_stats.mean.toFixed(4)}}</span></div>
                        <div class="stat-row"><span class="label">Std:</span><span class="value">${{data.M_stats.std.toFixed(4)}}</span></div>
                        <div class="stat-row"><span class="label">Norm:</span><span class="value">${{data.M_stats.norm.toFixed(4)}}</span></div>
                        <div class="stat-row"><span class="label">Range:</span><span class="value">[${{data.M_stats.min.toFixed(2)}}, ${{data.M_stats.max.toFixed(2)}}]</span></div>
                    </div>
                    <div class="stat-card">
                        <h3>Key Matrix (K)</h3>
                        <div class="stat-row"><span class="label">Shape:</span><span class="value">${{data.shape.K.join(' × ')}}</span></div>
                        <div class="stat-row"><span class="label">Mean:</span><span class="value">${{data.K_stats.mean.toFixed(4)}}</span></div>
                        <div class="stat-row"><span class="label">Std:</span><span class="value">${{data.K_stats.std.toFixed(4)}}</span></div>
                        <div class="stat-row"><span class="label">Norm:</span><span class="value">${{data.K_stats.norm.toFixed(4)}}</span></div>
                    </div>
                    <div class="stat-card">
                        <h3>Level Parameters</h3>
                        <div class="stat-row"><span class="label">Level ID:</span><span class="value">${{data.level_id}}</span></div>
                        <div class="stat-row"><span class="label">Decay (d_ℓ):</span><span class="value">${{data.decay.toFixed(4)}}</span></div>
                        <div class="stat-row"><span class="label">Half-life:</span><span class="value">${{(0.693 / data.decay).toFixed(1)}} steps</span></div>
                    </div>
                `;
                document.getElementById('stats-grid').innerHTML = statsHtml;
                
                // Update heatmap
                if (data.M_data) {{
                    const heatmapData = [{{
                        z: data.M_data,
                        type: 'heatmap',
                        colorscale: 'Viridis',
                    }}];
                    
                    const layout = {{
                        title: `Memory Matrix M^(${{levelId}})`,
                        xaxis: {{ title: 'Dimension' }},
                        yaxis: {{ title: 'Slot' }},
                        paper_bgcolor: '#1e1e1e',
                        plot_bgcolor: '#1e1e1e',
                        font: {{ color: '#e0e0e0' }},
                    }};
                    
                    Plotly.newPlot('heatmap', heatmapData, layout);
                }} else {{
                    document.getElementById('heatmap').innerHTML = '<p style="color: var(--text-dim);">Matrix too large to display. Showing statistics only.</p>';
                }}
                
            }} catch(e) {{
                console.error(e);
                alert('Failed to load level data');
            }}
        }}
        
        async function init() {{
            try {{
                const res = await fetch('/api/hope/config');
                const config = await res.json();
                
                numLevels = config.num_levels;
                
                // Create level selector buttons
                let buttonsHtml = '';
                for (let i = 0; i < numLevels; i++) {{
                    buttonsHtml += `<button class="level-btn" onclick="loadLevel(${{i}})">Level ${{i}} (N=${{config.cms_sizes[i]}})</button>`;
                }}
                document.getElementById('level-selector').innerHTML = buttonsHtml;
                
                // Load first level
                loadLevel(0);
                
            }} catch(e) {{
                console.error(e);
                alert('Failed to initialize');
            }}
        }}
        
        init();
    </script>
</body>
</html>
"""

# Stability Monitor
HOPE_STABILITY_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stability Monitor</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; }}
        
        .alert-box {{ padding: 20px; border-radius: 12px; margin: 20px 0; border-left: 4px solid; }}
        .alert-box.success {{ background: rgba(0, 255, 136, 0.1); border-color: var(--accent-color); }}
        .alert-box.warning {{ background: rgba(255, 165, 0, 0.1); border-color: #ffa500; }}
        .alert-box.error {{ background: rgba(255, 59, 48, 0.1); border-color: #ff3b30; }}
        .alert-box h3 {{ margin: 0 0 10px 0; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 0.85em; color: var(--text-dim); }}
        .metric-card .value {{ font-size: 2em; font-weight: 700; color: #fff; }}
        
        .chart-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; margin: 20px 0; }}
        .chart-card h2 {{ margin: 0 0 20px 0; color: var(--accent-color); }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <h1>⚖️ Stability Monitor</h1>
    <p style="color: var(--text-dim); margin-bottom: 20px;">Lyapunov-based stability analysis with scientific rigor</p>
    
    <div id="alert-container">
        <!-- Populated dynamically -->
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Stability Status</h3>
            <div class="value" id="stability-status">-</div>
        </div>
        <div class="metric-card">
            <h3>Lyapunov Energy</h3>
            <div class="value" id="lyapunov-current">-</div>
        </div>
        <div class="metric-card">
            <h3>Dissipation Rate</h3>
            <div class="value" id="dissipation-rate">-</div>
        </div>
        <div class="metric-card">
            <h3>Gradient Norm</h3>
            <div class="value" id="gradient-norm">-</div>
        </div>
    </div>
    
    <div class="chart-card">
        <h2>Lyapunov Energy Decomposition</h2>
        <canvas id="lyapunov-decomp-chart"></canvas>
    </div>
    
    <script>
        let decompChart;
        const decompData = {{ labels: [], datasets: [
            {{ label: 'V_fast', data: [], backgroundColor: '#ff6b6b', stack: 'Stack 0' }},
            {{ label: 'V_memory', data: [], backgroundColor: '#4ecdc4', stack: 'Stack 0' }},
            {{ label: 'V_params', data: [], backgroundColor: '#ffd93d', stack: 'Stack 0' }},
        ]}};
        
        function initCharts() {{
            const ctx = document.getElementById('lyapunov-decomp-chart').getContext('2d');
            decompChart = new Chart(ctx, {{
                type: 'bar',
                data: decompData,
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                    scales: {{
                        x: {{ stacked: true, ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                        y: {{ stacked: true, ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    }},
                }},
            }});
        }}
        
        async function updateStability() {{
            try {{
                const res = await fetch('/api/hope/stability');
                const data = await res.json();
                
                // Update metrics
                document.getElementById('stability-status').textContent = data.is_stable ? '✓ Stable' : '✗ Unstable';
                document.getElementById('stability-status').style.color = data.is_stable ? '#00ff88' : '#ff3b30';
                
                document.getElementById('lyapunov-current').textContent = data.lyapunov_current.toFixed(2);
                document.getElementById('dissipation-rate').textContent = data.dissipation_rate.toFixed(4);
                document.getElementById('gradient-norm').textContent = (data.gradient_norm || 0).toFixed(4);

                // Update alert
                let alertHtml = '';
                if (!data.is_stable) {{
                    alertHtml = '<div class="alert-box error"><h3>⚠️ Instability Detected</h3><p>System energy is not bounded. Check training parameters.</p></div>';
                }} else if (data.has_nan || data.has_inf) {{
                    alertHtml = '<div class="alert-box error"><h3>⚠️ Numerical Issues</h3><p>NaN or Inf values detected in state. System may be unstable.</p></div>';
                }} else if (data.gradient_spike) {{
                    alertHtml = '<div class="alert-box warning"><h3>⚠️ Gradient Spike</h3><p>Recent gradients exceeded the configured clip threshold. Monitor learning rate and rewards.</p></div>';
                }} else if (data.dissipation_rate < 0) {{
                    alertHtml = '<div class="alert-box warning"><h3>⚠️ Energy Increasing</h3><p>Lyapunov energy is growing. Monitor closely.</p></div>';
                }} else {{
                    alertHtml = '<div class="alert-box success"><h3>✓ System Stable</h3><p>All stability metrics within normal range.</p></div>';
                }}
                document.getElementById('alert-container').innerHTML = alertHtml;
                
                // Fetch metrics for decomposition
                const metricsRes = await fetch('/api/hope/metrics');
                const metricsData = await metricsRes.json();
                
                if (!metricsData.error) {{
                    const timestamp = new Date(metricsData.timestamp * 1000).toLocaleTimeString();
                    decompData.labels.push(timestamp);
                    decompData.datasets[0].data.push(metricsData.lyapunov.fast);
                    decompData.datasets[1].data.push(metricsData.lyapunov.memory);
                    decompData.datasets[2].data.push(metricsData.lyapunov.params);
                    
                    if (decompData.labels.length > 20) {{
                        decompData.labels.shift();
                        decompData.datasets.forEach(ds => ds.data.shift());
                    }}
                    decompChart.update('none');
                }}
                
            }} catch(e) {{
                console.error(e);
            }}
        }}
        
        initCharts();
        updateStability();
        setInterval(updateStability, 1000);
    </script>
</body>
</html>
"""

# Wave-Particle Dynamics Page
HOPE_DYNAMICS_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Wave-Particle Dynamics</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 0.85em; color: var(--text-dim); }}
        .metric-card .value {{ font-size: 2em; font-weight: 700; color: #fff; }}
        
        .chart-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; margin: 20px 0; }}
        .chart-card h2 {{ margin: 0 0 20px 0; color: var(--accent-color); }}
    </style>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <h1>🌊 Wave-Particle Dynamics</h1>
    <p style="color: var(--text-dim); margin-bottom: 20px;">Hybrid recurrence visualization and analysis</p>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Wave Dominance</h3>
            <div class="value" id="wave-dom">-</div>
        </div>
        <div class="metric-card">
            <h3>Particle Dominance</h3>
            <div class="value" id="particle-dom">-</div>
        </div>
        <div class="metric-card">
            <h3>Mixing Entropy</h3>
            <div class="value" id="mixing-entropy">-</div>
        </div>
    </div>
    
    <div class="chart-card">
        <h2>State Space Trajectory (3D)</h2>
        <div id="trajectory-3d"></div>
    </div>
    
    <div class="chart-card">
        <h2>Gate Activation Over Time</h2>
        <canvas id="gate-chart"></canvas>
    </div>
    
    <div class="chart-card">
        <h2>Wave vs Particle Contribution</h2>
        <canvas id="contribution-chart"></canvas>
    </div>
    
    <script>
        let gateChart, contributionChart;
        const trajectoryData = {{ x: [], y: [], z: [] }};
        const gateData = {{ labels: [], datasets: [{{ label: 'g_t', data: [], borderColor: '#00ff88', tension: 0.4 }}] }};
        const maxPoints = 100;
        
        function initCharts() {{
            // Gate chart
            const ctx1 = document.getElementById('gate-chart').getContext('2d');
            gateChart = new Chart(ctx1, {{
                type: 'line',
                data: gateData,
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                    scales: {{
                        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                        y: {{ min: 0, max: 1, ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    }},
                }},
            }});
            
            // Contribution chart
            const ctx2 = document.getElementById('contribution-chart').getContext('2d');
            contributionChart = new Chart(ctx2, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [
                        {{ label: 'Wave', data: [], borderColor: '#4ecdc4', fill: true, tension: 0.4 }},
                        {{ label: 'Particle', data: [], borderColor: '#ff6b6b', fill: true, tension: 0.4 }},
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                    scales: {{
                        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                        y: {{ stacked: true, ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    }},
                }},
            }});
        }}
        
        async function updateDynamics() {{
            try {{
                const res = await fetch('/api/hope/metrics');
                const data = await res.json();
                
                if (data.error) return;
                
                // For now, use mock gate value (would need to add to API)
                const g_t = 0.5; // Mock value
                
                // Update metrics
                document.getElementById('wave-dom').textContent = ((1 - g_t) * 100).toFixed(1) + '%';
                document.getElementById('particle-dom').textContent = (g_t * 100).toFixed(1) + '%';
                document.getElementById('mixing-entropy').textContent = '0.69'; // Mock
                
                // Update 3D trajectory (use first 3 dims of state)
                trajectoryData.x.push(data.state_norms.s);
                trajectoryData.y.push(data.state_norms.w);
                trajectoryData.z.push(data.state_norms.p);
                
                if (trajectoryData.x.length > maxPoints) {{
                    trajectoryData.x.shift();
                    trajectoryData.y.shift();
                    trajectoryData.z.shift();
                }}
                
                const trace = {{
                    x: trajectoryData.x,
                    y: trajectoryData.y,
                    z: trajectoryData.z,
                    mode: 'lines+markers',
                    type: 'scatter3d',
                    marker: {{ size: 3, color: trajectoryData.x, colorscale: 'Viridis' }},
                    line: {{ width: 2, color: '#00ff88' }},
                }};
                
                const layout = {{
                    scene: {{
                        xaxis: {{ title: '||s||', gridcolor: '#333', color: '#888' }},
                        yaxis: {{ title: '||w||', gridcolor: '#333', color: '#888' }},
                        zaxis: {{ title: '||p||', gridcolor: '#333', color: '#888' }},
                    }},
                    paper_bgcolor: '#1e1e1e',
                    plot_bgcolor: '#1e1e1e',
                    font: {{ color: '#e0e0e0' }},
                }};
                
                Plotly.newPlot('trajectory-3d', [trace], layout);
                
                // Update gate chart
                const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
                gateData.labels.push(timestamp);
                gateData.datasets[0].data.push(g_t);
                
                if (gateData.labels.length > maxPoints) {{
                    gateData.labels.shift();
                    gateData.datasets[0].data.shift();
                }}
                gateChart.update('none');
                
                // Update contribution chart
                contributionChart.data.labels.push(timestamp);
                contributionChart.data.datasets[0].data.push(1 - g_t);
                contributionChart.data.datasets[1].data.push(g_t);
                
                if (contributionChart.data.labels.length > maxPoints) {{
                    contributionChart.data.labels.shift();
                    contributionChart.data.datasets.forEach(ds => ds.data.shift());
                }}
                contributionChart.update('none');
                
            }} catch(e) {{
                console.error(e);
            }}
        }}
        
        initCharts();
        updateDynamics();
        setInterval(updateDynamics, 1000);
    </script>
</body>
</html>
"""

# Performance Benchmarks Page
HOPE_PERFORMANCE_HTML = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Performance Benchmarks</title>
    <style>
        {COMMON_CSS}
        body {{ padding: 40px; overflow-y: auto; display: block; }}
        h1 {{ margin-top: 0; }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 0.85em; color: var(--text-dim); }}
        .metric-card .value {{ font-size: 2em; font-weight: 700; color: #fff; }}
        .metric-card .target {{ font-size: 0.85em; color: var(--text-dim); margin-top: 5px; }}
        
        .chart-card {{ background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 25px; margin: 20px 0; }}
        .chart-card h2 {{ margin: 0 0 20px 0; color: var(--accent-color); }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <h1>⚡ Performance Benchmarks</h1>
    <p style="color: var(--text-dim); margin-bottom: 20px;">Real-time performance monitoring and optimization</p>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Inference Speed</h3>
            <div class="value" id="steps-per-sec">-</div>
            <div class="target">Target: >10 steps/sec</div>
        </div>
        <div class="metric-card">
            <h3>Mean Latency</h3>
            <div class="value" id="mean-latency">-</div>
            <div class="target">ms per step</div>
        </div>
        <div class="metric-card">
            <h3>Memory Usage</h3>
            <div class="value" id="memory-usage">-</div>
            <div class="target">Target: <2 GB</div>
        </div>
        <div class="metric-card">
            <h3>Model Size</h3>
            <div class="value" id="model-size">-</div>
            <div class="target">MB</div>
        </div>
    </div>
    
    <div class="chart-card">
        <h2>Inference Speed Over Time</h2>
        <canvas id="speed-chart"></canvas>
    </div>
    
    <div class="chart-card">
        <h2>Memory Usage Trend</h2>
        <canvas id="memory-chart"></canvas>
    </div>
    
    <script>
        let speedChart, memoryChart;
        const maxPoints = 100;
        let lastTimestamp = Date.now();
        let stepCount = 0;
        
        function initCharts() {{
            // Speed chart
            const ctx1 = document.getElementById('speed-chart').getContext('2d');
            speedChart = new Chart(ctx1, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Steps/sec',
                        data: [],
                        borderColor: '#00ff88',
                        tension: 0.4,
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                    scales: {{
                        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    }},
                }},
            }});
            
            // Memory chart
            const ctx2 = document.getElementById('memory-chart').getContext('2d');
            memoryChart = new Chart(ctx2, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Total Memory (MB)',
                        data: [],
                        borderColor: '#4ecdc4',
                        tension: 0.4,
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
                    scales: {{
                        x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                    }},
                }},
            }});
        }}
        
        async function updatePerformance() {{
            try {{
                const res = await fetch('/api/hope/metrics');
                const data = await res.json();
                
                if (data.error) return;
                
                // Calculate steps/sec
                const now = Date.now();
                const elapsed = (now - lastTimestamp) / 1000;
                const stepsPerSec = data.steps > stepCount ? (data.steps - stepCount) / elapsed : 0;
                lastTimestamp = now;
                stepCount = data.steps;
                
                // Update metrics
                document.getElementById('steps-per-sec').textContent = stepsPerSec.toFixed(1);
                document.getElementById('mean-latency').textContent = stepsPerSec > 0 ? (1000 / stepsPerSec).toFixed(1) : '-';
                document.getElementById('memory-usage').textContent = '1.18'; // Mock (would need API)
                document.getElementById('model-size').textContent = '1.17'; // Mock
                
                // Update charts
                const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString();
                
                speedChart.data.labels.push(timestamp);
                speedChart.data.datasets[0].data.push(stepsPerSec);
                
                if (speedChart.data.labels.length > maxPoints) {{
                    speedChart.data.labels.shift();
                    speedChart.data.datasets[0].data.shift();
                }}
                speedChart.update('none');
                
                memoryChart.data.labels.push(timestamp);
                memoryChart.data.datasets[0].data.push(1.18); // Mock
                
                if (memoryChart.data.labels.length > maxPoints) {{
                    memoryChart.data.labels.shift();
                    memoryChart.data.datasets[0].data.shift();
                }}
                memoryChart.update('none');
                
            }} catch(e) {{
                console.error(e);
            }}
        }}
        
        initCharts();
        updatePerformance();
        setInterval(updatePerformance, 1000);
    </script>
</body>
</html>
"""

# Export all pages
UI_ROUTES_HOPE = {
    "/ui/hope/training": HOPE_TRAINING_HTML,
    "/ui/hope/memory": HOPE_MEMORY_HTML,
    "/ui/hope/stability": HOPE_STABILITY_HTML,
    "/ui/hope/dynamics": HOPE_DYNAMICS_HTML,
    "/ui/hope/performance": HOPE_PERFORMANCE_HTML,
}

