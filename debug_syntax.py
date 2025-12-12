COMMON_CSS = 'css'
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
