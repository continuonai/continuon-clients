
function startAutonomousTraining() {
    fetch('/api/training/run', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert('Autonomous Training Started: ' + JSON.stringify(data));
            refreshMetrics();
        })
        .catch(err => alert('Error starting training: ' + err));
}

function startManualTraining() {
    fetch('/api/training/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "episodes": 1 })
    })
        .then(response => response.json())
        .then(data => {
            alert('Manual Training Step Completed: ' + JSON.stringify(data));
            refreshMetrics();
        })
        .catch(err => alert('Error running manual training: ' + err));
}

function stopTraining() {
    // There isn't an explicit "Stop" endpoint exposed in routes.py yet for the background process,
    // but usually 'switching mode' or similar might help. For now we will just log it.
    // Ideally we would POST to /api/mode/manual_control to stop autonomous loops.
    fetch('/api/training/control/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "mode": "manual_control" })
    })
        .then(res => res.json())
        .then(data => {
            alert('Switched to Manual Control (Stopping Training Loop): ' + JSON.stringify(data));
        });
}

function runWaveCoreLoop(speed) {
    let payload = {};
    // Configure to run ONLY the requested loop type if possible, or all if defaults.
    // Based on WavecoreTrainer logic, it runs all 3. We can try to minimize others by setting steps=0?
    // Let's try to just emphasize the one we want.
    if (speed === 'fast') {
        payload = { "fast": { "max_steps": 20 }, "mid": { "max_steps": 0 }, "slow": { "max_steps": 0 } };
    } else if (speed === 'mid') {
        payload = { "fast": { "max_steps": 0 }, "mid": { "max_steps": 20 }, "slow": { "max_steps": 0 } };
    } else if (speed === 'slow') {
        payload = { "fast": { "max_steps": 0 }, "mid": { "max_steps": 0 }, "slow": { "max_steps": 20 } };
    }

    fetch('/api/training/wavecore_loops', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
        .then(response => response.json())
        .then(data => {
            console.log('WaveCore Result:', data);
            alert('WaveCore ' + speed + ' loop finished. Check logs.');
            refreshMetrics(); // Status should update
        })
        .catch(err => alert('Error running WaveCore: ' + err));
}

function refreshMetrics() {
    fetch('/api/training/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('full-metrics-json').innerText = JSON.stringify(data, null, 2);
            document.getElementById('full-metrics-json').style.display = 'block';

            // Try to extract key metrics if available
            // Structure varies, but usually data.fast.result.metrics or similar
            if (data.fast && data.fast.result && data.fast.result.metrics) {
                const m = data.fast.result.metrics;
                document.getElementById('metric-loss').innerText = m.loss ? m.loss.toFixed(4) : '--';
                document.getElementById('metric-accuracy').innerText = m.accuracy ? (m.accuracy * 100).toFixed(1) + '%' : '--';
            }
            // If data is just from the status file, it might contain "trainer_status"
        })
        .catch(err => console.error('Error fetching metrics:', err));
}

function refreshLogs() {
    // For now, listing log files.
    fetch('/api/training/logs')
        .then(response => response.json())
        .then(data => {
            let logText = "Available Logs:\n" + data.map(f => f.path).join('\n');
            document.getElementById('training-logs').innerText = logText;
        })
        .catch(err => {
            document.getElementById('training-logs').innerText = 'Error fetching log list: ' + err;
        });
}

// Auto-refresh on load
document.addEventListener('DOMContentLoaded', function () {
    refreshMetrics();
    // Poll every 10 seconds?
    // setInterval(refreshMetrics, 10000);
});
