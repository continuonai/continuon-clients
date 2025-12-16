
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

function testManualTraining() {
    fetch('/api/training/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "force_mock_train": true, "episodes": 1 })
    })
        .then(response => response.json())
        .then(data => {
            alert('Manual Training Test Completed: ' + JSON.stringify(data));
            refreshMetrics();
        })
        .catch(err => alert('Error running manual training test: ' + err));
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

let selectedLogPath = null;
let logRefreshIntervalId = null;
const LOG_TAIL_LINES = 200;
const LOG_REFRESH_MS = 5000;

function setLogStatus(message) {
    const logContent = document.getElementById('training-logs');
    if (logContent) {
        logContent.innerText = message;
    }
}

function renderLogList(logs) {
    const listEl = document.getElementById('training-log-list');
    if (!listEl) {
        return;
    }

    listEl.innerHTML = '';

    if (!Array.isArray(logs) || logs.length === 0) {
        selectedLogPath = null;
        listEl.innerText = 'No logs found.';
        setLogStatus('Waiting for logs...');
        stopLogAutoRefresh();
        return;
    }

    logs.forEach((log) => {
        const button = document.createElement('button');
        button.textContent = `${new Date(log.mtime * 1000).toLocaleString()}\n${log.path}`;
        button.style.display = 'block';
        button.style.width = '100%';
        button.style.textAlign = 'left';
        button.style.marginBottom = '6px';
        button.style.fontFamily = 'monospace';
        button.style.whiteSpace = 'pre-wrap';
        button.className = selectedLogPath === log.path ? 'rail-btn primary' : 'rail-btn';
        button.onclick = () => selectLog(log.path);
        listEl.appendChild(button);
    });

    if (!selectedLogPath) {
        selectLog(logs[0].path);
    }
}

function refreshLogs() {
    fetch('/api/training/logs')
        .then(response => response.json())
        .then(data => {
            renderLogList(data);
        })
        .catch(err => {
            setLogStatus('Error fetching log list: ' + err);
        });
}

function selectLog(path) {
    selectedLogPath = path;
    refreshLogs();
    fetchSelectedLogTail();
    startLogAutoRefresh();
}

function fetchSelectedLogTail() {
    if (!selectedLogPath) {
        setLogStatus('Select a log to view content.');
        return;
    }

    const url = `/api/training/logs/tail?path=${encodeURIComponent(selectedLogPath)}&lines=${LOG_TAIL_LINES}`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'ok') {
                setLogStatus('Error: ' + (data.message || 'Unable to read log'));
                return;
            }
            const header = `Tailing ${data.path} (last ${data.lines} lines)\n\n`;
            setLogStatus(header + (data.content || ''));
            const contentEl = document.getElementById('training-logs');
            if (contentEl) {
                contentEl.scrollTop = contentEl.scrollHeight;
            }
        })
        .catch(err => {
            setLogStatus('Error fetching log tail: ' + err);
        });
}

function startLogAutoRefresh() {
    if (logRefreshIntervalId) {
        return;
    }
    logRefreshIntervalId = setInterval(fetchSelectedLogTail, LOG_REFRESH_MS);
}

function stopLogAutoRefresh() {
    if (logRefreshIntervalId) {
        clearInterval(logRefreshIntervalId);
        logRefreshIntervalId = null;
    }
}

// Auto-refresh on load
document.addEventListener('DOMContentLoaded', function () {
    refreshMetrics();
    refreshLogs();
});
