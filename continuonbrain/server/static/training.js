
const seedBundleState = {
    readiness: null,
    baseDir: "/opt/continuonos/brain",
};

const seedIterationPlan = [
    {
        id: 'collect-rlds',
        title: 'Capture + triage RLDS episodes',
        detail: 'Opt-in logging with PII-safe RLDS JSONL under /rlds/episodes; keep camera-only acceptable when PCA is down.',
        check: (r) => r?.rlds?.episodes_present === true || (typeof r?.rlds?.count === 'number' && r.rlds.count > 0),
    },
    {
        id: 'prime-small-corpora',
        title: 'Prime with small HF corpora (VLM/VLA/LM)',
        detail: 'Blend compact Hugging Face datasets into a TFRecord cache before local replay.',
        check: (r) => r?.seed?.tfrecord_cache_present === true || r?.seed?.tfrecord_ready === true,
    },
    {
        id: 'train-eval',
        title: 'Run iterative train + eval loops',
        detail: 'Manual step or WaveCore fast/mid/slow with HOPE eval episodes logging.',
        check: (r) => r?.seed?.checkpoint_exists === true || r?.trainer?.last_run_ok === true,
    },
    {
        id: 'export-install',
        title: 'Export bundle + reinstall on edge',
        detail: 'Build handoff zip, verify manifest, then install as candidate/core seed.',
        check: (r) => r?.ready_for_cloud_handoff === true,
    },
];

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
            const statusBadge = document.getElementById('training-status-badge');
            const state = data.state || data.status || 'unknown';
            statusBadge.innerText = state;
            document.getElementById('trainer-state').innerText = state;

            const steps = data.steps ?? data?.fast?.result?.steps ?? data?.fast?.steps;
            document.getElementById('metric-steps').innerText = steps !== undefined ? steps : '--';

            const lossValue = typeof data.avg_loss === 'number'
                ? data.avg_loss
                : data?.fast?.result?.metrics?.loss;
            document.getElementById('metric-loss').innerText = typeof lossValue === 'number'
                ? lossValue.toFixed(4)
                : '--';

            const adapterPath = data.adapter_path || data.adapterPath || '--';
            document.getElementById('adapter-path').innerText = adapterPath;

            document.getElementById('full-metrics-json').innerText = JSON.stringify(data, null, 2);
            document.getElementById('full-metrics-json').style.display = 'block';
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
    // refreshLogs(); // Removed to prevent unnecessary API call and re-render
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
    // Poll every 10 seconds?
    // setInterval(refreshMetrics, 10000);

    refreshSeedReadiness();
    listSeedExports();
});

/**
 * Derives the base directory from the payload.
 * Precedence: RLDS dir takes precedence over seed export dir.
 */
function deriveBaseDir(payload) {
    const rldsDir = payload?.rlds?.dir;
    if (typeof rldsDir === 'string' && rldsDir.includes('/rlds/episodes')) {
        // RLDS dir has highest precedence
        return rldsDir.split('/rlds/episodes')[0];
    }
    const seedExport = payload?.seed?.export_dir;
    if (typeof seedExport === 'string' && seedExport.includes('/model/adapters/candidate/core_model_seed')) {
        // Fallback to seed export dir if RLDS dir is not available
        return seedExport.split('/model/adapters/candidate/core_model_seed')[0];
    }
    // Default fallback
    return seedBundleState.baseDir;
}

function renderGates(gates) {
    if (!Array.isArray(gates) || gates.length === 0) {
        return '<div class="stack-item"><span class="stack-meta">No readiness gates reported.</span></div>';
    }
    return gates.map(g => {
        const ok = g?.ok === true;
        const chipClass = ok ? 'status-chip active' : 'status-chip warning';
        const chipText = ok ? 'OK' : 'BLOCKED';
        return (
            '<div class="stack-item">' +
            '<div>' +
            `<h4>${g?.name || 'gate'}</h4>` +
            `<div class="stack-meta">${g?.detail || ''}</div>` +
            '</div>' +
            `<div><span class="${chipClass}">${chipText}</span></div>` +
            '</div>'
        );
    }).join('');
}

function renderIterationPlan(readiness) {
    const planEl = document.getElementById('seed-iteration-plan');
    const chipEl = document.getElementById('seed-iteration-chip');
    if (!planEl) return;

    const items = seedIterationPlan.map((step) => {
        const ok = step?.check?.(readiness) === true;
        const chipClass = ok ? 'status-chip active' : 'status-chip warning';
        const chipText = ok ? 'OK' : 'PENDING';
        return (
            '<div class="stack-item">' +
            '<div>' +
            `<h4>${step.title}</h4>` +
            `<div class="stack-meta">${step.detail}</div>` +
            '</div>' +
            `<div><span class="${chipClass}">${chipText}</span></div>` +
            '</div>'
        );
    }).join('');

    planEl.innerHTML = items || '<div class="stack-item"><span class="stack-meta">Plan unavailable.</span></div>';

    if (chipEl) {
        const completed = seedIterationPlan.every((step) => step?.check?.(readiness) === true);
        const partial = seedIterationPlan.some((step) => step?.check?.(readiness) === true);
        chipEl.textContent = completed ? 'Loop ready' : partial ? 'In progress' : 'Staging';
        chipEl.className = `status-badge ${completed ? 'active' : partial ? '' : 'warning'}`;
    }
}

function applySeedPathHelper(value) {
    const pathInput = document.getElementById('cloud-install-path');
    if (!pathInput) return;
    const base = seedBundleState.baseDir || '/opt/continuonos/brain';
    if (value === 'candidate-manifest') {
        pathInput.value = `${base}/model/adapters/candidate/core_model_seed/model_manifest.json`;
    } else if (value === 'default-manifest') {
        pathInput.value = `${base}/model/manifest.pi5.example.json`;
    } else if (value === 'adapter-dir') {
        pathInput.value = `${base}/model/adapters/current`;
    }
    // Reset the helper dropdown to its default value
    const helperSelect = document.getElementById('cloud-install-path-helper');
    if (helperSelect) {
        helperSelect.value = "";
    }
}

async function refreshSeedReadiness() {
    const gatesEl = document.getElementById('seed-readiness-gates');
    const metaEl = document.getElementById('seed-readiness-meta');
    const chipEl = document.getElementById('seed-readiness-chip');
    if (gatesEl) {
        gatesEl.innerHTML = '<div class="stack-item"><span class="stack-meta">Running readiness check…</span></div>';
    }
    if (chipEl) {
        chipEl.textContent = 'Checking…';
        chipEl.className = 'status-badge';
    }
    if (metaEl) metaEl.textContent = '';
    try {
        const res = await fetch('/api/training/cloud_readiness');
        const data = await res.json();
        seedBundleState.readiness = data;
        seedBundleState.baseDir = deriveBaseDir(data);
        renderIterationPlan(data);
        if (gatesEl) {
            gatesEl.innerHTML = renderGates(data.gates);
        }
        if (chipEl) {
            const ok = data.ready_for_cloud_handoff === true;
            chipEl.textContent = ok ? 'Ready' : 'Blocked';
            chipEl.className = `status-badge ${ok ? 'active' : 'warning'}`;
        }
        if (metaEl) {
            const seeds = data.seed || {};
            const manifestStatus = seeds.manifest_exists ? 'found' : 'missing';
            metaEl.textContent = `Base dir ${seedBundleState.baseDir} · Seed manifest ${manifestStatus}`;
        }
    } catch (err) {
        console.error('refreshSeedReadiness failed', err);
        if (gatesEl) gatesEl.innerHTML = '<div class="stack-item"><span class="stack-meta">Readiness check failed.</span></div>';
        if (chipEl) {
            chipEl.textContent = 'Error';
            chipEl.className = 'status-badge warning';
        }
        if (metaEl) metaEl.textContent = 'Unable to read readiness; offline checks not updated.';
        renderIterationPlan(null);
    }
}

async function buildSeedExportZip() {
    const statusEl = document.getElementById('cloud-export-status');
    if (statusEl) statusEl.textContent = 'Building zip…';
    const include = {
        episodes: !!document.getElementById('cloud-export-episodes')?.checked,
        tfrecord: !!document.getElementById('cloud-export-tfrecord')?.checked,
        seed_export: !!document.getElementById('cloud-export-seed')?.checked,
        checkpoints: !!document.getElementById('cloud-export-checkpoints')?.checked,
        trainer_status: true,
    };
    const limitRaw = document.getElementById('cloud-export-episode-limit')?.value;
    const nameRaw = document.getElementById('seed-export-name')?.value;
    const episode_limit = limitRaw ? Number(limitRaw) : null;
    const name = nameRaw && nameRaw.trim() ? nameRaw.trim() : null;
    try {
        const res = await fetch('/api/training/export_zip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ include, episode_limit, name }),
        });
        const data = await res.json();
        if (!res.ok || data.status === 'error') throw new Error(data.message || 'export failed');
        if (statusEl) {
            statusEl.innerHTML = `Built <strong>${data.zip_name}</strong> (${data.size_bytes} bytes)`;
        }
        await listSeedExports();
    } catch (err) {
        console.warn('buildSeedExportZip failed', err);
        if (statusEl) statusEl.textContent = 'Export failed: ' + (err?.message || err);
    }
}

async function listSeedExports() {
    const listEl = document.getElementById('cloud-export-list');
    const statusEl = document.getElementById('cloud-export-status');
    if (statusEl && !statusEl.textContent) statusEl.textContent = 'Listing exports…';
    if (listEl) listEl.innerHTML = '';
    try {
        const res = await fetch('/api/training/exports');
        const data = await res.json();
        if (!res.ok || data.status === 'error') throw new Error(data.message || 'list failed');
        const items = Array.isArray(data.items) ? data.items : [];
        if (!items.length) {
            if (statusEl) statusEl.textContent = 'No exports yet.';
            if (listEl) listEl.innerHTML = '';
            return;
        }
        if (listEl) {
            listEl.innerHTML = items.map(item => {
                const ts = new Date(item.mtime * 1000).toLocaleString();
                return (
                    '<div class="stack-item">' +
                    `<div><div>${item.name}</div><div class="stack-meta">${item.size_bytes} bytes · ${ts}</div></div>` +
                    `<div><a class="rail-btn" href="${item.download_url}" target="_blank">Download</a></div>` +
                    '</div>'
                );
            }).join('');
        }
        if (statusEl) statusEl.textContent = '';
    } catch (err) {
        console.warn('listSeedExports failed', err);
        if (statusEl) statusEl.textContent = 'List failed: ' + (err?.message || err);
    }
}

async function installSeedBundle() {
    const statusEl = document.getElementById('cloud-install-status');
    if (statusEl) statusEl.textContent = 'Installing…';
    const kind = document.getElementById('cloud-install-kind')?.value || 'jax_seed_manifest';
    const source_url = document.getElementById('cloud-install-url')?.value?.trim();
    const source_path = document.getElementById('cloud-install-path')?.value?.trim();
    // Validate source_url if provided
    if (source_url) {
        let urlObj;
        try {
            urlObj = new URL(source_url);
        } catch (e) {
            if (statusEl) statusEl.textContent = 'Invalid URL format. Please enter a valid http(s) URL.';
            return;
        }
        if (urlObj.protocol !== 'http:' && urlObj.protocol !== 'https:') {
            if (statusEl) statusEl.textContent = 'Only http(s) URLs are allowed.';
            return;
        }
    }
    if (!source_url && !source_path) {
        if (statusEl) statusEl.textContent = 'Provide a source URL or local path to install.';
        return;
    }
    if (source_url && seedBundleState.readiness?.ready_for_cloud_handoff !== true) {
        if (statusEl) statusEl.textContent = 'External bundle downloads are blocked until the readiness check passes. Please verify local RLDS episodes and seed manifests exist, then re-run the readiness check.';
        return;
    }
    try {
        const res = await fetch('/api/training/install_bundle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kind, source_url: source_url || null, source_path: source_path || null }),
        });
        const data = await res.json();
        if (!res.ok || data.status === 'error') throw new Error(data.message || 'install failed');
        if (statusEl) {
            if (data.installed_to) {
                statusEl.textContent = `Installed to ${data.installed_to}`;
            } else if (data.edge_manifest) {
                statusEl.textContent = `Installed to ${data.edge_manifest}`;
            } else {
                statusEl.textContent = 'Installed, but the location is unknown. Please check the backend logs or contact support.';
            }
        }
        await refreshSeedReadiness();
        await listSeedExports();
    } catch (err) {
        console.warn('installSeedBundle failed', err);
        if (statusEl) statusEl.textContent = 'Install failed: ' + (err?.message || err);
    }
}

// Agent Chat Learning Test Functions

const chatLearnTestConfigs = {
    'basic': {
        turns: 5,
        model_hint: 'hope-v1',
        delegate_model_hint: null,
        topic: 'system architecture and learning mechanisms'
    },
    'hope-gemma': {
        turns: 6,
        model_hint: 'hope-v1',
        delegate_model_hint: 'google/gemma-370m',
        topic: 'CMS compaction and memory management'
    },
    'jax': {
        turns: 4,
        model_hint: 'hope-v1',
        delegate_model_hint: 'consult:google/gemma-370m',
        topic: 'JAX training pipeline and TPU deployment'
    },
    'extended': {
        turns: 10,
        model_hint: 'hope-v1',
        delegate_model_hint: 'google/gemma-3n-2b',
        topic: 'comprehensive system understanding and improvement'
    },
    'multi-topic': {
        turns: 3,
        model_hint: 'hope-v1',
        delegate_model_hint: 'google/gemma-370m',
        topic: 'safety policies and intervention'
    }
};

function updateChatLearnStatus(message, type = 'info') {
    const statusEl = document.getElementById('chat-learn-status');
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.className = `status-badge ${type === 'error' ? 'warning' : type === 'success' ? 'active' : ''}`;
    }
}

function appendChatLearnResult(message) {
    const resultsEl = document.getElementById('chat-learn-results');
    if (resultsEl) {
        const timestamp = new Date().toLocaleTimeString();
        resultsEl.textContent += `[${timestamp}] ${message}\n`;
        resultsEl.scrollTop = resultsEl.scrollHeight;
    }
}

async function runChatLearnTest(testType) {
    const config = chatLearnTestConfigs[testType];
    if (!config) {
        appendChatLearnResult(`ERROR: Unknown test type: ${testType}`);
        return;
    }
    
    updateChatLearnStatus('Running...', 'info');
    appendChatLearnResult(`Starting ${testType} test: ${config.topic}`);
    appendChatLearnResult(`  Turns: ${config.turns}, Model: ${config.model_hint}`);
    if (config.delegate_model_hint) {
        appendChatLearnResult(`  Delegate: ${config.delegate_model_hint}`);
    }
    
    try {
        const t0 = Date.now();
        const res = await fetch('/api/training/chat_learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                turns: config.turns,
                model_hint: config.model_hint,
                delegate_model_hint: config.delegate_model_hint || null,
                topic: config.topic,
                session_id: `test_${testType}_${Date.now()}`
            })
        });
        
        const dt = ((Date.now() - t0) / 1000).toFixed(2);
        const data = await res.json();
        
        // Handle wrapped response
        const result = data.result || data;
        
        if (res.ok && result.status !== 'error') {
            const history = result.history || [];
            const outputs = result.outputs || [];
            
            appendChatLearnResult(`✓ Completed in ${dt}s`);
            appendChatLearnResult(`  History: ${history.length} turns, Outputs: ${outputs.length}`);
            
            // Show conversation preview
            if (history.length > 0) {
                appendChatLearnResult(`  Conversation preview:`);
                history.slice(0, 3).forEach((turn, i) => {
                    const role = turn.role || 'unknown';
                    const content = (turn.content || '').substring(0, 100);
                    appendChatLearnResult(`    [${i+1}] ${role}: ${content}...`);
                });
                if (history.length > 3) {
                    appendChatLearnResult(`    ... (${history.length - 3} more turns)`);
                }
            }
            
            updateChatLearnStatus('Success', 'success');
        } else {
            const error = result.error || result.message || 'Unknown error';
            appendChatLearnResult(`✗ Failed: ${error}`);
            updateChatLearnStatus('Error', 'error');
        }
    } catch (err) {
        appendChatLearnResult(`✗ Exception: ${err.message}`);
        updateChatLearnStatus('Error', 'error');
        console.error('Chat learn test failed:', err);
    }
}

async function runChatLearnCustom() {
    const turns = parseInt(document.getElementById('chat-learn-turns')?.value || '5');
    const model_hint = document.getElementById('chat-learn-model')?.value || 'hope-v1';
    const delegate = document.getElementById('chat-learn-delegate')?.value || null;
    const topic = document.getElementById('chat-learn-topic')?.value || 'tool use + planning + safety';
    
    updateChatLearnStatus('Running...', 'info');
    appendChatLearnResult(`Starting custom test`);
    appendChatLearnResult(`  Turns: ${turns}, Model: ${model_hint}, Topic: ${topic}`);
    if (delegate) {
        appendChatLearnResult(`  Delegate: ${delegate}`);
    }
    
    try {
        const t0 = Date.now();
        const res = await fetch('/api/training/chat_learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                turns: Math.max(1, Math.min(50, turns)),
                model_hint: model_hint,
                delegate_model_hint: delegate || null,
                topic: topic,
                session_id: `test_custom_${Date.now()}`
            })
        });
        
        const dt = ((Date.now() - t0) / 1000).toFixed(2);
        const data = await res.json();
        const result = data.result || data;
        
        if (res.ok && result.status !== 'error') {
            const history = result.history || [];
            appendChatLearnResult(`✓ Completed in ${dt}s (${history.length} turns)`);
            updateChatLearnStatus('Success', 'success');
        } else {
            appendChatLearnResult(`✗ Failed: ${result.error || result.message || 'Unknown error'}`);
            updateChatLearnStatus('Error', 'error');
        }
    } catch (err) {
        appendChatLearnResult(`✗ Exception: ${err.message}`);
        updateChatLearnStatus('Error', 'error');
    }
}

async function checkChatLearnRLDS() {
    appendChatLearnResult('Checking RLDS logging status...');
    try {
        // Check if RLDS logging is enabled via settings
        const res = await fetch('/api/status');
        const data = await res.json();
        const chatSettings = data?.chat || {};
        const rldsEnabled = chatSettings.log_rlds === true;
        
        appendChatLearnResult(`RLDS Logging: ${rldsEnabled ? 'ENABLED' : 'DISABLED'}`);
        if (!rldsEnabled) {
            appendChatLearnResult('  Set CONTINUON_LOG_CHAT_RLDS=1 or chat.log_rlds=true to enable');
        } else {
            appendChatLearnResult('  Conversations will be logged to RLDS episodes directory');
        }
    } catch (err) {
        appendChatLearnResult(`✗ Error checking RLDS status: ${err.message}`);
    }
}
