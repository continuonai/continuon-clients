// Tab switching for Home Page
window.switchHomeTab = function (tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.tab === tabName) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    document.querySelectorAll('.home-panel').forEach(panel => {
        if (panel.id === tabName + '-panel') panel.style.display = 'block';
        else panel.style.display = 'none';
    });
};
// Global functions for onclick handlers
window.showMessage = function (message, isError) {
    if (typeof isError === 'undefined') { isError = false; }
    var msgDiv = document.getElementById('status-message');
    msgDiv.textContent = message;
    msgDiv.style.display = 'block';
    msgDiv.style.background = isError ? '#ff3b30' : '#34c759';
    msgDiv.style.color = 'white';
    msgDiv.style.textAlign = 'center';
    setTimeout(function () {
        msgDiv.style.display = 'none';
    }, 3000);
};

// NEW: Action Logger
window.logAction = function (msg) {
    const logContainer = document.getElementById('logger-content');
    if (logContainer) {
        const line = document.createElement('div');
        line.className = 'log-line action';
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logContainer.insertBefore(line, logContainer.firstChild); // Prepend for reverse order
        // Limit history
        if (logContainer.children.length > 100) {
            logContainer.removeChild(logContainer.lastChild);
        }
    } else {
        // Fallback for missing container
        console.log('[Action Log]', msg);
    }
}

// Auto-attach logger to buttons
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
        const btn = e.target.tagName === 'BUTTON' ? e.target : e.target.closest('button');
        const label = btn.innerText || btn.title || 'Button';
        // Avoid logging chat send/mic as generic clicks if handled elsewhere, but fine for now
        window.logAction(`Clicked: ${label}`);
    }
});

// --- Settings Inline Logic ---
const settingsFormInline = document.getElementById('settings-form-inline');
const settingsStatusInline = document.getElementById('settings-status-inline');

function setSettingsStatus(message, isError) {
    if (!settingsStatusInline) return;
    settingsStatusInline.textContent = message;
    settingsStatusInline.style.color = isError ? '#ff7b7b' : 'var(--muted)';
}

function populateSettingsForm(settings) {
    const context = settings?.context || 'hybrid';
    const radios = document.getElementsByName('runtime-context-inline');
    for (const r of radios) {
        if (r.value === context) r.checked = true;
    }

    if (document.getElementById('chat-persona-inline'))
        document.getElementById('chat-persona-inline').value = settings?.chat?.persona ?? 'operator';
    if (document.getElementById('chat-temperature-inline'))
        document.getElementById('chat-temperature-inline').value = settings?.chat?.temperature ?? 0.35;
}

async function fetchSettings() {
    const response = await fetch('/api/settings');
    if (!response.ok) throw new Error('Server responded with ' + response.status);
    const payload = await response.json();
    if (!payload.success) throw new Error(payload.message || 'Unable to load settings');
    return payload.settings || {};
}

// Alias for existing calls, redirects to view mode
window.openSettingsModal = async function () {
    window.setViewMode('settings'); // Switch view

    setSettingsStatus('Loading current settings...');
    try {
        const settings = await fetchSettings();
        populateSettingsForm(settings);
        setSettingsStatus('');
    } catch (err) {
        console.error(err);
        setSettingsStatus(err.message || 'Failed to load settings', true);
    }
};

window.closeSettingsModal = function () {
    // Just switch back to owner view
    window.setViewMode('owner');
};

settingsFormInline?.addEventListener('submit', async function (event) {
    event.preventDefault();

    // Determine context
    let context = 'hybrid';
    const radios = document.getElementsByName('runtime-context-inline');
    for (const r of radios) { if (r.checked) context = r.value; }

    const isTraining = context === 'training';
    const isInference = context === 'inference';
    const isHybrid = context === 'hybrid';

    const persona = document.getElementById('chat-persona-inline')?.value || 'operator';
    const temp = parseFloat(document.getElementById('chat-temperature-inline')?.value || '0.35');

    // Construct payload
    const payload = {
        context: context,
        safety: {
            allow_motion: true,
            record_episodes: isTraining || isHybrid,
            require_supervision: isInference ? false : true,
        },
        telemetry: { rate_hz: isTraining ? 5.0 : 2.0 },
        chat: {
            persona: persona,
            temperature: temp,
        }
    };

    setSettingsStatus('Saving...');
    window.logAction(`Saving settings: ${context.toUpperCase()}`);

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.message || 'Save failed');
        }

        populateSettingsForm(result.settings || payload);
        setSettingsStatus('Saved successfully.');
        window.showMessage(`Settings updated`);
    } catch (err) {
        console.error('Save failed', err);
        setSettingsStatus(err.message || 'Unable to save settings', true);
    }
});

const agentManagerState = {
    humanMode: false,
    agents: [
        { name: 'Safety Guardian', status: 'active', threads: 2, focus: 'Gates + envelopes', milestone: 'Monitoring' },
        { name: 'Navigator', status: 'active', threads: 1, focus: 'Drive pathfinding', milestone: 'Updating map' },
        { name: 'Trainer', status: 'paused', threads: 1, focus: 'Self-learning batches', milestone: 'Queued' },
    ],
    modelStack: {
        primary: { name: 'HOPE model', role: 'Default Agent Manager', status: 'active', latency: '80 ms', accelerator: 'JAX/CPU' },
        fallbacks: [
            { name: 'Gemma 3n', role: 'On-device fallback', status: 'ready', latency: '120 ms', accelerator: 'CPU/GPU auto' },
            { name: 'Cloud LLM', role: 'Long-context fallback', status: 'standby', latency: 'network' },
        ],
    },
    toolchain: [
        { name: 'Safety tools', detail: 'Gates, envelopes, e-stop intents', status: 'ready' },
        { name: 'RLDS curator', detail: 'Episode triage + tagging', status: 'ready' },
        { name: 'Skill linker', detail: 'Chains task to reusable skills', status: 'active' },
    ],
    trainingSummary: {
        episodes: 24,
        pendingImports: 3,
        localMemories: 128,
        lastRun: '12m ago',
        queueState: 'Idle',
    },
    learningEvents: [
        { title: 'Episode flagged for replay', time: '2m ago', detail: 'Marked safe drive lane for offline finetune' },
        { title: 'Gate alignment saved', time: '8m ago', detail: 'Motion + recording gates synced to idle baseline' },
        { title: 'Autonomy pulse', time: '12m ago', detail: 'Wave/particle balance stable; buffer widened' },
    ],
};

const safetyPersonaPresets = [
    {
        id: 'owner_safe',
        label: 'Owner',
        apiMode: 'manual_control',
        fallbackMode: 'manual_control',
        description: 'Primary owners with full controls; telemetry pinned.',
        gates: { allow_motion: true, record_episodes: true, require_supervision: false },
    },
    {
        id: 'renter_safe',
        label: 'Renter / Leasee',
        apiMode: 'autonomous',
        fallbackMode: 'autonomous',
        description: 'Temporary operators with supervised motion and recorded evidence.',
        gates: { allow_motion: true, record_episodes: true, require_supervision: true },
    },
    {
        id: 'enterprise_maintenance',
        label: 'Enterprise Maintenance',
        apiMode: 'sleep_learning',
        fallbackMode: 'sleep_learning',
        description: 'Technicians with diagnostics on and motion held unless cleared.',
        gates: { allow_motion: false, record_episodes: true, require_supervision: true },
    },
    {
        id: 'researcher',
        label: 'Researcher',
        apiMode: 'autonomous',
        fallbackMode: 'autonomous',
        description: 'Research sandboxes with cautious motion and logged changes.',
        gates: { allow_motion: true, record_episodes: true, require_supervision: true },
    },
    {
        id: 'creator',
        label: 'Creator',
        apiMode: 'idle',
        fallbackMode: 'idle',
        description: 'System creator with design-time control; motion starts locked.',
        gates: { allow_motion: false, record_episodes: false, require_supervision: false },
    },
];

const safetyPersonaState = { selected: null, gates: {} };

const taskDeckState = {
    selectedGroup: 'all',
    expandedDetails: {},
    textFilter: '',
};

let taskLibraryPayload = null;
const skillDeckState = {
    textFilter: '',
    skills: [],
};
let skillLibraryPayload = null;

const realityProofState = { status: null, loops: null, surprises: 0 };

function setProofText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function updateRealityProof() {
    const status = realityProofState.status || {};
    const loopsWrapper = realityProofState.loops || {};
    const loops = loopsWrapper.loop_metrics || loopsWrapper.metrics || loopsWrapper || {};
    const hopeLoops = loops.hope_loops || {};
    const fast = hopeLoops.fast || loops.fast || {};
    const hz = fast.hz || 0;

    const tokens = hz ? Math.round(hz * 16) : null; // rough est: tokens per frame * hz
    setProofText('visual-token-rate', tokens ? `${tokens} tok/s (est)` : '—');
    setProofText('temporal-resolution', hz ? `${Math.round(1000 / hz)} ms/frame` : '—');
    const latency = fast.latency_ms || null;
    setProofText('dream-latency', latency ? `${latency} ms` : '—');

    const accel = (status.detected_hardware && status.detected_hardware.primary && status.detected_hardware.primary.ai_accelerator) ||
        (status.capabilities && status.capabilities.has_ai_accelerator && 'AI accelerator') ||
        'CPU/JAX fallback';
    setProofText('compute-budget', accel || 'Unknown');

    const drift = (status.learning && (status.learning.recent_surprise || status.learning.surprise)) || null;
    setProofText('reality-drift', drift != null ? drift : 'Awaiting reality check');

    setProofText('context-updates', realityProofState.surprises > 0 ? `${realityProofState.surprises} surprise events logged` : 'No surprises yet');
    setProofText('surprise-counter-chip', `Surprises: ${realityProofState.surprises}`);

    setProofText('proof-vision-line', status.capabilities && status.capabilities.has_vision ? 'Vision tokens streaming' : 'Waiting for vision…');
    setProofText('proof-dream-line', status.model_stack ? 'Model stack active' : 'Model stack not detected');
    setProofText('proof-reality-line', status.allow_motion ? 'Reality checks live (motion allowed)' : 'Reality checks pending motion');
    setProofText('proof-context-line', realityProofState.surprises > 0 ? 'Context updated after surprises' : 'No context updates logged');
}

window.logRealitySurprise = function () {
    realityProofState.surprises += 1;
    const log = document.getElementById('surprise-log');
    const entry = `<div class="surprise-row">Surprise @ ${new Date().toLocaleTimeString()}</div>`;
    if (log) {
        log.innerHTML = entry + log.innerHTML;
    }
    updateRealityProof();
}

function renderAgentChips(agents) {
    const chipRow = document.getElementById('agent-chip-row');
    if (!chipRow) return;

    chipRow.innerHTML = agents.map(agent => {
        const normalizedStatus = (agent.status || 'active').toLowerCase();
        return `<span class="status-chip ${normalizedStatus}"><span class="badge-dot"></span>${agent.name} — ${agent.threads || 1} thread${(agent.threads || 1) > 1 ? 's' : ''}</span>`;
    }).join('');
}

function renderModelStack(status) {
    const container = document.getElementById('model-stack-list');
    if (!container) return;

    const stack = (status && status.model_stack) || agentManagerState.modelStack || {};
    const primary = stack.primary || { name: 'Gemma 3n Agent Manager', role: 'On-device', status: 'active', latency: '—' };
    const fallbacks = Array.isArray(stack.fallbacks) ? stack.fallbacks : [];
    const items = [
        { ...primary, primary: true },
        ...fallbacks,
    ];

    if (!items.length) {
        container.innerHTML = '<div class="stack-item"><span class="stack-meta">No models detected</span></div>';
        return;
    }

    container.innerHTML = items.map(item => {
        const statusChip = `<span class="status-chip ${item.status || 'active'}">${(item.status || '').toUpperCase() || 'ACTIVE'}</span>`;
        const flags = [item.role, item.latency, item.accelerator].filter(Boolean).map(flag => `<span class="task-tag">${flag}</span>`).join('');
        return `<div class="stack-item ${item.primary ? 'primary' : ''}">` +
            `<div>` +
            `<h4>${item.name || 'Model'}</h4>` +
            `<div class="stack-meta">${item.description || 'Primary + fallback routing for Agent Manager'}</div>` +
            (flags ? `<div class="stack-flags">${flags}</div>` : '') +
            `</div>` +
            `<div>${statusChip}</div>` +
            `</div>`;
    }).join('');
}

function renderToolchain(status) {
    const container = document.getElementById('toolchain-list');
    if (!container) return;

    const tools = (status && status.toolchain) || agentManagerState.toolchain || [];
    if (!tools.length) {
        container.innerHTML = '<div class="stack-item"><span class="stack-meta">No sub-agents/tools registered</span></div>';
        return;
    }

    container.innerHTML = tools.map(tool => {
        const statusChip = `<span class="status-chip ${tool.status || 'info'}">${(tool.status || 'info').toUpperCase()}</span>`;
        const flags = [tool.detail, tool.scope].filter(Boolean).map(flag => `<span class="task-tag">${flag}</span>`).join('');
        return `<div class="stack-item">` +
            `<div>` +
            `<h4>${tool.name || 'Tool'}</h4>` +
            `<div class="stack-meta">${tool.detail || 'Sub-agent or tool connector'}</div>` +
            (flags ? `<div class="stack-flags">${flags}</div>` : '') +
            `</div>` +
            `<div>${statusChip}</div>` +
            `</div>`;
    }).join('');
}

function renderTrainingSummary(status) {
    const container = document.getElementById('training-memory-list');
    if (!container) return;

    const summary = (status && status.training) || agentManagerState.trainingSummary || {};
    const metrics = [
        { label: 'RLDS episodes', value: summary.episodes ?? '—' },
        { label: 'Pending imports', value: summary.pendingImports ?? '—' },
        { label: 'Local memories', value: summary.localMemories ?? '—' },
        { label: 'Last run', value: summary.lastRun || 'n/a' },
        { label: 'Queue state', value: summary.queueState || 'Idle' },
    ];

    container.innerHTML = metrics.map(metric => {
        return `<div class="metric-row">` +
            `<div class="metric-label">${metric.label}</div>` +
            `<div class="metric-value">${metric.value}</div>` +
            `</div>`;
    }).join('');
}

function renderGateStatus(gates) {
    const grid = document.getElementById('gate-status-grid');
    if (!grid) return;

    const snapshot = gates || safetyPersonaState.gates || {};
    safetyPersonaState.gates = snapshot;

    const rows = [
        { label: 'Motion gate', value: snapshot.allow_motion ? 'Open' : 'Locked', status: snapshot.allow_motion },
        { label: 'Recording', value: snapshot.record_episodes ? 'Recording' : 'Stopped', status: snapshot.record_episodes },
        { label: 'Supervision', value: snapshot.require_supervision ? 'Human in loop' : 'Autonomy allowed', status: !snapshot.require_supervision },
    ];

    grid.innerHTML = rows.map(row => {
        const statusClass = row.status ? 'status-value status-good' : 'status-value';
        return `<div class="status-item">` +
            `<span class="status-label">${row.label}</span>` +
            `<span class="${statusClass}">${row.value}</span>` +
            `</div>`;
    }).join('');

    const trainingChip = document.getElementById('training-safe-chip');
    if (trainingChip) {
        const label = snapshot.require_supervision ? 'Supervised safe mode' : 'Autonomy cleared';
        trainingChip.textContent = label;
    }
}

function renderSafeModeGrid(selectedId) {
    const grid = document.getElementById('safe-mode-grid');
    if (!grid) return;

    grid.innerHTML = safetyPersonaPresets.map(preset => {
        const selected = preset.id === selectedId;
        const gates = preset.gates || {};
        const gateLine = `${gates.allow_motion ? 'Motion allowed' : 'Motion held'} • ${gates.record_episodes ? 'Recording on' : 'Recording off'} • ${gates.require_supervision ? 'Supervision required' : 'Autonomy ok'}`;
        return `<div class="safe-mode-card ${selected ? 'selected' : ''}">` +
            `<div class="panel-eyebrow">${preset.label}</div>` +
            `<div class="safe-mode-meta">${preset.description}</div>` +
            `<div class="status-chip info">${gateLine}</div>` +
            `<button class="command-btn" onclick="window.setSafeModePreset('${preset.id}')">${selected ? 'Active' : 'Set safe mode'}</button>` +
            `</div>`;
    }).join('');

    const statusChip = document.getElementById('safe-mode-status');
    if (statusChip) {
        const active = safetyPersonaPresets.find(p => p.id === selectedId);
        statusChip.textContent = active ? `${active.label} guardrails loaded` : 'Persona guardrails synced';
    }
}

window.setSafeModePreset = async function (presetId) {
    const preset = safetyPersonaPresets.find(p => p.id === presetId);
    const statusChip = document.getElementById('safe-mode-status');
    if (!preset) return;

    if (statusChip) statusChip.textContent = `Applying ${preset.label} safeguards...`;

    try {
        const mode = preset.apiMode || preset.fallbackMode || 'idle';
        const res = await fetch(`/api/mode/${mode}`);
        if (!res.ok && preset.fallbackMode && preset.fallbackMode !== mode) {
            await fetch(`/api/mode/${preset.fallbackMode}`);
        }
        safetyPersonaState.selected = preset.id;
        const personaBadge = document.getElementById('persona-badge');
        if (personaBadge) personaBadge.textContent = `Persona: ${preset.label}`;
        renderSafeModeGrid(preset.id);
        renderGateStatus(preset.gates);
        if (statusChip) statusChip.textContent = `${preset.label} safe mode ready`;
        window.showMessage(`${preset.label} guardrails applied`);
    } catch (err) {
        console.warn('Safe mode failed', err);
        if (statusChip) statusChip.textContent = 'Safe mode failed';
        window.showMessage('Unable to set safe mode', true);
    }

    window.fetchGates();
    window.updateStatus();
}

window.applyGatesPayload = function (payload) {
    if (!payload) return;
    renderGateStatus(payload);
}

window.fetchGates = async function (payload) {
    if (payload) {
        window.applyGatesPayload(payload);
        return;
    }
    try {
        const res = await fetch('/api/gates');
        if (!res.ok) return;
        const data = await res.json();
        window.applyGatesPayload(data);
    } catch (err) {
        console.warn('Gate fetch failed', err);
    }
}

function renderAgentThreads(agents) {
    const list = document.getElementById('agent-thread-list');
    if (!list) return;

    if (!agents || !agents.length) {
        list.innerHTML = '<div class="agent-thread"><span class="agent-meta">No active agents</span></div>';
        return;
    }

    list.innerHTML = agents.map(agent => {
        const normalizedStatus = (agent.status || 'active').toLowerCase();
        const chip = `<span class="status-chip ${normalizedStatus}">${normalizedStatus === 'active' ? '🟢' : '⏸️'} ${normalizedStatus}</span>`;
        return `<div class="agent-thread">` +
            `<div>` +
            `<h4>${agent.name || 'Agent'}</h4>` +
            `<div class="agent-meta">${agent.focus || 'Monitoring'} • ${agent.threads || 1} thread${(agent.threads || 1) > 1 ? 's' : ''}</div>` +
            `</div>` +
            `<div>${chip}</div>` +
            `</div>`;
    }).join('');
}

function renderLearningMilestones(status) {
    const container = document.getElementById('learning-milestones');
    if (!container) return;

    const gateSnapshot = status.gate_snapshot || status.gates || {};
    const modeLabel = (status.mode || 'idle').replace(/_/g, ' ');
    const uptime = gateSnapshot.mode_uptime_seconds || 0;
    const progress = Math.min(100, Math.round((uptime % 180) / 180 * 100));

    const milestones = [
        { label: 'Motion gate open', done: !!gateSnapshot.allow_motion, detail: gateSnapshot.allow_motion ? 'Ready to move' : 'Waiting for clearance' },
        { label: 'Recording allowed', done: !!gateSnapshot.record_episodes, detail: gateSnapshot.record_episodes ? 'Episodes gated in' : 'Recording gated off' },
        { label: 'Self-train enabled', done: !!gateSnapshot.self_train, detail: gateSnapshot.self_train ? 'Robot learning live' : 'Manual/human-in-loop' },
        { label: `${modeLabel} uptime`, done: progress > 40, detail: `${Math.round(uptime)}s in mode`, progress },
    ];

    container.innerHTML = milestones.map(item => {
        const chipClass = item.done ? 'status-chip active' : 'status-chip paused';
        const progressBar = typeof item.progress === 'number'
            ? `<div class="progress-bar"><div class="progress-fill" style="width:${item.progress}%"></div></div>`
            : '';
        return `<div class="milestone-row">` +
            `<div>` +
            `<div>${item.label}</div>` +
            `<div class="learning-meta">${item.detail}</div>` +
            `${progressBar}` +
            `</div>` +
            `<span class="${chipClass}">${item.done ? '✅' : '…'} ${item.done ? 'Complete' : 'Pending'}</span>` +
            `</div>`;
    }).join('');
}

function renderLearningEvents(events) {
    const list = document.getElementById('learning-events');
    if (!list) return;
    const items = events && events.length ? events : [{ title: 'No learning events yet', time: '', detail: 'Agents will surface new checkpoints here.' }];

    list.innerHTML = items.map(event => {
        return `<div class="learning-item">` +
            `<strong>${event.title}</strong>` +
            `<div class="learning-meta">${event.detail || 'Update pending'}</div>` +
            (event.time ? `<div class="learning-meta">${event.time}</div>` : '') +
            `</div>`;
    }).join('');
}

function renderSelectedTask(selectedTask) {
    const pill = document.getElementById('current-task-pill');
    if (!pill) return;

    if (!selectedTask || !selectedTask.entry) {
        pill.textContent = 'No task selected';
        pill.className = 'selected-task-pill warning';
        return;
    }

    const entry = selectedTask.entry;
    const eligible = entry.eligibility ? entry.eligibility.eligible : false;
    pill.textContent = `${entry.title || 'Task'} • ${entry.group || 'Task Library'}`;
    pill.className = 'selected-task-pill ' + (eligible ? 'success' : 'warning');
}

function renderTaskLibrary(payload) {
    const container = document.getElementById('task-groups');
    const controls = document.getElementById('task-group-segments');
    taskLibraryPayload = payload || null;
    if (!container) return;

    const tasks = (payload && payload.tasks) || [];
    const selectedId = payload ? payload.selected_task_id : null;
    if (!tasks.length) {
        container.innerHTML = '<div class="status-item"><span class="status-label">No tasks available</span></div>';
        if (controls) { controls.innerHTML = ''; }
        return;
    }

    const groups = {};
    const filterText = (taskDeckState.textFilter || '').trim();
    tasks.forEach(task => {
        const groupKey = task.group || 'Tasks';
        if (!groups[groupKey]) {
            groups[groupKey] = [];
        }
        if (!filterText || (task.title && task.title.toLowerCase().includes(filterText)) || (task.description && task.description.toLowerCase().includes(filterText))) {
            groups[groupKey].push(task);
        }
    });

    const groupEntries = Object.entries(groups);
    const groupLabels = groupEntries.map(([label]) => label);

    if (controls) {
        controls.innerHTML = ['all', ...groupLabels].map(label => {
            const active = taskDeckState.selectedGroup === label;
            const displayLabel = label === 'all' ? 'All' : label;
            return `<button class="${active ? 'active' : ''}" data-group="${label}" onclick="window.filterTaskGroup('${label}')">${displayLabel}</button>`;
        }).join('');
    }

    container.innerHTML = groupEntries.map(([groupLabel, entries]) => {
        const cards = entries.map(task => {
            const eligibility = task.eligibility || {};
            const markers = (eligibility.markers || []).map(marker => {
                const severity = marker.blocking ? 'blocking' : (marker.severity || 'info');
                const remediation = marker.remediation ? ` — ${marker.remediation}` : '';
                const label = marker.label || marker.code || 'marker';
                return `<div class="eligibility-marker ${severity}">${label}${remediation}</div>`;
            }).join('') || '<div class="eligibility-marker success">Eligible</div>';

            const tagRow = (task.tags || []).map(tag => `<span class="task-tag">${tag}</span>`).join('');
            const isSelected = selectedId && selectedId === task.id;
            const selectLabel = isSelected ? 'Selected' : 'Select task';
            const isEligible = eligibility.eligible !== false;
            const disabledAttr = isEligible ? '' : 'disabled';
            const highlightStyle = isSelected ? 'style="border-color: rgba(122,215,255,0.7);"' : '';
            const detailOpen = !!taskDeckState.expandedDetails[task.id];
            const detailLabel = detailOpen ? 'Hide details' : 'Details';
            const detailMeta = [task.estimated_duration, task.recommended_mode].filter(Boolean).join(' • ');

            return `<div class="task-card ${isSelected ? 'selected' : ''} ${detailOpen ? 'expanded' : ''}" ${highlightStyle}>` +
                `<div>` +
                `<h3>${task.title || 'Task'}</h3>` +
                `<div class="task-description">${task.description || ''}</div>` +
                `</div>` +
                `<div class="task-actions">` +
                `<button onclick="window.selectTask('${task.id}')" ${disabledAttr}>${selectLabel}</button>` +
                `<button class="ghost" onclick="window.viewTaskSummary('${task.id}')">Summary</button>` +
                `<button class="ghost" onclick="window.toggleTaskDetails('${task.id}')">${detailLabel}</button>` +
                `<div class="agent-shortcuts">` +
                `<span class="task-tag">Model: <span id="agent-model-chip">${localStorage.getItem('agent_manager_model') || 'gemma-3n'}</span></span>` +
                `<button class="ghost" onclick="window.openAgentMenu('${task.id}')">Agent menu</button>` +
                `</div>` +
                `</div>` +
                `<div class="task-detail-drawer ${detailOpen ? 'open' : ''}" id="task-detail-${task.id}">` +
                `<div class="task-meta">${detailMeta || 'No timing hints'}</div>` +
                `${tagRow ? `<div class="task-tags">${tagRow}</div>` : ''}` +
                `<div class="eligibility-stack">${markers}</div>` +
                `</div>` +
                `</div>`;
        }).join('');

        const isVisible = taskDeckState.selectedGroup === 'all' || taskDeckState.selectedGroup === groupLabel;
        const openAttr = taskDeckState.selectedGroup === groupLabel ? 'open' : '';
        const hiddenStyle = isVisible ? '' : 'style="display:none;"';

        return `<details class="task-group" data-group="${groupLabel}" ${openAttr} ${hiddenStyle}>` +
            `<summary>` +
            `<div>` +
            `<div class="panel-eyebrow">${groupLabel}</div>` +
            `<h3>${entries.length} task${entries.length === 1 ? '' : 's'}</h3>` +
            `</div>` +
            `<div class="status-chip info">${taskDeckState.selectedGroup === groupLabel ? 'Focused' : 'Browse'}</div>` +
            `</summary>` +
            `<div class="task-panel-grid">${cards}</div>` +
            `</details>`;
    }).join('');
}

window.filterTaskGroup = function (group) {
    taskDeckState.selectedGroup = group || 'all';
    renderTaskLibrary(taskLibraryPayload || {});
};

window.filterTaskText = function (text) {
    taskDeckState.textFilter = (text || '').toLowerCase();
    renderTaskLibrary(taskLibraryPayload || {});
};

window.setViewMode = function (mode) {
    viewState.mode = mode; // 'owner' | 'research' | 'docs'
    localStorage.setItem('studio_view_mode', viewState.mode);

    const ownerPanels = document.querySelectorAll('.owner-only');
    const researchPanels = document.querySelectorAll('.research-only');
    const standardPanels = document.querySelectorAll('.standard-view');
    const docsPanels = document.querySelectorAll('.docs-view');

    // Reset all
    ownerPanels.forEach(p => p.style.display = 'none');
    researchPanels.forEach(p => p.style.display = 'none');
    standardPanels.forEach(p => p.style.display = 'none');
    docsPanels.forEach(p => p.style.display = 'none');

    if (mode === 'docs') {
        docsPanels.forEach(p => p.style.display = 'block');
    } else if (mode === 'owner') {
        standardPanels.forEach(p => p.style.display = 'block');
        ownerPanels.forEach(p => p.style.display = 'block');
    } else if (mode === 'research') {
        standardPanels.forEach(p => p.style.display = 'block');
        researchPanels.forEach(p => p.style.display = 'block');
    }

    const ownerBtn = document.getElementById('owner-view-btn');
    const researchBtn = document.getElementById('research-view-btn');
    const docsBtn = document.getElementById('docs-view-btn');
    if (ownerBtn) ownerBtn.classList.toggle('active', viewState.mode === 'owner');
    if (researchBtn) researchBtn.classList.toggle('active', viewState.mode === 'research');
    if (docsBtn) docsBtn.classList.toggle('active', viewState.mode === 'docs');
};

(function initViewMode() {
    const saved = localStorage.getItem('studio_view_mode');
    if (saved) {
        window.setViewMode(saved);
    } else {
        window.setViewMode(viewState.mode);
    }
})();

window.toggleTaskDetails = function (taskId) {
    if (!taskId) return;
    taskDeckState.expandedDetails[taskId] = !taskDeckState.expandedDetails[taskId];
    renderTaskLibrary(taskLibraryPayload || {});
};

function renderAgentRail(status) {
    const agents = (status && Array.isArray(status.agent_threads) && status.agent_threads.length)
        ? status.agent_threads
        : agentManagerState.agents;

    const events = (status && Array.isArray(status.learning_events) && status.learning_events.length)
        ? status.learning_events
        : agentManagerState.learningEvents;

    const managerStatus = (status && (status.agent_manager || status.agent_manager_status)) || {};

    renderAgentThreads(agents);
    renderAgentChips(agents);
    renderModelStack(managerStatus);
    renderToolchain(managerStatus);
    renderTrainingSummary(managerStatus);
    renderLearningMilestones(status || {});
    renderLearningEvents(events);

    const toggle = document.getElementById('human-toggle');
    const toggleState = document.getElementById('human-toggle-state');
    const personaBadgeEl = document.getElementById('persona-badge');
    if (personaBadgeEl) {
        const persona = managerStatus.persona || 'default';
        personaBadgeEl.textContent = `Persona: ${persona}`;
    }
    if (toggle) {
        toggle.classList.toggle('active', agentManagerState.humanMode);
    }
    if (toggleState) {
        toggleState.textContent = agentManagerState.humanMode ? 'On' : 'Off';
    }

    // Persona badge from saved persona sliders
    if (personaBadgeEl) {
        try {
            const saved = localStorage.getItem('agent_persona');
            if (saved) {
                const parsed = JSON.parse(saved);
                const persona = parsed.persona || 'default';
                const style = parsed.response_style || 'concise';
                personaBadgeEl.textContent = `Persona: ${persona} • ${style}`;
            }
        } catch (err) {
            console.warn('Failed to read persona badge', err);
        }
    }
}

function renderModeList(status) {
    const container = document.getElementById('mode-pill-container');
    if (!container) return;
    const current = (status && status.mode) || 'unknown';
    const modes = [
        { id: 'idle', label: 'Idle' },
        { id: 'manual_control', label: 'Manual Control' },
        { id: 'manual_training', label: 'Manual Training' },
        { id: 'autonomous', label: 'Autonomous' },
        { id: 'sleep_learning', label: 'Sleep Learning' },
        { id: 'emergency_stop', label: 'Emergency Stop' },
    ];
    container.innerHTML = modes.map(m => {
        const isCurrent = current === m.id;
        return `<span class="mode-pill ${isCurrent ? 'current' : ''}">${m.label}</span>`;
    }).join('');

    const stackChip = document.getElementById('model-stack-chip');
    if (stackChip) {
        const stack = (status && status.model_stack) || agentManagerState.modelStack || {};
        const primary = stack.primary?.name || 'HOPE v0';
        const fallbacks = (stack.fallbacks || []).map(f => f.name).filter(Boolean);
        const summary = [primary, ...fallbacks].join(' → ') || 'HOPE → Gemini → Gemma';
        stackChip.textContent = 'Model stack: ' + summary;
    }
}

window.toggleHumanMode = function () {
    agentManagerState.humanMode = !agentManagerState.humanMode;
    window.showMessage(agentManagerState.humanMode ? 'Human guidance injected' : 'Human mode disabled');
    renderAgentRail();
};

window.pauseAgents = function () {
    agentManagerState.agents = agentManagerState.agents.map(agent => ({ ...agent, status: 'paused' }));
    window.showMessage('Agents paused for inspection');
    renderAgentRail();
};

window.resumeAgents = function () {
    agentManagerState.agents = agentManagerState.agents.map(agent => ({ ...agent, status: 'active' }));
    window.showMessage('Agents resumed');
    renderAgentRail();
};

window.reviewLearning = function () {
    window.showMessage('Learning feed refreshed');
    renderAgentRail();
    const events = document.getElementById('learning-events');
    if (events) {
        events.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
};

window.triggerSafetyHold = async function () {
    window.showMessage('Engaging safety hold...');
    try {
        const response = await fetch('/api/safety/hold', { method: 'POST' });
        const data = await response.json();
        if (data && data.success) {
            window.showMessage('Safety hold engaged');
            window.updateStatus();
        } else {
            window.showMessage('Hold failed', true);
        }
    } catch (err) {
        console.error(err);
        window.showMessage('Hold request failed', true);
    }
};

window.resetSafetyGates = async function () {
    window.showMessage('Resetting gates to idle baseline...');
    try {
        const response = await fetch('/api/safety/reset', { method: 'POST' });
        const data = await response.json();
        if (data && data.success) {
            window.showMessage('Gates reset to idle');
            window.updateStatus();
        } else {
            window.showMessage('Reset failed', true);
        }
    } catch (err) {
        console.error(err);
        window.showMessage('Reset request failed', true);
    }
};

function renderLoopTelemetry(status) {
    var loops = status.loop_metrics || status.metrics || {};
    var gates = status.gate_snapshot || status.gates || {};
    var safety = status.safety_head || {};

    var wave = (typeof loops.wave_particle_balance === 'number') ? Math.min(1, Math.max(0, loops.wave_particle_balance)) : 0;
    var waveFill = document.getElementById('wave-meter');
    if (waveFill) {
        waveFill.style.width = Math.round(wave * 100) + '%';
    }
    var waveLabel = document.getElementById('wave-value');
    if (waveLabel) {
        var wavePercent = Math.round(wave * 100);
        waveLabel.textContent = wavePercent + '% wave / ' + (100 - wavePercent) + '% particle';
    }

    var hope = loops.hope_loops || {};
    var fast = hope.fast || {};
    var mid = hope.mid || {};
    var slow = hope.slow || {};
    var hopeFast = document.getElementById('hope-fast');
    if (hopeFast) { hopeFast.textContent = 'Fast: ' + (fast.hz ? fast.hz + ' Hz (' + fast.latency_ms + ' ms)' : '--'); }
    var hopeMid = document.getElementById('hope-mid');
    if (hopeMid) { hopeMid.textContent = 'Mid: ' + (mid.hz ? mid.hz + ' Hz (' + mid.latency_ms + ' ms)' : '--'); }
    var hopeSlow = document.getElementById('hope-slow');
    if (hopeSlow) { hopeSlow.textContent = 'Slow: ' + (slow.hz ? slow.hz + ' Hz (' + slow.latency_ms + ' ms)' : '--'); }

    var cms = loops.cms || {};
    var cmsRatio = document.getElementById('cms-ratio');
    if (cmsRatio) {
        cmsRatio.textContent = cms.policy_ratio ? 'Policy ' + cms.policy_ratio + ' | Maintenance ' + cms.maintenance_ratio : '--';
    }
    var cmsBuffer = document.getElementById('cms-buffer');
    if (cmsBuffer) {
        cmsBuffer.textContent = cms.buffer_fill ? 'Buffer fill: ' + Math.round(cms.buffer_fill * 100) + '%' : 'Buffer fill: --';
    }

    var heartbeat = loops.heartbeat || {};
    var heartbeatBadge = document.getElementById('heartbeat-badge');
    if (heartbeatBadge) {
        var beatAgeMs = heartbeat.last_beat ? (Date.now() - heartbeat.last_beat * 1000) : null;
        var beatAgeLabel = beatAgeMs ? ' • ' + Math.round(beatAgeMs) + 'ms ago' : '';
        heartbeatBadge.textContent = heartbeat.ok ? 'Heartbeat stable' + beatAgeLabel : 'Heartbeat delayed';
        heartbeatBadge.className = 'chip ' + (heartbeat.ok ? 'success' : 'danger');
    }

    var gateAllow = document.getElementById('gate-allow');
    if (gateAllow) { gateAllow.textContent = gates.allow_motion ? 'Open' : 'Locked'; }
    var gateRecord = document.getElementById('gate-record');
    if (gateRecord) { gateRecord.textContent = gates.recording_gate ? 'Armed' : 'Off'; }

    var safetyHead = document.getElementById('safety-head-path');
    if (safetyHead) { safetyHead.textContent = safety.head_path || 'stub'; }
    var safetyEnvelope = document.getElementById('safety-envelope');
    if (safetyEnvelope) {
        var env = safety.envelope || {};
        safetyEnvelope.textContent = (env.status || 'simulated') + ' • ' + (env.radius_m || '?') + 'm radius';
    }
    var safetyHeartbeat = document.getElementById('safety-heartbeat');
    if (safetyHeartbeat) {
        var safetyBeat = safety.heartbeat || {};
        var beatDelta = safetyBeat.timestamp_ns ? ((Date.now() * 1e6 - safetyBeat.timestamp_ns) / 1e9).toFixed(1) : null;
        var beatLabel = beatDelta ? ' • ' + beatDelta + 's ago' : '';
        safetyHeartbeat.textContent = safetyBeat.ok ? 'Online (' + (safetyBeat.source || 'safety') + beatLabel + ')' : 'Simulated';
    }
}

window.selectTask = async function (taskId) {
    if (!taskId) return;
    try {
        const response = await fetch('/api/tasks/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId, reason: 'studio-selection' })
        });
        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }
        const payload = await response.json();
        window.showMessage(payload.message || 'Task selection updated', !payload.accepted);
        if (payload.selected_task) {
            renderSelectedTask({ entry: payload.selected_task });
        }
        window.fetchTaskLibrary();
        window.updateStatus();
    } catch (err) {
        console.warn('Task selection failed', err);
        window.showMessage('Failed to select task', true);
    }
};

window.applyTaskLibraryPayload = function (payload) {
    if (!payload) return;
    renderTaskLibrary(payload);
    if (payload && payload.selected_task_id && Array.isArray(payload.tasks)) {
        const selected = payload.tasks.find(task => task.id === payload.selected_task_id);
        if (selected) {
            renderSelectedTask({ entry: selected });
        }
    }
}

window.fetchTaskLibrary = async function (payload) {
    if (payload) {
        window.applyTaskLibraryPayload(payload);
        return;
    }
    try {
        const response = await fetch('/api/tasks?include_ineligible=true');
        if (!response.ok) { return; }
        const json = await response.json();
        window.applyTaskLibraryPayload(json);
    } catch (err) {
        console.warn('Task library fetch failed', err);
    }
};

window.applySkillLibraryPayload = function (payload) {
    const container = document.getElementById('skill-groups');
    if (!container) return;
    const skills = (payload && payload.skills) || [];
    skillDeckState.skills = skills;
    skillLibraryPayload = payload;
    const filterText = (skillDeckState.textFilter || '').trim();
    if (!skills.length) {
        container.innerHTML = '<div class="status-item"><span class="status-label">No skills available</span></div>';
        return;
    }
    const pinned = localStorage.getItem('pinned_skill');
    const pinnedPill = document.getElementById('pinned-skill-pill');
    if (pinnedPill) {
        pinnedPill.textContent = 'Pinned skill: ' + (pinned || 'none');
    }

    const groups = {};
    skills.forEach(skill => {
        const groupKey = skill.group || 'Skills';
        if (!groups[groupKey]) groups[groupKey] = [];
        if (!filterText || (skill.title && skill.title.toLowerCase().includes(filterText)) || (skill.description && skill.description.toLowerCase().includes(filterText))) {
            groups[groupKey].push(skill);
        }
    });

    container.innerHTML = Object.entries(groups).map(([groupLabel, entries]) => {
        const cards = entries.map(skill => {
            const eligibility = skill.eligibility || {};
            const markers = (eligibility.markers || []).map(marker => {
                const severity = marker.blocking ? 'blocking' : (marker.severity || 'info');
                const remediation = marker.remediation ? ` — ${marker.remediation}` : '';
                const label = marker.label || marker.code || 'marker';
                return `<div class="eligibility-marker ${severity}">${label}${remediation}</div>`;
            }).join('') || '<div class="eligibility-marker success">Eligible</div>';

            const tagRow = (skill.tags || []).map(tag => `<span class="task-tag">${tag}</span>`).join('');
            const capRow = (skill.capabilities || []).map(cap => `<span class="task-tag">${cap}</span>`).join('');

            return `
                        <div class="skill-card">
                            <h3>${skill.title || 'Skill'}</h3>
                            <div class="task-description">${skill.description || ''}</div>
                            <div class="skill-meta">
                                <span>Publisher: ${skill.publisher || 'local'}</span>
                                <span>v${skill.version || '0.1.0'}</span>
                            </div>
                            <div class="task-tags">${tagRow}</div>
                            <div class="task-tags">${capRow}</div>
                            <div style="margin-top:6px;">${markers}</div>
                            <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;">
                                <button onclick="window.viewSkillSummary('${skill.id}')">Summary</button>
                                <button class="ghost" onclick="window.pinSkill('${skill.id}')">Pin</button>
                            </div>
                        </div>
                    `;
        }).join('');

        return `
                    <div>
                        <div class="panel-eyebrow" style="margin-bottom:4px;">${groupLabel}</div>
                        <div class="skill-grid">${cards}</div>
                    </div>
                `;
    }).join('');
}

window.fetchSkillLibrary = async function (payload) {
    if (payload) {
        window.applySkillLibraryPayload(payload);
        return;
    }
    try {
        const response = await fetch('/api/skills?include_ineligible=true');
        if (!response.ok) { return; }
        const json = await response.json();
        window.applySkillLibraryPayload(json);
    } catch (err) {
        console.warn('Skill library fetch failed', err);
    }
};

window.filterSkillText = function (text) {
    skillDeckState.textFilter = (text || '').toLowerCase();
    window.applySkillLibraryPayload(skillLibraryPayload || { skills: skillDeckState.skills });
}

window.viewTaskSummary = async function (taskId) {
    try {
        const res = await fetch(`/api/tasks/summary/${taskId}`);
        if (!res.ok) return;
        const payload = await res.json();
        alert(JSON.stringify(payload.summary || payload, null, 2));
    } catch (err) {
        console.warn('Task summary failed', err);
    }
}

window.viewSkillSummary = async function (skillId) {
    try {
        const res = await fetch(`/api/skills/summary/${skillId}`);
        if (!res.ok) return;
        const payload = await res.json();
        alert(JSON.stringify(payload.summary || payload, null, 2));
    } catch (err) {
        console.warn('Skill summary failed', err);
    }
};

window.pinSkill = function (skillId) {
    localStorage.setItem('pinned_skill', skillId);
    window.showMessage('Pinned skill: ' + skillId);
    const pill = document.getElementById('pinned-skill-pill');
    if (pill) {
        pill.textContent = 'Pinned skill: ' + skillId;
    }
};

window.startJaxTraining = async function () {
    try {
        window.showMessage('Starting JAX Training cycle...');
        document.getElementById('training-status').className = 'status-badge warning';
        document.getElementById('training-status').textContent = 'Training...';

        const res = await fetch('/api/training/jax/start', { method: 'POST' });
        const json = await res.json();

        if (json.status === 'success') {
            window.showMessage('Training step complete!');
            document.getElementById('training-status').className = 'status-badge success';
            document.getElementById('training-status').textContent = 'Idle';

            if (json.metrics) {
                document.getElementById('train-step').textContent = json.metrics.step || 0;
                document.getElementById('train-loss').textContent = (json.metrics.loss || 0).toFixed(4);
                const log = document.getElementById('training-log');
                if (log) {
                    log.innerHTML = `<div class="log-entry system">Step ${json.metrics.step}: Loss ${json.metrics.loss.toFixed(4)}</div>` + log.innerHTML;
                }
            }
        } else {
            window.showMessage('Training failed: ' + json.message, true);
            document.getElementById('training-status').className = 'status-badge error';
            document.getElementById('training-status').textContent = 'Error';
        }
    } catch (e) {
        console.error(e);
        window.showMessage('Training request failed', true);
        document.getElementById('training-status').className = 'status-badge error';
        document.getElementById('training-status').textContent = 'Error';
    }
};

window.startSymbolicSearch = async function () {
    try {
        window.showMessage('Starting Symbolic Search...');
        document.getElementById('training-status').className = 'status-badge warning';
        document.getElementById('training-status').textContent = 'Imagine...';

        const val = document.getElementById('imagination-depth');
        if (val) val.textContent = '...';

        // Default payload for demo
        const payload = {
            start_joints: [0, 0, 0, 0, 0, 0],
            target_joints: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            depth: 5
        };

        const res = await fetch('/api/imagination/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const json = await res.json();

        if (json.status === 'success') {
            window.showMessage('Search complete!');
            document.getElementById('training-status').className = 'status-badge success';
            document.getElementById('training-status').textContent = 'Idle';

            if (json.metrics) {
                document.getElementById('plan-score').textContent = (json.metrics.plan_score || 0).toFixed(2);
                document.getElementById('imagination-depth').textContent = json.metrics.imagination_depth || 0;
                const log = document.getElementById('training-log');
                if (log) {
                    log.innerHTML = `<div class="log-entry success">Plan found! Score: ${(json.metrics.plan_score).toFixed(2)}</div>` + log.innerHTML;
                }
            }
        } else {
            window.showMessage('Search failed: ' + json.message, true);
            document.getElementById('training-status').className = 'status-badge error';
            document.getElementById('training-status').textContent = 'Error';
        }
    } catch (e) {
        console.error(e);
        window.showMessage('Search request failed', true);
    }
};

// Tab switching logic// --- View Mode Logic ---
const viewState = {
    mode: 'owner',
};

window.setViewMode = function (mode) {
    viewState.mode = mode; // 'owner' | 'research' | 'docs' | 'settings'
    localStorage.setItem('studio_view_mode', viewState.mode);

    const ownerPanels = document.querySelectorAll('.owner-only');
    const researchPanels = document.querySelectorAll('.research-only');
    const standardPanels = document.querySelectorAll('.standard-view');
    const docsPanels = document.querySelectorAll('.docs-view');
    const settingsPanels = document.querySelectorAll('.settings-view');

    // Hide all first
    ownerPanels.forEach(p => p.style.display = 'none');
    researchPanels.forEach(p => p.style.display = 'none');
    standardPanels.forEach(p => p.style.display = 'none');
    docsPanels.forEach(p => p.style.display = 'none');
    settingsPanels.forEach(p => p.style.display = 'none');

    // Show based on mode
    if (mode === 'docs') {
        docsPanels.forEach(p => p.style.display = 'block');
    } else if (mode === 'settings') {
        settingsPanels.forEach(p => p.style.display = 'block');
    } else if (mode === 'owner') {
        standardPanels.forEach(p => p.style.display = 'block');
        ownerPanels.forEach(p => p.style.display = 'block');
    } else if (mode === 'research') {
        standardPanels.forEach(p => p.style.display = 'block');
        researchPanels.forEach(p => p.style.display = 'block');
    }

    // Update Top Bar Button States (Assuming IDs exist, though Top Bar doesn't have explicit mode buttons for all these yet except custom ones)
    // We can rely on visual state for now.

    // Also, if 'settings' mode, perhaps we want to highlight the settings button if we had one.
    // For now, this is sufficient.
};

(function initViewMode() {
    const saved = localStorage.getItem('studio_view_mode');
    if (saved) {
        window.setViewMode(saved);
    } else {
        window.setViewMode(viewState.mode);
    }
})();

// --- Status Orchestration ---
// Calls render functions defined in other modules
window.applyStatusPayload = function (statusPayload) {
    if (!statusPayload) return;
    var status = statusPayload.status ? statusPayload.status : statusPayload;
    var mode = status.mode || 'unknown';
    var modeText = mode.replace(/_/g, ' ').toUpperCase();

    const modeEl = document.getElementById('mode');
    if (modeEl) modeEl.innerHTML = '<span class="badge ' + mode + '">' + modeText + '</span>';

    const recEl = document.getElementById('recording');
    if (recEl) recEl.textContent = status.is_recording ? 'Recording' : 'Idle';

    var motionAllowed = typeof status.allow_motion !== 'undefined'
        ? status.allow_motion
        : (status.gate_snapshot ? status.gate_snapshot.allow_motion : false);

    const motionEl = document.getElementById('motion');
    if (motionEl) motionEl.textContent = motionAllowed ? 'Motion Enabled' : 'Motion Locked';

    var modeCard = document.getElementById('mode-card');
    if (modeCard) { modeCard.textContent = modeText; }
    var recordingCard = document.getElementById('recording-card');
    if (recordingCard) { recordingCard.textContent = status.is_recording ? 'Recording' : 'Idle'; }
    var motionCard = document.getElementById('motion-card');
    if (motionCard) { motionCard.textContent = motionAllowed ? 'Allowed' : 'Prevented'; }

    // Call module-specific renderers if they exist
    if (window.renderLoopTelemetry) window.renderLoopTelemetry(status); // Research
    if (window.renderGateStatus) window.renderGateStatus(status.gate_snapshot || status.gates); // Owner
    if (window.renderAgentRail) window.renderAgentRail(status); // Owner
    if (window.renderSelectedTask) window.renderSelectedTask(status.current_task); // Owner
    if (window.renderModeList) window.renderModeList(status); // Owner

    if (window.realityProofState) {
        window.realityProofState.status = status;
        if (window.updateRealityProof) window.updateRealityProof();
    }
    var batteryCard = document.getElementById('battery-status');
    if (batteryCard) {
        var batt = status.battery;
        batteryCard.textContent = batt && typeof batt.level !== 'undefined'
            ? Math.round((batt.level || 0) * 100) + '%'
            : (batt && batt.percent ? batt.percent : 'n/a');
    }

    // Update hardware sensors
    var hardwareDiv = document.getElementById('hardware-status');
    if (status.detected_hardware) {
        var hw = status.detected_hardware;
        var hwHtml = '';

        if (hw.depth_camera) {
            hwHtml += '<div class="status-item"><span class="status-label">📷 Depth Camera</span><span class="status-value">' + hw.depth_camera + '</span></div>';
        }
        if (hw.depth_camera_driver) {
            hwHtml += '<div class="status-item"><span class="status-label">Camera Driver</span><span class="status-value">' + hw.depth_camera_driver + '</span></div>';
        }
        if (hw.servo_controller) {
            hwHtml += '<div class="status-item"><span class="status-label">🦾 Servo Controller</span><span class="status-value">' + hw.servo_controller + '</span></div>';
        }
        if (hw.servo_controller_address) {
            hwHtml += '<div class="status-item"><span class="status-label">I2C Address</span><span class="status-value">' + hw.servo_controller_address + '</span></div>';
        }

        if (hwHtml) {
            hardwareDiv.innerHTML = hwHtml;
        } else {
            hardwareDiv.innerHTML = '<div class="status-item"><span class="status-label">No hardware detected</span></div>';
        }
    } else {
        hardwareDiv.innerHTML = '<div class="status-item"><span class="status-label">Hardware info not available</span></div>';
    }

    renderModeList(status);

    var agiModel = document.getElementById('agi-model-stack');
    if (agiModel) {
        var stack = (status && status.model_stack) || agentManagerState.modelStack || {};
        var primary = stack.primary ? stack.primary.name : 'HOPE';
        var fallbacks = (stack.fallbacks || []).map(function (f) { return f.name; }).filter(Boolean);
        agiModel.textContent = [primary].concat(fallbacks).join(' → ') || 'HOPE → Gemma';
    }
};

window.updateStatus = function (payload) {
    if (payload) {
        window.applyStatusPayload(payload);
        return;
    }
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/status', true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            try {
                var data = JSON.parse(xhr.responseText);
                if (data.status) {
                    window.applyStatusPayload(data.status);
                }
            } catch (e) {
                console.error('Parse error:', e);
            }
        }
    };
    xhr.onerror = function () {
        console.error('Connection failed');
        window.showMessage('Failed to connect to robot', true);
    };
    xhr.send();
};

window.setMode = function (mode) {
    console.log('Setting mode to:', mode);
    window.showMessage('Changing mode to ' + mode.replace(/_/g, ' ') + '...');

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/mode/' + mode, true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            try {
                var data = JSON.parse(xhr.responseText);
                if (data.success) {
                    window.showMessage('Mode changed to ' + mode.replace(/_/g, ' ').toUpperCase());
                    setTimeout(window.updateStatus, 500);
                } else {
                    window.showMessage('Failed: ' + (data.message || 'Unknown error'), true);
                }
            } catch (e) {
                console.error('Parse error:', e);
                window.showMessage('Error parsing response', true);
            }
        } else {
            window.showMessage('Server error: ' + xhr.status, true);
        }
    };
    xhr.onerror = function () {
        console.error('Connection failed');
        window.showMessage('Connection failed', true);
    };
    xhr.send();
};

window.startManualControl = async function (buttonEl) {
    const manualButton = buttonEl || document.getElementById('manual-control-btn');
    if (!manualButton) {
        window.showMessage('Manual Control button not found', true);
        return;
    }

    if (manualButton.disabled) {
        return;
    }

    const originalText = manualButton.textContent;
    manualButton.disabled = true;
    manualButton.textContent = 'Switching...';

    try {
        window.showMessage('Switching to manual control...');
        const response = await fetch('/api/mode/manual_control', { method: 'POST' });
        if (!response.ok) {
            throw new Error('Server responded with ' + response.status);
        }

        const payload = await response.json();
        if (payload && payload.success) {
            window.showMessage('Manual control enabled, redirecting...');
            window.updateStatus();
            setTimeout(() => { window.location.href = '/control'; }, 300);
        } else {
            const message = (payload && payload.message) ? payload.message : 'Unable to enable manual control';
            window.showMessage(message, true);
            window.updateStatus();
            manualButton.disabled = false;
            manualButton.textContent = originalText;
        }
    } catch (err) {
        console.error('Manual control switch failed', err);
        window.showMessage('Failed to switch to manual control', true);
        window.updateStatus();
        manualButton.disabled = false;
        manualButton.textContent = originalText;
    }
};

window.applyLoopHealthPayload = function (payload) {
    if (!payload) return;
    const source = payload.metrics || payload.loop_metrics ? payload : payload.status || payload;
    if (source.metrics || source.loop_metrics) {
        renderLoopTelemetry({
            loop_metrics: source.metrics || source.loop_metrics,
            gate_snapshot: source.gates || source.gate_snapshot,
            safety_head: source.safety_head
        });
    }
    realityProofState.loops = source;
    updateRealityProof();
}

window.pollLoopHealth = async function (payload) {
    if (payload) {
        window.applyLoopHealthPayload(payload);
        return;
    }
    try {
        const response = await fetch('/api/loops');
        if (!response.ok) { return; }
        const json = await response.json();
        if (json) {
            window.applyLoopHealthPayload(json);
        }
    } catch (err) {
        console.warn('Loop telemetry fetch failed', err);
    }
};

// Column resizers for full-width IDE layout
// Column resizers for full-width IDE layout (Nav | Content | Agent)
(function initColumnResizers() {
    const grid = document.getElementById('workspace-grid');
    if (!grid) return;
    const leftHandle = document.querySelector('[data-resize="left"]');
    const rightHandle = document.querySelector('[data-resize="right"]');
    const leftCol = document.querySelector('.left-column');
    const centerCol = document.querySelector('.center-column');

    function startDrag(type) {
        return function (event) {
            event.preventDefault();
            const startX = event.clientX;
            // Capture current widths (whether % or px) as pixels for smooth dragging
            const startLeftW = leftCol.getBoundingClientRect().width;
            const startCenterW = centerCol.getBoundingClientRect().width;

            function onMove(e) {
                const delta = e.clientX - startX;
                let newLeft = startLeftW;
                let newCenter = startCenterW;

                if (type === 'left') {
                    newLeft = Math.max(200, startLeftW + delta);
                    // Lock Center to its current pixel width so it doesn't jump
                    grid.style.gridTemplateColumns = `${newLeft}px 10px ${startCenterW}px 10px 1fr`;
                } else if (type === 'right') {
                    newCenter = Math.max(300, startCenterW + delta);
                    // Lock Left to its current pixel width
                    grid.style.gridTemplateColumns = `${startLeftW}px 10px ${newCenter}px 10px 1fr`;
                }
            }

            function onUp() {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                // Grid remains locked in pixels after drag, which is expected behavior
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        };
    }

    leftHandle?.addEventListener('mousedown', startDrag('left'));
    rightHandle?.addEventListener('mousedown', startDrag('right'));
})();

window.openAgentMenu = function (taskId) {
    const menu = document.createElement('div');
    menu.className = 'task-card expanded';
    menu.style.position = 'fixed';
    menu.style.right = '16px';
    menu.style.bottom = '16px';
    menu.style.zIndex = 9999;
    menu.innerHTML = `
                <h3>Agent Manager</h3>
                <div class="task-description">Context for task ${taskId}</div>
                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:8px;">
                    <button onclick="window.setAgentModel('gemma-3n')">Gemma 3n</button>
                    <button onclick="window.setAgentModel('gemma-transformer')" class="ghost">Transformers</button>
                    <button onclick="window.setAgentModel('hailo-runtime')" class="ghost">Hailo</button>
                </div>
                <div style="margin-top:10px;">
                    <div class="task-description" style="margin-bottom:6px;">Sub-agents</div>
                    <div style="display:flex; gap:6px; flex-wrap:wrap;">
                        <button class="ghost" onclick="alert('Planner toggled (stub)')">Planner</button>
                        <button class="ghost" onclick="alert('Navigator toggled (stub)')">Navigator</button>
                        <button class="ghost" onclick="alert('Safety toggled (stub)')">Safety</button>
                        <button class="ghost" onclick="alert('Recorder toggled (stub)')">Recorder</button>
                    </div>
                </div>
                <div class="task-description" style="margin-top:10px;">(Local-only stub; wire to backend to persist.)</div>
                <button class="ghost" style="margin-top:10px;" onclick="this.parentElement.remove()">Close</button>
            `;
    document.body.appendChild(menu);
};

window.setAgentModel = function (model) {
    localStorage.setItem('agent_manager_model', model);
    const chip = document.getElementById('agent-model-chip');
    if (chip) chip.textContent = model;
    alert('Agent Manager model set (local): ' + model);
};

// Chat overlay persistence (shared between /ui and /control)
var chatMinimized = false;
var chatHistory = [];
var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
var chatHistoryKey = chatStoragePrefix + '_history';
var chatMinimizedKey = chatStoragePrefix + '_minimized';

function persistChatState() {
    try {
        localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory.slice(-50)));
        localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
    } catch (e) {
        console.warn('Unable to persist chat state', e);
    }
}

function applyChatMinimized() {
    var panel = document.getElementById('chat-panel');
    var toggle = document.getElementById('chat-toggle');
    if (!panel || !toggle) return;

    if (chatMinimized) {
        panel.classList.add('minimized');
        toggle.textContent = '+';
    } else {
        panel.classList.remove('minimized');
        toggle.textContent = '−';
    }
}


hydrateChatOverlay();

// Chat overlay persistence (shared between /ui and /control)
var chatMinimized = false;
var chatHistory = [];
var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
var chatHistoryKey = chatStoragePrefix + '_history';
var chatMinimizedKey = chatStoragePrefix + '_minimized';

function persistChatState() {
    try {
        localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory.slice(-50)));
        localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
    } catch (e) {
        console.warn('Unable to persist chat state', e);
    }
}

function applyChatMinimized() {
    var panel = document.getElementById('chat-panel');
    var toggle = document.getElementById('chat-toggle');
    if (!panel || !toggle) return;

    if (chatMinimized) {
        panel.classList.add('minimized');
        toggle.textContent = '+';
    } else {
        panel.classList.remove('minimized');
        toggle.textContent = '−';
    }
}

function renderChatMessage(text, role, shouldPersist) {
    if (typeof shouldPersist === 'undefined') shouldPersist = true;

    var messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return;

    var messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ' + role;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (shouldPersist) {
        chatHistory.push({ role: role, content: text });
        persistChatState();
    }
}


hydrateChatOverlay();

// Chat overlay persistence (/ui + /control)
var chatMinimized = false;
var chatHistory = [];
var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
var chatHistoryKey = chatStoragePrefix + '_history';
var chatMinimizedKey = chatStoragePrefix + '_minimized';
var MAX_MESSAGE_LENGTH = 10000; // Maximum allowed message length for DoS protection
var initialChatMessage = 'Chat with Gemma 3n about robot control';

// Sanitize text to prevent XSS attacks
function sanitizeText(text) {
    if (typeof text !== 'string') {
        return '';
    }

    // Remove any HTML tags first
    var sanitized = text.replace(/<[^>]*>/g, '');

    // Remove any remaining < or > characters that might be part of incomplete tags
    sanitized = sanitized.replace(/[<>]/g, '');

    // Remove javascript: and data: URL schemes
    sanitized = sanitized.replace(/javascript:/gi, '');
    sanitized = sanitized.replace(/data:/gi, '');

    // Remove common XSS event handlers
    sanitized = sanitized.replace(/on\w+\s*=/gi, '');

    // Limit length to prevent DOS attacks
    return sanitized.substring(0, 10000);
}

function persistChatState() {
    try {
        // Trim in-memory history as well
        if (chatHistory.length > 50) {
            chatHistory = chatHistory.slice(-50);
        }
        localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory));
        localStorage.setItem(chatMinimizedKey, chatMinimized ? 'true' : 'false');
    } catch (e) {
        console.warn('Unable to persist chat state', e);
        // Show a user-visible warning
        alert('Warning: Unable to save chat history. Your messages may not be saved.');
    }
}

function applyChatMinimized() {
    var panel = document.getElementById('chat-panel');
    var toggle = document.getElementById('chat-toggle');
    if (!panel || !toggle) return;

    if (chatMinimized) {
        panel.classList.add('minimized');
        toggle.textContent = '+';
    } else {
        panel.classList.remove('minimized');
        toggle.textContent = '−';
    }
}

function validateChatContent(content) {
    // Validate chat content to prevent injection attacks
    // Ensure content is a string and within reasonable bounds
    if (typeof content !== 'string') return '';
    // Limit message length to prevent DoS
    if (content.length > MAX_MESSAGE_LENGTH) return content.substring(0, MAX_MESSAGE_LENGTH);
    return content;
}

function validateChatRole(role) {
    // Validate role to prevent class injection
    var validRoles = ['user', 'assistant', 'system'];
    return validRoles.indexOf(role) !== -1 ? role : 'system';
}

function renderChatMessage(text, role, shouldPersist) {
    if (typeof shouldPersist === 'undefined') shouldPersist = true;

    var messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return;

    // Validate inputs to prevent injection attacks
    var validatedText = validateChatContent(text);
    var safeRole = (role || 'system').toLowerCase();

    // Map backend roles to safe CSS classes
    var validClasses = ['user', 'assistant', 'system', 'agent-manager', 'subagent'];
    var cssClass = validClasses.indexOf(safeRole) !== -1 ? safeRole : 'system';

    var messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ' + cssClass;
    messageDiv.textContent = validatedText;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (shouldPersist) {
        chatHistory.push({ role: cssClass, content: validatedText });
        persistChatState();
    }
}

window.renderIncomingChatMessage = function (evt) {
    console.log("[StudioClient] Received chat event:", evt);
    // evt = { role: "subagent"|"agent_manager", name: "...", text: "...", ... }
    if (!evt || !evt.text) return;
    // Map the event role to our CSS class
    var cssRole = 'system';
    if (evt.role === 'subagent') cssRole = 'subagent';
    else if (evt.role === 'agent_manager') cssRole = 'agent-manager';
    else if (evt.role === 'user') cssRole = 'user';

    // Optional: Prepend name if it's a subagent/agent-manager to make it clear
    var displayText = evt.text;
    if (evt.name && (cssRole === 'subagent' || cssRole === 'agent-manager')) {
        displayText = '[' + evt.name + '] ' + displayText;
    }

    renderChatMessage(displayText, cssRole, true);
};

function hydrateChatOverlay() {
    try {
        var storedHistory = localStorage.getItem(chatHistoryKey);
        if (storedHistory) {
            try {
                chatHistory = JSON.parse(storedHistory) || [];
                if (!Array.isArray(chatHistory)) {
                    chatHistory = [];
                }
            } catch (parseError) {
                console.warn('Failed to parse chat history from localStorage, using empty history instead:', parseError);
                chatHistory = [];
            }
            chatHistory.forEach(function (msg) {
                if (msg && typeof msg === 'object' && msg.role && msg.content) {
                    renderChatMessage(msg.content, msg.role, false);
                }
            });
        } else {
            chatHistory = [];
        }

        var storedMinimized = localStorage.getItem(chatMinimizedKey);
        if (storedMinimized === 'true') {
            chatMinimized = true;
        }
    } catch (e) {
        console.warn('Unable to hydrate chat state', e);
        chatHistory = [];
    }

    applyChatMinimized();
}

window.toggleChat = function () {
    chatMinimized = !chatMinimized;
    persistChatState();
    applyChatMinimized();
};

window.addChatMessage = function (text, role) {
    renderChatMessage(text, role, true);
};

window.sendChatMessage = function () {
    var input = document.getElementById('chat-input');
    var sendBtn = document.getElementById('chat-send');
    var message = input ? input.value.trim() : '';

    if (!message) return;

    // Add user message
    addChatMessage(message, 'user');
    if (input) input.value = '';

    // Disable input while processing
    if (input) input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    // Send to Gemma endpoint
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/chat', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function () {
        if (input) input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;

        if (xhr.status === 200) {
            try {
                var data = JSON.parse(xhr.responseText);
                if (data.response) {
                    addChatMessage(data.response, 'assistant');
                } else if (data.error) {
                    addChatMessage('Error: ' + data.error, 'system');
                }
            } catch (e) {
                addChatMessage('Error parsing response', 'system');
            }
        } else {
            addChatMessage('Server error: ' + xhr.status, 'system');
        }

        if (input) input.focus();
    };
    xhr.onerror = function () {
        if (input) input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        addChatMessage('Connection error', 'system');
        if (input) input.focus();
    };

    // Include chat history for context
    xhr.send(JSON.stringify({
        message: message,
        history: chatHistory.slice(-10) // Last 10 messages for context
    }));
};

hydrateChatOverlay();

// Kick off initial renders
renderSafeModeGrid(safetyPersonaState.selected);
renderGateStatus(safetyPersonaState.gates);
renderAgentRail();
window.updateStatus();
window.pollLoopHealth();
window.fetchGates();
window.fetchTaskLibrary();
window.fetchSkillLibrary();

// Realtime updates via SSE with polling fallback
if (window.StudioClient && window.StudioClient.startRealtime) {
    window.StudioClient.startRealtime({
        onStatus: (payload) => window.updateStatus(payload),
        onLoops: (payload) => window.pollLoopHealth(payload),
        onTasks: (payload) => window.fetchTaskLibrary(payload),
        onSkills: (payload) => window.fetchSkillLibrary({ skills: payload }),
        onChat: (payload) => window.renderIncomingChatMessage(payload),
        fallbackPollMs: 8000,
    });
} else {
    setInterval(window.updateStatus, 4000);
    setInterval(window.pollLoopHealth, 4000);
    setInterval(window.fetchTaskLibrary, 8000);
    setInterval(window.fetchSkillLibrary, 8000);
}

// --- Teacher Mode (HITL) Implementation ---
var teacherModeActive = false;
var pendingTeacherQuestion = null;
var teacherPollInterval = null;

window.toggleTeacherMode = async function () {
    teacherModeActive = !teacherModeActive;
    const stateLabel = document.getElementById('teacher-toggle-state');
    const toggleBtn = document.getElementById('teacher-toggle');

    if (stateLabel) stateLabel.textContent = teacherModeActive ? 'On' : 'Off';
    if (toggleBtn) {
        if (teacherModeActive) toggleBtn.classList.add('active');
        else toggleBtn.classList.remove('active');
    }

    try {
        const response = await fetch('/api/training/teacher/mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ active: teacherModeActive })
        });
        const res = await response.json();
        window.showMessage(res.message || 'Teacher mode updated');

        if (teacherModeActive && !teacherPollInterval) {
            teacherPollInterval = setInterval(pollTeacherStatus, 2000);
        } else if (!teacherModeActive && teacherPollInterval) {
            clearInterval(teacherPollInterval);
            teacherPollInterval = null;
            document.body.classList.remove('teacher-intervention-active');
        }
    } catch (err) {
        console.error('Failed to toggle teacher mode', err);
        window.showMessage('Failed to toggle teacher mode', true);
        // Revert UI state on error
        teacherModeActive = !teacherModeActive;
        if (stateLabel) stateLabel.textContent = teacherModeActive ? 'On' : 'Off';
    }
};

async function pollTeacherStatus() {
    if (!teacherModeActive) return;
    try {
        const response = await fetch('/api/training/teacher/status');
        if (!response.ok) return;
        const status = await response.json();

        const wasPending = !!pendingTeacherQuestion;
        pendingTeacherQuestion = status.pending_question;

        if (status.waiting_for_answer) {
            document.body.classList.add('teacher-intervention-active');
            // Ensure the UI shows the question if it's new
            if (!wasPending && pendingTeacherQuestion) {
                const chatMessages = document.getElementById('chat-messages');
                const questionHtml = `
                            <div class="message-row assistant intervention">
                                <div class="message-bubble system-alert">
                                    <strong>👨‍🏫 Teacher Intervention Requested</strong><br/>
                                    The Agent Manager is asking:<br/>
                                    <em>"${pendingTeacherQuestion}"</em><br/>
                                    <br/>
                                    Please type your answer below to teach the agent.
                                </div>
                            </div>
                         `;
                chatMessages.insertAdjacentHTML('beforeend', questionHtml);
                chatMessages.scrollTop = chatMessages.scrollHeight;

                // Focus input
                const input = document.getElementById('chat-input');
                if (input) {
                    input.focus();
                    input.placeholder = "Type your answer to teach the agent...";
                    input.classList.add('highlight');
                }
            }
        } else {
            document.body.classList.remove('teacher-intervention-active');
            const input = document.getElementById('chat-input');
            if (input && input.classList.contains('highlight')) {
                input.classList.remove('highlight');
                input.placeholder = "Ask about robot status, control tips...";
            }
        }
    } catch (err) {
        console.warn('Teacher poll failed', err);
    }
}

// Hook into sendChatMessage to intercept answers
const originalSendChatMessage = window.sendChatMessage;
window.sendChatMessage = async function () {
    if (teacherModeActive && pendingTeacherQuestion) {
        var input = document.getElementById('chat-input');
        var message = input ? input.value.trim() : '';
        if (!message) return;

        // Optimistic UI update
        addChatMessage(message, 'user teacher-answer');
        if (input) input.value = '';

        try {
            const response = await fetch('/api/training/teacher/answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ answer: message })
            });
            const res = await response.json();
            if (res.success) {
                // Clear pending state locally until next poll confirms
                pendingTeacherQuestion = null;
                document.body.classList.remove('teacher-intervention-active');
                const msgDiv = document.getElementById('chat-messages');
                msgDiv.insertAdjacentHTML('beforeend',
                    `<div class="message-row system"><div class="message-bubble success">✅ Answer submitted to Agent Manager. Resuming training...</div></div>`
                );
                msgDiv.scrollTop = msgDiv.scrollHeight;

                const inputEl = document.getElementById('chat-input');
                if (inputEl) {
                    inputEl.classList.remove('highlight');
                    inputEl.placeholder = "Ask about robot status, control tips...";
                }
            } else {
                window.showMessage(res.message || 'Failed to submit answer', true);
            }
        } catch (err) {
            console.error('Answer submission error', err);
            window.showMessage('Error submitting answer', true);
        }
        return;
    }
    // Fallback to original
    if (originalSendChatMessage) originalSendChatMessage();
};

// Add some CSS for the teacher mode
const style = document.createElement('style');
style.textContent = `
            .teacher-intervention-active .chat-input {
                border: 2px solid #FFD700 !important;
                background: rgba(255, 215, 0, 0.1);
            }
            .chat-message.agent-manager {
            align-self: flex-start;
            background: #2d3748;
            border-left: 3px solid #63b3ed;
        }
        .chat-message.subagent {
            align-self: flex-start;
            background: #1a3c5e; /* distinct blue-ish tint */
            border-left: 3px solid #90cdf4;
            margin-left: 20px; /* Indent to show it's a sub-call */
        }
        .chat-message.assistant { align-self: flex-start; background: #2d3342; border-radius: 4px 12px 12px 4px; color: #cfd7ff; }
    .chat-message.agent_manager { align-self: flex-start; background: #2d4233; border-radius: 4px 12px 12px 4px; color: #d7ffcf; border-left: 3px solid #4caf50; }
    .chat-message.subagent { align-self: flex-start; background: #2d3855; border-radius: 4px 12px 12px 4px; color: #d0e1ff; border-left: 3px solid #4c89af; font-family: 'Roboto Mono', monospace; font-size: 0.9em; }
    .chat-message.user { align-self: flex-end; background: #4a5a75; border-radius: 12px 4px 4px 12px; color: #fff; }
            .chat-message.system-alert {
            background: #742a2a;
            color: #fff5f5;
            border: 1px solid #ffbcbc;
        }        color: #fff;
            }
            .message-bubble.success {
                background: rgba(50, 205, 50, 0.2);
                color: #fff;
                font-size: 0.9em;
            }
            #teacher-toggle.active {
                background: rgba(255, 215, 0, 0.2);
                border-color: #FFD700;
            }
            #teacher-toggle.active strong {
                color: #FFD700;
            }
            /* Docs Styles */
            .step-item { display:flex; gap:12px; margin-bottom:12px; }
            .step-number { background:#4a5568; color:#fff; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px; flex-shrink:0; }
            .step-content strong { color:#e2e8f0; display:block; margin-bottom:2px; }
            .step-content p { color:#a0aec0; margin:0; font-size:0.9em; line-height:1.4; }
        `;
document.head.appendChild(style);

