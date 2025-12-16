/**
 * ui_owner.js - Owner View & Agent Management Logic
 */

window.agentManagerState = {
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

window.safetyPersonaPresets = [
    {
        id: 'owner_safe', label: 'Owner', apiMode: 'manual_control', fallbackMode: 'manual_control',
        description: 'Primary owners with full controls; telemetry pinned.',
        gates: { allow_motion: true, record_episodes: true, require_supervision: false },
    },
    {
        id: 'renter_safe', label: 'Renter / Leasee', apiMode: 'autonomous', fallbackMode: 'autonomous',
        description: 'Temporary operators with supervised motion and recorded evidence.',
        gates: { allow_motion: true, record_episodes: true, require_supervision: true },
    },
    {
        id: 'enterprise_maintenance', label: 'Enterprise Maintenance', apiMode: 'sleep_learning', fallbackMode: 'sleep_learning',
        description: 'Technicians with diagnostics on and motion held unless cleared.',
        gates: { allow_motion: false, record_episodes: true, require_supervision: true },
    },
    {
        id: 'researcher', label: 'Researcher', apiMode: 'autonomous', fallbackMode: 'autonomous',
        description: 'Research sandboxes with cautious motion and logged changes.',
        gates: { allow_motion: true, record_episodes: true, require_supervision: true },
    },
    {
        id: 'creator', label: 'Creator', apiMode: 'idle', fallbackMode: 'idle',
        description: 'System creator with design-time control; motion starts locked.',
        gates: { allow_motion: false, record_episodes: false, require_supervision: false },
    },
];

window.safetyPersonaState = { selected: null, gates: {} };

window.taskDeckState = { selectedGroup: 'all', expandedDetails: {}, textFilter: '' };
window.taskLibraryPayload = null;
window.skillDeckState = { textFilter: '', skills: [] };
window.skillLibraryPayload = null;


// --- Agent Rail Rendering ---

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
    const stack = (status && status.model_stack) || window.agentManagerState.modelStack || {};
    const primary = stack.primary || { name: 'Gemma 3n Agent Manager', role: 'On-device', status: 'active', latency: '—' };
    const fallbacks = Array.isArray(stack.fallbacks) ? stack.fallbacks : [];
    const items = [{ ...primary, primary: true }, ...fallbacks];

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
    const tools = (status && status.toolchain) || window.agentManagerState.toolchain || [];
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
    const summary = (status && status.training) || window.agentManagerState.trainingSummary || {};
    const metrics = [
        { label: 'RLDS episodes', value: summary.episodes ?? '—' },
        { label: 'Pending imports', value: summary.pendingImports ?? '—' },
        { label: 'Local memories', value: summary.localMemories ?? '—' },
        { label: 'Last run', value: summary.lastRun || 'n/a' },
        { label: 'Queue state', value: summary.queueState || 'Idle' },
    ];
    container.innerHTML = metrics.map(metric => `<div class="metric-row"><div class="metric-label">${metric.label}</div><div class="metric-value">${metric.value}</div></div>`).join('');
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

window.renderAgentRail = function (status) {
    const agents = (status && Array.isArray(status.agent_threads) && status.agent_threads.length)
        ? status.agent_threads
        : window.agentManagerState.agents;

    const events = (status && Array.isArray(status.learning_events) && status.learning_events.length)
        ? status.learning_events
        : window.agentManagerState.learningEvents;

    const managerStatus = (status && (status.agent_manager || status.agent_manager_status)) || {};

    renderAgentThreads(agents);
    renderAgentChips(agents);
    renderModelStack(managerStatus);
    renderToolchain(managerStatus);
    renderTrainingSummary(managerStatus);
    renderLearningMilestones(status || {});
    renderLearningEvents(events);

    // Persona rendering
    const personaBadgeEl = document.getElementById('persona-badge');
    if (personaBadgeEl) {
        const persona = managerStatus.persona || 'default';
        personaBadgeEl.textContent = `Persona: ${persona}`;
    }
    // Local storage persona override
    if (personaBadgeEl) {
        try {
            const saved = localStorage.getItem('agent_persona');
            if (saved) {
                const parsed = JSON.parse(saved);
                const persona = parsed.persona || 'default';
                const style = parsed.response_style || 'concise';
                personaBadgeEl.textContent = `Persona: ${persona} • ${style}`;
            }
        } catch (err) { }
    }
    const toggle = document.getElementById('human-toggle');
    if (toggle) toggle.classList.toggle('active', window.agentManagerState.humanMode);
    const toggleState = document.getElementById('human-toggle-state');
    if (toggleState) toggleState.textContent = window.agentManagerState.humanMode ? 'On' : 'Off';
};


// --- Safety & Gates ---

window.renderGateStatus = function (gates) {
    const grid = document.getElementById('gate-status-grid');
    if (!grid) return;
    const snapshot = gates || window.safetyPersonaState.gates || {};
    window.safetyPersonaState.gates = snapshot;

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
};

window.renderSafeModeGrid = function (selectedId) {
    const grid = document.getElementById('safe-mode-grid');
    if (!grid) return;
    grid.innerHTML = window.safetyPersonaPresets.map(preset => {
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
        const active = window.safetyPersonaPresets.find(p => p.id === selectedId);
        statusChip.textContent = active ? `${active.label} guardrails loaded` : 'Persona guardrails synced';
    }
};

window.setSafeModePreset = async function (presetId) {
    const preset = window.safetyPersonaPresets.find(p => p.id === presetId);
    const statusChip = document.getElementById('safe-mode-status');
    if (!preset) return;

    if (statusChip) statusChip.textContent = `Applying ${preset.label} safeguards...`;

    try {
        const mode = preset.apiMode || preset.fallbackMode || 'idle';
        const res = await fetch(`/api/mode/${mode}`);
        if (!res.ok && preset.fallbackMode && preset.fallbackMode !== mode) {
            await fetch(`/api/mode/${preset.fallbackMode}`);
        }
        window.safetyPersonaState.selected = preset.id;
        const personaBadge = document.getElementById('persona-badge');
        if (personaBadge) personaBadge.textContent = `Persona: ${preset.label}`;
        window.renderSafeModeGrid(preset.id);
        window.renderGateStatus(preset.gates);
        if (statusChip) statusChip.textContent = `${preset.label} safe mode ready`;
        window.showMessage(`${preset.label} guardrails applied`);
    } catch (err) {
        console.warn('Safe mode failed', err);
        if (statusChip) statusChip.textContent = 'Safe mode failed';
        window.showMessage('Unable to set safe mode', true);
    }

    window.fetchGates();
    window.updateStatus();
};

window.applyGatesPayload = function (payload) {
    if (!payload) return;
    window.renderGateStatus(payload);
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


// --- Task & Skill Libraries ---

window.renderSelectedTask = function (selectedTask) {
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

window.renderTaskLibrary = function (payload) {
    const container = document.getElementById('task-groups');
    const controls = document.getElementById('task-group-segments');
    window.taskLibraryPayload = payload || null;
    if (!container) return;

    const tasks = (payload && payload.tasks) || [];
    const selectedId = payload ? payload.selected_task_id : null;
    if (!tasks.length) {
        container.innerHTML = '<div class="status-item"><span class="status-label">No tasks available</span></div>';
        if (controls) { controls.innerHTML = ''; }
        return;
    }

    const groups = {};
    const filterText = (window.taskDeckState.textFilter || '').trim();
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
            const active = window.taskDeckState.selectedGroup === label;
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
            const detailOpen = !!window.taskDeckState.expandedDetails[task.id];
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

        const isVisible = window.taskDeckState.selectedGroup === 'all' || window.taskDeckState.selectedGroup === groupLabel;
        const openAttr = window.taskDeckState.selectedGroup === groupLabel ? 'open' : '';
        const hiddenStyle = isVisible ? '' : 'style="display:none;"';

        return `<details class="task-group" data-group="${groupLabel}" ${openAttr} ${hiddenStyle}>` +
            `<summary>` +
            `<div>` +
            `<div class="panel-eyebrow">${groupLabel}</div>` +
            `<h3>${entries.length} task${entries.length === 1 ? '' : 's'}</h3>` +
            `</div>` +
            `<div class="status-chip info">${window.taskDeckState.selectedGroup === groupLabel ? 'Focused' : 'Browse'}</div>` +
            `</summary>` +
            `<div class="task-panel-grid">${cards}</div>` +
            `</details>`;
    }).join('');
}

window.filterTaskGroup = function (group) {
    window.taskDeckState.selectedGroup = group || 'all';
    window.renderTaskLibrary(window.taskLibraryPayload || {});
};

window.filterTaskText = function (text) {
    window.taskDeckState.textFilter = (text || '').toLowerCase();
    window.renderTaskLibrary(window.taskLibraryPayload || {});
};

window.toggleTaskDetails = function (taskId) {
    if (!taskId) return;
    window.taskDeckState.expandedDetails[taskId] = !window.taskDeckState.expandedDetails[taskId];
    window.renderTaskLibrary(window.taskLibraryPayload || {});
};

window.renderSkillLibrary = function (payload) {
    const container = document.getElementById('skill-groups');
    if (!container) return;
    const skills = (payload && payload.skills) || [];
    window.skillDeckState.skills = skills;
    window.skillLibraryPayload = payload;
    const filterText = (window.skillDeckState.textFilter || '').trim();
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
            window.renderSelectedTask({ entry: payload.selected_task });
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
    window.renderTaskLibrary(payload);
    if (payload && payload.selected_task_id && Array.isArray(payload.tasks)) {
        const selected = payload.tasks.find(task => task.id === payload.selected_task_id);
        if (selected) {
            window.renderSelectedTask({ entry: selected });
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
    window.renderSkillLibrary(payload);
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
    window.skillDeckState.textFilter = (text || '').toLowerCase();
    window.applySkillLibraryPayload(window.skillLibraryPayload || { skills: window.skillDeckState.skills });
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


// --- Manual Control & Other ---

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

window.switchTab = function (tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.tab === tabName) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    document.querySelectorAll('.control-panel').forEach(panel => {
        if (panel.id === tabName + '-panel') panel.style.display = 'block';
        else panel.style.display = 'none';
    });
};

window.renderModeList = function (status) {
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
        const stack = (status && status.model_stack) || window.agentManagerState.modelStack || {};
        const primary = stack.primary?.name || 'HOPE v0';
        const fallbacks = (stack.fallbacks || []).map(f => f.name).filter(Boolean);
        const summary = [primary, ...fallbacks].join(' → ') || 'HOPE → Gemini → Gemma';
        stackChip.textContent = 'Model stack: ' + summary;
    }
}


// --- Agent Menu Stub ---
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

window.toggleHumanMode = function () {
    window.agentManagerState.humanMode = !window.agentManagerState.humanMode;
    window.showMessage(window.agentManagerState.humanMode ? 'Human guidance injected' : 'Human mode disabled');
    window.renderAgentRail();
};

window.pauseAgents = function () {
    window.agentManagerState.agents = window.agentManagerState.agents.map(agent => ({ ...agent, status: 'paused' }));
    window.showMessage('Agents paused for inspection');
    window.renderAgentRail();
};

window.resumeAgents = function () {
    window.agentManagerState.agents = window.agentManagerState.agents.map(agent => ({ ...agent, status: 'active' }));
    window.showMessage('Agents resumed');
    window.renderAgentRail();
};

window.reviewLearning = function () {
    window.showMessage('Learning feed refreshed');
    window.renderAgentRail();
};

window.requestTraining = async function () {
    const statusEl = document.getElementById('training-status');
    if (statusEl) statusEl.textContent = 'Starting...';
    try {
        const res = await fetch('/api/training/run', { method: 'POST' });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        if (statusEl) statusEl.textContent = 'Training started (async)';
    } catch (err) {
        console.warn('Training start failed', err);
        if (statusEl) statusEl.textContent = 'Training failed: ' + err;
    }
};

window.openEpisodeImports = function () {
    fetch('/api/training/tool_dataset_summary')
        .then((r) => r.json().then((j) => ({ ok: r.ok, j })))
        .then(({ ok, j }) => {
            alert("Episode Summary:\n" + JSON.stringify(j, null, 2));
        })
        .catch((err) => {
            console.warn(err);
            alert("Failed to fetch episode exports");
        });
};

window.viewTrainingLogs = function () {
    alert("Training Logs are not yet connected to list view.");
};
