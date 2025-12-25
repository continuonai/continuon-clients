/**
 * ui_core.js - Core utilities and orchestration for ContinuonXR UI
 */

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

// --- Settings Modal Logic ---
const settingsModal = document.getElementById('settings-modal');
const settingsBackdrop = document.getElementById('settings-backdrop');
const settingsForm = document.getElementById('settings-form');
const settingsStatus = document.getElementById('settings-status');

function setSettingsStatus(message, isError) {
    if (!settingsStatus) return;
    settingsStatus.textContent = message;
    settingsStatus.style.color = isError ? '#ff7b7b' : 'var(--muted)';
}

function populateSettingsForm(settings) {
    if (document.getElementById('settings-allow-motion'))
        document.getElementById('settings-allow-motion').checked = !!settings?.safety?.allow_motion;
    if (document.getElementById('settings-record-episodes'))
        document.getElementById('settings-record-episodes').checked = !!settings?.safety?.record_episodes;
    if (document.getElementById('settings-require-supervision'))
        document.getElementById('settings-require-supervision').checked = !!settings?.safety?.require_supervision;
    if (document.getElementById('telemetry-rate'))
        document.getElementById('telemetry-rate').value = settings?.telemetry?.rate_hz ?? 2.0;
    if (document.getElementById('chat-persona'))
        document.getElementById('chat-persona').value = settings?.chat?.persona ?? 'operator';
    if (document.getElementById('chat-temperature'))
        document.getElementById('chat-temperature').value = settings?.chat?.temperature ?? 0.35;
}

async function fetchSettings() {
    const response = await fetch('/api/settings');
    if (!response.ok) {
        throw new Error('Server responded with ' + response.status);
    }
    const payload = await response.json();
    if (!payload.success) {
        throw new Error(payload.message || 'Unable to load settings');
    }
    return payload.settings || {};
}

window.openSettingsModal = async function () {
    settingsModal?.classList.add('open');
    settingsBackdrop?.classList.add('open');
    settingsModal?.setAttribute('aria-hidden', 'false');
    setSettingsStatus('Loading current settings...');

    try {
        const settings = await fetchSettings();
        populateSettingsForm(settings);
        setSettingsStatus('Loaded from config directory.');
    } catch (err) {
        console.error(err);
        setSettingsStatus(err.message || 'Failed to load settings', true);
    }
};

window.closeSettingsModal = function () {
    settingsModal?.classList.remove('open');
    settingsBackdrop?.classList.remove('open');
    settingsModal?.setAttribute('aria-hidden', 'true');
};

settingsBackdrop?.addEventListener('click', closeSettingsModal);

settingsForm?.addEventListener('submit', async function (event) {
    event.preventDefault();

    const telemetryRate = parseFloat(document.getElementById('telemetry-rate').value);
    const chatTemperature = parseFloat(document.getElementById('chat-temperature').value);

    if (Number.isNaN(telemetryRate) || telemetryRate <= 0 || telemetryRate > 30) {
        setSettingsStatus('Telemetry rate must be between 0.1 and 30 Hz', true);
        return;
    }

    if (Number.isNaN(chatTemperature) || chatTemperature < 0 || chatTemperature > 1) {
        setSettingsStatus('Chat temperature must be between 0 and 1', true);
        return;
    }

    const payload = {
        safety: {
            allow_motion: document.getElementById('settings-allow-motion').checked,
            record_episodes: document.getElementById('settings-record-episodes').checked,
            require_supervision: document.getElementById('settings-require-supervision').checked,
        },
        telemetry: { rate_hz: telemetryRate },
        chat: {
            persona: document.getElementById('chat-persona').value,
            temperature: chatTemperature,
        },
    };

    setSettingsStatus('Saving...');
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
        setSettingsStatus('Settings saved to config directory.');
        window.showMessage('Settings updated successfully');
        setTimeout(closeSettingsModal, 300);
    } catch (err) {
        console.error('Save failed', err);
        setSettingsStatus(err.message || 'Unable to save settings', true);
        window.showMessage('Unable to save settings', true);
    }
});


// --- View Mode Logic ---
const viewState = {
    mode: 'owner',
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
    if (hardwareDiv) {
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
    }

    var agiModel = document.getElementById('agi-model-stack');
    if (agiModel) {
        var stack = (status && status.model_stack) || (window.agentManagerState && window.agentManagerState.modelStack) || {};
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

// --- Column Resizers ---
(function initColumnResizers() {
    const grid = document.getElementById('workspace-grid');
    if (!grid) return;
    const leftHandle = document.querySelector('[data-resize="left"]');
    const rightHandle = document.querySelector('[data-resize="right"]');
    let leftWidth = 320;
    let rightWidth = 360;
    const clamp = (val, min, max) => Math.max(min, Math.min(max, val));
    const apply = () => {
        grid.style.gridTemplateColumns = `${leftWidth}px 10px 1fr 10px ${rightWidth}px`;
    };
    apply();

    function startDrag(type) {
        return function (event) {
            event.preventDefault();
            const startX = event.clientX;
            const startLeft = leftWidth;
            const startRight = rightWidth;
            function onMove(e) {
                const delta = e.clientX - startX;
                if (type === 'left') {
                    leftWidth = clamp(startLeft + delta, 240, 520);
                } else {
                    rightWidth = clamp(startRight - delta, 260, 520);
                }
                apply();
            }
            function onUp() {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        };
    }

    leftHandle?.addEventListener('mousedown', startDrag('left'));
    rightHandle?.addEventListener('mousedown', startDrag('right'));
})();


// --- CHAT & TEACHER MODE ---
var chatMinimized = false;
var chatHistory = [];
var chatStoragePrefix = 'gemma_chat_' + (window.location.host || 'local');
var chatHistoryKey = chatStoragePrefix + '_history';
var chatMinimizedKey = chatStoragePrefix + '_minimized';
var chatSessionKey = chatStoragePrefix + '_session_id';
var MAX_MESSAGE_LENGTH = 10000;

function getChatSessionId() {
    let sid = localStorage.getItem(chatSessionKey);
    if (!sid) {
        sid = 'session_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
        localStorage.setItem(chatSessionKey, sid);
    }
    return sid;
}

window.clearChatSession = async function() {
    const sid = getChatSessionId();
    if (!confirm("Are you sure you want to clear the conversation history?")) return;

    try {
        const response = await fetch('/api/chat/session/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sid })
        });
        const data = await response.json();
        if (data.success) {
            // Reset local state
            chatHistory = [];
            localStorage.removeItem(chatHistoryKey);
            document.getElementById('chat-messages').innerHTML = '';
            
            // Generate new session ID
            const newSid = 'session_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
            localStorage.setItem(chatSessionKey, newSid);
            
            window.showMessage("Conversation cleared.");
        }
    } catch (e) {
        console.error("Failed to clear session", e);
        window.showMessage("Failed to clear session", true);
    }
};

function persistChatState() {
    try {
        if (chatHistory.length > 50) chatHistory = chatHistory.slice(-50);
        localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory));
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

function validateChatContent(content) {
    if (typeof content !== 'string') return '';
    if (content.length > MAX_MESSAGE_LENGTH) return content.substring(0, MAX_MESSAGE_LENGTH);
    return content;
}

function renderChatMessage(text, role, shouldPersist, conversation_id, session_id) {
    if (typeof shouldPersist === 'undefined') shouldPersist = true;
    var messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return;

    var validatedText = validateChatContent(text);
    var safeRole = (role || 'system').toLowerCase();
    var validClasses = ['user', 'assistant', 'system', 'agent-manager', 'subagent', 'teacher-answer'];
    var cssClass = validClasses.indexOf(safeRole) !== -1 ? safeRole : 'system';

    var messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ' + cssClass;
    
    // Add context badge if part of a session
    const currentSid = localStorage.getItem(chatSessionKey);
    if (session_id === currentSid && (safeRole === 'user' || safeRole === 'assistant' || safeRole === 'agent-manager')) {
        var badge = document.createElement('span');
        badge.className = 'context-badge';
        badge.textContent = '🧵 In Thread';
        badge.style.fontSize = '9px';
        badge.style.opacity = '0.6';
        badge.style.display = 'block';
        badge.style.marginBottom = '2px';
        messageDiv.appendChild(badge);
    }

    // Create text container
    var textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = validatedText;
    messageDiv.appendChild(textDiv);

    // Add feedback buttons for assistant messages
    if (safeRole === 'assistant' || safeRole === 'agent-manager') {
        var actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        actionsDiv.style.display = 'flex';
        actionsDiv.style.gap = '8px';
        actionsDiv.style.marginTop = '4px';
        actionsDiv.style.fontSize = '12px';

        var upBtn = document.createElement('button');
        upBtn.className = 'feedback-btn ghost tight';
        upBtn.innerHTML = '👍';
        upBtn.title = 'Accurate/Helpful';
        upBtn.onclick = () => window.validateResponse(conversation_id, true, upBtn);

        var downBtn = document.createElement('button');
        downBtn.className = 'feedback-btn ghost tight';
        downBtn.innerHTML = '👎';
        downBtn.title = 'Inaccurate/Unhelpful';
        downBtn.onclick = () => window.validateResponse(conversation_id, false, downBtn);

        actionsDiv.appendChild(upBtn);
        actionsDiv.appendChild(downBtn);
        messageDiv.appendChild(actionsDiv);
    }

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (shouldPersist) {
        chatHistory.push({ role: cssClass, content: validatedText, conversation_id: conversation_id, session_id: session_id });
        persistChatState();
    }
}

window.validateResponse = async function(conversation_id, isValid, btnEl) {
    if (!conversation_id) {
        console.warn("No conversation_id for validation");
        return;
    }

    try {
        const response = await fetch('/api/agent/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                conversation_id: conversation_id,
                validated: isValid
            })
        });
        const data = await response.json();
        if (data.success) {
            // Visual feedback
            if (btnEl) {
                const parent = btnEl.parentElement;
                parent.querySelectorAll('.feedback-btn').forEach(b => b.classList.add('disabled'));
                btnEl.classList.remove('disabled');
                btnEl.classList.add('active');
                btnEl.style.background = isValid ? 'rgba(52, 199, 89, 0.2)' : 'rgba(255, 59, 48, 0.2)';
            }
            window.showMessage(isValid ? 'Feedback recorded! Memory promoted.' : 'Feedback recorded. Memory flagged.');
        }
    } catch (e) {
        console.error("Validation failed", e);
        window.showMessage("Failed to send feedback", true);
    }
};

window.renderIncomingChatMessage = function (evt) {
    console.log("[StudioClient] Received chat event:", evt);
    if (!evt || !evt.text) return;
    var cssRole = 'system';
    if (evt.role === 'subagent') cssRole = 'subagent';
    else if (evt.role === 'agent_manager') cssRole = 'agent_manager';
    else if (evt.role === 'user') cssRole = 'user';

    var displayText = evt.text;
    if (evt.name && (cssRole === 'subagent' || cssRole === 'agent_manager')) {
        displayText = '[' + evt.name + '] ' + displayText;
    }
    renderChatMessage(displayText, cssRole, true, evt.conversation_id, evt.session_id);
};

function hydrateChatOverlay() {
    try {
        var storedHistory = localStorage.getItem(chatHistoryKey);
        if (storedHistory) {
            try {
                chatHistory = JSON.parse(storedHistory) || [];
                if (!Array.isArray(chatHistory)) chatHistory = [];
            } catch (parseError) {
                chatHistory = [];
            }
            chatHistory.forEach(function (msg) {
                renderChatMessage(msg.content, msg.role, false, msg.conversation_id, msg.session_id);
            });
        }
        var storedMinimized = localStorage.getItem(chatMinimizedKey);
        if (storedMinimized === 'true') chatMinimized = true;
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
    renderChatMessage(text, role, true, null, getChatSessionId());
};

window.sendChatMessage = async function () {
    // Intercept for teacher mode check (logic appended further down)
    if (teacherModeActive && pendingTeacherQuestion) {
        await handleTeacherAnswer();
        return;
    }

    var input = document.getElementById('chat-input');
    var sendBtn = document.getElementById('chat-send');
    var message = input ? input.value.trim() : '';

    if (!message) return;

    const sid = getChatSessionId();
    renderChatMessage(message, 'user', true, null, sid);
    if (input) input.value = '';
    if (input) input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: sid,
                history: chatHistory.slice(-10)
            })
        });
        const data = await response.json();
        if (data.response) {
            renderChatMessage(data.response, 'assistant', true, data.conversation_id, sid);
        } else if (data.error) {
            addChatMessage('Error: ' + data.error, 'system');
        }
    } catch (e) {
        addChatMessage('Connection or parse error', 'system');
    } finally {
        if (input) {
            input.disabled = false;
            input.focus();
        }
        if (sendBtn) sendBtn.disabled = false;
    }
};

hydrateChatOverlay();

// Teacher Mode Logic
var teacherModeActive = false;
var pendingTeacherQuestion = null;
var teacherPollInterval = null;

window.toggleTeacherMode = async function () {
    teacherModeActive = !teacherModeActive;
    const stateLabel = document.getElementById('teacher-toggle-state');
    const toggleBtn = document.getElementById('teacher-toggle');

    if (stateLabel) stateLabel.textContent = teacherModeActive ? 'On' : 'Off';
    toggleBtn?.classList.toggle('active', teacherModeActive);

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
        // Revert
        teacherModeActive = !teacherModeActive;
        if (stateLabel) stateLabel.textContent = teacherModeActive ? 'On' : 'Off';
        toggleBtn?.classList.toggle('active', teacherModeActive);
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
            if (!wasPending && pendingTeacherQuestion) {
                renderChatMessage("👨‍🏫 Teacher Intervention: " + pendingTeacherQuestion, "system-alert", false);
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
    } catch (err) { }
}

async function handleTeacherAnswer() {
    var input = document.getElementById('chat-input');
    var message = input ? input.value.trim() : '';
    if (!message) return;

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
            pendingTeacherQuestion = null;
            document.body.classList.remove('teacher-intervention-active');
            renderChatMessage("✅ Answer submitted. Resuming...", "assistant success", false);
            const inputEl = document.getElementById('chat-input');
            if (inputEl) {
                inputEl.classList.remove('highlight');
                inputEl.placeholder = "Ask about robot status, control tips...";
            }
        } else {
            window.showMessage(res.message || 'Failed to submit answer', true);
        }
    } catch (err) {
        window.showMessage('Error submitting answer', true);
    }
}

// --- Initialization ---
// This will run when the DOM is fully loaded, ensuring all scripts are present.
window.addEventListener('load', function () {
    console.log("ContinuonXR UI Initializing...");

    // Initial fetches
    if (window.renderSafeModeGrid && window.safetyPersonaState) window.renderSafeModeGrid(window.safetyPersonaState.selected);
    if (window.renderGateStatus && window.safetyPersonaState) window.renderGateStatus(window.safetyPersonaState.gates);
    if (window.renderAgentRail) window.renderAgentRail();

    window.updateStatus();
    if (window.pollLoopHealth) window.pollLoopHealth();
    if (window.fetchGates) window.fetchGates();
    if (window.fetchTaskLibrary) window.fetchTaskLibrary();
    if (window.fetchSkillLibrary) window.fetchSkillLibrary();

    // Realtime SSE setup
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
        if (window.pollLoopHealth) setInterval(window.pollLoopHealth, 4000);
        if (window.fetchTaskLibrary) setInterval(window.fetchTaskLibrary, 8000);
        if (window.fetchSkillLibrary) setInterval(window.fetchSkillLibrary, 8000);
    }
});
