// Shared chat overlay logic for UI + Control pages.
// Provides consistent chat UX, history persistence, and live status feed hooks.
(function () {
  const storageKeyPrefix = 'continuon_chat_' + (window.location.host || 'local');
  const historyKey = storageKeyPrefix + '_history';
  const minimizedKey = storageKeyPrefix + '_minimized';
  const positionKey = storageKeyPrefix + '_position';
  const sizeKey = storageKeyPrefix + '_size';
  const expandedKey = storageKeyPrefix + '_expanded';
  const speechKey = storageKeyPrefix + '_speak_replies';
  const MAX_HISTORY = 50;

  let state = {
    minimized: false,
    history: [],
    pendingId: null,
    lastMode: null,
    lastGate: null,
    expanded: false,
    position: null,
    size: null,
    lastTaskCount: 0,
    lastHeartbeat: null,
    sessionId: null,
    lastPosted: {
      mode: 0,
      gate: 0,
    },
  };
  let dragTracker = null;
  let resizeTracker = null;

  function qs(id) {
    return document.getElementById(id);
  }

  function getSpeakEnabled() {
    try {
      return localStorage.getItem(speechKey) === '1';
    } catch (e) {
      return false;
    }
  }

  function setSpeakEnabled(v) {
    try {
      localStorage.setItem(speechKey, v ? '1' : '0');
    } catch (e) { }
  }

  async function speakOnRobot(text) {
    const msg = String(text || '').trim();
    if (!msg) return;
    try {
      await fetch('/api/audio/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: msg.slice(0, 500), rate_wpm: 175, voice: 'en' }),
      });
    } catch (e) { }
  }

  function startBrowserSTT() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      appendMessage('Speech recognition unavailable in this browser. Tip: use Chrome, or add a server-side STT backend later.', 'system');
      return;
    }
    const rec = new SpeechRecognition();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (ev) => {
      try {
        const text = ev.results && ev.results[0] && ev.results[0][0] ? ev.results[0][0].transcript : '';
        const input = qs('chat-input');
        if (input && text) {
          input.value = String(text).trim();
          input.focus();
        }
      } catch (e) { }
    };
    rec.onerror = (ev) => {
      appendMessage('Mic error: ' + (ev && ev.error ? ev.error : 'unknown'), 'system');
    };
    try {
      rec.start();
    } catch (e) {
      appendMessage('Mic start failed: ' + (e && e.message ? e.message : String(e)), 'system');
    }
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(historyKey);
      state.history = raw ? JSON.parse(raw) || [] : [];
    } catch (e) {
      console.warn('Chat history load failed', e);
      state.history = [];
    }
    try {
      const sessionKey = storageKeyPrefix + '_session';
      let sessionId = localStorage.getItem(sessionKey);
      if (!sessionId) {
        sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(16).slice(2);
        localStorage.setItem(sessionKey, sessionId);
      }
      state.sessionId = sessionId;
    } catch (e) {
      console.warn('Chat session load failed', e);
      state.sessionId = null;
    }
    try {
      state.minimized = localStorage.getItem(minimizedKey) === 'true';
      const storedPosition = localStorage.getItem(positionKey);
      state.position = storedPosition ? JSON.parse(storedPosition) : null;
      const storedSize = localStorage.getItem(sizeKey);
      state.size = storedSize ? JSON.parse(storedSize) : null;
      state.expanded = localStorage.getItem(expandedKey) === 'true';
    } catch (e) {
      console.warn('Chat minimized load failed', e);
    }
  }

  function saveState() {
    try {
      const trimmed = state.history.slice(-MAX_HISTORY);
      localStorage.setItem(historyKey, JSON.stringify(trimmed));
      localStorage.setItem(minimizedKey, state.minimized ? 'true' : 'false');
      if (state.position) {
        localStorage.setItem(positionKey, JSON.stringify(state.position));
      }
      if (state.size) {
        localStorage.setItem(sizeKey, JSON.stringify(state.size));
      }
      localStorage.setItem(expandedKey, state.expanded ? 'true' : 'false');
    } catch (e) {
      console.warn('Chat state save failed', e);
    }
  }

  function renderHistory() {
    const container = qs('chat-messages');
    if (!container) return;
    container.innerHTML = '';
    state.history.forEach((msg) => appendMessage(msg.content, msg.role, false));
    container.scrollTop = container.scrollHeight;
  }

  function applyMinimized() {
    const panel = qs('chat-panel');
    const toggle = qs('chat-toggle');
    if (!panel || !toggle) return;
    if (state.minimized) {
      panel.classList.add('minimized');
      toggle.textContent = '+';
      toggle.setAttribute('aria-expanded', 'false');
    } else {
      panel.classList.remove('minimized');
      toggle.textContent = '−';
      toggle.setAttribute('aria-expanded', 'true');
    }
    applyLayout();
  }

  function applyLayout() {
    const panel = qs('chat-panel');
    if (!panel) return;

    if (state.position && typeof state.position.left === 'number' && typeof state.position.top === 'number') {
      panel.style.left = `${state.position.left}px`;
      panel.style.top = `${state.position.top}px`;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    }

    if (state.size) {
      if (state.size.width) panel.style.width = `${Math.max(state.size.width, 280)}px`;
      if (state.size.height) panel.style.height = `${Math.max(state.size.height, 200)}px`;
    }

    panel.classList.toggle('expanded', !!state.expanded);
  }

  function appendMessage(text, role = 'assistant', persist = true) {
    const container = qs('chat-messages');
    if (!container) return;
    const div = document.createElement('div');
    
    // Normalize role to safe CSS class
    let safeRole = String(role || 'assistant').toLowerCase().replace(/[^a-z0-9]+/g, '_');
    
    // Map roles to positioning categories
    // HOPE agent roles (left side)
    if (safeRole === 'hope' || safeRole === 'hope_v1' || safeRole === 'hope-v1' || 
        safeRole === 'agent_manager' || safeRole === 'agent-manager' || 
        safeRole === 'hope_agent' || safeRole === 'hope-agent') {
      safeRole = 'hope';
    }
    // 3rd party models (center) - subagent, assistant (when not HOPE), gemma, phi, etc.
    else if (safeRole === 'subagent' || safeRole === 'assistant' || 
             safeRole === 'gemma' || safeRole === 'phi' || safeRole === 'third_party' ||
             safeRole.includes('gemma') || safeRole.includes('phi') || 
             safeRole.includes('llm') || safeRole.includes('model')) {
      // Keep as subagent or assistant for 3rd party models
      if (safeRole === 'assistant' && !safeRole.includes('hope')) {
        safeRole = 'subagent'; // Default assistant to subagent (3rd party) unless explicitly HOPE
      }
    }
    // User messages (right side)
    else if (safeRole === 'user') {
      safeRole = 'user';
    }
    // System messages (center)
    else if (safeRole === 'system' || safeRole === 'system-alert') {
      safeRole = 'system';
    }
    // Default unknown roles to assistant (3rd party, centered)
    else {
      safeRole = 'subagent';
    }
    
    div.className = 'chat-message ' + safeRole;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    if (persist) {
      state.history.push({ role, content: text });
      saveState();
    }
  }

  function startDrag(e) {
    const panel = qs('chat-panel');
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    dragTracker = {
      offsetX: e.clientX - rect.left,
      offsetY: e.clientY - rect.top,
    };
    panel.classList.add('dragging');
    window.addEventListener('mousemove', onDrag);
    window.addEventListener('mouseup', endDrag);
  }

  function onDrag(e) {
    if (!dragTracker) return;
    const panel = qs('chat-panel');
    if (!panel) return;
    const left = Math.max(0, e.clientX - dragTracker.offsetX);
    const top = Math.max(0, e.clientY - dragTracker.offsetY);
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.right = 'auto';
    panel.style.bottom = 'auto';
    state.position = { left, top };
  }

  function endDrag() {
    const panel = qs('chat-panel');
    if (panel) panel.classList.remove('dragging');
    window.removeEventListener('mousemove', onDrag);
    window.removeEventListener('mouseup', endDrag);
    dragTracker = null;
    saveState();
  }

  function startResize(e) {
    const panel = qs('chat-panel');
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    resizeTracker = {
      startWidth: rect.width,
      startHeight: rect.height,
      startX: e.clientX,
      startY: e.clientY,
    };
    window.addEventListener('mousemove', onResize);
    window.addEventListener('mouseup', endResize);
    e.preventDefault();
  }

  function onResize(e) {
    if (!resizeTracker) return;
    const panel = qs('chat-panel');
    if (!panel) return;
    const width = Math.min(Math.max(resizeTracker.startWidth + (e.clientX - resizeTracker.startX), 280), 800);
    const height = Math.max(resizeTracker.startHeight + (e.clientY - resizeTracker.startY), 220);
    panel.style.width = `${width}px`;
    panel.style.height = `${height}px`;
    state.size = { width, height };
  }

  function endResize() {
    window.removeEventListener('mousemove', onResize);
    window.removeEventListener('mouseup', endResize);
    resizeTracker = null;
    saveState();
  }

  function renderStructured(data) {
    const pre = qs('chat-structured');
    if (!pre) return;
    const structured = data?.structured || data?.plan || null;
    if (!structured) {
      pre.textContent = '(no structured plan yet)';
      return;
    }
    try {
      pre.textContent = JSON.stringify(structured, null, 2);
    } catch (e) {
      pre.textContent = String(structured);
    }
  }

  async function sendChatMessage() {
    const input = qs('chat-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;

    appendMessage(text, 'user');
    input.value = '';

    const placeholderId = 'pending-' + Date.now();
    state.pendingId = placeholderId;
    appendMessage('Thinking...', 'assistant', false);

    try {
      const sel = qs('chat-agent-select');
      const choice = sel ? String(sel.value || 'agent_manager') : 'agent_manager';
      let model_hint = null;
      let delegate_model_hint = null;
      if (choice === 'hope-v1') {
        model_hint = 'hope-v1';
      } else if (choice.startsWith('consult:')) {
        // HOPE v1 is the main agent; consult runs a subagent turn first.
        model_hint = 'hope-v1';
        delegate_model_hint = choice; // keep "consult:" prefix for backend behavior
      } else if (choice.startsWith('direct:')) {
        model_hint = choice.replace('direct:', '').trim() || null;
      }
      const attachEl = qs('chat-attach-camera');
      const attach_camera_frame = !!(attachEl && attachEl.checked);
      const thumb = qs('chat-camera-thumb');
      if (thumb) {
        if (attach_camera_frame) {
          thumb.style.display = '';
          thumb.src = `/api/camera/frame?t=${Date.now()}`;
        } else {
          thumb.style.display = 'none';
          thumb.removeAttribute('src');
        }
      }

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: state.history.slice(-10),
          session_id: state.sessionId,
          model_hint,
          delegate_model_hint,
          attach_camera_frame,
        }),
      });
      const data = await res.json();
      const reply = data?.response || data?.message || JSON.stringify(data);
      
      // Determine role based on model_hint and response metadata
      let responseRole = 'assistant';
      if (model_hint === 'hope-v1' || model_hint === 'hope') {
        // HOPE agent response - left side
        responseRole = 'hope';
      } else if (delegate_model_hint && delegate_model_hint.startsWith('consult:')) {
        // 3rd party model consulted by HOPE - center
        responseRole = 'subagent';
      } else if (model_hint && !model_hint.includes('hope')) {
        // Direct 3rd party model (Gemma, Phi-2, etc.) - center
        responseRole = 'subagent';
      } else if (data?.agent || data?.model) {
        // Check response metadata for agent/model info
        const agentInfo = String(data.agent || data.model || '').toLowerCase();
        if (agentInfo.includes('hope') || agentInfo.includes('agent_manager')) {
          responseRole = 'hope';
        } else {
          responseRole = 'subagent';
        }
      }
      
      appendMessage(reply, responseRole);
      renderStructured(data);
      const speakEl = qs('chat-speak-replies');
      if (speakEl && speakEl.checked) {
        speakOnRobot(reply);
      }
    } catch (err) {
      const href = (window.location && window.location.href) ? window.location.href : '';
      const host = (window.location && window.location.host) ? window.location.host : '';
      const msg = err?.message || 'chat failed';
      const nameNotResolved = /ERR_NAME_NOT_RESOLVED/i.test(msg);
      if (!host || nameNotResolved) {
        appendMessage(`Error: cannot resolve robot host from this page. Open the UI using the robot IP, e.g. http://<robot-ip>:8080/ui (current: ${href})`, 'system');
      } else {
        appendMessage('Error: ' + msg, 'system');
      }
    }
  }

  async function runChatLearn() {
    // Manual agent↔subagent learning loop (server-side).
    // Writes RLDS only when enabled (privacy gate).
    try {
      const topicEl = qs('chat-learn-topic');
      const topic = topicEl ? String(topicEl.value || '').trim() : '';
      const sel = qs('chat-agent-select');
      const choice = sel ? String(sel.value || 'hope-v1') : 'hope-v1';

      let model_hint = 'hope-v1';
      let delegate_model_hint = null;
      if (choice.startsWith('consult:')) {
        delegate_model_hint = choice; // keep consult: prefix
      } else if (choice.startsWith('direct:')) {
        // Direct models are not "learning loops" in the intended sense; still allow.
        model_hint = choice.replace('direct:', '').trim() || 'hope-v1';
      } else if (choice === 'agent_manager') {
        model_hint = null;
      }

      appendMessage(`Starting learn loop${topic ? `: ${topic}` : ''}...`, 'system');
      const res = await fetch('/api/training/chat_learn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          turns: 10,
          topic: topic || null,
          model_hint,
          delegate_model_hint,
          session_id: state.sessionId ? `chat_learn_ui_${state.sessionId}` : null,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.status === 'error') {
        throw new Error(data.message || ('HTTP ' + res.status));
      }
      appendMessage('Learn loop queued/completed (see Training dashboard + RLDS if enabled).', 'system');
      renderStructured(data);
    } catch (err) {
      appendMessage('Learn loop failed: ' + (err?.message || err), 'system');
    }
  }

  function toggleChat() {
    state.minimized = !state.minimized;
    applyMinimized();
    saveState();
  }

  function toggleChatSize() {
    state.expanded = !state.expanded;
    applyLayout();
    saveState();
  }

  function wireDom() {
    const toggleBtn = qs('chat-toggle');
    if (toggleBtn && !toggleBtn.dataset.wired) {
      toggleBtn.dataset.wired = '1';
      toggleBtn.addEventListener('click', toggleChat);
    }
    const expandBtn = qs('chat-expand');
    if (expandBtn && !expandBtn.dataset.wired) {
      expandBtn.dataset.wired = '1';
      expandBtn.addEventListener('click', toggleChatSize);
    }
    const dragHandle = qs('chat-drag-handle');
    if (dragHandle && !dragHandle.dataset.wired) {
      dragHandle.dataset.wired = '1';
      dragHandle.addEventListener('mousedown', startDrag);
      dragHandle.addEventListener('dblclick', toggleChatSize);
    }
    const resizeHandle = qs('chat-resize-handle');
    if (resizeHandle && !resizeHandle.dataset.wired) {
      resizeHandle.dataset.wired = '1';
      resizeHandle.addEventListener('mousedown', startResize);
    }
    const sendBtn = qs('chat-send');
    if (sendBtn && !sendBtn.dataset.wired) {
      sendBtn.dataset.wired = '1';
      sendBtn.addEventListener('click', sendChatMessage);
    }
    const input = qs('chat-input');
    if (input && !input.dataset.wired) {
      input.dataset.wired = '1';
      input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
      });
    }

    const micBtn = qs('chat-mic');
    if (micBtn && !micBtn.dataset.wired) {
      micBtn.dataset.wired = '1';
      micBtn.addEventListener('click', startBrowserSTT);
    }

    const speakToggle = qs('chat-speak-replies');
    if (speakToggle && !speakToggle.dataset.wired) {
      speakToggle.dataset.wired = '1';
      speakToggle.checked = getSpeakEnabled();
      speakToggle.addEventListener('change', () => setSpeakEnabled(!!speakToggle.checked));
    }

    const learnBtn = qs('chat-learn');
    if (learnBtn && !learnBtn.dataset.wired) {
      learnBtn.dataset.wired = '1';
      learnBtn.addEventListener('click', runChatLearn);
    }
  }

  function attachStatusFeed() {
    if (!window.StudioClient || typeof window.StudioClient.startRealtime !== 'function') return;
    const DEBOUNCE_MS = 1500;

    function maybeAnnounce(type, message) {
      const now = Date.now();
      if (now - state.lastPosted[type] < DEBOUNCE_MS) return;
      state.lastPosted[type] = now;
      appendMessage(message, 'system');
    }

    window.StudioClient.startRealtime({
      onStatus: (payload) => {
        const status = payload?.status || payload;
        if (!status) return;
        if (status.mode && status.mode !== state.lastMode) {
          maybeAnnounce('mode', `Mode changed to ${status.mode}`);
          state.lastMode = status.mode;
        }
        const gateSnapshot = status.gate_snapshot || status.gates;
        if (gateSnapshot && gateSnapshot.allow_motion !== state.lastGate) {
          maybeAnnounce(
            'gate',
            `Motion gate: ${gateSnapshot.allow_motion ? 'ENABLED' : 'LOCKED'}`
          );
          state.lastGate = gateSnapshot.allow_motion;
        }
      },
      onLoops: (loops) => {
        if (!loops) return;
        const heartbeat = loops.heartbeat;
        if (heartbeat && typeof heartbeat.ok === 'boolean' && heartbeat.ok !== state.lastHeartbeat) {
          maybeAnnounce('loop', heartbeat.ok ? 'Loop heartbeat stable' : 'Loop heartbeat delayed');
          state.lastHeartbeat = heartbeat.ok;
        }
      },
      onTasks: (tasks) => {
        const taskList = (tasks && tasks.tasks) || tasks || [];
        const count = Array.isArray(taskList) ? taskList.length : 0;
        if (count && count !== state.lastTaskCount) {
          maybeAnnounce('tasks', `Task library updated (${count})`);
          state.lastTaskCount = count;
        }
      },
      onChat: (msg) => {
        if (!msg) return;
        // Ignore user messages reflected back to avoid duplicates if local echo handled it
        if (msg.role === 'user') return;
        
        // Map incoming message role to our positioning system
        let mappedRole = msg.role;
        const roleLower = String(msg.role || '').toLowerCase();
        if (roleLower.includes('hope') || roleLower === 'agent_manager' || roleLower === 'agent-manager') {
          mappedRole = 'hope';
        } else if (roleLower === 'subagent' || roleLower === 'assistant' || roleLower.includes('gemma') || roleLower.includes('phi')) {
          mappedRole = 'subagent';
        }
        
        appendMessage(msg.content, mappedRole, true);
      },
      reconnectDelayMs: 3000,
    });
  }

  function init() {
    loadState();
    applyLayout();
    renderHistory();
    applyMinimized();
    wireDom();
    attachStatusFeed();
  }

  // Expose globals expected by templates
  window.toggleChat = toggleChat;
  window.toggleChatSize = toggleChatSize;
  window.sendChatMessage = sendChatMessage;
  window.runChatLearn = runChatLearn;
  window.renderChatMessage = appendMessage;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

