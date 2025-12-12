// Shared chat overlay logic for UI + Control pages.
// Provides consistent chat UX, history persistence, and live status feed hooks.
(function () {
  const storageKeyPrefix = 'continuon_chat_' + (window.location.host || 'local');
  const historyKey = storageKeyPrefix + '_history';
  const minimizedKey = storageKeyPrefix + '_minimized';
  const positionKey = storageKeyPrefix + '_position';
  const sizeKey = storageKeyPrefix + '_size';
  const expandedKey = storageKeyPrefix + '_expanded';
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
    div.className = 'chat-message ' + role;
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
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: state.history.slice(-10),
          session_id: state.sessionId,
        }),
      });
      const data = await res.json();
      const reply = data?.response || data?.message || JSON.stringify(data);
      appendMessage(reply, 'assistant');
      renderStructured(data);
    } catch (err) {
      appendMessage('Error: ' + (err?.message || 'chat failed'), 'system');
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
  window.renderChatMessage = appendMessage;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

