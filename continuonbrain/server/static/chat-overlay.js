// Shared chat overlay logic for UI + Control pages.
// Provides consistent chat UX, history persistence, and live status feed hooks.
(function () {
  const storageKeyPrefix = 'continuon_chat_' + (window.location.host || 'local');
  const historyKey = storageKeyPrefix + '_history';
  const minimizedKey = storageKeyPrefix + '_minimized';
  const MAX_HISTORY = 50;

  let state = {
    minimized: false,
    history: [],
    pendingId: null,
    lastMode: null,
    lastGate: null,
    lastPosted: {
      mode: 0,
      gate: 0,
    },
  };

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
      state.minimized = localStorage.getItem(minimizedKey) === 'true';
    } catch (e) {
      console.warn('Chat minimized load failed', e);
    }
  }

  function saveState() {
    try {
      const trimmed = state.history.slice(-MAX_HISTORY);
      localStorage.setItem(historyKey, JSON.stringify(trimmed));
      localStorage.setItem(minimizedKey, state.minimized ? 'true' : 'false');
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
        }),
      });
      const data = await res.json();
      const reply = data?.response || data?.message || JSON.stringify(data);
      appendMessage(reply, 'assistant');
    } catch (err) {
      appendMessage('Error: ' + (err?.message || 'chat failed'), 'system');
    }
  }

  function toggleChat() {
    state.minimized = !state.minimized;
    applyMinimized();
    saveState();
  }

  function wireDom() {
    const toggleBtn = qs('chat-toggle');
    if (toggleBtn && !toggleBtn.dataset.wired) {
      toggleBtn.dataset.wired = '1';
      toggleBtn.addEventListener('click', toggleChat);
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
        if (!loops || !loops.metrics) return;
      },
      reconnectDelayMs: 3000,
    });
  }

  function init() {
    loadState();
    renderHistory();
    applyMinimized();
    wireDom();
    attachStatusFeed();
  }

  // Expose globals expected by templates
  window.toggleChat = toggleChat;
  window.sendChatMessage = sendChatMessage;
  window.renderChatMessage = appendMessage;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

