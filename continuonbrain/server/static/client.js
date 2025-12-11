(function () {
  const subscribers = { status: [], loops: [], tasks: [], skills: [] };
  let eventSource = null;

  function emit(type, payload) {
    (subscribers[type] || []).forEach((fn) => {
      try {
        fn(payload);
      } catch (err) {
        console.warn("StudioClient subscriber error", err);
      }
    });
  }

  async function fetchJson(url, options = {}) {
    const controller = new AbortController();
    const timeoutMs = options.timeoutMs || 8000;
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      const text = await response.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = text;
      }
      if (!response.ok) {
        const err = new Error(`HTTP ${response.status}`);
        err.status = response.status;
        err.data = data;
        throw err;
      }
      return data;
    } finally {
      clearTimeout(timeout);
    }
  }

  function startEventStream({ onStatus, onLoops, onTasks, onSkills, reconnectDelayMs = 2000 }) {
    if (eventSource) {
      eventSource.close();
    }
    const connect = () => {
      eventSource = new EventSource("/api/events");
      eventSource.onmessage = (evt) => {
        if (!evt.data) return;
        try {
          const payload = JSON.parse(evt.data);
          if (payload.status) emit("status", payload.status);
          if (payload.loops) emit("loops", payload.loops);
          if (payload.tasks) emit("tasks", payload.tasks);
          if (payload.skills) emit("skills", payload.skills);
        } catch (err) {
          console.warn("SSE parse error", err);
        }
      };
      eventSource.onerror = () => {
        eventSource.close();
        setTimeout(connect, reconnectDelayMs);
      };
    };

    if (onStatus) subscribers.status.push(onStatus);
    if (onLoops) subscribers.loops.push(onLoops);
    if (onTasks) subscribers.tasks.push(onTasks);
    if (onSkills) subscribers.skills.push(onSkills);
    connect();
  }

  function startRealtime(options = {}) {
    const { onStatus, onLoops, onTasks, onSkills, fallbackPollMs = 8000 } = options;
    startEventStream({ onStatus, onLoops, onTasks, onSkills });

    if (fallbackPollMs && fallbackPollMs > 0) {
      setInterval(() => {
        if (onStatus && typeof window.updateStatus === "function") {
          window.updateStatus();
        }
        if (onLoops && typeof window.pollLoopHealth === "function") {
          window.pollLoopHealth();
        }
        if (onTasks && typeof window.fetchTaskLibrary === "function") {
          window.fetchTaskLibrary();
        }
        if (onSkills && typeof window.fetchSkillLibrary === "function") {
          window.fetchSkillLibrary();
        }
      }, fallbackPollMs);
    }
  }

  window.StudioClient = {
    fetchJson,
    startRealtime,
  };
})();

window.requestTraining = async function() {
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

window.viewTrainingLogs = async function() {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Fetching logs...';
  try {
    const res = await fetch('/api/training/logs');
    const data = await res.json();
    alert(JSON.stringify(data, null, 2));
    if (statusEl) statusEl.textContent = 'Logs loaded';
  } catch (err) {
    console.warn('Logs fetch failed', err);
    if (statusEl) statusEl.textContent = 'Logs fetch failed';
  }
};

window.openEpisodeImports = function() {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Episode import UI not yet wired';
  alert('Episode import UI not yet wired to backend.');
};

window.runManualTraining = async function(payload = {}) {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Submitting manual training run...';
  try {
    const res = await fetch('/api/training/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'Manual training failed');
    if (statusEl) statusEl.textContent = 'Manual training queued';
  } catch (err) {
    console.warn('Manual training failed', err);
    if (statusEl) statusEl.textContent = 'Manual training failed: ' + (err?.message || err);
  }
};

window.runWavecoreLoops = async function(payload = {}) {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Starting WaveCore loops...';
  try {
    const res = await fetch('/api/training/wavecore_loops', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'WaveCore run failed');
    if (statusEl) statusEl.textContent = 'WaveCore loops running';
  } catch (err) {
    console.warn('WaveCore loops failed', err);
    if (statusEl) statusEl.textContent = 'WaveCore loops failed: ' + (err?.message || err);
  }
};

window.runHopeEval = async function(payload = {}) {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Running HOPE eval...';
  try {
    const res = await fetch('/api/training/hope_eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'HOPE eval failed');
    if (statusEl) statusEl.textContent = 'HOPE eval completed';
  } catch (err) {
    console.warn('HOPE eval failed', err);
    if (statusEl) statusEl.textContent = 'HOPE eval failed: ' + (err?.message || err);
  }
};

window.runFactsEval = async function(payload = {}) {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Running facts eval...';
  try {
    const res = await fetch('/api/training/hope_eval_facts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'Facts eval failed');
    if (statusEl) statusEl.textContent = 'Facts eval completed';
  } catch (err) {
    console.warn('Facts eval failed', err);
    if (statusEl) statusEl.textContent = 'Facts eval failed: ' + (err?.message || err);
  }
};

window.checkTrainingStatus = async function() {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Reading trainer status...';
  try {
    const res = await fetch('/api/training/status');
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const label = data.message || data.status || JSON.stringify(data);
    if (statusEl) statusEl.textContent = 'Trainer: ' + label;
  } catch (err) {
    console.warn('Training status failed', err);
    if (statusEl) statusEl.textContent = 'Status fetch failed: ' + (err?.message || err);
  }
};

// Initialize view mode from storage on load (requires setViewMode in page)
(function initViewModeClient() {
  try {
    const saved = localStorage.getItem('studio_view_mode');
    if (saved && typeof window.setViewMode === 'function') {
      window.setViewMode(saved);
    }
  } catch (err) {
    console.warn('Unable to init view mode', err);
  }
})();

