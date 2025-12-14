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

window.runArmPlanner = async function () {
  const out = document.getElementById('plan-output');
  const g0 = document.getElementById('plan-goal-j0');
  const g1 = document.getElementById('plan-goal-j1');
  const g2 = document.getElementById('plan-goal-j2');
  const g3 = document.getElementById('plan-goal-j3');
  const g4 = document.getElementById('plan-goal-j4');
  const g5 = document.getElementById('plan-goal-j5');
  const execute = document.getElementById('plan-execute');
  const goal = [
    g0 ? Number(g0.value || 0) : 0.2,
    g1 ? Number(g1.value || 0) : 0,
    g2 ? Number(g2.value || 0) : 0,
    g3 ? Number(g3.value || 0) : 0,
    g4 ? Number(g4.value || 0) : 0,
    g5 ? Number(g5.value || 0) : 0,
  ];
  const payload = {
    goal_joint_pos: goal,
    execute: !!(execute && execute.checked),
    horizon: 6,
    beam_width: 6,
    action_step: 0.05,
    time_budget_ms: 150,
  };
  if (out) out.textContent = 'Planning...';
  try {
    const res = await fetch('/api/planning/arm_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    window.__lastArmPlan = data;
    window.__lastArmPlanIdx = 0;
    if (out) out.textContent = JSON.stringify(data, null, 2);
    if (typeof window.renderPlanSteps === 'function') window.renderPlanSteps(data);
    if (typeof window.setPlanRawVisible === 'function') window.setPlanRawVisible();
  } catch (err) {
    console.warn('runArmPlanner failed', err);
    if (out) out.textContent = 'Planner failed: ' + (err?.message || err);
  }
};

function describeConnectivityHint(err) {
  const href = (typeof window !== 'undefined' && window.location && window.location.href) ? window.location.href : '(unknown)';
  const host = (typeof window !== 'undefined' && window.location && window.location.host) ? window.location.host : '';
  const msg = err && err.message ? err.message : String(err || 'fetch failed');
  const nameNotResolved = /ERR_NAME_NOT_RESOLVED/i.test(msg);
  if (!host || nameNotResolved) {
    return `Cannot resolve the robot host from this page (${href}). Open the UI using the robot IP or a resolvable hostname, e.g. http://<robot-ip>:8080/ui`;
  }
  return `Cannot reach robot API from ${href} (${msg}).`;
}

window.updateStatus = async function () {
  const badge = document.getElementById('connection-status');
  const sys = document.getElementById('system-info-content');
  if (badge) badge.textContent = 'Connecting…';

  try {
    const data = await window.StudioClient.fetchJson('/api/status', { timeoutMs: 4000 });
    const status = data && (data.status || data);
    if (badge) badge.textContent = 'Connected';
    if (sys && status && typeof status === 'object') {
      const hw = status.hardware_mode || status.hardwareMode || 'unknown';
      const mode = status.mode || status.robot_mode || 'unknown';
      sys.innerHTML = `
        <div class="metric-row"><div class="metric-label">Mode</div><div class="metric-value">${mode}</div></div>
        <div class="metric-row"><div class="metric-label">Hardware</div><div class="metric-value">${hw}</div></div>
      `;
    }
    // Best-effort: populate the legacy Home/Status widgets if present.
    try {
      applyStatusToHomePanels(status);
    } catch (e) {
      // ignore
    }
    return data;
  } catch (err) {
    if (badge) badge.textContent = 'Disconnected';
    if (sys) {
      sys.innerHTML = `<div class="panel-subtitle" style="color:#ffb3c0">${describeConnectivityHint(err)}</div>`;
    }
    return null;
  }
};

// Initialize status quickly so users immediately see the agent/server connection state.
document.addEventListener('DOMContentLoaded', () => {
  if (typeof window.updateStatus === 'function') {
    window.updateStatus();
    setInterval(() => window.updateStatus(), 12000);
  }

  // Page-specific auto-wiring
  if (document.getElementById('task-list') && typeof window.fetchTaskLibrary === 'function') {
    window.fetchTaskLibrary();
  }
  if (document.getElementById('skill-list') && typeof window.fetchSkillLibrary === 'function') {
    window.fetchSkillLibrary();
  }
  if (document.getElementById('safety-gates-list') && typeof window.pollLoopHealth === 'function') {
    window.pollLoopHealth();
    setInterval(() => window.pollLoopHealth(), 12000);
  }

  if (document.getElementById('reality-proof-score') || document.getElementById('capability-matrix')) {
    // Research page: best-effort populate.
    window.refreshResearchPanels?.();
  }

  // Agent details panel: keep lightweight lists fresh.
  if (typeof window.updateAgentDetailsPanel === 'function') {
    window.updateAgentDetailsPanel();
    setInterval(() => window.updateAgentDetailsPanel(), 15000);
  }
});

function applyStatusToHomePanels(status) {
  if (!status || typeof status !== 'object') return;
  const modeBadge = document.getElementById('robot-mode-badge');
  const hwEl = document.getElementById('hardware-mode');
  const battEl = document.getElementById('battery-status');
  const uptimeEl = document.getElementById('uptime');
  const healthEl = document.getElementById('health-metrics');

  const mode = status.mode || status.robot_mode || 'unknown';
  const hw = status.hardware_mode || status.hardwareMode || 'unknown';
  const battery = status.battery || {};
  const battLabel = battery && typeof battery === 'object'
    ? (battery.percent != null ? `${battery.percent}%` : (battery.available === false ? 'unavailable' : (battery.error || '—')))
    : '—';
  const uptimeS = Number(status.mode_duration ?? status.uptime_s ?? status.uptime_seconds);
  const uptimeLabel = Number.isFinite(uptimeS) ? `${Math.max(0, uptimeS).toFixed(0)}s` : '—';

  if (modeBadge) modeBadge.textContent = String(mode);
  if (hwEl) hwEl.textContent = String(hw);
  if (battEl) battEl.textContent = String(battLabel);
  if (uptimeEl) uptimeEl.textContent = String(uptimeLabel);

  if (healthEl) {
    const gates = status.gate_snapshot || status.gates || {};
    const safety = status.safety_head || {};
    const cap = status.capabilities || {};
    const lines = [];
    if (gates && typeof gates === 'object') {
      lines.push(`<div class="metric-row"><div class="metric-label">Motion gate</div><div class="metric-value">${gates.allow_motion ? 'ENABLED' : 'LOCKED'}</div></div>`);
      lines.push(`<div class="metric-row"><div class="metric-label">Recording</div><div class="metric-value">${gates.record_episodes ? 'ON' : 'OFF'}</div></div>`);
      lines.push(`<div class="metric-row"><div class="metric-label">Inference</div><div class="metric-value">${gates.run_inference ? 'ON' : 'OFF'}</div></div>`);
    }
    if (safety && typeof safety === 'object') {
      lines.push(`<div class="metric-row"><div class="metric-label">Safety head</div><div class="metric-value">${escapeHtml(String(safety.status || '—'))}</div></div>`);
    }
    if (cap && typeof cap === 'object') {
      lines.push(`<div class="metric-row"><div class="metric-label">Vision</div><div class="metric-value">${cap.has_vision ? 'yes' : 'no'}</div></div>`);
      lines.push(`<div class="metric-row"><div class="metric-label">Manipulator</div><div class="metric-value">${cap.has_manipulator ? 'yes' : 'no'}</div></div>`);
    }
    healthEl.innerHTML = lines.length ? lines.join('') : '<div class="stack-item"><span class="stack-meta">No health telemetry yet</span></div>';
  }
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload ? JSON.stringify(payload) : '{}',
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const err = new Error('HTTP ' + res.status);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

// Agent rail actions (wired)
window.toggleAgentDetails = function () {
  const panel = document.getElementById('agent-details-panel');
  const btn = document.getElementById('agent-details-toggle');
  if (!panel) return;
  const open = panel.classList.toggle('open');
  if (btn) {
    btn.textContent = open ? 'Hide details' : 'Show details';
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
};

window.pauseAgents = async function () {
  try {
    await postJson('/api/mode/idle');
  } catch (err) {
    console.warn('pauseAgents failed', err);
  } finally {
    if (typeof window.updateStatus === 'function') window.updateStatus();
  }
};

window.resumeAgents = async function () {
  try {
    await postJson('/api/mode/autonomous');
  } catch (err) {
    console.warn('resumeAgents failed', err);
  } finally {
    if (typeof window.updateStatus === 'function') window.updateStatus();
  }
};

window.reviewLearning = function () {
  try {
    const panel = document.getElementById('agent-details-panel');
    if (panel && !panel.classList.contains('open')) {
      window.toggleAgentDetails();
    }
    const target = document.getElementById('training-dashboard-list') || document.getElementById('training-memory-list');
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    console.warn('reviewLearning failed', err);
  }
};

window.toggleHumanMode = async function () {
  const toggle = document.getElementById('human-toggle');
  const stateEl = document.getElementById('human-toggle-state');
  const key = 'continuon_human_mode';
  let on = false;
  try {
    on = localStorage.getItem(key) === '1';
  } catch {
    on = false;
  }
  const next = !on;
  try {
    localStorage.setItem(key, next ? '1' : '0');
  } catch {}
  if (toggle) toggle.classList.toggle('active', next);
  if (stateEl) stateEl.textContent = next ? 'On' : 'Off';
  if (next) {
    await window.pauseAgents();
  }
};

window.clearPlan = function () {
  window.__lastArmPlan = null;
  window.__lastArmPlanIdx = 0;
  const out = document.getElementById('plan-output');
  const steps = document.getElementById('plan-steps');
  if (steps) steps.innerHTML = '<div class="stack-item"><span class="stack-meta">(no plan yet)</span></div>';
  if (out) out.textContent = '(no plan yet)';
};

window.executeNextPlanStep = async function () {
  const out = document.getElementById('plan-output');
  const plan = window.__lastArmPlan;
  const idx = window.__lastArmPlanIdx || 0;
  const steps = plan?.plan?.steps || [];
  if (!steps.length) {
    if (out) out.textContent = 'No plan steps available. Click Plan first.';
    return;
  }
  if (idx >= steps.length) {
    if (out) out.textContent = 'Plan complete (no more steps).';
    return;
  }
  const joint_delta = steps[idx]?.joint_delta;
  if (!Array.isArray(joint_delta) || joint_delta.length !== 6) {
    if (out) out.textContent = 'Invalid joint_delta at step ' + idx;
    return;
  }
  try {
    const res = await fetch('/api/planning/arm_execute_delta', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ joint_delta }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    window.__lastArmPlanIdx = idx + 1;
    if (typeof window.renderPlanSteps === 'function') window.renderPlanSteps(plan);
    const rawToggle = document.getElementById('plan-show-raw');
    if (out && rawToggle && rawToggle.checked) {
      out.textContent = `Executed step ${idx}: ` + JSON.stringify(data, null, 2) + '\n\n' + JSON.stringify(plan, null, 2);
    }
  } catch (err) {
    console.warn('executeNextPlanStep failed', err);
    if (out) out.textContent = 'Execute failed: ' + (err?.message || err);
  }
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

window.viewTrainingLogs = async function () {
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

window.openEpisodeImports = function () {
  const statusEl = document.getElementById('training-status');
  if (statusEl) statusEl.textContent = 'Loading episode import summary…';
  // "Episode imports" = imported tool-use datasets (toolchat_hf* folders) summary.
  fetch('/api/training/tool_dataset_summary?limit=2000')
    .then((r) => r.json().then((j) => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      if (!ok || j?.status === 'error') throw new Error(j?.message || 'tool dataset summary failed');
      const sources = Array.isArray(j.sources) ? j.sources : [];
      const brief = sources.map((s) => {
        const name = (s.dir || '').split('/').slice(-1)[0] || 'dataset';
        const rate = (typeof s.tool_call_rate === 'number') ? (s.tool_call_rate * 100).toFixed(1) + '%' : '—';
        return `${name}: episodes=${s.episodes ?? '—'} steps=${s.steps_total ?? '—'} tool_call_rate=${rate}`;
      }).join('\n');
      if (statusEl) statusEl.textContent = sources.length ? `Episode imports: ${sources.length} dataset(s) detected` : 'Episode imports: none found';
      alert(brief || 'No toolchat_hf_* datasets found under /opt/continuonos/brain/rlds/episodes.');
    })
    .catch((err) => {
      console.warn('openEpisodeImports failed', err);
      if (statusEl) statusEl.textContent = 'Episode import summary failed';
      alert('Episode import summary failed: ' + (err?.message || err));
    });
};

window.runManualTraining = async function (payload = {}) {
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

window.runWavecoreLoops = async function (payload = {}) {
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

window.runHopeEval = async function (payload = {}) {
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

window.runFactsEval = async function (payload = {}) {
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

window.checkTrainingStatus = async function () {
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

// --- Page wiring for Tasks / Skills / Safety ---
window.fetchTaskLibrary = async function () {
  const container = document.getElementById('task-list');
  if (!container) return null;
  container.innerHTML = '<div class="loading-spinner">Loading tasks...</div>';
  try {
    const data = await window.StudioClient.fetchJson('/api/tasks?include_ineligible=false', { timeoutMs: 6000 });
    const tasks = data?.tasks || data || [];
    if (!Array.isArray(tasks) || tasks.length === 0) {
      container.innerHTML = '<div class="stack-item"><span class="stack-meta">No tasks available</span></div>';
      return data;
    }
    container.innerHTML = tasks.map((t) => {
      const id = escapeHtml(t.id || t.task_id || '');
      const title = escapeHtml(t.title || t.name || id || 'task');
      const desc = escapeHtml(t.description || '');
      const eligible = (t.eligible === true) ? 'eligible' : ((t.eligible === false) ? 'ineligible' : 'unknown');
      return (
        `<div class="stack-item" style="cursor:pointer" onclick="window.selectTask('${id}')">` +
          `<div style="min-width:0;"><h4>${title}</h4><div class="stack-meta">${desc}</div></div>` +
          `<div><span class="status-chip ${eligible === 'eligible' ? 'active' : 'warning'}">${eligible.toUpperCase()}</span></div>` +
        `</div>`
      );
    }).join('');
    return data;
  } catch (err) {
    console.warn('fetchTaskLibrary failed', err);
    container.innerHTML = '<div class="stack-item"><span class="stack-meta">Failed to load tasks</span></div>';
    return null;
  }
};

window.selectTask = async function (taskId) {
  const details = document.getElementById('task-details');
  if (details) details.innerHTML = '<div class="loading-spinner">Loading task...</div>';
  try {
    const res = await window.StudioClient.fetchJson(`/api/tasks/summary/${encodeURIComponent(taskId)}`, { timeoutMs: 6000 });
    const summary = res?.summary || res || {};
    if (!details) return;
    details.innerHTML =
      `<div class="stack-item"><div><h4>${escapeHtml(summary.title || summary.name || taskId)}</h4><div class="stack-meta">${escapeHtml(summary.description || '')}</div></div>` +
      `<div><button class="rail-btn primary" type="button" onclick="window.activateTask('${escapeHtml(taskId)}')">Select</button></div></div>` +
      `<pre style="white-space:pre-wrap; font-size:12px; color:#cfd7ff; background:#121826; padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.08); max-height:320px; overflow:auto;">${escapeHtml(JSON.stringify(res, null, 2))}</pre>`;
  } catch (err) {
    console.warn('selectTask failed', err);
    if (details) details.innerHTML = '<div class="stack-item"><span class="stack-meta">Failed to load task summary</span></div>';
  }
};

window.activateTask = async function (taskId) {
  try {
    await postJson('/api/tasks/select', { task_id: taskId, reason: 'selected from web UI' });
  } catch (err) {
    console.warn('activateTask failed', err);
  } finally {
    if (typeof window.fetchTaskLibrary === 'function') window.fetchTaskLibrary();
  }
};

window.refreshTasks = function () {
  return window.fetchTaskLibrary();
};

window.fetchSkillLibrary = async function () {
  const container = document.getElementById('skill-list');
  if (!container) return null;
  container.innerHTML = '<div class="loading-spinner">Loading skills...</div>';
  try {
    const data = await window.StudioClient.fetchJson('/api/skills?include_ineligible=false', { timeoutMs: 6000 });
    const skills = data?.skills || data || [];
    if (!Array.isArray(skills) || skills.length === 0) {
      container.innerHTML = '<div class="stack-item"><span class="stack-meta">No skills available</span></div>';
      return data;
    }
    container.innerHTML = skills.map((s) => {
      const id = escapeHtml(s.id || s.skill_id || '');
      const title = escapeHtml(s.title || s.name || id || 'skill');
      const desc = escapeHtml(s.description || '');
      const eligible = (s.eligible === true) ? 'eligible' : ((s.eligible === false) ? 'ineligible' : 'unknown');
      return (
        `<div class="stack-item" style="cursor:pointer" onclick="window.selectSkill('${id}')">` +
          `<div style="min-width:0;"><h4>${title}</h4><div class="stack-meta">${desc}</div></div>` +
          `<div><span class="status-chip ${eligible === 'eligible' ? 'active' : 'warning'}">${eligible.toUpperCase()}</span></div>` +
        `</div>`
      );
    }).join('');
    return data;
  } catch (err) {
    console.warn('fetchSkillLibrary failed', err);
    container.innerHTML = '<div class="stack-item"><span class="stack-meta">Failed to load skills</span></div>';
    return null;
  }
};

window.selectSkill = async function (skillId) {
  const details = document.getElementById('skill-details');
  if (details) details.innerHTML = '<div class="loading-spinner">Loading skill...</div>';
  try {
    const res = await window.StudioClient.fetchJson(`/api/skills/summary/${encodeURIComponent(skillId)}`, { timeoutMs: 6000 });
    if (!details) return;
    const summary = res?.summary || res || {};
    details.innerHTML =
      `<div class="stack-item"><div><h4>${escapeHtml(summary.title || summary.name || skillId)}</h4><div class="stack-meta">${escapeHtml(summary.description || '')}</div></div></div>` +
      `<pre style="white-space:pre-wrap; font-size:12px; color:#cfd7ff; background:#121826; padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.08); max-height:320px; overflow:auto;">${escapeHtml(JSON.stringify(res, null, 2))}</pre>`;
  } catch (err) {
    console.warn('selectSkill failed', err);
    if (details) details.innerHTML = '<div class="stack-item"><span class="stack-meta">Failed to load skill summary</span></div>';
  }
};

window.refreshSkills = function () {
  return window.fetchSkillLibrary();
};

window.pollLoopHealth = async function () {
  const list = document.getElementById('safety-gates-list');
  if (!list) return null;
  try {
    const data = await window.StudioClient.fetchJson('/api/gates', { timeoutMs: 5000 });
    const gates = data?.gates || data || {};
    if (!gates || typeof gates !== 'object') {
      list.innerHTML = '<div class="stack-item"><span class="stack-meta">No gates reported</span></div>';
      return data;
    }
    const rows = Object.entries(gates).map(([k, v]) => {
      const val = String(v);
      const locked = (val === 'locked' || val === 'closed' || val === 'false');
      return (
        `<div class="stack-item">` +
          `<div><h4>${escapeHtml(k)}</h4><div class="stack-meta">${escapeHtml(val)}</div></div>` +
          `<div><span class="status-chip ${locked ? 'warning' : 'active'}">${locked ? 'LOCKED' : 'OK'}</span></div>` +
        `</div>`
      );
    });
    list.innerHTML = rows.join('') || '<div class="stack-item"><span class="stack-meta">No gates reported</span></div>';
    return data;
  } catch (err) {
    console.warn('pollLoopHealth failed', err);
    if (list) list.innerHTML = '<div class="stack-item"><span class="stack-meta">Failed to load safety gates</span></div>';
    return null;
  }
};

window.triggerEStop = async function () {
  try {
    await postJson('/api/safety/hold');
  } catch (err) {
    console.warn('triggerEStop failed', err);
  } finally {
    if (typeof window.pollLoopHealth === 'function') window.pollLoopHealth();
    if (typeof window.updateStatus === 'function') window.updateStatus();
  }
};

// Safety page buttons map onto the same mode actions as the agent rail.
window.pauseRobot = window.pauseAgents;
window.resumeRobot = window.resumeAgents;

// Research page wiring (best-effort, offline-first)
window.refreshResearchPanels = async function () {
  const proofEl = document.getElementById('reality-proof-score');
  const dreamEl = document.getElementById('dream-consistency-score');
  const matrixEl = document.getElementById('capability-matrix');
  const chartEl = document.getElementById('world-model-chart');

  try {
    const [statusRes, dqRes, archRes, tasksRes, skillsRes] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/training/data_quality?limit=40&step_cap=2500'),
      fetch('/api/training/architecture_status'),
      fetch('/api/tasks?include_ineligible=false'),
      fetch('/api/skills?include_ineligible=false'),
    ]);
    const status = await statusRes.json().catch(() => ({}));
    const dq = await dqRes.json().catch(() => ({}));
    const arch = await archRes.json().catch(() => ({}));
    const tasks = await tasksRes.json().catch(() => ({}));
    const skills = await skillsRes.json().catch(() => ({}));

    // Simple heuristic scores (placeholders, but connected to real signals)
    const actNonzero = Number(dq?.action?.nonzero_rate);
    const obsNumeric = Number(dq?.observation?.numeric_rate);
    const proofScore = (Number.isFinite(actNonzero) && Number.isFinite(obsNumeric))
      ? Math.max(0, Math.min(100, Math.round(((actNonzero + obsNumeric) / 2) * 100)))
      : null;
    const dreamScore = (arch?.status === 'ok')
      ? 100
      : null;

    if (proofEl) proofEl.textContent = (proofScore == null) ? '--%' : `${proofScore}%`;
    if (dreamEl) dreamEl.textContent = (dreamScore == null) ? '--%' : `${dreamScore}%`;

    if (chartEl) {
      chartEl.innerHTML =
        `<pre style="white-space:pre-wrap; font-size:12px; color:#cfd7ff; background:#121826; padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.08); max-height:320px; overflow:auto;">` +
        `${escapeHtml(JSON.stringify({ architecture_status: arch, data_quality: dq }, null, 2))}` +
        `</pre>`;
    }

    if (matrixEl) {
      const cap = status?.status?.capabilities || status?.capabilities || {};
      const taskCount = Array.isArray(tasks?.tasks) ? tasks.tasks.length : (Array.isArray(tasks) ? tasks.length : null);
      const skillCount = Array.isArray(skills?.skills) ? skills.skills.length : (Array.isArray(skills) ? skills.length : null);
      matrixEl.innerHTML =
        `<div class="stack-item"><div><h4>Capabilities</h4><div class="stack-meta">Runtime-reported modality support</div></div></div>` +
        `<div class="metric-row"><div class="metric-label">Vision</div><div class="metric-value">${cap.has_vision ? 'yes' : 'no'}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Manipulator</div><div class="metric-value">${cap.has_manipulator ? 'yes' : 'no'}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Mobile base</div><div class="metric-value">${cap.has_mobile_base ? 'yes' : 'no'}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Audio</div><div class="metric-value">${cap.has_audio ? 'yes' : 'no'}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Tasks (eligible)</div><div class="metric-value">${taskCount == null ? '—' : String(taskCount)}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Skills (eligible)</div><div class="metric-value">${skillCount == null ? '—' : String(skillCount)}</div></div>`;
    }
  } catch (err) {
    console.warn('refreshResearchPanels failed', err);
    if (matrixEl) matrixEl.innerHTML = '<div class="stack-item"><span class="stack-meta">Research panels unavailable</span></div>';
  }
};

window.updateAgentDetailsPanel = async function () {
  const threadList = document.getElementById('agent-thread-list');
  const modelList = document.getElementById('model-stack-list');
  const toolList = document.getElementById('toolchain-list');
  const milestones = document.getElementById('learning-milestones');
  const eventsEl = document.getElementById('learning-events');

  if (!threadList && !modelList && !toolList && !milestones && !eventsEl) return;

  try {
    const [statusRes, archRes, eventsRes] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/training/architecture_status'),
      fetch('/api/system/events?limit=40'),
    ]);
    const statusPayload = await statusRes.json().catch(() => ({}));
    const status = statusPayload?.status || statusPayload || {};
    const arch = await archRes.json().catch(() => ({}));
    const events = await eventsRes.json().catch(() => ({}));

    const caps = status?.capabilities || {};
    const gates = status?.gate_snapshot || {};
    const threads = arch?.tasks || {};

    if (threadList) {
      const rows = [];
      for (const [k, v] of Object.entries(threads || {})) {
        const alive = !!v?.alive;
        rows.push(
          `<div class="stack-item">` +
            `<div><h4>${escapeHtml(k)}</h4><div class="stack-meta">${escapeHtml(v?.present ? (alive ? 'alive' : 'stopped') : 'absent')}</div></div>` +
            `<div><span class="status-chip ${alive ? 'active' : 'warning'}">${alive ? 'OK' : 'OFF'}</span></div>` +
          `</div>`
        );
      }
      threadList.innerHTML = rows.length ? rows.join('') : '<div class="stack-item"><span class="stack-meta">No agent threads reported</span></div>';
    }

    if (modelList) {
      const hw = status?.hardware_mode || 'unknown';
      const det = status?.detected_hardware || {};
      const hailo = arch?.hailo || {};
      modelList.innerHTML =
        `<div class="stack-item"><div><h4>Runtime</h4><div class="stack-meta">hardware_mode=${escapeHtml(hw)}</div></div></div>` +
        `<div class="metric-row"><div class="metric-label">Vision</div><div class="metric-value">${caps.has_vision ? 'yes' : 'no'}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Manipulator</div><div class="metric-value">${caps.has_manipulator ? 'yes' : 'no'}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Depth cam</div><div class="metric-value">${escapeHtml(det.depth_camera || '—')}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Accelerator</div><div class="metric-value">${escapeHtml(det.ai_accelerator || '—')}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Hailo devices</div><div class="metric-value">${escapeHtml(String((hailo.hardware?.dev_nodes || []).join(', ') || '—'))}</div></div>`;
    }

    if (toolList) {
      const toolRouter = arch?.tool_router || {};
      toolList.innerHTML =
        `<div class="metric-row"><div class="metric-label">Chat API</div><div class="metric-value">/api/chat</div></div>` +
        `<div class="metric-row"><div class="metric-label">SSE stream</div><div class="metric-value">/api/events</div></div>` +
        `<div class="metric-row"><div class="metric-label">Tool-router export</div><div class="metric-value">${escapeHtml(toolRouter.export_dir || '—')}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Tool-router metrics</div><div class="metric-value">${escapeHtml(toolRouter.metrics_path || '—')}</div></div>` +
        `<div class="metric-row"><div class="metric-label">Tool-router eval</div><div class="metric-value">${escapeHtml(toolRouter.eval_metrics_path || '—')}</div></div>`;
    }

    if (milestones) {
      milestones.innerHTML =
        `<div class="milestone-row"><div>Mode</div><div>${escapeHtml(String(status?.mode || '—'))}</div></div>` +
        `<div class="milestone-row"><div>Allow motion</div><div>${gates.allow_motion ? 'yes' : 'no'}</div></div>` +
        `<div class="milestone-row"><div>Record episodes</div><div>${gates.record_episodes ? 'yes' : 'no'}</div></div>` +
        `<div class="milestone-row"><div>Run inference</div><div>${gates.run_inference ? 'yes' : 'no'}</div></div>`;
    }

    if (eventsEl) {
      const items = Array.isArray(events?.items) ? events.items : [];
      if (!items.length) {
        eventsEl.innerHTML = '<div class="stack-item"><span class="stack-meta">No system events yet</span></div>';
      } else {
        eventsEl.innerHTML = items.slice().reverse().map((ev) => {
          const t = ev.timestamp || ev.time || ev.ts || '';
          const kind = ev.event_type || ev.type || 'event';
          const msg = ev.message || ev.detail || ev.summary || '';
          return `<div class="learning-item"><strong>${escapeHtml(kind)}</strong><div class="learning-meta">${escapeHtml(String(t))} ${escapeHtml(String(msg))}</div></div>`;
        }).join('');
      }
    }
  } catch (err) {
    console.warn('updateAgentDetailsPanel failed', err);
  }
};

function formatTs(isoOrUnix) {
  if (!isoOrUnix) return '—';
  try {
    if (typeof isoOrUnix === 'number') {
      return new Date(isoOrUnix * 1000).toLocaleString();
    }
    // iso string
    return new Date(isoOrUnix).toLocaleString();
  } catch {
    return String(isoOrUnix);
  }
}

function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function fmtVec(vec) {
  if (!Array.isArray(vec)) return '—';
  return '[' + vec.map((v) => {
    const n = Number(v);
    if (Number.isFinite(n)) return n.toFixed(2);
    return String(v);
  }).join(', ') + ']';
}

window.setPlanRawVisible = function () {
  const rawToggle = document.getElementById('plan-show-raw');
  const pre = document.getElementById('plan-output');
  if (!pre) return;
  const on = !!(rawToggle && rawToggle.checked);
  pre.style.display = on ? 'block' : 'none';
};

window.renderPlanSteps = function (planResp) {
  const container = document.getElementById('plan-steps');
  if (!container) return;
  const steps = planResp?.plan?.steps || [];
  const idx = window.__lastArmPlanIdx || 0;

  if (!Array.isArray(steps) || steps.length === 0) {
    container.innerHTML = '<div class="stack-item"><span class="stack-meta">(no steps yet)</span></div>';
    return;
  }

  container.innerHTML = steps.map((s, i) => {
    const active = i === idx ? ' primary' : '';
    const delta = fmtVec(s?.joint_delta);
    const pred = fmtVec(s?.predicted_joint_pos);
    const score = (s?.score !== undefined) ? String(Number(s.score).toFixed(3)) : '—';
    const unc = (s?.uncertainty !== undefined) ? String(Number(s.uncertainty).toFixed(3)) : '—';
    return (
      `<div class="stack-item${active}">` +
        `<div>` +
          `<h4>Step ${i}</h4>` +
          `<div class="stack-meta">Δ ${escapeHtml(delta)}</div>` +
          `<div class="stack-meta">Pred ${escapeHtml(pred)}</div>` +
          `<div class="stack-flags">` +
            `<span class="task-tag">score:${escapeHtml(score)}</span>` +
            `<span class="task-tag">unc:${escapeHtml(unc)}</span>` +
          `</div>` +
        `</div>` +
        `<div style="display:flex; gap:8px; align-items:center;">` +
          `<button class="rail-btn" onclick="window.executePlanStep(${i})">Execute</button>` +
        `</div>` +
      `</div>`
    );
  }).join('');
};

window.executePlanStep = async function (stepIdx) {
  const out = document.getElementById('plan-output');
  const plan = window.__lastArmPlan;
  const steps = plan?.plan?.steps || [];
  const s = steps[stepIdx];
  const joint_delta = s?.joint_delta;
  if (!Array.isArray(joint_delta) || joint_delta.length !== 6) {
    if (out) out.textContent = 'Invalid joint_delta at step ' + stepIdx;
    return;
  }
  try {
    const res = await fetch('/api/planning/arm_execute_delta', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ joint_delta }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    window.__lastArmPlanIdx = stepIdx + 1;
    window.renderPlanSteps(plan);
    const rawToggle = document.getElementById('plan-show-raw');
    if (out && rawToggle && rawToggle.checked) {
      out.textContent = `Executed step ${stepIdx}: ` + JSON.stringify(data, null, 2) + '\n\n' + JSON.stringify(plan, null, 2);
    }
  } catch (err) {
    console.warn('executePlanStep failed', err);
    if (out) out.textContent = 'Execute failed: ' + (err?.message || err);
  }
};

window.updatePlannerGate = async function () {
  const chip = document.getElementById('plan-motion-gate');
  if (!chip) return;
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const status = data?.status || data;
    const allow = !!(status?.allow_motion);
    chip.textContent = allow ? 'MOTION ENABLED' : 'MOTION LOCKED';
    chip.className = 'status-chip ' + (allow ? 'active' : 'warning');
  } catch {
    chip.textContent = '--';
    chip.className = 'status-chip info';
  }
};

(function initPlannerUi() {
  function wire() {
    const rawToggle = document.getElementById('plan-show-raw');
    if (rawToggle && !rawToggle.dataset.wired) {
      rawToggle.dataset.wired = '1';
      rawToggle.addEventListener('change', window.setPlanRawVisible);
    }
    window.setPlanRawVisible();
    window.updatePlannerGate();
    setInterval(window.updatePlannerGate, 5000);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();

async function copyToClipboard(value) {
  try {
    await navigator.clipboard.writeText(String(value || ''));
    return true;
  } catch {
    return false;
  }
}

function renderPromotionAudit(container, status) {
  const promo = status?.promotion_audit || status?.promotionAudit || null;
  if (!promo) {
    return '<div class="stack-item"><span class="stack-meta">No promotion audit yet</span></div>';
  }

  const promoted = promo.promoted === true;
  const badge = `<span class="status-chip ${promoted ? 'active' : 'warning'}">${promoted ? 'PROMOTED' : 'REJECTED'}</span>`;
  const reason = promo.reason || '—';
  const ts = formatTs(promo.timestamp_unix_s);
  const currentPath = promo.current_path || status?.current_adapter?.path || '—';
  const sha256 = promo.sha256 || '—';
  const auditPath = promo.audit_log_path || '—';
  const rldsEpisodeDir = promo.rlds_episode_dir || '—';

  function row(label, value, copyable = false) {
    const safeVal = escapeHtml(value);
    const copyBtn = copyable
      ? `<button class="rail-btn" data-copy="${escapeHtml(value)}" style="padding:6px 10px; font-size:12px;">Copy</button>`
      : '';
    return `<div class="metric-row">` +
      `<div class="metric-label">${escapeHtml(label)}</div>` +
      `<div class="metric-value" style="display:flex; gap:8px; align-items:center; justify-content:flex-end;">` +
        `<span style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${safeVal}</span>` +
        `${copyBtn}` +
      `</div>` +
    `</div>`;
  }

  const parts = [
    `<div class="stack-item">` +
      `<div>` +
        `<h4>Adapter promotion</h4>` +
        `<div class="stack-meta">Latest candidate → current decision</div>` +
      `</div>` +
      `<div>${badge}</div>` +
    `</div>`,
    row('When', ts, false),
    row('Reason', reason, false),
    row('Current adapter', currentPath, true),
    row('SHA-256', sha256, true),
    row('Audit log', auditPath, true),
    row('RLDS episode dir', rldsEpisodeDir, true),
  ];

  return parts.join('');
}

window.updateTrainingRail = async function () {
  const container = document.getElementById('training-memory-list');
  if (!container) return;
  try {
    const res = await fetch('/api/training/status');
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    container.innerHTML = renderPromotionAudit(container, data);

    // Wire copy buttons inside this container.
    const buttons = container.querySelectorAll('[data-copy]');
    buttons.forEach((btn) => {
      if (btn.dataset.wired) return;
      btn.dataset.wired = '1';
      btn.addEventListener('click', async () => {
        const ok = await copyToClipboard(btn.getAttribute('data-copy'));
        const old = btn.textContent;
        btn.textContent = ok ? 'Copied' : 'Copy failed';
        setTimeout(() => (btn.textContent = old), 900);
      });
    });
  } catch (err) {
    console.warn('updateTrainingRail failed', err);
    container.innerHTML = '<div class="stack-item"><span class="stack-meta">Trainer status unavailable</span></div>';
  }
};

function renderCloudReadiness(payload) {
  if (!payload || payload.status !== 'ok') {
    return '<div class="stack-item"><span class="stack-meta">Cloud readiness unavailable</span></div>';
  }
  const ready = !!payload.ready_for_cloud_handoff;
  const chip = `<span class="status-chip ${ready ? 'active' : 'warning'}">${ready ? 'READY' : 'NOT READY'}</span>`;

  function row(label, value, copyable = false) {
    const safeVal = escapeHtml(value);
    const copyBtn = copyable
      ? `<button class="rail-btn" data-copy="${escapeHtml(value)}" style="padding:6px 10px; font-size:12px;">Copy</button>`
      : '';
    return `<div class="metric-row">` +
      `<div class="metric-label">${escapeHtml(label)}</div>` +
      `<div class="metric-value" style="display:flex; gap:8px; align-items:center; justify-content:flex-end;">` +
        `<span style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${safeVal}</span>` +
        `${copyBtn}` +
      `</div>` +
    `</div>`;
  }

  const gates = Array.isArray(payload.gates) ? payload.gates : [];
  const gateHtml = gates.map((g) => {
    const ok = g && g.ok === true;
    return (
      `<div class="stack-item">` +
        `<div>` +
          `<h4>${escapeHtml(g?.name || 'gate')}</h4>` +
          `<div class="stack-meta">${escapeHtml(g?.detail || '')}</div>` +
        `</div>` +
        `<div><span class="status-chip ${ok ? 'active' : 'warning'}">${ok ? 'OK' : 'BLOCKED'}</span></div>` +
      `</div>`
    );
  }).join('');

  const rlds = payload.rlds || {};
  const seed = payload.seed || {};
  const cmds = payload.commands || {};
  const dist = payload.distribution || {};
  const vertex = dist.vertex_templates || {};

  return [
    `<div class="stack-item">` +
      `<div>` +
        `<h4>Cloud handoff</h4>` +
        `<div class="stack-meta">Seed + episodes → TPU v1 training</div>` +
      `</div>` +
      `<div>${chip}</div>` +
    `</div>`,
    row('RLDS dir', rlds.dir || '—', true),
    row('Episodes', String(rlds.episodes_total ?? '—'), false),
    row('Seed export', seed.export_dir || '—', true),
    row('Seed manifest', seed.manifest_path || '—', true),
    row('Checkpoint dir', seed.checkpoint_dir || '—', true),
    row('TFRecord dir', payload.optional?.tfrecord_dir?.path || '—', true),
    `<div class="panel-subtitle" style="margin-top:8px;">Gates</div>`,
    gateHtml || '<div class="stack-item"><span class="stack-meta">No gates reported</span></div>',
    `<div class="panel-subtitle" style="margin-top:8px;">Copyable commands</div>`,
    row('Zip episodes', cmds.zip_episodes || '—', true),
    row('TFRecord convert', cmds.tfrecord_convert || '—', true),
    row('Cloud train template', cmds.cloud_tpu_train_template || '—', true),
    `<div class="panel-subtitle" style="margin-top:8px;">Vertex / Edge distribution (templates)</div>`,
    row('Upload to GCS', vertex.upload_to_gcs || '—', true),
    row('Signed URL (gcloud)', vertex.sign_url_gcloud || '—', true),
    row('Signed URL (gsutil)', vertex.sign_url_gsutil_legacy || '—', true),
  ].join('');
}

async function wireCopyButtons(container) {
  if (!container) return;
  const buttons = container.querySelectorAll('[data-copy]');
  buttons.forEach((btn) => {
    if (btn.dataset.wired) return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', async () => {
      const ok = await copyToClipboard(btn.getAttribute('data-copy'));
      const old = btn.textContent;
      btn.textContent = ok ? 'Copied' : 'Copy failed';
      setTimeout(() => (btn.textContent = old), 900);
    });
  });
}

window.refreshCloudReadiness = async function () {
  const list = document.getElementById('cloud-readiness-list');
  if (list) list.innerHTML = '<div class="stack-item"><span class="stack-meta">Checking readiness…</span></div>';
  try {
    const res = await fetch('/api/training/cloud_readiness');
    const data = await res.json();
    if (!res.ok) throw new Error('HTTP ' + res.status);
    if (list) list.innerHTML = renderCloudReadiness(data);
    await wireCopyButtons(list);
  } catch (err) {
    console.warn('refreshCloudReadiness failed', err);
    if (list) list.innerHTML = '<div class="stack-item"><span class="stack-meta">Readiness check failed</span></div>';
  }
};

window.buildCloudExportZip = async function () {
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
  const episode_limit = limitRaw ? Number(limitRaw) : null;
  try {
    const res = await fetch('/api/training/export_zip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include, episode_limit }),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'export failed');
    if (statusEl) statusEl.innerHTML = `Built <strong>${escapeHtml(data.zip_name)}</strong> (${escapeHtml(String(data.size_bytes))} bytes) <button class="rail-btn" style="padding:6px 10px; font-size:12px;" onclick="window.open('${escapeHtml(data.download_url)}','_blank')">Download</button>`;
  } catch (err) {
    console.warn('buildCloudExportZip failed', err);
    if (statusEl) statusEl.textContent = 'Export failed: ' + (err?.message || err);
  }
};

window.listCloudExports = async function () {
  const statusEl = document.getElementById('cloud-export-status');
  if (statusEl) statusEl.textContent = 'Listing exports…';
  try {
    const res = await fetch('/api/training/exports');
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'list failed');
    const items = Array.isArray(data.items) ? data.items : [];
    if (!items.length) {
      if (statusEl) statusEl.textContent = 'No exports yet.';
      return;
    }
    const first = items[0];
    if (statusEl) statusEl.innerHTML = `Latest: <strong>${escapeHtml(first.name)}</strong> <button class="rail-btn" style="padding:6px 10px; font-size:12px;" onclick="window.open('${escapeHtml(first.download_url)}','_blank')">Download</button>`;
  } catch (err) {
    console.warn('listCloudExports failed', err);
    if (statusEl) statusEl.textContent = 'List failed: ' + (err?.message || err);
  }
};

window.installCloudBundle = async function () {
  const statusEl = document.getElementById('cloud-install-status');
  if (statusEl) statusEl.textContent = 'Installing…';
  const kind = document.getElementById('cloud-install-kind')?.value || 'jax_seed_manifest';
  const source_url = document.getElementById('cloud-install-url')?.value?.trim();
  const source_path = document.getElementById('cloud-install-path')?.value?.trim();
  try {
    const res = await fetch('/api/training/install_bundle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, source_url: source_url || null, source_path: source_path || null }),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'install failed');
    if (statusEl) statusEl.textContent = `Installed: ${data.installed_to || data.edge_manifest || 'ok'}`;
    // Refresh readiness after install.
    if (typeof window.refreshCloudReadiness === 'function') window.refreshCloudReadiness();
  } catch (err) {
    console.warn('installCloudBundle failed', err);
    if (statusEl) statusEl.textContent = 'Install failed: ' + (err?.message || err);
  }
};

function sparklineSvg(points, { width = 220, height = 46 } = {}) {
  if (!Array.isArray(points) || points.length < 2) {
    return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><path d="" fill="none" stroke="rgba(255,255,255,0.35)" /></svg>`;
  }
  const xs = points.map((p, i) => i);
  const ys = points.map((p) => Number(p));
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const ySpan = (yMax - yMin) || 1;
  const xSpan = (points.length - 1) || 1;
  const pad = 2;
  const mapX = (i) => pad + (i / xSpan) * (width - pad * 2);
  const mapY = (v) => {
    const t = (v - yMin) / ySpan;
    return pad + (1 - t) * (height - pad * 2);
  };
  const d = points.map((v, i) => `${i === 0 ? 'M' : 'L'} ${mapX(i).toFixed(2)} ${mapY(Number(v)).toFixed(2)}`).join(' ');
  return (
    `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">` +
      `<path d="${d}" fill="none" stroke="rgba(124, 196, 255, 0.95)" stroke-width="2" stroke-linecap="round" />` +
      `<text x="${width - 6}" y="${height - 8}" text-anchor="end" fill="rgba(255,255,255,0.55)" font-size="10">${escapeHtml(yMin.toExponential(2))}–${escapeHtml(yMax.toExponential(2))}</text>` +
    `</svg>`
  );
}

function pct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return (n * 100).toFixed(0) + '%';
}

function fmtNum(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 1) return n.toFixed(3);
  return n.toExponential(2);
}

function renderTrainingDashboard({ status, metrics, evals, tasks, skills }) {
  const wavecore = metrics?.wavecore || {};
  const fastPts = (wavecore.fast?.points || []).map((p) => p.loss);
  const midPts = (wavecore.mid?.points || []).map((p) => p.loss);
  const slowPts = (wavecore.slow?.points || []).map((p) => p.loss);

  const latestHope = evals?.hope_eval?.latest || null;
  const latestFacts = evals?.facts_eval?.latest || null;
  const latestWiki = evals?.wiki_learn?.latest || null;

  const tasksCount = Array.isArray(tasks?.tasks) ? tasks.tasks.length : null;
  const skillsCount = Array.isArray(skills?.skills) ? skills.skills.length : null;

  function chip(label, kind) {
    const cls = kind === 'good' ? 'active' : (kind === 'warn' ? 'warning' : 'info');
    return `<span class="status-chip ${cls}">${escapeHtml(label)}</span>`;
  }

  function card(title, subtitle, rightHtml, bodyHtml) {
    return (
      `<div class="stack-item">` +
        `<div style="min-width:0;">` +
          `<h4>${escapeHtml(title)}</h4>` +
          `<div class="stack-meta">${escapeHtml(subtitle || '')}</div>` +
        `</div>` +
        `<div>${rightHtml || ''}</div>` +
      `</div>` +
      (bodyHtml ? `<div class="panel-ghost" style="margin-top:10px;">${bodyHtml}</div>` : '')
    );
  }

  const loopSummary = (loopName) => {
    const block = status?.[loopName] || {};
    const res = block?.result || {};
    const steps = res?.steps;
    const avg = res?.avg_loss;
    const fin = res?.final_loss;
    const wall = res?.wall_time_s;
    return `steps:${escapeHtml(String(steps ?? '—'))} avg:${escapeHtml(fmtNum(avg))} final:${escapeHtml(fmtNum(fin))} time:${escapeHtml(fmtNum(wall))}s`;
  };

  const lossGrid = (
    `<div style="display:grid; grid-template-columns: 1fr; gap:10px;">` +
      `<div class="metric-row"><div class="metric-label">Fast</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(fastPts)}<span class="stack-meta">${escapeHtml(loopSummary('fast'))}</span></div></div>` +
      `<div class="metric-row"><div class="metric-label">Mid</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(midPts)}<span class="stack-meta">${escapeHtml(loopSummary('mid'))}</span></div></div>` +
      `<div class="metric-row"><div class="metric-label">Slow</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(slowPts)}<span class="stack-meta">${escapeHtml(loopSummary('slow'))}</span></div></div>` +
    `</div>`
  );

  const toolRouter = metrics?.tool_router || {};
  const toolRouterLossPts = (toolRouter.loss?.points || []).map((p) => p.loss);
  const toolRouterAccPts = (toolRouter.acc?.points || []).map((p) => p.acc);
  const toolRouterEval = metrics?.tool_router_eval || {};
  const toolRouterTop1Pts = (toolRouterEval.top1?.points || []).map((p) => p.top1);
  const toolRouterTop5Pts = (toolRouterEval.top5?.points || []).map((p) => p.top5);
  const toolRouterStatus = status?.tool_router || null;
  const toolRouterRes = toolRouterStatus?.status === 'ok' ? toolRouterStatus?.tool_router : toolRouterStatus;
  const toolRouterManifest = toolRouterRes?.manifest || toolRouterRes?.tool_router?.manifest || null;
  const toolRouterChip = (() => {
    if (toolRouterStatus?.status === 'running') return chip('RUNNING', 'warn');
    if (toolRouterStatus?.status === 'error') return chip('ERROR', 'warn');
    if (toolRouterRes?.status === 'ok' || toolRouterRes?.export_dir) return chip('READY', 'good');
    return chip('—', 'info');
  })();

  const toolRouterBody = (
    `<div class="metric-row"><div class="metric-label">Loss</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(toolRouterLossPts)}<span class="stack-meta">latest:${escapeHtml(fmtNum(toolRouterLossPts.at(-1)))}</span></div></div>` +
    `<div class="metric-row"><div class="metric-label">Accuracy</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(toolRouterAccPts)}<span class="stack-meta">latest:${escapeHtml(pct(toolRouterAccPts.at(-1)))}</span></div></div>` +
    `<div class="metric-row"><div class="metric-label">Eval top-1</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(toolRouterTop1Pts)}<span class="stack-meta">latest:${escapeHtml(pct(toolRouterTop1Pts.at(-1)))}</span></div></div>` +
    `<div class="metric-row"><div class="metric-label">Eval top-5</div><div class="metric-value" style="display:flex; gap:10px; align-items:center;">${sparklineSvg(toolRouterTop5Pts)}<span class="stack-meta">latest:${escapeHtml(pct(toolRouterTop5Pts.at(-1)))}</span></div></div>` +
    `<div class="metric-row"><div class="metric-label">Manifest</div><div class="metric-value" style="display:flex; gap:8px; align-items:center; justify-content:flex-end;"><span style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(toolRouterManifest || '—')}</span>${toolRouterManifest ? `<button class="rail-btn" data-copy="${escapeHtml(toolRouterManifest)}" style="padding:6px 10px; font-size:12px;">Copy</button>` : ''}</div></div>` +
    `<div class="metric-row"><div class="metric-label"></div><div class="metric-value" style="display:flex; gap:8px; justify-content:flex-end;"><button class="rail-btn" type="button" onclick="window.trainToolRouter()">Train tool-router</button><button class="rail-btn" type="button" onclick="window.evalToolRouter()">Run heldout eval</button></div></div>` +
    `<div class="panel-subtitle" style="margin-top:8px;">Try a prompt</div>` +
    `<div class="metric-row"><div class="metric-label">Prompt</div><div class="metric-value"><input id="tool-router-prompt" class="input-compact" type="text" placeholder="e.g. Find instagram user info for nike" style="width:260px;"></div></div>` +
    `<div class="metric-row"><div class="metric-label"></div><div class="metric-value"><button class="rail-btn" type="button" onclick="window.predictToolRouter()">Suggest tools</button></div></div>` +
    `<div class="stack-list" id="tool-router-predictions" style="margin-top:8px;"></div>`
  );

  const evalCardBody = (() => {
    const rows = [];
    if (latestHope) {
      rows.push(`<div class="metric-row"><div class="metric-label">HOPE eval</div><div class="metric-value">${chip(pct(latestHope.success_rate), (latestHope.success_rate ?? 0) > 0.9 ? 'good' : 'warn')} ${chip('fallback ' + pct(latestHope.fallback_rate), (latestHope.fallback_rate ?? 1) < 0.05 ? 'good' : 'warn')}</div></div>`);
      rows.push(`<div class="metric-row"><div class="metric-label">Latest episode</div><div class="metric-value"><span style="max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(latestHope.path || '')}</span></div></div>`);
    } else {
      rows.push(`<div class="stack-item"><span class="stack-meta">No hope_eval episodes found</span></div>`);
    }
    if (latestFacts) {
      rows.push(`<div class="metric-row"><div class="metric-label">FACTS eval</div><div class="metric-value">${chip(pct(latestFacts.success_rate), (latestFacts.success_rate ?? 0) > 0.9 ? 'good' : 'warn')} ${chip('fallback ' + pct(latestFacts.fallback_rate), (latestFacts.fallback_rate ?? 1) < 0.05 ? 'good' : 'warn')}</div></div>`);
    }
    if (latestWiki) {
      rows.push(`<div class="metric-row"><div class="metric-label">Wiki learn</div><div class="metric-value">${chip(pct(latestWiki.success_rate), (latestWiki.success_rate ?? 0) > 0.9 ? 'good' : 'warn')} ${chip('fallback ' + pct(latestWiki.fallback_rate), (latestWiki.fallback_rate ?? 1) < 0.05 ? 'good' : 'warn')}</div></div>`);
      rows.push(`<div class="metric-row"><div class="metric-label">Latest wiki episode</div><div class="metric-value"><span style="max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(latestWiki.path || '')}</span></div></div>`);
    }
    return rows.join('');
  })();

  const capabilityBody = (
    `<div class="metric-row"><div class="metric-label">Tasks (eligible)</div><div class="metric-value">${escapeHtml(tasksCount == null ? '—' : String(tasksCount))}</div></div>` +
    `<div class="metric-row"><div class="metric-label">Skills (eligible)</div><div class="metric-value">${escapeHtml(skillsCount == null ? '—' : String(skillsCount))}</div></div>` +
    `<div class="stack-item"><span class="stack-meta">This is “capability coverage” (what the runtime thinks it can do). We can add richer “can-do” probes next (e.g., driving, planning, tool use) as logged RLDS tests.</span></div>`
  );

  const toolSum = metrics?.tool_dataset_summary || null;
  const toolBody = (() => {
    if (!toolSum || toolSum.status !== 'ok') {
      return '<div class="stack-item"><span class="stack-meta">Tool dataset summary unavailable</span></div>';
    }
    const sources = Array.isArray(toolSum.sources) ? toolSum.sources : [];
    if (!sources.length) {
      return '<div class="stack-item"><span class="stack-meta">No toolchat_hf_* folders found under RLDS episodes.</span></div>';
    }
    const rows = [];
    for (const src of sources) {
      const topTools = Array.isArray(src.top_tools) ? src.top_tools : [];
      const topLine = topTools.slice(0, 6).map((t) => `${t.name}:${t.count}`).join('  ');
      const name = (src.dir || '').split('/').slice(-1)[0] || 'tool dataset';
      rows.push(
        `<div class="stack-item">` +
          `<div style="min-width:0;">` +
            `<h4>${escapeHtml(name)}</h4>` +
            `<div class="stack-meta">episodes:${escapeHtml(String(src.episodes ?? '—'))} steps:${escapeHtml(String(src.steps_total ?? '—'))} tool-call rate:${escapeHtml(pct(src.tool_call_rate))}</div>` +
          `</div>` +
          `<div>${chip(pct(src.tool_call_rate), (src.tool_call_rate ?? 0) > 0.12 ? 'good' : 'warn')}</div>` +
        `</div>` +
        `<div class="metric-row"><div class="metric-label">Top tools</div><div class="metric-value"><span class="stack-meta">${escapeHtml(topLine || '—')}</span></div></div>`
      );
    }
    return rows.join('');
  })();

  const exportInfo = status?.export || null;
  const exportChip = exportInfo?.manifest ? chip('seed exported', 'good') : chip('no export', 'warn');

  const dq = metrics?.data_quality || null;
  const dqBody = (() => {
    if (!dq || dq.status !== 'ok') {
      return '<div class="stack-item"><span class="stack-meta">Data quality unavailable</span></div>';
    }
    const warn = Array.isArray(dq.warnings) ? dq.warnings : [];
    const act = dq.action || {};
    const obs = dq.observation || {};
    const topKeys = Array.isArray(obs.top_keys) ? obs.top_keys : [];
    const keysLine = topKeys.slice(0, 6).map((k) => `${k.key}:${k.count}`).join('  ');
    return (
      `<div class="metric-row"><div class="metric-label">Steps scanned</div><div class="metric-value">${escapeHtml(String(dq.steps_scanned ?? '—'))}</div></div>` +
      `<div class="metric-row"><div class="metric-label">Action present</div><div class="metric-value">${chip(pct(act.present_rate), (act.present_rate ?? 0) > 0.8 ? 'good' : 'warn')}</div></div>` +
      `<div class="metric-row"><div class="metric-label">Action non-zero</div><div class="metric-value">${chip(pct(act.nonzero_rate), (act.nonzero_rate ?? 0) > 0.5 ? 'good' : 'warn')} <span class="stack-meta">mean|a|:${escapeHtml(fmtNum(act.mean_abs_sum))} σ:${escapeHtml(fmtNum(act.std_abs_sum))}</span></div></div>` +
      `<div class="metric-row"><div class="metric-label">Obs present</div><div class="metric-value">${chip(pct(obs.present_rate), (obs.present_rate ?? 0) > 0.8 ? 'good' : 'warn')}</div></div>` +
      `<div class="metric-row"><div class="metric-label">Obs numeric</div><div class="metric-value">${chip(pct(obs.numeric_rate), (obs.numeric_rate ?? 0) > 0.5 ? 'good' : 'warn')} <span class="stack-meta">avg scalars:${escapeHtml(fmtNum(obs.avg_numeric_scalars))}</span></div></div>` +
      `<div class="metric-row"><div class="metric-label">Top obs keys</div><div class="metric-value"><span class="stack-meta">${escapeHtml(keysLine || '—')}</span></div></div>` +
      (warn.length ? `<div class="stack-item"><span class="stack-meta">${escapeHtml(warn.join(' '))}</span></div>` : '')
    );
  })();

  return [
    card('WaveCore training', 'Loss curves (lower is better) from fast/mid/slow loops', exportChip, lossGrid),
    card('Intelligence signals', 'HOPE/FACTS eval health (heuristic: non-error answers + fallback usage)', '', evalCardBody),
    card('Data quality', 'Does the dataset contain non-trivial actions/observations?', '', dqBody),
    card('Tool-router (JAX)', 'Text → tool-name classifier trained from toolchat_hf_* RLDS episodes', toolRouterChip, toolRouterBody),
    card('Tool-use dataset coverage', 'Imported tool-calling corpora (ToolBench/Glaive/xLAM): tool-call density + top tools', '', toolBody),
    card('Capabilities', 'Eligibility counts from Task/Skill libraries', '', capabilityBody),
  ].join('');
}

window.trainToolRouter = async function () {
  try {
    await fetch('/api/training/tool_router_train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_steps: 300, batch_size: 64, learning_rate: 0.003, max_episodes_scan: 20000, top_k_tools: 128, include_dirs_prefix: 'toolchat_hf' }),
    });
  } catch (err) {
    console.warn('trainToolRouter failed', err);
  } finally {
    if (typeof window.updateTrainingDashboard === 'function') window.updateTrainingDashboard();
  }
};

window.evalToolRouter = async function () {
  try {
    await fetch('/api/training/tool_router_eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ eval_mod: 10, eval_bucket: 0, k: 5, max_episodes_scan: 30000, include_dirs_prefix: 'toolchat_hf' }),
    });
  } catch (err) {
    console.warn('evalToolRouter failed', err);
  } finally {
    if (typeof window.updateTrainingDashboard === 'function') window.updateTrainingDashboard();
  }
};

window.refreshOwnershipStatus = async function () {
  const list = document.getElementById('pairing-status-list');
  try {
    const res = await fetch('/api/ownership/status');
    const data = await res.json();
    const own = data?.ownership || {};
    if (list) {
      list.innerHTML =
        `<div class="stack-item"><div class="stack-title">Owned</div><div class="stack-meta">${escapeHtml(String(!!own.owned))}</div></div>` +
        `<div class="stack-item"><div class="stack-title">Owner</div><div class="stack-meta">${escapeHtml(own.owner_id || '—')}</div></div>` +
        `<div class="stack-item"><div class="stack-title">Account type</div><div class="stack-meta">${escapeHtml(own.account_type || '—')}</div></div>`;
    }
  } catch (e) {
    if (list) list.innerHTML = `<div class="stack-item"><span class="stack-meta">Ownership status unavailable</span></div>`;
  }
};

window.startPhonePairing = async function () {
  const codeEl = document.getElementById('pairing-confirm-code');
  const img = document.getElementById('pairing-qr');
  const urlEl = document.getElementById('pairing-url');
  try {
    const base_url = window.location && window.location.origin ? window.location.origin : '';
    const res = await fetch('/api/ownership/pair/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url, ttl_s: 300 }),
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') throw new Error(data.message || 'pair start failed');
    if (codeEl) codeEl.textContent = data.confirm_code || '—';
    if (urlEl) urlEl.textContent = data.url || '(no url)';
    if (img) {
      img.style.display = '';
      img.src = `/api/ownership/pair/qr?t=${Date.now()}`;
    }
    if (typeof window.refreshOwnershipStatus === 'function') window.refreshOwnershipStatus();
  } catch (e) {
    if (codeEl) codeEl.textContent = '—';
    if (urlEl) urlEl.textContent = 'Error: ' + (e && e.message ? e.message : String(e));
    if (img) img.style.display = 'none';
  }
};

window.predictToolRouter = async function () {
  const prompt = document.getElementById('tool-router-prompt')?.value || '';
  const out = document.getElementById('tool-router-predictions');
  if (out) out.innerHTML = '<div class="stack-item"><span class="stack-meta">Predicting…</span></div>';
  try {
    const res = await fetch('/api/training/tool_router_predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, k: 6 }),
    });
    const data = await res.json();
    if (!res.ok || data.status === 'error') throw new Error(data.message || 'predict failed');
    const preds = Array.isArray(data.predictions) ? data.predictions : [];
    if (!preds.length) {
      if (out) out.innerHTML = '<div class="stack-item"><span class="stack-meta">No predictions</span></div>';
      return;
    }
    if (out) out.innerHTML = preds.map((p) => (
      `<div class="stack-item">` +
        `<div><h4>${escapeHtml(p.tool || 'tool')}</h4><div class="stack-meta">score ${escapeHtml(fmtNum(p.score))}</div></div>` +
        `<div>${chip(pct(p.score), (p.score ?? 0) > 0.35 ? 'good' : 'info')}</div>` +
      `</div>`
    )).join('');
  } catch (err) {
    console.warn('predictToolRouter failed', err);
    if (out) out.innerHTML = '<div class="stack-item"><span class="stack-meta">Predict failed</span></div>';
  }
};

window.updateTrainingDashboard = async function () {
  const container = document.getElementById('training-dashboard-list');
  if (!container) return;
  container.innerHTML = '<div class="stack-item"><span class="stack-meta">Loading training dashboard…</span></div>';
  try {
    const [statusRes, metricsRes, evalRes, dqRes, toolRes, tasksRes, skillsRes] = await Promise.all([
      fetch('/api/training/status'),
      fetch('/api/training/metrics?limit=140'),
      fetch('/api/training/eval_summary?limit=6'),
      fetch('/api/training/data_quality?limit=40&step_cap=2500'),
      fetch('/api/training/tool_dataset_summary?limit=2000'),
      fetch('/api/tasks?include_ineligible=false'),
      fetch('/api/skills?include_ineligible=false'),
    ]);
    const status = await statusRes.json();
    const metrics = await metricsRes.json();
    const evals = await evalRes.json();
    const data_quality = await dqRes.json();
    const tool_dataset_summary = await toolRes.json();
    const tasks = await tasksRes.json();
    const skills = await skillsRes.json();
    const mergedMetrics = { ...(metrics || {}), data_quality, tool_dataset_summary };
    container.innerHTML = renderTrainingDashboard({ status, metrics: mergedMetrics, evals, tasks: tasks?.tasks || tasks, skills: skills?.skills || skills });
    await wireCopyButtons(container);
  } catch (err) {
    console.warn('updateTrainingDashboard failed', err);
    container.innerHTML = '<div class="stack-item"><span class="stack-meta">Training dashboard unavailable</span></div>';
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

// Keep the rail fresh even on non-home pages.
(function initTrainingRail() {
  function tick() {
    if (typeof window.updateTrainingRail === 'function') {
      window.updateTrainingRail();
    }
    if (typeof window.updateTrainingDashboard === 'function') {
      window.updateTrainingDashboard();
    }
    if (typeof window.refreshCloudReadiness === 'function') {
      window.refreshCloudReadiness();
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', tick);
  } else {
    tick();
  }
  // Lightweight polling: training status updates are infrequent.
  setInterval(tick, 15000);
})();

