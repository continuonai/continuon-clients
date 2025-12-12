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
});

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
  if (statusEl) statusEl.textContent = 'Episode import UI not yet wired';
  alert('Episode import UI not yet wired to backend.');
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

