(function () {
  const body = document.body;
  const railToggle = document.getElementById('mobile-rail-toggle');
  const refreshBtn = document.getElementById('mobile-refresh');
  const statusChip = document.getElementById('mobile-status-chip');
  const healthChip = document.getElementById('mobile-health-chip');
  const summaryEndpoint = '/api/mobile/summary';
  const STORAGE_KEY = 'continuonbrain_mobile_rail_open';

  function setRailVisible(visible) {
    if (!body) return;
    body.classList.toggle('mobile-rail-hidden', !visible);
    if (railToggle) {
      railToggle.textContent = visible ? 'Hide agent rail' : 'Show agent rail';
      railToggle.setAttribute('aria-pressed', visible ? 'false' : 'true');
    }
    try {
      localStorage.setItem(STORAGE_KEY, visible ? '1' : '0');
    } catch (err) {
      console.warn('Unable to persist rail state', err);
    }
  }

  function initRailToggle() {
    if (!railToggle) return;
    let initialVisible = true;
    try {
      initialVisible = localStorage.getItem(STORAGE_KEY) !== '0';
    } catch (err) {
      initialVisible = true;
    }
    setRailVisible(initialVisible);
    railToggle.addEventListener('click', () => setRailVisible(body?.classList.contains('mobile-rail-hidden')));
  }

  function renderStatus(summary) {
    if (!summary) return;
    const statusPayload = summary.status?.status || {};
    const mode = statusPayload.mode || statusPayload.robot_mode || 'Unknown mode';
    const safety = summary.loops?.gates || summary.loops?.metrics || {};
    const battery = statusPayload.battery;

    if (statusChip) {
      const batteryText = battery?.percent != null ? `${battery.percent}% battery` : 'battery unknown';
      statusChip.textContent = `${mode} · ${batteryText}`;
    }

    if (healthChip) {
      const gateEntries = typeof safety === 'object' ? Object.entries(safety) : [];
      const lockedGates = gateEntries.filter(([, val]) => val === 'closed' || val === 'locked');
      const gatesText = lockedGates.length > 0
        ? `${lockedGates.length} safety gate(s) locked`
        : 'Safety gates clear';
      healthChip.textContent = gatesText;
      healthChip.dataset.state = lockedGates.length > 0 ? 'warning' : 'ok';
    }
  }

  async function refreshSummary() {
    try {
      if (refreshBtn) refreshBtn.disabled = true;
      const response = await fetch(summaryEndpoint, { headers: { 'Accept': 'application/json' } });
      const data = await response.json();
      renderStatus(data);
    } catch (err) {
      console.warn('Mobile summary fetch failed', err);
      if (statusChip) {
        const href = (window.location && window.location.href) ? window.location.href : '';
        const host = (window.location && window.location.host) ? window.location.host : '';
        const msg = err && err.message ? err.message : String(err || 'fetch failed');
        const nameNotResolved = /ERR_NAME_NOT_RESOLVED/i.test(msg);
        if (!host || nameNotResolved) {
          statusChip.textContent = 'Disconnected · open via robot IP (http://<robot-ip>:8080/ui)';
        } else {
          statusChip.textContent = 'Unable to load status';
        }
        statusChip.title = href;
      }
    } finally {
      if (refreshBtn) refreshBtn.disabled = false;
    }
  }

  function initServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.register('/static/service-worker.js').catch((err) => {
      console.warn('Service worker registration failed', err);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initRailToggle();
    initServiceWorker();
    if (refreshBtn) refreshBtn.addEventListener('click', refreshSummary);
    refreshSummary();
    setInterval(refreshSummary, 12000);
  });
})();
