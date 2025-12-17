/**
 * ui_research.js - Research View & Experimental Features
 */

window.realityProofState = { status: null, loops: null, surprises: 0 };

function setProofText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

window.updateRealityProof = function () {
    const status = window.realityProofState.status || {};
    const loopsWrapper = window.realityProofState.loops || {};
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

    setProofText('context-updates', window.realityProofState.surprises > 0 ? `${window.realityProofState.surprises} surprise events logged` : 'No surprises yet');
    setProofText('surprise-counter-chip', `Surprises: ${window.realityProofState.surprises}`);

    setProofText('proof-vision-line', status.capabilities && status.capabilities.has_vision ? 'Vision tokens streaming' : 'Waiting for frames…');
    setProofText('proof-dream-line', status.model_stack ? 'Model stack active' : 'Model stack not detected');
    setProofText('proof-reality-line', status.allow_motion ? 'Reality checks live (motion allowed)' : 'Reality checks pending motion');
    setProofText('proof-context-line', window.realityProofState.surprises > 0 ? 'Context updated after surprises' : 'No context updates logged');
};

window.logRealitySurprise = function () {
    window.realityProofState.surprises += 1;
    const log = document.getElementById('surprise-log');
    const entry = `<div class="surprise-row">Surprise @ ${new Date().toLocaleTimeString()}</div>`;
    if (log) {
        log.innerHTML = entry + log.innerHTML;
    }
    window.updateRealityProof();
}

window.renderLoopTelemetry = function (status) {
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
};

window.applyLoopHealthPayload = function (payload) {
    if (!payload) return;
    const source = payload.metrics || payload.loop_metrics ? payload : payload.status || payload;
    if (source.metrics || source.loop_metrics) {
        window.renderLoopTelemetry({
            loop_metrics: source.metrics || source.loop_metrics,
            gate_snapshot: source.gates || source.gate_snapshot,
            safety_head: source.safety_head
        });
    }
    window.realityProofState.loops = source;
    window.updateRealityProof();
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
