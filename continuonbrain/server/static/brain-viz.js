// 3D + "time" (4D) HOPE brain visualization for /research.
//
// This is best-effort and must never break the UI. It dynamically imports Three.js
// from a CDN; if unavailable, we show a readable fallback message.

async function init() {
  const container = document.getElementById('brain-viz-container');
  const fallback = document.getElementById('brain-viz-fallback');
  const slider = document.getElementById('brain-viz-time');
  const sliderLabel = document.getElementById('brain-viz-time-label');
  const overlaySel = document.getElementById('brain-overlay');
  const layoutBtns = {
    graph: document.getElementById('brain-layout-graph'),
    columnar: document.getElementById('brain-layout-columnar'),
    hybrid: document.getElementById('brain-layout-hybrid'),
  };
  if (!container || !fallback || !slider) return;

  fallback.textContent = 'Loading 3D brain…';

  let THREE;
  let OrbitControls;
  try {
    THREE = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
    ({ OrbitControls } = await import('https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js'));
  } catch (e) {
    fallback.textContent =
      '3D brain view needs Three.js (CDN). Network offline or blocked — use the evidence cards above.';
    return;
  }

  fallback.style.display = 'none';

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0c111b, 10, 60);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(
    55,
    container.clientWidth / container.clientHeight,
    0.1,
    200,
  );
  camera.position.set(0, 8, 26);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.target.set(0, 2, 0);

  // Lighting
  scene.add(new THREE.AmbientLight(0x9bb3ff, 0.55));
  const key = new THREE.DirectionalLight(0x7ad7ff, 1.0);
  key.position.set(8, 10, 6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8a52ff, 0.6);
  fill.position.set(-10, 6, -8);
  scene.add(fill);

  // A subtle "field" grid to make it feel 4D/spacey.
  const grid = new THREE.GridHelper(60, 30, 0x355070, 0x1f2a3d);
  grid.position.y = -2.5;
  grid.material.opacity = 0.25;
  grid.material.transparent = true;
  scene.add(grid);

  // Node graph: represent subsystems as spheres with edges.
  const nodes = [
    { id: 'agent_manager', label: 'Agent Manager', color: 0x7ad7ff, pos: [0, 6, 0], size: 1.3 },
    { id: 'safety_head', label: 'SafetyHead', color: 0x38d996, pos: [-6, 3, 4], size: 1.0 },
    { id: 'tool_router', label: 'Symbolic Search\n(tool router)', color: 0x4f9dff, pos: [7, 4, 2], size: 1.1 },
    { id: 'cms_fast', label: 'CMS Fast', color: 0xff8c42, pos: [-5, 1, -4], size: 0.9 },
    { id: 'cms_mid', label: 'CMS Mid', color: 0x8a52ff, pos: [0, 1.5, -6], size: 1.0 },
    { id: 'cms_slow', label: 'CMS Slow', color: 0x67d3ff, pos: [5, 1, -4], size: 0.9 },
    { id: 'wavecore_fast', label: 'WaveCore Fast', color: 0xff4d6d, pos: [-10, 0.5, 0], size: 0.85 },
    { id: 'wavecore_mid', label: 'WaveCore Mid', color: 0xffb703, pos: [-8, 2.2, -1], size: 0.85 },
    { id: 'wavecore_slow', label: 'WaveCore Slow', color: 0x7ad7ff, pos: [-8, 4.0, -2], size: 0.85 },
    { id: 'sensors', label: 'Sensors', color: 0xa9c7ff, pos: [10, 0.0, 0], size: 0.75 },
    { id: 'actuators', label: 'Actuators', color: 0x8df5c7, pos: [10, -1.8, 3], size: 0.75 },
  ];

  const edges = [
    ['agent_manager', 'tool_router'],
    ['agent_manager', 'safety_head'],
    ['agent_manager', 'cms_mid'],
    ['cms_fast', 'cms_mid'],
    ['cms_mid', 'cms_slow'],
    ['wavecore_fast', 'wavecore_mid'],
    ['wavecore_mid', 'wavecore_slow'],
    ['wavecore_slow', 'cms_slow'],
    ['sensors', 'agent_manager'],
    ['agent_manager', 'actuators'],
    ['safety_head', 'actuators'],
  ];

  const nodeMeshes = new Map();
  const nodeGlow = new Map();
  let layoutMode = 'graph';

  const sphereGeo = new THREE.SphereGeometry(1, 28, 28);
  for (const n of nodes) {
    const mat = new THREE.MeshStandardMaterial({
      color: n.color,
      roughness: 0.35,
      metalness: 0.25,
      emissive: 0x000000,
      emissiveIntensity: 0.0,
    });
    const mesh = new THREE.Mesh(sphereGeo, mat);
    mesh.userData.baseScale = n.size;
    mesh.userData.basePos = new THREE.Vector3(n.pos[0], n.pos[1], n.pos[2]);
    mesh.userData.targetPos = mesh.userData.basePos.clone();
    mesh.scale.setScalar(n.size);
    mesh.position.set(n.pos[0], n.pos[1], n.pos[2]);
    scene.add(mesh);
    nodeMeshes.set(n.id, mesh);

    // A soft halo (billboard-ish) using a sprite.
    const haloMat = new THREE.SpriteMaterial({
      color: n.color,
      opacity: 0.18,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const halo = new THREE.Sprite(haloMat);
    halo.position.copy(mesh.position);
    halo.scale.set(6 * n.size, 6 * n.size, 1);
    scene.add(halo);
    nodeGlow.set(n.id, halo);
  }

  // Edge lines
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x1f2a3d, transparent: true, opacity: 0.9 });
  const edgeLines = [];
  for (const [a, b] of edges) {
    const ma = nodeMeshes.get(a);
    const mb = nodeMeshes.get(b);
    if (!ma || !mb) continue;
    const geo = new THREE.BufferGeometry().setFromPoints([ma.position, mb.position]);
    const line = new THREE.Line(geo, edgeMat);
    scene.add(line);
    edgeLines.push({ a, b, line });
  }

  // "4D": we scrub recent metrics onto node intensities.
  // We keep a small cache of time series, plus a rolling history of snapshots.
  const metricCache = {
    controlLoop: [], // period_ms points
    waveFast: [],
    waveMid: [],
    waveSlow: [],
    toolTop1: [],
    hopeSuccess: [],
  };
  const history = [];
  const historyMax = 120; // ~10 minutes at 5s refresh

  async function refreshMetrics() {
    try {
      const [loopRes, metricsRes, evalRes] = await Promise.all([
        fetch('/api/runtime/control_loop?limit=180'),
        fetch('/api/training/metrics?limit=180'),
        fetch('/api/training/eval_summary?limit=10'),
      ]);
      const loop = await loopRes.json();
      const metrics = await metricsRes.json();
      const evals = await evalRes.json();

      const safe = (arr) => Array.isArray(arr) ? arr.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];

      metricCache.controlLoop = safe(loop?.period_ms?.points || []);
      metricCache.waveFast = safe((metrics?.wavecore?.fast?.points || []).map((p) => p?.loss));
      metricCache.waveMid = safe((metrics?.wavecore?.mid?.points || []).map((p) => p?.loss));
      metricCache.waveSlow = safe((metrics?.wavecore?.slow?.points || []).map((p) => p?.loss));
      metricCache.toolTop1 = safe((metrics?.tool_router_eval?.top1?.points || []).map((p) => p?.top1));
      metricCache.hopeSuccess = safe((evals?.hope_eval?.episodes || []).map((e) => e?.success_rate));

      // Append latest snapshot so the slider can scrub real evolution over time.
      const snap = {
        t: Date.now(),
        tick: metricCache.controlLoop.length ? metricCache.controlLoop[metricCache.controlLoop.length - 1] : null,
        fast: metricCache.waveFast.length ? metricCache.waveFast[metricCache.waveFast.length - 1] : null,
        mid: metricCache.waveMid.length ? metricCache.waveMid[metricCache.waveMid.length - 1] : null,
        slow: metricCache.waveSlow.length ? metricCache.waveSlow[metricCache.waveSlow.length - 1] : null,
        top1: metricCache.toolTop1.length ? metricCache.toolTop1[metricCache.toolTop1.length - 1] : null,
        hope: metricCache.hopeSuccess.length ? metricCache.hopeSuccess[metricCache.hopeSuccess.length - 1] : null,
      };
      history.push(snap);
      while (history.length > historyMax) history.shift();

      slider.min = '0';
      slider.max = String(Math.max(0, history.length - 1));
      // Default to latest
      slider.value = String(Math.max(0, history.length - 1));
    } catch (e) {
      // ignore; keep last values
    }
  }

  function sampleHistory(idx) {
    if (!history.length) return null;
    const i = Math.max(0, Math.min(history.length - 1, Number(idx)));
    return history[i];
  }

  function overlayMode() {
    return (overlaySel && overlaySel.value) ? overlaySel.value : 'all';
  }

  function norm(value, min, max) {
    if (value == null || !Number.isFinite(value)) return 0.0;
    const span = (max - min) || 1.0;
    return Math.max(0, Math.min(1, (value - min) / span));
  }

  function apply4DFromSnapshot(snap) {
    if (!snap) return;
    const timeStr = new Date(snap.t).toLocaleTimeString();
    const isLatest = history.length && Number(slider.value) === (history.length - 1);
    if (sliderLabel) sliderLabel.textContent = (isLatest ? `latest • ${timeStr}` : timeStr);

    // Lower is better for losses and control-loop period; higher is better for toolTop1/hope.
    const tick = snap.tick;
    const fast = snap.fast;
    const mid = snap.mid;
    const slow = snap.slow;
    const top1 = snap.top1;
    const hope = snap.hope;

    const tickBad = norm(tick, 30, 160); // 30ms..160ms
    const lossBad = (x) => norm(x, 0.2, 1.5);
    const good = (x) => norm(x, 0.2, 0.95);

    // Emissive intensities
    const setGlow = (id, intensity) => {
      const mesh = nodeMeshes.get(id);
      const halo = nodeGlow.get(id);
      if (mesh && mesh.material) {
        mesh.material.emissiveIntensity = Math.max(0, Math.min(1.2, intensity));
        mesh.material.emissive = new THREE.Color(0xffffff);
      }
      if (halo && halo.material) {
        halo.material.opacity = 0.10 + 0.35 * Math.max(0, Math.min(1, intensity));
      }
    };

    const mode = overlayMode();
    const show = (name) => mode === 'all' || mode === name;

    // Baseline glow.
    setGlow('agent_manager', 0.15);
    setGlow('tool_router', 0.12);
    setGlow('wavecore_fast', 0.10);
    setGlow('wavecore_mid', 0.10);
    setGlow('wavecore_slow', 0.10);
    setGlow('cms_fast', 0.10);
    setGlow('cms_mid', 0.10);
    setGlow('cms_slow', 0.10);
    setGlow('safety_head', 0.12);

    if (show('hope_eval')) setGlow('agent_manager', 0.35 + 0.65 * good(hope ?? 0.0));
    if (show('tool_search')) setGlow('tool_router', 0.25 + 0.85 * good(top1 ?? 0.0));
    if (show('wavecore')) {
      setGlow('wavecore_fast', 0.20 + 0.75 * (1 - lossBad(fast ?? 1.0)));
      setGlow('wavecore_mid', 0.20 + 0.75 * (1 - lossBad(mid ?? 1.0)));
      setGlow('wavecore_slow', 0.20 + 0.75 * (1 - lossBad(slow ?? 1.0)));
    }
    if (show('control_loop')) {
      setGlow('cms_fast', 0.25 + 0.75 * (1 - tickBad));
      setGlow('cms_mid', 0.25 + 0.75 * (1 - tickBad));
      setGlow('cms_slow', 0.25 + 0.75 * (1 - tickBad));
      setGlow('safety_head', 0.35 + 0.55 * (1 - tickBad));
    }
  }

  // Update on slider
  function updateFromSlider() {
    const idx = Number(slider.value || 0);
    apply4DFromSnapshot(sampleHistory(idx));
  }
  slider.addEventListener('input', updateFromSlider);
  if (overlaySel) overlaySel.addEventListener('change', updateFromSlider);

  function setLayout(mode) {
    layoutMode = mode;
    for (const [m, btn] of Object.entries(layoutBtns)) {
      if (!btn) continue;
      btn.classList.toggle('current', m === mode);
    }
    const targets = {
      graph: {
        agent_manager: [0, 6, 0],
        safety_head: [-6, 3, 4],
        tool_router: [7, 4, 2],
        cms_fast: [-5, 1, -4],
        cms_mid: [0, 1.5, -6],
        cms_slow: [5, 1, -4],
        wavecore_fast: [-10, 0.5, 0],
        wavecore_mid: [-8, 2.2, -1],
        wavecore_slow: [-8, 4.0, -2],
        sensors: [10, 0.0, 0],
        actuators: [10, -1.8, 3],
      },
      columnar: {
        sensors: [-12, 1.0, 0],
        cms_fast: [-6, 1.0, 0],
        cms_mid: [-2, 2.3, 0],
        cms_slow: [2, 3.8, 0],
        wavecore_fast: [-6, -1.0, 0],
        wavecore_mid: [-2, -0.2, 0],
        wavecore_slow: [2, 0.8, 0],
        tool_router: [8, 3.0, 0],
        agent_manager: [6, 5.3, 0],
        safety_head: [8, 1.0, 0],
        actuators: [12, 0.2, 0],
      },
      hybrid: {
        sensors: [-12, 0.5, -2],
        cms_fast: [-6, 1.2, -3],
        cms_mid: [-1, 2.8, -4],
        cms_slow: [4, 3.5, -3],
        wavecore_fast: [-7, -0.8, 1],
        wavecore_mid: [-2, 0.1, 1],
        wavecore_slow: [3, 1.2, 1],
        tool_router: [9, 3.4, 0],
        agent_manager: [6, 6.2, 0],
        safety_head: [10, 1.4, 1],
        actuators: [12, -0.4, 2],
      },
    };
    const t = targets[mode] || targets.graph;
    for (const [id, mesh] of nodeMeshes.entries()) {
      const p = t[id] || targets.graph[id] || [0, 0, 0];
      mesh.userData.targetPos.set(p[0], p[1], p[2]);
    }
  }

  if (layoutBtns.graph) layoutBtns.graph.addEventListener('click', () => setLayout('graph'));
  if (layoutBtns.columnar) layoutBtns.columnar.addEventListener('click', () => setLayout('columnar'));
  if (layoutBtns.hybrid) layoutBtns.hybrid.addEventListener('click', () => setLayout('hybrid'));
  setLayout('graph');

  // Resize handling
  const onResize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener('resize', onResize);

  // Initial refresh and periodic updates
  await refreshMetrics();
  updateFromSlider();
  setInterval(refreshMetrics, 5000);

  // Animate
  let t = 0;
  function animate() {
    t += 0.01;
    controls.update();
    // A subtle breathing motion to imply "living" state.
    for (const [id, mesh] of nodeMeshes.entries()) {
      const halo = nodeGlow.get(id);
      const base = 1.0 + 0.03 * Math.sin(t + mesh.position.x * 0.2 + mesh.position.y * 0.1);
      const s = (mesh.userData.baseScale || 1.0) * base;
      mesh.scale.setScalar(s);
      if (mesh.userData.targetPos) {
        mesh.position.lerp(mesh.userData.targetPos, 0.06);
      }
      if (halo) halo.scale.set(halo.scale.x, halo.scale.y, 1);
    }
    for (const e of edgeLines) {
      const a = nodeMeshes.get(e.a);
      const b = nodeMeshes.get(e.b);
      if (!a || !b) continue;
      e.line.geometry.setFromPoints([a.position, b.position]);
      e.line.geometry.attributes.position.needsUpdate = true;
    }
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
}

// Run best-effort init.
init();

