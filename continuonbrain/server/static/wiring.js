(() => {
  const canvas = document.getElementById("scene");
  const hud = document.getElementById("hud");
  let renderer, scene, camera, nodes = [];

  function init() {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    resize();
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0c10);
    camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 16);

    const light = new THREE.PointLight(0xffffff, 1.2);
    light.position.set(5, 8, 10);
    scene.add(light);

    window.addEventListener("resize", resize);
    animate();
  }

  function resize() {
    if (!renderer) return;
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h);
    if (camera) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }

  function clearNodes() {
    nodes.forEach(n => scene.remove(n));
    nodes = [];
  }

  function addNode(name, x, y, color) {
    const geom = new THREE.SphereGeometry(0.5, 24, 24);
    const mat = new THREE.MeshStandardMaterial({ color });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(x, y, 0);
    scene.add(mesh);
    nodes.push(mesh);

    const div = document.createElement("div");
    div.style.position = "absolute";
    div.style.color = "#e0e6ed";
    div.style.fontSize = "12px";
    div.style.pointerEvents = "none";
    div.innerText = name;
    document.body.appendChild(div);
    mesh.__label = div;
    return mesh;
  }

  function addEdge(a, b, intensity) {
    const points = [a.position, b.position];
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({ color: 0x58c4ff, linewidth: 2 * intensity });
    const line = new THREE.Line(geom, mat);
    scene.add(line);
    nodes.push(line);
  }

  function updateLabels() {
    nodes.forEach(n => {
      if (!n.__label) return;
      const vec = n.position.clone().project(camera);
      const x = (vec.x * 0.5 + 0.5) * window.innerWidth;
      const y = (-vec.y * 0.5 + 0.5) * window.innerHeight;
      n.__label.style.left = `${x}px`;
      n.__label.style.top = `${y}px`;
    });
  }

  async function loadStats() {
    try {
      const res = await fetch("/api/wiring");
      const stats = await res.json();
      renderGraph(stats);
      renderHud(stats);
    } catch (err) {
      hud.innerText = "Failed to load wiring stats";
    }
  }

  function renderHud(stats) {
    hud.innerHTML = `
      <div><strong>Hardware:</strong> ${stats.hardware_mode}</div>
      <div><strong>Episodes:</strong> ${stats.episodes_total}</div>
      <div><strong>HOPE evals:</strong> ${stats.hope_eval_episodes}</div>
      <div><strong>FACTS evals:</strong> ${stats.facts_eval_episodes}</div>
      <div><strong>Compact:</strong> ${stats.compact ? "available" : "n/a"}</div>
    `;
  }

  function renderGraph(stats) {
    clearNodes();
    const fast = addNode("Fast", -4, 2, 0x7dd3fc);
    const mid = addNode("Mid", 0, 3, 0x38bdf8);
    const slow = addNode("Slow", 4, 2, 0x0ea5e9);
    const wave = addNode("Wave", -2, -1, 0x22d3ee);
    const particle = addNode("Particle", 2, -1, 0xfca5a5);
    addEdge(fast, mid, 1.0);
    addEdge(mid, slow, 1.2);
    addEdge(fast, wave, 0.8);
    addEdge(mid, wave, 0.6);
    addEdge(slow, wave, 0.5);
    addEdge(fast, particle, 0.6);
    addEdge(mid, particle, 0.5);
    addEdge(slow, particle, 0.4);
  }

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    updateLabels();
  }

  init();
  loadStats();
})();
