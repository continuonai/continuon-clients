/**
 * House 3D POV Viewer - Three.js Implementation
 *
 * Split-view 3D training environment:
 * - Robot POV camera (main view)
 * - Overhead map (sidebar)
 * - Sensor data panels
 * - WebSocket sync with Python training
 */

// =============================================================================
// Global State
// =============================================================================

let povScene, povCamera, povRenderer, povControls;
let overheadScene, overheadCamera, overheadRenderer;
let depthScene, depthCamera, depthRenderer, depthTarget;

let sceneData = null;
let robotState = {
    position: { x: 3, y: 0, z: 3 },
    yaw: 0,
    pitch: 0,
};

let isExploring = false;
let isRecording = false;
let websocket = null;
let animationId = null;

// Robot marker in overhead view
let robotMarker = null;
let robotDirectionArrow = null;

// Movement state
const moveState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
};

const MOVE_SPEED = 0.1;
const TURN_SPEED = 2;

// =============================================================================
// Initialization
// =============================================================================

function init() {
    initPOVView();
    initOverheadView();
    initDepthView();
    initControls();
    connectWebSocket();
    loadTemplate('studio_apartment');
    animate();
}

function initPOVView() {
    const canvas = document.getElementById('pov-canvas');
    const container = canvas.parentElement;

    // Scene
    povScene = new THREE.Scene();
    povScene.background = new THREE.Color(0x0a0a1a);
    povScene.fog = new THREE.Fog(0x0a0a1a, 10, 30);

    // Camera (robot POV)
    const aspect = container.clientWidth / container.clientHeight;
    povCamera = new THREE.PerspectiveCamera(75, aspect, 0.1, 100);
    povCamera.position.set(3, 0.8, 3);

    // Renderer
    povRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    povRenderer.setSize(container.clientWidth, container.clientHeight);
    povRenderer.shadowMap.enabled = true;
    povRenderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Pointer lock controls (for FPS-style exploration)
    povControls = new THREE.PointerLockControls(povCamera, document.body);

    povControls.addEventListener('lock', () => {
        document.getElementById('controls-hint').classList.add('visible');
    });

    povControls.addEventListener('unlock', () => {
        document.getElementById('controls-hint').classList.remove('visible');
        isExploring = false;
        document.getElementById('btn-start').textContent = 'Start Exploration';
    });

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    povScene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff8e8, 0.6);
    sunLight.position.set(5, 10, 5);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 50;
    sunLight.shadow.camera.left = -15;
    sunLight.shadow.camera.right = 15;
    sunLight.shadow.camera.top = 15;
    sunLight.shadow.camera.bottom = -15;
    povScene.add(sunLight);

    // Handle resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        povCamera.aspect = width / height;
        povCamera.updateProjectionMatrix();
        povRenderer.setSize(width, height);
    });
}

function initOverheadView() {
    const canvas = document.getElementById('overhead-canvas');
    const container = canvas.parentElement;

    // Scene (shared objects added later)
    overheadScene = new THREE.Scene();
    overheadScene.background = new THREE.Color(0x12122a);

    // Orthographic camera looking down
    const aspect = container.clientWidth / container.clientHeight;
    const viewSize = 10;
    overheadCamera = new THREE.OrthographicCamera(
        -viewSize * aspect / 2, viewSize * aspect / 2,
        viewSize / 2, -viewSize / 2,
        0.1, 100
    );
    overheadCamera.position.set(5, 20, 5);
    overheadCamera.lookAt(5, 0, 5);

    // Renderer
    overheadRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    overheadRenderer.setSize(container.clientWidth, container.clientHeight);

    // Ambient light
    overheadScene.add(new THREE.AmbientLight(0xffffff, 0.8));

    // Robot marker
    const markerGeom = new THREE.CircleGeometry(0.3, 16);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0xe94560 });
    robotMarker = new THREE.Mesh(markerGeom, markerMat);
    robotMarker.rotation.x = -Math.PI / 2;
    robotMarker.position.set(3, 0.1, 3);
    overheadScene.add(robotMarker);

    // Direction arrow
    const arrowDir = new THREE.Vector3(0, 0, 1);
    const arrowOrigin = new THREE.Vector3(3, 0.2, 3);
    robotDirectionArrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, 0.8, 0x4ade80, 0.3, 0.2);
    overheadScene.add(robotDirectionArrow);

    // Grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x2a2a5a, 0x1a1a3a);
    overheadScene.add(gridHelper);
}

function initDepthView() {
    const canvas = document.getElementById('depth-canvas');

    depthScene = new THREE.Scene();
    depthCamera = povCamera.clone();

    depthRenderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    depthRenderer.setSize(200, 150);

    // Depth render target
    depthTarget = new THREE.WebGLRenderTarget(200, 150, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
    });
}

function initControls() {
    // Keyboard controls
    document.addEventListener('keydown', (e) => {
        if (!isExploring) return;

        switch (e.code) {
            case 'KeyW': moveState.forward = true; break;
            case 'KeyS': moveState.backward = true; break;
            case 'KeyA': moveState.left = true; break;
            case 'KeyD': moveState.right = true; break;
            case 'Space':
                e.preventDefault();
                toggleRecording();
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        switch (e.code) {
            case 'KeyW': moveState.forward = false; break;
            case 'KeyS': moveState.backward = false; break;
            case 'KeyA': moveState.left = false; break;
            case 'KeyD': moveState.right = false; break;
        }
    });
}

// =============================================================================
// Scene Building
// =============================================================================

function buildScene(data) {
    sceneData = data;

    // Clear existing objects
    while (povScene.children.length > 2) {  // Keep lights
        povScene.remove(povScene.children[povScene.children.length - 1]);
    }
    while (overheadScene.children.length > 4) {  // Keep lights, marker, arrow, grid
        overheadScene.remove(overheadScene.children[overheadScene.children.length - 1]);
    }

    // Build objects
    for (const obj of data.objects || []) {
        const mesh = createObjectMesh(obj);
        if (mesh) {
            povScene.add(mesh);

            // Add simplified version to overhead
            const overheadMesh = createOverheadMesh(obj);
            if (overheadMesh) {
                overheadScene.add(overheadMesh);
            }
        }
    }

    // Add lights from scene data
    for (const light of data.lights || []) {
        addLight(light);
    }

    // Update overhead camera to fit scene
    if (data.bounds) {
        const bounds = data.bounds;
        const centerX = (bounds.min[0] + bounds.max[0]) / 2;
        const centerZ = (bounds.min[2] + bounds.max[2]) / 2;
        const size = Math.max(
            bounds.max[0] - bounds.min[0],
            bounds.max[2] - bounds.min[2]
        ) * 1.2;

        overheadCamera.left = -size / 2;
        overheadCamera.right = size / 2;
        overheadCamera.top = size / 2;
        overheadCamera.bottom = -size / 2;
        overheadCamera.position.set(centerX, 20, centerZ);
        overheadCamera.lookAt(centerX, 0, centerZ);
        overheadCamera.updateProjectionMatrix();
    }

    console.log(`Scene loaded: ${data.objects?.length || 0} objects, ${data.lights?.length || 0} lights`);
}

function createObjectMesh(obj) {
    const { transform, geometryType, geometryParams, material } = obj;
    const size = geometryParams.size || [1, 1, 1];

    let geometry;

    if (geometryType === 'plane') {
        geometry = new THREE.PlaneGeometry(
            geometryParams.width || size[0],
            geometryParams.depth || size[2]
        );
    } else if (geometryType === 'custom') {
        // Handle custom geometries (simplified as boxes for now)
        geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
    } else {
        // Default box
        geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
    }

    // Get material
    const mat = getMaterial(material);
    const mesh = new THREE.Mesh(geometry, mat);

    // Apply transform
    mesh.position.set(
        transform.position[0],
        transform.position[1],
        transform.position[2]
    );

    mesh.rotation.set(
        THREE.MathUtils.degToRad(transform.rotation[0]),
        THREE.MathUtils.degToRad(transform.rotation[1]),
        THREE.MathUtils.degToRad(transform.rotation[2])
    );

    if (transform.scale) {
        mesh.scale.set(transform.scale[0], transform.scale[1], transform.scale[2]);
    }

    mesh.castShadow = obj.castShadow !== false;
    mesh.receiveShadow = obj.receiveShadow !== false;
    mesh.name = obj.name;

    return mesh;
}

function createOverheadMesh(obj) {
    if (obj.tags?.includes('ceiling')) return null;  // Skip ceilings in overhead

    const { transform, geometryParams, material } = obj;
    const size = geometryParams.size || [1, 1, 1];

    // Create 2D representation
    const geometry = new THREE.PlaneGeometry(size[0], size[2]);
    const color = getOverheadColor(obj.tags || []);
    const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.7,
    });

    const mesh = new THREE.Mesh(geometry, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(
        transform.position[0],
        0.05,
        transform.position[2]
    );
    mesh.rotation.z = THREE.MathUtils.degToRad(transform.rotation[1]);

    return mesh;
}

function getOverheadColor(tags) {
    if (tags.includes('floor')) return 0x3a3a5a;
    if (tags.includes('wall')) return 0x5a5a8a;
    if (tags.includes('furniture')) return 0x4a8a4a;
    if (tags.includes('seating')) return 0x4a6a9a;
    if (tags.includes('table')) return 0x8a6a4a;
    if (tags.includes('bed')) return 0x8a4a6a;
    if (tags.includes('appliance')) return 0x4a8a8a;
    return 0x6a6a6a;
}

function getMaterial(materialName) {
    // Material library (simplified)
    const materials = {
        'white_paint': { color: 0xffffff, roughness: 0.8 },
        'cream_paint': { color: 0xfffaf0, roughness: 0.8 },
        'light_gray_paint': { color: 0xdcdce0, roughness: 0.75 },
        'sage_green': { color: 0xb4c8b4, roughness: 0.8 },
        'navy_blue': { color: 0x283250, roughness: 0.75 },
        'oak_floor': { color: 0xb48c64, roughness: 0.4 },
        'dark_hardwood': { color: 0x3c2820, roughness: 0.35 },
        'walnut': { color: 0x5a3c28, roughness: 0.3 },
        'gray_fabric': { color: 0x828288, roughness: 0.9 },
        'beige_fabric': { color: 0xd2c3af, roughness: 0.85 },
        'brown_leather': { color: 0x64412d, roughness: 0.5 },
        'white_ceramic': { color: 0xf5f5f5, roughness: 0.2 },
        'gray_tile': { color: 0x8c8c90, roughness: 0.25 },
        'stainless_steel': { color: 0xc8c8cd, roughness: 0.3, metalness: 0.9 },
        'chrome': { color: 0xe6e6eb, roughness: 0.1, metalness: 1.0 },
        'granite_gray': { color: 0x828288, roughness: 0.25 },
        'matte_black_metal': { color: 0x232328, roughness: 0.7, metalness: 0.8 },
        'screen_off': { color: 0x0f0f14, roughness: 0.1 },
        'screen_on': { color: 0x283c50, emissive: 0x6496c8, emissiveIntensity: 0.3 },
        'velvet_blue': { color: 0x283c64, roughness: 0.95 },
        'beige_carpet': { color: 0xc3b4a0, roughness: 1.0 },
    };

    const props = materials[materialName] || { color: 0x969696, roughness: 0.5 };

    return new THREE.MeshStandardMaterial({
        color: props.color,
        roughness: props.roughness || 0.5,
        metalness: props.metalness || 0,
        emissive: props.emissive || 0x000000,
        emissiveIntensity: props.emissiveIntensity || 0,
    });
}

function addLight(lightData) {
    let light;

    if (lightData.type === 'ambient') {
        light = new THREE.AmbientLight(lightData.color || 0xffffff, lightData.intensity || 0.3);
    } else if (lightData.type === 'directional') {
        light = new THREE.DirectionalLight(lightData.color || 0xffffff, lightData.intensity || 0.5);
        if (lightData.position) {
            light.position.set(lightData.position[0], lightData.position[1], lightData.position[2]);
        }
        light.castShadow = lightData.castShadow !== false;
    } else if (lightData.type === 'point') {
        light = new THREE.PointLight(
            lightData.color || 0xfff8e0,
            lightData.intensity || 1,
            lightData.range || 10
        );
        if (lightData.position) {
            light.position.set(lightData.position[0], lightData.position[1], lightData.position[2]);
        }
        light.castShadow = lightData.castShadow !== false;
    }

    if (light) {
        povScene.add(light);
    }
}

// =============================================================================
// Animation Loop
// =============================================================================

function animate() {
    animationId = requestAnimationFrame(animate);

    // Update robot movement
    if (isExploring) {
        updateMovement();
    }

    // Update robot state display
    updateRobotInfo();

    // Render views
    povRenderer.render(povScene, povCamera);
    overheadRenderer.render(overheadScene, overheadCamera);

    // Send state to WebSocket if recording
    if (isRecording && websocket?.readyState === WebSocket.OPEN) {
        sendRobotState();
    }
}

function updateMovement() {
    const direction = new THREE.Vector3();
    povCamera.getWorldDirection(direction);
    direction.y = 0;
    direction.normalize();

    const right = new THREE.Vector3();
    right.crossVectors(direction, new THREE.Vector3(0, 1, 0));

    if (moveState.forward) {
        povCamera.position.addScaledVector(direction, MOVE_SPEED);
    }
    if (moveState.backward) {
        povCamera.position.addScaledVector(direction, -MOVE_SPEED);
    }
    if (moveState.left) {
        povCamera.position.addScaledVector(right, -MOVE_SPEED);
    }
    if (moveState.right) {
        povCamera.position.addScaledVector(right, MOVE_SPEED);
    }

    // Update robot state
    robotState.position.x = povCamera.position.x;
    robotState.position.z = povCamera.position.z;

    // Calculate yaw from camera direction
    robotState.yaw = Math.atan2(direction.x, direction.z) * 180 / Math.PI;

    // Update overhead marker
    robotMarker.position.x = povCamera.position.x;
    robotMarker.position.z = povCamera.position.z;

    robotDirectionArrow.position.x = povCamera.position.x;
    robotDirectionArrow.position.z = povCamera.position.z;
    robotDirectionArrow.setDirection(direction);
}

function updateRobotInfo() {
    document.getElementById('pos-x').textContent = robotState.position.x.toFixed(2);
    document.getElementById('pos-z').textContent = robotState.position.z.toFixed(2);
    document.getElementById('pos-yaw').textContent = `${Math.round(robotState.yaw)}\u00b0`;

    // Calculate forward distance (simple raycast)
    const raycaster = new THREE.Raycaster();
    const direction = new THREE.Vector3();
    povCamera.getWorldDirection(direction);
    raycaster.set(povCamera.position, direction);

    const intersects = raycaster.intersectObjects(povScene.children, true);
    if (intersects.length > 0) {
        const dist = intersects[0].distance;
        document.getElementById('forward-dist').textContent = `${dist.toFixed(2)}m`;
        document.getElementById('forward-bar').style.width = `${Math.min(100, dist * 10)}%`;
    } else {
        document.getElementById('forward-dist').textContent = '> 10m';
        document.getElementById('forward-bar').style.width = '100%';
    }

    // Object count
    document.getElementById('objects-count').textContent = sceneData?.objects?.length || 0;
}

// =============================================================================
// User Actions
// =============================================================================

function startExploration() {
    if (isExploring) {
        povControls.unlock();
        isExploring = false;
        document.getElementById('btn-start').textContent = 'Start Exploration';
    } else {
        povControls.lock();
        isExploring = true;
        document.getElementById('btn-start').textContent = 'Stop Exploration';
    }
}

function resetRobot() {
    povCamera.position.set(3, 0.8, 3);
    povCamera.rotation.set(0, 0, 0);
    robotState.position = { x: 3, y: 0, z: 3 };
    robotState.yaw = 0;

    robotMarker.position.set(3, 0.1, 3);
    robotDirectionArrow.position.set(3, 0.2, 3);
    robotDirectionArrow.setDirection(new THREE.Vector3(0, 0, 1));
}

function toggleRecording() {
    isRecording = !isRecording;
    const btn = document.getElementById('btn-record');
    const indicator = document.getElementById('training-indicator');
    const status = document.getElementById('training-status');

    if (isRecording) {
        btn.textContent = 'Stop Recording';
        btn.classList.add('active');
        indicator.classList.add('active');
        status.textContent = 'Recording...';
    } else {
        btn.textContent = 'Record Episode';
        btn.classList.remove('active');
        indicator.classList.remove('active');
        status.textContent = 'Training: Off';
    }
}

function toggleDepthView() {
    const canvas = document.getElementById('depth-canvas');
    canvas.classList.toggle('visible');
}

function loadTemplate(templateName) {
    fetch(`/house3d/template/${templateName}`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error('Failed to load template:', data.error);
                // Generate client-side fallback
                buildDefaultScene();
                return;
            }
            buildScene(data);
        })
        .catch(err => {
            console.error('Template load error:', err);
            buildDefaultScene();
        });
}

function loadFromScanner() {
    // Load last scan result
    fetch('/room/last_scan')
        .then(res => res.json())
        .then(data => {
            if (data.error || !data.result) {
                alert('No recent scan found. Please scan a room first.');
                return;
            }
            // Convert scan result to scene format
            fetch('/house3d/from_scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data.result),
            })
                .then(res => res.json())
                .then(sceneData => buildScene(sceneData))
                .catch(err => console.error('Scan conversion error:', err));
        })
        .catch(err => {
            console.error('Load scan error:', err);
            alert('Failed to load scan data');
        });
}

function buildDefaultScene() {
    // Fallback scene if server unavailable
    const scene = {
        name: 'Default Room',
        bounds: { min: [0, 0, 0], max: [10, 3, 10] },
        objects: [
            {
                name: 'Floor',
                transform: { position: [5, 0, 5], rotation: [0, 0, 0], scale: [1, 1, 1] },
                geometryType: 'plane',
                geometryParams: { width: 10, depth: 10 },
                material: 'oak_floor',
                tags: ['floor'],
            },
            {
                name: 'Wall North',
                transform: { position: [5, 1.35, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
                geometryType: 'box',
                geometryParams: { size: [10, 2.7, 0.15] },
                material: 'white_paint',
                tags: ['wall'],
            },
            {
                name: 'Wall South',
                transform: { position: [5, 1.35, 10], rotation: [0, 0, 0], scale: [1, 1, 1] },
                geometryType: 'box',
                geometryParams: { size: [10, 2.7, 0.15] },
                material: 'white_paint',
                tags: ['wall'],
            },
            {
                name: 'Wall East',
                transform: { position: [10, 1.35, 5], rotation: [0, 90, 0], scale: [1, 1, 1] },
                geometryType: 'box',
                geometryParams: { size: [10, 2.7, 0.15] },
                material: 'white_paint',
                tags: ['wall'],
            },
            {
                name: 'Wall West',
                transform: { position: [0, 1.35, 5], rotation: [0, 90, 0], scale: [1, 1, 1] },
                geometryType: 'box',
                geometryParams: { size: [10, 2.7, 0.15] },
                material: 'white_paint',
                tags: ['wall'],
            },
            {
                name: 'Sofa',
                transform: { position: [5, 0.425, 2], rotation: [0, 0, 0], scale: [1, 1, 1] },
                geometryType: 'box',
                geometryParams: { size: [2.2, 0.85, 0.9] },
                material: 'gray_fabric',
                tags: ['furniture', 'seating'],
            },
            {
                name: 'Coffee Table',
                transform: { position: [5, 0.225, 3.5], rotation: [0, 0, 0], scale: [1, 1, 1] },
                geometryType: 'box',
                geometryParams: { size: [1.2, 0.45, 0.6] },
                material: 'walnut',
                tags: ['furniture', 'table'],
            },
        ],
        lights: [
            { type: 'ambient', intensity: 0.3 },
            { type: 'point', position: [5, 2.5, 5], intensity: 1.0, range: 10 },
        ],
    };

    buildScene(scene);
}

// =============================================================================
// WebSocket Connection
// =============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/house3d/ws`;

    try {
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log('WebSocket connected');
            document.getElementById('ws-status').classList.add('connected');
            document.getElementById('ws-status-text').textContent = 'Connected';
        };

        websocket.onclose = () => {
            console.log('WebSocket disconnected');
            document.getElementById('ws-status').classList.remove('connected');
            document.getElementById('ws-status-text').textContent = 'Disconnected';

            // Reconnect after delay
            setTimeout(connectWebSocket, 5000);
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        };

        websocket.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    } catch (err) {
        console.error('WebSocket connection failed:', err);
        document.getElementById('ws-status-text').textContent = 'Offline';
    }
}

function sendRobotState() {
    if (websocket?.readyState !== WebSocket.OPEN) return;

    websocket.send(JSON.stringify({
        type: 'robot_state',
        position: robotState.position,
        yaw: robotState.yaw,
        timestamp: Date.now(),
    }));
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'scene_update':
            buildScene(data.scene);
            break;
        case 'training_frame':
            // Server sent a training frame
            if (data.pov) {
                // Could display server-rendered frame
            }
            break;
        case 'episode_complete':
            console.log('Training episode complete:', data.stats);
            break;
    }
}

// =============================================================================
// Start
// =============================================================================

document.addEventListener('DOMContentLoaded', init);
