"""
3D Home Exploration Web Server - FastAPI server with WebSocket for real-time game updates.

Provides:
- HTTP endpoints for game state and level management
- WebSocket for real-time command/response and state updates
- 3D home exploration with room navigation
- Integration with Brain B actor runtime
- RLDS episode logging
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.home_world import HomeWorld, get_level, list_levels, LEVELS
from simulator.home_handler import HomeHandler
from simulator.home_rlds_logger import HomeRLDSLogger
from actor_runtime import ActorRuntime


class Home3DSession:
    """Manages a 3D home game session with Brain B integration."""

    def __init__(self, data_dir: str = "brain_b_data"):
        self.data_dir = data_dir
        self.runtime = ActorRuntime(data_path=data_dir)
        self.world = get_level("simple_apartment")
        self.handler = HomeHandler(
            self.world,
            self.runtime,
            on_state_change=self._on_state_change
        )
        self.websockets: list[WebSocket] = []
        self.event_history: list[dict] = []

        # RLDS logging
        self.rlds_logger = HomeRLDSLogger(
            output_dir=os.path.join(data_dir, "home_rlds_episodes")
        )
        self._episode_active = False

    def _on_state_change(self, state: dict) -> None:
        """Called when game state changes - will broadcast to websockets."""
        pass  # Async broadcast handled in handle_command

    async def broadcast(self, message: dict) -> None:
        """Send message to all connected websockets."""
        dead = []
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.websockets.remove(ws)

    def start_episode(self) -> str:
        """Start RLDS episode logging."""
        if self._episode_active:
            return self._episode_id
        episode_id = self.rlds_logger.start_episode(
            self.world,
            level_id=self.world.level_id,
        )
        self._episode_active = True
        self._episode_id = episode_id
        return episode_id

    def end_episode(self) -> str:
        """End RLDS episode logging."""
        if not self._episode_active:
            return ""
        path = self.rlds_logger.end_episode(
            self.world,
            success=self.world.level_complete
        )
        self._episode_active = False
        return path

    def handle_command(self, command: str) -> dict:
        """Process a command and return response."""
        # Auto-start episode if not active
        if not self._episode_active:
            self.start_episode()

        response = self.handler.handle(command)

        # Log step to RLDS
        if self._episode_active:
            self.rlds_logger.log_step(
                world=self.world,
                action_command=response.action_type or "unknown",
                action_intent=response.action_type or "unknown",
                action_params={},
                raw_input=command,
                success=response.action_taken,
                sandbox_denied=response.sandbox_denied,
                item_collected=response.collected_item,
                interacted_object=response.interacted_object,
                level_complete=response.level_complete,
            )

        # Record event
        event = {
            "type": "command",
            "input": command,
            "response": response.text,
            "action_taken": response.action_taken,
            "action_type": response.action_type,
            "sandbox_denied": response.sandbox_denied,
            "level_complete": response.level_complete,
            "collected_item": response.collected_item,
            "interacted_object": response.interacted_object,
        }
        self.event_history.append(event)

        # End episode if level complete
        if response.level_complete and self._episode_active:
            self.end_episode()

        return {
            "type": "response",
            "text": response.text,
            "action_taken": response.action_taken,
            "sandbox_denied": response.sandbox_denied,
            "level_complete": response.level_complete,
            "collected_item": response.collected_item,
            "world_state": self.world.to_dict(),
            "top_down": self.world.render_top_down(),
            "observation": self.world.get_observation(),
        }

    def get_state(self) -> dict:
        """Get current game state."""
        return {
            "world": self.world.to_dict(),
            "top_down": self.world.render_top_down(),
            "observation": self.world.get_observation(),
            "level_id": self.world.level_id,
            "goal_description": self.world.goal_description,
            "robot": self.world.robot.to_dict(),
            "visible_objects": self.world.get_visible_objects(),
            "recording": self.runtime.teaching.is_recording,
            "recording_name": self.runtime.teaching.recording_name,
            "behaviors": self.runtime.teaching.list_behaviors(),
            "event_count": len(self.event_history),
            "episode_active": self._episode_active,
        }


# Global session (created on startup)
session: Optional[Home3DSession] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize game session on startup."""
    global session
    data_dir = os.environ.get("BRAIN_B_DATA", "brain_b_data")
    session = Home3DSession(data_dir)
    print(f"[Home3D] Server started. Data dir: {data_dir}")
    yield
    # End any active episode on shutdown
    if session and session._episode_active:
        session.end_episode()
    print("[Home3D] Server shutdown.")


app = FastAPI(
    title="3D Home Exploration",
    description="A 3D home environment for robot exploration training",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# === HTTP Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main game UI."""
    html_file = static_dir / "home.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    # Fallback to inline HTML
    return HTMLResponse(content=HOME_HTML_FALLBACK)


@app.get("/api/state")
async def get_state():
    """Get current game state."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return JSONResponse(content=session.get_state())


@app.get("/api/levels")
async def api_list_levels():
    """List available levels."""
    level_ids = list_levels()
    levels_data = {}
    for level_id in level_ids:
        lvl = get_level(level_id)
        levels_data[level_id] = {
            "id": level_id,
            "description": lvl.goal_description,
            "width": lvl.width,
            "depth": lvl.depth,
            "height": lvl.height,
        }
    return JSONResponse(content={"levels": levels_data})


@app.post("/api/command")
async def send_command(command: dict):
    """Send a command to the game."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")

    text = command.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No command provided")

    result = session.handle_command(text)

    # Broadcast to websockets
    await session.broadcast(result)

    return JSONResponse(content=result)


@app.post("/api/load/{level_id}")
async def load_level_endpoint(level_id: str):
    """Load a specific level."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")

    try:
        # End current episode if active
        if session._episode_active:
            session.end_episode()

        session.world = get_level(level_id)
        session.handler.world = session.world
        state = session.get_state()
        await session.broadcast({"type": "level_loaded", **state})
        return JSONResponse(content=state)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/reset")
async def reset_level():
    """Reset current level."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")

    # End and restart episode
    if session._episode_active:
        session.end_episode()

    level_id = session.world.level_id
    session.world = get_level(level_id)
    session.handler.world = session.world

    state = session.get_state()
    await session.broadcast({"type": "level_reset", **state})
    return JSONResponse(content=state)


@app.get("/api/events")
async def get_events():
    """Get event history."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return JSONResponse(content={"events": session.event_history[-100:]})


@app.get("/api/episodes")
async def get_episodes():
    """Get RLDS episode list."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    episodes = session.rlds_logger.list_episodes()
    return JSONResponse(content={"episodes": episodes})


@app.get("/api/behaviors")
async def get_behaviors():
    """Get learned behaviors."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    behaviors = session.runtime.teaching.list_behaviors()
    details = {}
    for name in behaviors:
        desc = session.runtime.teaching.describe(name)
        details[name] = desc
    return JSONResponse(content={"behaviors": behaviors, "details": details})


@app.get("/api/observation")
async def get_observation():
    """Get current observation for RLDS."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return JSONResponse(content=session.world.get_observation())


# === WebSocket Endpoint ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time game updates."""
    await websocket.accept()

    if not session:
        await websocket.close(code=1011, reason="Session not initialized")
        return

    session.websockets.append(websocket)

    # Send initial state
    await websocket.send_json({
        "type": "connected",
        **session.get_state()
    })

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "command":
                text = data.get("text", "").strip()
                if text:
                    result = session.handle_command(text)
                    # Broadcast to all clients
                    await session.broadcast(result)

            elif data.get("type") == "get_state":
                await websocket.send_json({
                    "type": "state",
                    **session.get_state()
                })

            elif data.get("type") == "load_level":
                level_id = data.get("level_id", "simple_apartment")
                try:
                    if session._episode_active:
                        session.end_episode()
                    session.world = get_level(level_id)
                    session.handler.world = session.world
                    await session.broadcast({
                        "type": "level_loaded",
                        **session.get_state()
                    })
                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            elif data.get("type") == "reset":
                if session._episode_active:
                    session.end_episode()
                level_id = session.world.level_id
                session.world = get_level(level_id)
                session.handler.world = session.world
                await session.broadcast({
                    "type": "level_reset",
                    **session.get_state()
                })

            elif data.get("type") == "start_episode":
                episode_id = session.start_episode()
                await websocket.send_json({
                    "type": "episode_started",
                    "episode_id": episode_id
                })

            elif data.get("type") == "end_episode":
                path = session.end_episode()
                await websocket.send_json({
                    "type": "episode_ended",
                    "path": path
                })

    except WebSocketDisconnect:
        session.websockets.remove(websocket)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        if websocket in session.websockets:
            session.websockets.remove(websocket)


# === Fallback HTML ===

HOME_HTML_FALLBACK = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Home Exploration</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: #16213e;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { color: #4ecca3; font-size: 1.5em; }
        .status { color: #888; font-size: 0.9em; }
        main {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .game-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .world-view {
            background: #0f0f23;
            border: 2px solid #4ecca3;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            white-space: pre;
            font-size: 16px;
            line-height: 1.2;
            flex: 1;
            overflow: auto;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .control-group {
            display: grid;
            grid-template-columns: repeat(3, 50px);
            gap: 5px;
        }
        .btn {
            background: #4ecca3;
            border: none;
            color: #1a1a2e;
            padding: 12px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }
        .btn:hover { background: #7ee8c2; }
        .btn:active { transform: scale(0.95); }
        .btn.empty { visibility: hidden; }
        .sidebar {
            width: 300px;
            background: #16213e;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow-y: auto;
        }
        .panel {
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
        }
        .panel h3 {
            color: #4ecca3;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .info { font-size: 0.85em; color: #aaa; }
        .info div { margin: 5px 0; }
        .command-input {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .command-input input {
            flex: 1;
            background: #0f0f23;
            border: 1px solid #4ecca3;
            color: #eee;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
        }
        .messages {
            max-height: 150px;
            overflow-y: auto;
            font-size: 0.85em;
        }
        .message {
            padding: 5px;
            border-bottom: 1px solid #333;
        }
        .message.response { color: #4ecca3; }
        .message.error { color: #e74c3c; }
        .levels {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .level-btn {
            background: #0f0f23;
            border: 1px solid #4ecca3;
            color: #4ecca3;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.8em;
        }
        .level-btn:hover { background: #4ecca3; color: #1a1a2e; }
        .level-btn.active { background: #4ecca3; color: #1a1a2e; }
    </style>
</head>
<body>
    <header>
        <h1>3D Home Exploration</h1>
        <div class="status" id="status">Connecting...</div>
    </header>
    <main>
        <div class="game-area">
            <div class="world-view" id="world-view">Loading...</div>
            <div class="controls">
                <div class="control-group">
                    <button class="btn empty"></button>
                    <button class="btn" onclick="send('forward')" title="W">W</button>
                    <button class="btn empty"></button>
                    <button class="btn" onclick="send('turn left')" title="A">A</button>
                    <button class="btn" onclick="send('backward')" title="S">S</button>
                    <button class="btn" onclick="send('turn right')" title="D">D</button>
                </div>
                <div class="control-group">
                    <button class="btn" onclick="send('strafe left')" title="Strafe Left">Q</button>
                    <button class="btn" onclick="send('look up')" title="Look Up">^</button>
                    <button class="btn" onclick="send('strafe right')" title="Strafe Right">E</button>
                    <button class="btn" onclick="send('look down')" title="Look Down">v</button>
                    <button class="btn" onclick="send('interact')" title="Interact">USE</button>
                    <button class="btn" onclick="send('look')" title="Look Around">LOOK</button>
                </div>
            </div>
            <div class="command-input">
                <input type="text" id="command-input" placeholder="Type command..." onkeypress="onKeyPress(event)">
                <button class="btn" onclick="sendInput()">Send</button>
            </div>
        </div>
        <div class="sidebar">
            <div class="panel">
                <h3>LEVELS</h3>
                <div class="levels" id="levels"></div>
            </div>
            <div class="panel">
                <h3>STATUS</h3>
                <div class="info" id="info">
                    <div>Position: -</div>
                    <div>Room: -</div>
                    <div>Moves: 0</div>
                    <div>Battery: 100%</div>
                </div>
            </div>
            <div class="panel">
                <h3>GOAL</h3>
                <div class="info" id="goal">-</div>
            </div>
            <div class="panel">
                <h3>INVENTORY</h3>
                <div class="info" id="inventory">Empty</div>
            </div>
            <div class="panel">
                <h3>MESSAGES</h3>
                <div class="messages" id="messages"></div>
            </div>
        </div>
    </main>
    <script>
        let ws = null;
        let currentLevel = '';

        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').style.color = '#4ecca3';
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').style.color = '#e74c3c';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function handleMessage(data) {
            if (data.top_down) {
                document.getElementById('world-view').textContent = data.top_down;
            }

            if (data.robot) {
                const r = data.robot;
                const pos = r.position;
                document.getElementById('info').innerHTML = `
                    <div>Position: (${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)})</div>
                    <div>Facing: ${r.rotation.yaw.toFixed(0)}deg</div>
                    <div>Moves: ${r.moves}</div>
                    <div>Battery: ${(r.battery * 100).toFixed(0)}%</div>
                `;

                if (r.inventory && r.inventory.length > 0) {
                    document.getElementById('inventory').textContent = r.inventory.join(', ');
                } else {
                    document.getElementById('inventory').textContent = 'Empty';
                }
            }

            if (data.observation && data.observation.current_room) {
                const room = data.observation.current_room.replace('_', ' ');
                document.getElementById('info').innerHTML += `<div>Room: ${room}</div>`;
            }

            if (data.goal_description) {
                document.getElementById('goal').textContent = data.goal_description;
            }

            if (data.level_id) {
                currentLevel = data.level_id;
                updateLevelButtons();
            }

            if (data.text) {
                addMessage(data.text, data.sandbox_denied ? 'error' : 'response');
            }

            if (data.level_complete) {
                addMessage('LEVEL COMPLETE!', 'response');
            }
        }

        function addMessage(text, type = '') {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.textContent = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function send(command) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'command', text: command }));
            }
        }

        function sendInput() {
            const input = document.getElementById('command-input');
            if (input.value.trim()) {
                send(input.value.trim());
                input.value = '';
            }
        }

        function onKeyPress(event) {
            if (event.key === 'Enter') {
                sendInput();
            }
        }

        function loadLevel(levelId) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'load_level', level_id: levelId }));
            }
        }

        function updateLevelButtons() {
            document.querySelectorAll('.level-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.level === currentLevel);
            });
        }

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (document.activeElement.tagName === 'INPUT') return;

            switch(e.key.toLowerCase()) {
                case 'w': send('forward'); break;
                case 's': send('backward'); break;
                case 'a': send('turn left'); break;
                case 'd': send('turn right'); break;
                case 'q': send('strafe left'); break;
                case 'e': send('strafe right'); break;
                case 'r': send('look up'); break;
                case 'f': send('look down'); break;
                case ' ': send('interact'); e.preventDefault(); break;
            }
        });

        // Load levels list
        fetch('/api/levels')
            .then(r => r.json())
            .then(data => {
                const container = document.getElementById('levels');
                for (const [id, info] of Object.entries(data.levels)) {
                    const btn = document.createElement('button');
                    btn.className = 'level-btn';
                    btn.dataset.level = id;
                    btn.textContent = id.replace('_', ' ');
                    btn.title = info.description;
                    btn.onclick = () => loadLevel(id);
                    container.appendChild(btn);
                }
            });

        connect();
    </script>
</body>
</html>
"""


# === CLI Entry Point ===

def main():
    """Run the server."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8083"))

    print(f"""
+===================================================+
|        3D Home Exploration Simulator v1.0         |
|    A 3D environment for robot training            |
+===================================================+
|  Server: http://{host}:{port}
|  API Docs: http://{host}:{port}/docs
|  WebSocket: ws://{host}:{port}/ws
+===================================================+
""")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
