"""
RobotGrid Web Server - FastAPI server with WebSocket for real-time game updates.

Provides:
- HTTP endpoints for game state and level management
- WebSocket for real-time command/response and state updates
- Three view modes: Grid, Event Log, Sandbox Audit
- Integration with Brain B actor runtime
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

from simulator.world import GridWorld, load_level, LEVELS
from simulator.game_handler import GameHandler
from actor_runtime import ActorRuntime


class GameSession:
    """Manages a game session with Brain B integration."""

    def __init__(self, data_dir: str = "brain_b_data"):
        self.data_dir = data_dir
        self.runtime = ActorRuntime(data_path=data_dir)
        self.world = load_level("tutorial")
        self.handler = GameHandler(
            self.world,
            self.runtime,
            on_state_change=self._on_state_change
        )
        self.websockets: list[WebSocket] = []
        self.event_history: list[dict] = []
        self.sandbox_audit: list[dict] = []

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

    def handle_command(self, command: str) -> dict:
        """Process a command and return response."""
        response = self.handler.handle(command)

        # Record event
        event = {
            "type": "command",
            "input": command,
            "response": response.text,
            "action_taken": response.action_taken,
            "action_type": response.action_type,
            "sandbox_denied": response.sandbox_denied,
            "level_complete": response.level_complete,
        }
        self.event_history.append(event)

        # Record sandbox audit if denied
        if response.sandbox_denied:
            self.sandbox_audit.append({
                "type": "DENIED",
                "action": command,
                "reason": "Restricted zone access attempted",
            })

        return {
            "type": "response",
            "text": response.text,
            "action_taken": response.action_taken,
            "sandbox_denied": response.sandbox_denied,
            "level_complete": response.level_complete,
            "world_state": self.world.to_dict(),
            "grid_render": self.world.render(),
        }

    def get_state(self) -> dict:
        """Get current game state."""
        return {
            "world": self.world.to_dict(),
            "grid_render": self.world.render(),
            "grid_bordered": self.world.render_with_border(),
            "level_name": self.world.level_name,
            "robot": self.world.robot.to_dict(),
            "recording": self.runtime.teaching.is_recording,
            "recording_name": self.runtime.teaching.recording_name,
            "behaviors": self.runtime.teaching.list_behaviors(),
            "event_count": len(self.event_history),
        }


# Global session (created on startup)
session: Optional[GameSession] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize game session on startup."""
    global session
    data_dir = os.environ.get("BRAIN_B_DATA", "brain_b_data")
    session = GameSession(data_dir)
    print(f"[RobotGrid] Server started. Data dir: {data_dir}")
    yield
    print("[RobotGrid] Server shutdown.")


app = FastAPI(
    title="RobotGrid Simulator",
    description="A tile-based game for testing Brain B systems",
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
    html_file = static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content="""
        <html>
            <body>
                <h1>RobotGrid Simulator</h1>
                <p>Static files not found. Run from brain_b directory.</p>
                <p>API available at <a href="/docs">/docs</a></p>
            </body>
        </html>
    """)


@app.get("/api/state")
async def get_state():
    """Get current game state."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return JSONResponse(content=session.get_state())


@app.get("/api/levels")
async def list_levels():
    """List available levels."""
    return JSONResponse(content={
        "levels": {k: {"name": v["name"], "description": v["description"]} for k, v in LEVELS.items()}
    })


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
        session.world = load_level(level_id)
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

    session.world.reset()
    state = session.get_state()
    await session.broadcast({"type": "level_reset", **state})
    return JSONResponse(content=state)


@app.get("/api/events")
async def get_events():
    """Get event history."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return JSONResponse(content={"events": session.event_history[-100:]})  # Last 100


@app.get("/api/sandbox-audit")
async def get_sandbox_audit():
    """Get sandbox audit log."""
    if not session:
        raise HTTPException(status_code=500, detail="Session not initialized")
    return JSONResponse(content={"audit": session.sandbox_audit[-50:]})  # Last 50


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
                level_id = data.get("level_id", "tutorial")
                try:
                    session.world = load_level(level_id)
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
                session.world.reset()
                await session.broadcast({
                    "type": "level_reset",
                    **session.get_state()
                })

    except WebSocketDisconnect:
        session.websockets.remove(websocket)
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        if websocket in session.websockets:
            session.websockets.remove(websocket)


# === CLI Entry Point ===

def main():
    """Run the server."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    print(f"""
╔═══════════════════════════════════════════════════╗
║           RobotGrid Simulator v1.0                ║
║  A tile-based game for testing Brain B systems    ║
╠═══════════════════════════════════════════════════╣
║  Server: http://{host}:{port}
║  API Docs: http://{host}:{port}/docs
║  WebSocket: ws://{host}:{port}/ws
╚═══════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
