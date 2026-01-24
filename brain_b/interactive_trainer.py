#!/usr/bin/env python3
"""
Interactive Training Server - Live training and conversation interface.

Features:
1. Watch AI play training games in real-time
2. Converse with Brain B using natural language
3. See training decisions and confidence levels
4. Provide live feedback to improve the model
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Brain B imports
from actor_runtime import ActorRuntime
from simulator.world import GridWorld, load_level, LEVELS
from simulator.game_handler import GameHandler
from trainer.conversation_trainer import ConversationTrainer, get_llm_backend

# Nav predictor
try:
    from simulator.simulator_training import get_simulator_predictor, IDX_TO_ACTION
    HAS_NAV = True
except ImportError:
    HAS_NAV = False
    IDX_TO_ACTION = {}


class InteractiveSession:
    """Interactive training session."""

    def __init__(self):
        print("[Session] Initializing...")

        # Brain B runtime
        self.runtime = ActorRuntime(data_path="brain_b_data")
        print("  ✓ Runtime")

        # Game world
        self.world = load_level("tutorial")
        self.handler = GameHandler(self.world, self.runtime)
        print(f"  ✓ World ({self.world.width}x{self.world.height})")

        # Conversation
        self.conversation = ConversationTrainer("brain_b_data")
        self.conversation.load_model()
        self.llm = get_llm_backend()
        print(f"  ✓ Conversation (LLM: {self.llm.backend})")

        # Nav predictor
        self.nav_ready = False
        if HAS_NAV:
            self.nav_predictor = get_simulator_predictor()
            self._load_nav_model()
        print(f"  ✓ Navigation (ready: {self.nav_ready})")

        # State
        self.auto_play = False
        self.feedback_count = 0
        self.history: List[Dict] = []

        print("[Session] Ready!")

    def _load_nav_model(self):
        """Load navigation model."""
        checkpoint_dir = Path("brain_b_data/simulator_checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("sim_model_*.json"))
            if checkpoints:
                try:
                    self.nav_predictor.load(str(checkpoints[-1]))
                    self.nav_ready = self.nav_predictor.is_ready
                except Exception as e:
                    print(f"  Nav load error: {e}")

    def get_state(self) -> dict:
        """Get full state."""
        return {
            "type": "state",
            "game": {
                "grid": self._get_grid(),
                "robot": {
                    "x": self.world.robot.x,
                    "y": self.world.robot.y,
                    "dir": self.world.robot.direction.name.lower(),
                },
                "level": self.world.level_name,
            },
            "conversation": {
                "trained": self.conversation.classifier.is_trained,
                "llm": self.llm.backend,
                "llm_ok": self.llm.is_available,
            },
            "nav_ready": self.nav_ready,
            "auto_play": self.auto_play,
            "feedback": self.feedback_count,
            "levels": list(LEVELS.keys()),
        }

    def _get_grid(self) -> List[List[int]]:
        """Get grid as 2D array."""
        # Map Tile enums to integers for frontend
        tile_map = {
            'FLOOR': 0, 'WALL': 1, 'GOAL': 2, 'LAVA': 3,
            'KEY': 4, 'DOOR': 5, 'BOX': 6, 'BUTTON': 7,
        }
        grid = []
        for y in range(self.world.height):
            row = []
            for x in range(self.world.width):
                tile = self.world.get_tile(x, y)
                tile_name = tile.name if hasattr(tile, 'name') else str(tile)
                row.append(tile_map.get(tile_name, 0))
            grid.append(row)
        return grid

    def game_command(self, cmd: str) -> dict:
        """Execute game command."""
        try:
            response = self.handler.handle(cmd)
            return {
                "type": "game_result",
                "cmd": cmd,
                "text": response.text,
                "action": response.action_type,
                "ok": response.action_taken,
            }
        except Exception as e:
            return {"type": "error", "msg": str(e)}

    def chat(self, text: str) -> dict:
        """Process chat message."""
        try:
            result = self.conversation.predict(text, use_llm=True)

            # Execute action if navigation intent
            action_done = False
            if result.get("action"):
                action_type = result["action"].get("type")
                if action_type in ("forward", "backward", "left", "right", "stop"):
                    self.game_command(action_type)
                    action_done = True

            return {
                "type": "chat_result",
                "input": text,
                "intent": result.get("intent", "unknown"),
                "confidence": result.get("confidence", 0),
                "response": result.get("response", ""),
                "backend": result.get("backend", "local"),
                "action": result.get("action"),
                "action_done": action_done,
            }
        except Exception as e:
            traceback.print_exc()
            return {"type": "error", "msg": str(e)}

    def get_ai_move(self) -> dict:
        """Get AI's suggested move."""
        if not self.nav_ready:
            return {
                "type": "ai_move",
                "action": "none",
                "confidence": 0,
                "reason": "No model loaded",
            }

        try:
            # Simple state: distances in each direction
            state = [0.0] * 48
            # Just use random for now since world doesn't have state_vector
            import random
            state = [random.random() for _ in range(48)]

            probs = self.nav_predictor.predict(state)
            best_idx = max(range(len(probs)), key=lambda i: probs[i])

            return {
                "type": "ai_move",
                "action": IDX_TO_ACTION.get(best_idx, "forward"),
                "confidence": probs[best_idx],
                "probs": {IDX_TO_ACTION.get(i, f"a{i}"): p for i, p in enumerate(probs)},
            }
        except Exception as e:
            return {"type": "error", "msg": str(e)}

    def load_level(self, name: str) -> dict:
        """Load a level."""
        if name in LEVELS:
            self.world = load_level(name)
            self.handler = GameHandler(self.world, self.runtime)
            return {"type": "level_loaded", "name": name}
        return {"type": "error", "msg": f"Unknown level: {name}"}

    def add_feedback(self, rating: str) -> dict:
        """Record feedback."""
        self.feedback_count += 1
        self.history.append({
            "type": "feedback",
            "rating": rating,
            "time": datetime.now().isoformat(),
        })
        return {"type": "feedback_ok", "total": self.feedback_count}


# FastAPI app
app = FastAPI()
session: Optional[InteractiveSession] = None


@app.on_event("startup")
async def startup():
    global session
    session = InteractiveSession()


INTERFACE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Brain B Interactive</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: monospace; background: #111; color: #0f0; padding: 20px; }
        .container { display: flex; gap: 20px; max-width: 1200px; margin: auto; }
        .panel { background: #1a1a1a; padding: 15px; border-radius: 8px; flex: 1; }
        h2 { color: #0ff; margin-bottom: 10px; font-size: 14px; }
        .grid { display: inline-grid; gap: 2px; background: #333; padding: 5px; border-radius: 4px; }
        .cell { width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
                background: #222; border-radius: 2px; font-size: 16px; }
        .cell.wall { background: #444; }
        .cell.robot { background: #0a0; color: #fff; }
        .cell.goal { background: #aa0; }
        .controls { display: grid; grid-template-columns: repeat(3, 50px); gap: 5px; margin: 15px 0; }
        button { padding: 10px; background: #222; color: #0f0; border: 1px solid #0f0;
                 cursor: pointer; border-radius: 4px; }
        button:hover { background: #0f0; color: #000; }
        button.on { background: #0f0; color: #000; }
        .chat { height: 200px; overflow-y: auto; background: #0a0a0a; padding: 10px;
                border-radius: 4px; margin-bottom: 10px; }
        .msg { margin: 5px 0; padding: 5px; border-radius: 4px; }
        .msg.user { background: #223; text-align: right; }
        .msg.bot { background: #232; }
        .msg .meta { font-size: 10px; color: #666; }
        input { width: 100%; padding: 10px; background: #222; color: #fff;
                border: 1px solid #333; border-radius: 4px; }
        .status { font-size: 11px; color: #666; margin-top: 10px; }
        .log { height: 150px; overflow-y: auto; background: #0a0a0a; padding: 10px;
               font-size: 11px; border-radius: 4px; }
        select { padding: 8px; background: #222; color: #fff; border: 1px solid #333;
                 border-radius: 4px; width: 100%; margin: 10px 0; }
        .ai-box { background: #112; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .ai-action { font-size: 20px; color: #ff0; }
        .ai-conf { color: #888; }
    </style>
</head>
<body>
    <h1 style="text-align:center; margin-bottom:20px;">🧠 Brain B - Interactive Training</h1>

    <div class="container">
        <!-- Game Panel -->
        <div class="panel">
            <h2>🎮 GAME</h2>
            <div id="grid" class="grid"></div>

            <div class="controls">
                <div></div>
                <button onclick="cmd('forward')">↑</button>
                <div></div>
                <button onclick="cmd('left')">←</button>
                <button onclick="cmd('stop')">■</button>
                <button onclick="cmd('right')">→</button>
                <div></div>
                <button onclick="cmd('backward')">↓</button>
                <div></div>
            </div>

            <select id="level" onchange="loadLevel(this.value)"></select>

            <button id="autoBtn" onclick="toggleAuto()" style="width:100%">🤖 Auto Play</button>

            <div class="ai-box">
                <div>AI suggests:</div>
                <div id="aiAction" class="ai-action">-</div>
                <div id="aiConf" class="ai-conf"></div>
            </div>
        </div>

        <!-- Chat Panel -->
        <div class="panel">
            <h2>💬 CHAT</h2>
            <div id="chat" class="chat"></div>
            <input id="chatInput" placeholder="Say something... (try 'hello' or 'go forward')"
                   onkeypress="if(event.key==='Enter')sendChat()">
            <div class="status">
                <span id="llmStatus">LLM: -</span> |
                <span id="convStatus">Conv: -</span>
            </div>
        </div>

        <!-- Log Panel -->
        <div class="panel">
            <h2>📋 LOG</h2>
            <div id="log" class="log"></div>

            <h2 style="margin-top:15px">👍 FEEDBACK</h2>
            <div style="display:flex; gap:10px;">
                <button onclick="feedback('good')" style="flex:1">👍 Good</button>
                <button onclick="feedback('bad')" style="flex:1">👎 Bad</button>
            </div>
            <div id="feedbackCount" class="status">Feedback: 0</div>
        </div>
    </div>

    <script>
        let ws;
        let autoPlay = false;
        let autoInterval;

        function connect() {
            ws = new WebSocket('ws://' + location.host + '/ws');

            ws.onopen = () => {
                log('Connected');
                send({type: 'get_state'});
            };

            ws.onmessage = (e) => {
                try {
                    handle(JSON.parse(e.data));
                } catch(err) {
                    log('Parse error: ' + err);
                }
            };

            ws.onerror = (e) => log('WS error');
            ws.onclose = () => {
                log('Disconnected, retrying...');
                setTimeout(connect, 2000);
            };
        }

        function send(data) {
            if (ws && ws.readyState === 1) {
                ws.send(JSON.stringify(data));
            }
        }

        function handle(data) {
            console.log('Received:', data);

            if (data.type === 'state') {
                renderGrid(data.game);
                updateStatus(data);
            }
            else if (data.type === 'game_result') {
                log(data.cmd + ' → ' + data.text);
                send({type: 'get_state'});
            }
            else if (data.type === 'chat_result') {
                addChat('user', data.input);
                addChat('bot', data.response, data.intent, data.confidence, data.backend);
                if (data.action_done) {
                    log('Action: ' + data.action.type);
                    send({type: 'get_state'});
                }
            }
            else if (data.type === 'ai_move') {
                document.getElementById('aiAction').textContent = data.action;
                document.getElementById('aiConf').textContent =
                    data.confidence ? (data.confidence * 100).toFixed(0) + '% confident' : data.reason || '';
            }
            else if (data.type === 'level_loaded') {
                log('Loaded: ' + data.name);
                send({type: 'get_state'});
            }
            else if (data.type === 'feedback_ok') {
                document.getElementById('feedbackCount').textContent = 'Feedback: ' + data.total;
            }
            else if (data.type === 'error') {
                log('Error: ' + data.msg);
            }
        }

        function renderGrid(game) {
            const grid = game.grid;
            const robot = game.robot;
            const container = document.getElementById('grid');

            const h = grid.length;
            const w = grid[0].length;
            container.style.gridTemplateColumns = 'repeat(' + w + ', 28px)';
            container.innerHTML = '';

            const dirs = {north:'↑', south:'↓', east:'→', west:'←'};

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';

                    if (x === robot.x && y === robot.y) {
                        cell.className += ' robot';
                        cell.textContent = dirs[robot.dir] || '●';
                    } else {
                        const v = grid[y][x];
                        if (v === 1) { cell.className += ' wall'; cell.textContent = '▓'; }
                        else if (v === 2) { cell.className += ' goal'; cell.textContent = '★'; }
                    }

                    container.appendChild(cell);
                }
            }

            // Update level select
            const select = document.getElementById('level');
            if (select.options.length === 0 && game.levels) {
                // This doesn't exist in game, get from state
            }
        }

        function updateStatus(data) {
            document.getElementById('llmStatus').textContent =
                'LLM: ' + data.conversation.llm + (data.conversation.llm_ok ? ' ✓' : ' ✗');
            document.getElementById('convStatus').textContent =
                'Conv: ' + (data.conversation.trained ? 'trained ✓' : 'not trained');
            document.getElementById('feedbackCount').textContent = 'Feedback: ' + data.feedback;

            // Update level select
            const select = document.getElementById('level');
            if (select.options.length === 0 && data.levels) {
                data.levels.forEach(l => {
                    const opt = document.createElement('option');
                    opt.value = l;
                    opt.textContent = l;
                    select.appendChild(opt);
                });
            }

            // Get AI move
            send({type: 'get_ai_move'});
        }

        function cmd(c) {
            send({type: 'game_cmd', cmd: c});
        }

        function sendChat() {
            const input = document.getElementById('chatInput');
            const text = input.value.trim();
            if (text) {
                send({type: 'chat', text: text});
                input.value = '';
            }
        }

        function addChat(who, text, intent, conf, backend) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'msg ' + who;

            if (who === 'bot') {
                div.innerHTML = text +
                    '<div class="meta">' + intent + ' (' + (conf*100).toFixed(0) + '%) via ' + backend + '</div>';
            } else {
                div.textContent = text;
            }

            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function toggleAuto() {
            autoPlay = !autoPlay;
            document.getElementById('autoBtn').className = autoPlay ? 'on' : '';

            if (autoPlay) {
                autoInterval = setInterval(() => {
                    send({type: 'get_ai_move'});
                    // Execute the suggested move after a short delay
                    setTimeout(() => {
                        const action = document.getElementById('aiAction').textContent;
                        if (action && action !== '-' && action !== 'none') {
                            const cmdMap = {
                                'move_forward': 'forward',
                                'move_backward': 'backward',
                                'rotate_left': 'left',
                                'rotate_right': 'right',
                            };
                            cmd(cmdMap[action] || action);
                        }
                    }, 200);
                }, 1500);
                log('Auto-play ON');
            } else {
                clearInterval(autoInterval);
                log('Auto-play OFF');
            }
        }

        function loadLevel(name) {
            send({type: 'load_level', name: name});
        }

        function feedback(rating) {
            send({type: 'feedback', rating: rating});
            log('Feedback: ' + rating);
        }

        function log(msg) {
            const logDiv = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            logDiv.innerHTML += '[' + time + '] ' + msg + '<br>';
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        connect();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return INTERFACE_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60)
            except asyncio.TimeoutError:
                # Send ping to keep alive
                await websocket.send_json({"type": "ping"})
                continue

            msg_type = data.get("type", "")
            print(f"[WS] Received: {msg_type}")

            try:
                if msg_type == "get_state":
                    await websocket.send_json(session.get_state())

                elif msg_type == "game_cmd":
                    result = session.game_command(data.get("cmd", ""))
                    await websocket.send_json(result)

                elif msg_type == "chat":
                    result = session.chat(data.get("text", ""))
                    await websocket.send_json(result)

                elif msg_type == "get_ai_move":
                    result = session.get_ai_move()
                    await websocket.send_json(result)

                elif msg_type == "load_level":
                    result = session.load_level(data.get("name", "tutorial"))
                    await websocket.send_json(result)

                elif msg_type == "feedback":
                    result = session.add_feedback(data.get("rating", ""))
                    await websocket.send_json(result)

                else:
                    await websocket.send_json({"type": "error", "msg": f"Unknown: {msg_type}"})

            except Exception as e:
                traceback.print_exc()
                await websocket.send_json({"type": "error", "msg": str(e)})

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()


def main():
    print("=" * 50)
    print("🧠 Brain B - Interactive Training")
    print("=" * 50)
    print()
    print("Open: http://localhost:8765")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")


if __name__ == "__main__":
    main()
