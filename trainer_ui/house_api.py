"""
House 3D API for WebSocket and REST endpoints.

Provides:
- Scene templates (studio, 2-bedroom)
- Scene conversion from room scanner
- WebSocket for real-time robot state sync
- Training frame generation
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Add brain_b to path
brain_b_path = Path(__file__).parent.parent / 'brain_b' / 'simulator'
sys.path.insert(0, str(brain_b_path))

router = APIRouter(prefix="/house3d", tags=["house3d"])

# Active WebSocket connections
active_connections: List[WebSocket] = []


class RobotStateUpdate(BaseModel):
    """Robot state from web client."""
    position: Dict[str, float]
    yaw: float
    timestamp: int


class TrainingFrameRequest(BaseModel):
    """Request for rendered training frame."""
    position: Dict[str, float]
    yaw: float
    pitch: float = 0.0
    width: int = 640
    height: int = 480
    include_depth: bool = True


# =============================================================================
# Scene Templates
# =============================================================================

@router.get("/template/{template_name}")
async def get_template(template_name: str):
    """
    Get a pre-built house scene template.

    Args:
        template_name: 'studio_apartment' or 'two_bedroom'
    """
    try:
        from house_3d import HouseScene

        scene = HouseScene.from_template(template_name)
        return scene.to_json()
    except ImportError:
        # Fallback if house_3d not available
        return get_fallback_scene(template_name)
    except ValueError as e:
        return {"error": str(e)}


def get_fallback_scene(template_name: str) -> dict:
    """Fallback scene when house_3d module unavailable."""
    if template_name == 'studio_apartment':
        return {
            "name": "studio_apartment",
            "bounds": {"min": [0, 0, 0], "max": [8, 2.7, 9]},
            "objects": [
                # Floor
                {
                    "name": "Floor",
                    "transform": {"position": [4, 0, 4.5], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "plane",
                    "geometryParams": {"width": 8, "depth": 9},
                    "material": "oak_floor",
                    "tags": ["floor"],
                },
                # Walls
                {
                    "name": "Wall_North",
                    "transform": {"position": [4, 1.35, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [8, 2.7, 0.15]},
                    "material": "cream_paint",
                    "tags": ["wall"],
                },
                {
                    "name": "Wall_South",
                    "transform": {"position": [4, 1.35, 9], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [8, 2.7, 0.15]},
                    "material": "cream_paint",
                    "tags": ["wall"],
                },
                {
                    "name": "Wall_East",
                    "transform": {"position": [8, 1.35, 4.5], "rotation": [0, 90, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [9, 2.7, 0.15]},
                    "material": "cream_paint",
                    "tags": ["wall"],
                },
                {
                    "name": "Wall_West",
                    "transform": {"position": [0, 1.35, 4.5], "rotation": [0, 90, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [9, 2.7, 0.15]},
                    "material": "cream_paint",
                    "tags": ["wall"],
                },
                # Living area
                {
                    "name": "Sofa",
                    "transform": {"position": [4, 0.425, 2.5], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [2.2, 0.85, 0.9]},
                    "material": "gray_fabric",
                    "tags": ["furniture", "seating"],
                },
                {
                    "name": "Coffee_Table",
                    "transform": {"position": [4, 0.225, 3.5], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [1.2, 0.45, 0.6]},
                    "material": "walnut",
                    "tags": ["furniture", "table"],
                },
                {
                    "name": "TV_Console",
                    "transform": {"position": [4, 0.25, 5.5], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [1.5, 0.5, 0.45]},
                    "material": "matte_black_metal",
                    "tags": ["furniture", "storage"],
                },
                {
                    "name": "TV",
                    "transform": {"position": [4, 0.75, 5.6], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [1.4, 0.8, 0.08]},
                    "material": "screen_on",
                    "tags": ["furniture", "electronics"],
                },
                # Kitchen area
                {
                    "name": "Refrigerator",
                    "transform": {"position": [0.5, 0.9, 8], "rotation": [0, 180, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [0.9, 1.8, 0.8]},
                    "material": "stainless_steel",
                    "tags": ["furniture", "appliance"],
                },
                {
                    "name": "Counter",
                    "transform": {"position": [2.5, 0.45, 8], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [2.0, 0.9, 0.6]},
                    "material": "granite_gray",
                    "tags": ["furniture", "storage"],
                },
                # Bed area
                {
                    "name": "Bed",
                    "transform": {"position": [1.5, 0.3, 1], "rotation": [0, 90, 0], "scale": [1, 1, 1]},
                    "geometryType": "box",
                    "geometryParams": {"size": [1.6, 0.6, 2.1]},
                    "material": "beige_fabric",
                    "tags": ["furniture", "bed"],
                },
            ],
            "lights": [
                {"type": "ambient", "intensity": 0.3},
                {"type": "point", "position": [4, 2.5, 4.5], "intensity": 1.0, "range": 10},
                {"type": "directional", "position": [5, 10, 5], "direction": [-1, -2, -1], "intensity": 0.5},
            ],
        }
    else:
        return {
            "name": template_name,
            "bounds": {"min": [0, 0, 0], "max": [10, 2.7, 10]},
            "objects": [],
            "lights": [{"type": "ambient", "intensity": 0.5}],
        }


# =============================================================================
# Scene Conversion
# =============================================================================

@router.post("/from_scan")
async def scene_from_scan(scan_result: dict):
    """
    Convert room scanner result to 3D scene.

    Args:
        scan_result: Result from /room/scan endpoint
    """
    try:
        from house_3d import HouseScene

        scene = HouseScene.from_room_scan(scan_result)
        return scene.to_json()
    except ImportError:
        # Manual conversion
        return convert_scan_to_scene(scan_result)


def convert_scan_to_scene(scan_result: dict) -> dict:
    """Manual conversion of scan result to scene format."""
    room = scan_result.get('room', {})
    width = room.get('width', 5.0)
    depth = room.get('depth', 5.0)
    height = room.get('height', 2.7)

    objects = [
        {
            "name": "Floor",
            "transform": {"position": [width/2, 0, depth/2], "rotation": [0, 0, 0], "scale": [1, 1, 1]},
            "geometryType": "plane",
            "geometryParams": {"width": width, "depth": depth},
            "material": room.get('floor_material', 'oak_floor'),
            "tags": ["floor"],
        },
    ]

    # Add walls
    for i, (pos, rot, size) in enumerate([
        ([width/2, height/2, 0], [0, 0, 0], [width, height, 0.15]),
        ([width/2, height/2, depth], [0, 0, 0], [width, height, 0.15]),
        ([0, height/2, depth/2], [0, 90, 0], [depth, height, 0.15]),
        ([width, height/2, depth/2], [0, 90, 0], [depth, height, 0.15]),
    ]):
        objects.append({
            "name": f"Wall_{i}",
            "transform": {"position": pos, "rotation": rot, "scale": [1, 1, 1]},
            "geometryType": "box",
            "geometryParams": {"size": size},
            "material": room.get('wall_material', 'white_paint'),
            "tags": ["wall"],
        })

    # Add detected objects
    for asset in scan_result.get('generated_assets', []):
        pos = asset.get('position', [width/2, 0.5, depth/2])
        size = asset.get('size', [1, 1, 1])
        objects.append({
            "name": asset.get('name', 'Object'),
            "transform": {"position": pos, "rotation": [0, 0, 0], "scale": [1, 1, 1]},
            "geometryType": "box",
            "geometryParams": {"size": size},
            "material": asset.get('material', 'gray_fabric'),
            "tags": ["furniture"],
        })

    return {
        "name": "Scanned Room",
        "bounds": {"min": [0, 0, 0], "max": [width, height, depth]},
        "objects": objects,
        "lights": [
            {"type": "ambient", "intensity": 0.4},
            {"type": "point", "position": [width/2, height - 0.2, depth/2], "intensity": 1.0},
        ],
    }


# =============================================================================
# Training Frame Generation
# =============================================================================

@router.post("/render_frame")
async def render_training_frame(request: TrainingFrameRequest):
    """
    Render a training frame from Python renderer.

    Returns base64-encoded RGB and depth images.
    """
    import base64
    import io

    try:
        from house_3d import HouseScene, HouseRenderer

        # Load or cache scene
        scene = HouseScene.from_template('studio_apartment')

        # Render
        renderer = HouseRenderer(width=request.width, height=request.height)

        position = (request.position['x'], request.position.get('y', 0), request.position['z'])
        result = renderer.render_split_view(
            scene,
            position,
            request.yaw,
            request.pitch,
        )

        # Encode to base64
        from PIL import Image

        def encode_image(arr):
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()

        response = {
            "pov": encode_image(result['pov']),
            "pov_mime": "image/png",
        }

        if request.include_depth:
            # Convert depth to visual (normalize and convert to uint8)
            import numpy as np
            depth = result['depth']
            depth_vis = np.clip(depth / 10.0, 0, 1)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = np.stack([depth_vis] * 3, axis=-1)  # Convert to RGB
            response["depth"] = encode_image(depth_vis)
            response["depth_mime"] = "image/png"

        return response

    except ImportError as e:
        return {"error": f"Renderer not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# WebSocket for Real-time Sync
# =============================================================================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time robot state synchronization.

    Receives:
    - robot_state: Position/yaw updates from web client

    Sends:
    - scene_update: Scene changes
    - training_frame: Rendered frames
    - episode_complete: Training statistics
    """
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            await handle_websocket_message(websocket, message)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def handle_websocket_message(websocket: WebSocket, message: dict):
    """Handle incoming WebSocket message."""
    msg_type = message.get('type')

    if msg_type == 'robot_state':
        # Record state for training
        state = RobotStateUpdate(**message)
        # Could save to training buffer here
        pass

    elif msg_type == 'request_frame':
        # Client requesting a rendered frame
        try:
            frame_data = await render_training_frame(TrainingFrameRequest(**message))
            await websocket.send_json({
                "type": "training_frame",
                **frame_data,
            })
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})

    elif msg_type == 'load_scene':
        # Load a specific scene
        template = message.get('template', 'studio_apartment')
        scene_data = await get_template(template)
        await websocket.send_json({
            "type": "scene_update",
            "scene": scene_data,
        })


async def broadcast_scene_update(scene_data: dict):
    """Broadcast scene update to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json({
                "type": "scene_update",
                "scene": scene_data,
            })
        except Exception:
            pass


# =============================================================================
# Scene Persistence
# =============================================================================

SCENES_DIR = Path(__file__).parent / 'house_scenes'
SCENES_DIR.mkdir(exist_ok=True)


@router.get("/scenes")
async def list_scenes():
    """List saved scenes."""
    scenes = []
    for f in SCENES_DIR.glob('*.json'):
        scenes.append({
            "name": f.stem,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return {"scenes": scenes}


@router.get("/scenes/{name}")
async def load_scene(name: str):
    """Load a saved scene."""
    path = SCENES_DIR / f"{name}.json"
    if not path.exists():
        return {"error": f"Scene '{name}' not found"}

    with open(path) as f:
        return json.load(f)


@router.post("/scenes/{name}")
async def save_scene(name: str, scene: dict):
    """Save a scene."""
    path = SCENES_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(scene, f, indent=2)
    return {"success": True, "path": str(path)}
