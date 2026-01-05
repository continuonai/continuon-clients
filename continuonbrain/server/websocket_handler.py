"""
WebSocket handler for real-time bi-directional communication.

Provides:
- Connection management with automatic cleanup
- Event broadcasting to all connected clients
- Message routing for commands from clients
- Heartbeat/ping-pong for connection health
"""

import asyncio
import base64
import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("WebSocketHandler")


class WebSocketOpcode(IntEnum):
    """WebSocket frame opcodes."""
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA


@dataclass
class WebSocketConnection:
    """Represents a single WebSocket connection."""
    id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    subscriptions: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)
    is_alive: bool = True

    def __hash__(self):
        return hash(self.id)


class WebSocketHandler:
    """
    Manages WebSocket connections and message routing.

    Event channels:
    - status: Robot status updates
    - training: Training progress events
    - cognitive: Thoughts, tool use, reasoning traces
    - chat: Chat message events
    - loops: Control loop health metrics
    - camera: Camera frame availability (metadata only)
    """

    WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    HEARTBEAT_INTERVAL = 30  # seconds

    def __init__(self, service=None):
        self.service = service
        self.connections: Dict[str, WebSocketConnection] = {}
        self._lock = asyncio.Lock()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        # Event queues for different channels
        self.event_queues: Dict[str, asyncio.Queue] = {
            "status": asyncio.Queue(maxsize=50),
            "training": asyncio.Queue(maxsize=100),
            "cognitive": asyncio.Queue(maxsize=100),
            "chat": asyncio.Queue(maxsize=100),
            "loops": asyncio.Queue(maxsize=50),
            "camera": asyncio.Queue(maxsize=10),
        }

        # Message handlers for incoming client messages
        self._message_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default message handlers."""
        self._message_handlers["subscribe"] = self._handle_subscribe
        self._message_handlers["unsubscribe"] = self._handle_unsubscribe
        self._message_handlers["ping"] = self._handle_client_ping
        self._message_handlers["command"] = self._handle_command

    def register_handler(self, message_type: str, handler: Callable):
        """Register a custom message handler."""
        self._message_handlers[message_type] = handler

    @staticmethod
    def is_websocket_upgrade(headers: Dict[str, str]) -> bool:
        """Check if the request is a WebSocket upgrade request."""
        connection = headers.get("connection", "").lower()
        upgrade = headers.get("upgrade", "").lower()
        return "upgrade" in connection and upgrade == "websocket"

    @staticmethod
    def compute_accept_key(key: str) -> str:
        """Compute the Sec-WebSocket-Accept response header value."""
        combined = key + WebSocketHandler.WEBSOCKET_GUID
        sha1 = hashlib.sha1(combined.encode()).digest()
        return base64.b64encode(sha1).decode()

    async def handle_upgrade(
        self,
        headers: Dict[str, str],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> bool:
        """
        Handle WebSocket upgrade handshake.

        Returns True if upgrade succeeded, False otherwise.
        """
        key = headers.get("sec-websocket-key")
        if not key:
            logger.warning("WebSocket upgrade missing Sec-WebSocket-Key")
            return False

        # Send upgrade response
        accept_key = self.compute_accept_key(key)
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key}\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()

        # Create connection
        conn_id = f"ws_{int(time.time() * 1000)}_{id(writer)}"
        conn = WebSocketConnection(
            id=conn_id,
            reader=reader,
            writer=writer,
            subscriptions={"status", "cognitive"},  # Default subscriptions
        )

        async with self._lock:
            self.connections[conn_id] = conn

        logger.info(f"WebSocket connection established: {conn_id}")

        # Send welcome message
        await self._send_json(conn, {
            "type": "welcome",
            "connection_id": conn_id,
            "channels": list(self.event_queues.keys()),
            "subscribed": list(conn.subscriptions),
        })

        # Start handling messages
        try:
            await self._handle_connection(conn)
        except Exception as e:
            logger.error(f"WebSocket error for {conn_id}: {e}")
        finally:
            await self._close_connection(conn)

        return True

    async def _handle_connection(self, conn: WebSocketConnection):
        """Handle incoming messages for a connection."""
        while conn.is_alive:
            try:
                frame = await self._read_frame(conn.reader)
                if frame is None:
                    break

                opcode, payload = frame

                if opcode == WebSocketOpcode.TEXT:
                    await self._handle_text_message(conn, payload.decode("utf-8"))
                elif opcode == WebSocketOpcode.BINARY:
                    await self._handle_binary_message(conn, payload)
                elif opcode == WebSocketOpcode.PING:
                    await self._send_frame(conn.writer, WebSocketOpcode.PONG, payload)
                elif opcode == WebSocketOpcode.PONG:
                    conn.last_ping = time.time()
                elif opcode == WebSocketOpcode.CLOSE:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket frame: {e}")
                break

    async def _read_frame(self, reader: asyncio.StreamReader) -> Optional[tuple]:
        """Read and parse a WebSocket frame."""
        try:
            # Read first 2 bytes
            header = await asyncio.wait_for(reader.read(2), timeout=60)
            if len(header) < 2:
                return None

            fin = (header[0] >> 7) & 1
            opcode = header[0] & 0x0F
            masked = (header[1] >> 7) & 1
            payload_len = header[1] & 0x7F

            # Extended payload length
            if payload_len == 126:
                ext = await reader.read(2)
                payload_len = struct.unpack(">H", ext)[0]
            elif payload_len == 127:
                ext = await reader.read(8)
                payload_len = struct.unpack(">Q", ext)[0]

            # Read mask if present
            mask = None
            if masked:
                mask = await reader.read(4)

            # Read payload
            payload = await reader.read(payload_len)

            # Unmask if needed
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            return (WebSocketOpcode(opcode), payload)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.debug(f"Frame read error: {e}")
            return None

    async def _send_frame(
        self,
        writer: asyncio.StreamWriter,
        opcode: WebSocketOpcode,
        payload: bytes,
    ):
        """Send a WebSocket frame."""
        frame = bytearray()

        # First byte: FIN + opcode
        frame.append(0x80 | opcode)

        # Second byte: payload length (no masking for server->client)
        payload_len = len(payload)
        if payload_len < 126:
            frame.append(payload_len)
        elif payload_len < 65536:
            frame.append(126)
            frame.extend(struct.pack(">H", payload_len))
        else:
            frame.append(127)
            frame.extend(struct.pack(">Q", payload_len))

        frame.extend(payload)

        try:
            writer.write(bytes(frame))
            await writer.drain()
        except Exception as e:
            logger.debug(f"Failed to send frame: {e}")

    async def _send_json(self, conn: WebSocketConnection, data: Dict[str, Any]):
        """Send JSON data to a connection."""
        try:
            payload = json.dumps(data, default=str).encode("utf-8")
            await self._send_frame(conn.writer, WebSocketOpcode.TEXT, payload)
        except Exception as e:
            logger.debug(f"Failed to send JSON to {conn.id}: {e}")
            conn.is_alive = False

    async def _handle_text_message(self, conn: WebSocketConnection, message: str):
        """Handle incoming text message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            handler = self._message_handlers.get(msg_type)
            if handler:
                await handler(conn, data)
            else:
                await self._send_json(conn, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
        except json.JSONDecodeError:
            await self._send_json(conn, {
                "type": "error",
                "message": "Invalid JSON",
            })

    async def _handle_binary_message(self, conn: WebSocketConnection, payload: bytes):
        """Handle incoming binary message."""
        # Binary messages could be used for camera frames, audio, etc.
        logger.debug(f"Received binary message: {len(payload)} bytes")

    async def _handle_subscribe(self, conn: WebSocketConnection, data: Dict[str, Any]):
        """Handle subscribe message."""
        channels = data.get("channels", [])
        if isinstance(channels, str):
            channels = [channels]

        valid_channels = set(channels) & set(self.event_queues.keys())
        conn.subscriptions.update(valid_channels)

        await self._send_json(conn, {
            "type": "subscribed",
            "channels": list(valid_channels),
            "all_subscriptions": list(conn.subscriptions),
        })

    async def _handle_unsubscribe(self, conn: WebSocketConnection, data: Dict[str, Any]):
        """Handle unsubscribe message."""
        channels = data.get("channels", [])
        if isinstance(channels, str):
            channels = [channels]

        for ch in channels:
            conn.subscriptions.discard(ch)

        await self._send_json(conn, {
            "type": "unsubscribed",
            "channels": channels,
            "all_subscriptions": list(conn.subscriptions),
        })

    async def _handle_client_ping(self, conn: WebSocketConnection, data: Dict[str, Any]):
        """Handle client ping message."""
        await self._send_json(conn, {
            "type": "pong",
            "timestamp": time.time(),
            "echo": data.get("timestamp"),
        })

    async def _handle_command(self, conn: WebSocketConnection, data: Dict[str, Any]):
        """Handle command message from client."""
        if not self.service:
            await self._send_json(conn, {
                "type": "error",
                "message": "No service available",
            })
            return

        command = data.get("command", "")
        args = data.get("args", {})
        request_id = data.get("request_id")

        try:
            # Route to appropriate service method
            result = await self._execute_command(command, args)
            response = {
                "type": "command_response",
                "command": command,
                "success": True,
                "result": result,
            }
            if request_id:
                response["request_id"] = request_id
            await self._send_json(conn, response)
        except Exception as e:
            response = {
                "type": "command_response",
                "command": command,
                "success": False,
                "error": str(e),
            }
            if request_id:
                response["request_id"] = request_id
            await self._send_json(conn, response)

    async def _execute_command(self, command: str, args: Dict[str, Any]) -> Any:
        """Execute a command via the service."""
        # Map commands to service methods
        command_map = {
            "status": lambda: self.service.GetRobotStatus(),
            "drive": lambda: self.service.Drive(args.get("steering"), args.get("throttle")),
            "mode": lambda: self.service.SetRobotMode(args.get("mode", "")),
            "safety_hold": lambda: self.service.TriggerSafetyHold(),
            "safety_reset": lambda: self.service.ResetSafetyGates(),
            "chat": lambda: self.service.ChatWithGemma(
                args.get("message", ""),
                args.get("history", []),
                session_id=args.get("session_id"),
            ),
        }

        handler = command_map.get(command)
        if not handler:
            raise ValueError(f"Unknown command: {command}")

        return await handler()

    async def _close_connection(self, conn: WebSocketConnection):
        """Close and cleanup a connection."""
        conn.is_alive = False

        async with self._lock:
            self.connections.pop(conn.id, None)

        try:
            await self._send_frame(conn.writer, WebSocketOpcode.CLOSE, b"")
            conn.writer.close()
            await conn.writer.wait_closed()
        except Exception:
            pass

        logger.info(f"WebSocket connection closed: {conn.id}")

    async def broadcast(self, channel: str, event: Dict[str, Any]):
        """Broadcast an event to all subscribers of a channel."""
        if channel not in self.event_queues:
            return

        message = {
            "type": "event",
            "channel": channel,
            "timestamp": time.time(),
            "data": event,
        }

        async with self._lock:
            connections = list(self.connections.values())

        for conn in connections:
            if channel in conn.subscriptions and conn.is_alive:
                await self._send_json(conn, message)

    async def push_event(self, channel: str, event: Dict[str, Any]):
        """Push an event to a channel queue for broadcasting."""
        if channel in self.event_queues:
            try:
                self.event_queues[channel].put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event
                try:
                    self.event_queues[channel].get_nowait()
                    self.event_queues[channel].put_nowait(event)
                except Exception:
                    pass

    async def start_background_tasks(self):
        """Start background broadcast and heartbeat tasks."""
        if self._running:
            return

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_background_tasks(self):
        """Stop background tasks."""
        self._running = False

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _broadcast_loop(self):
        """Background task to broadcast queued events."""
        while self._running:
            try:
                for channel, queue in self.event_queues.items():
                    try:
                        event = queue.get_nowait()
                        await self.broadcast(channel, event)
                    except asyncio.QueueEmpty:
                        pass

                await asyncio.sleep(0.05)  # 20 Hz broadcast rate
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
                await asyncio.sleep(1)

    async def _heartbeat_loop(self):
        """Background task to send heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)

                async with self._lock:
                    connections = list(self.connections.values())

                now = time.time()
                for conn in connections:
                    if not conn.is_alive:
                        continue

                    # Check for stale connections
                    if now - conn.last_ping > self.HEARTBEAT_INTERVAL * 3:
                        logger.warning(f"Connection {conn.id} timed out")
                        await self._close_connection(conn)
                        continue

                    # Send ping
                    try:
                        await self._send_frame(conn.writer, WebSocketOpcode.PING, b"ping")
                    except Exception:
                        conn.is_alive = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)

    def get_connections_info(self) -> List[Dict[str, Any]]:
        """Get info about all connections."""
        return [
            {
                "id": conn.id,
                "subscriptions": list(conn.subscriptions),
                "created_at": conn.created_at,
                "is_alive": conn.is_alive,
            }
            for conn in self.connections.values()
        ]
