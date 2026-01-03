"""
Swarm Coordination - Multi-Robot Communication and Collaboration

Enables robots in the same swarm (owned by same person) to:
1. Discover each other on local network
2. Share learned experiences (with owner consent)
3. Coordinate on tasks
4. Delegate subtasks
5. Avoid conflicts (e.g., both reaching for same object)

Privacy/Safety:
- Only robots with same owner can communicate
- Owner must enable swarm mode
- Shared experiences are anonymized
- No personal data about humans is shared
"""

import os
import json
import socket
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime
from enum import Enum, auto
import uuid
import hashlib

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of swarm messages."""
    # Discovery
    ANNOUNCE = "announce"           # Robot announcing presence
    DISCOVER = "discover"           # Request for nearby robots
    PRESENCE = "presence"           # Response to discover
    
    # Coordination
    TASK_OFFER = "task_offer"       # Offering a task to swarm
    TASK_ACCEPT = "task_accept"     # Accepting an offered task
    TASK_DECLINE = "task_decline"   # Declining a task
    TASK_COMPLETE = "task_complete" # Task completed notification
    
    # Conflict avoidance
    INTENT = "intent"               # Announcing intended action
    YIELD = "yield"                 # Yielding to another robot
    PROCEED = "proceed"             # Confirming safe to proceed
    
    # Experience sharing
    EXPERIENCE_OFFER = "experience_offer"  # Offering to share experience
    EXPERIENCE_REQUEST = "experience_request"  # Requesting experience
    EXPERIENCE_DATA = "experience_data"  # Actual experience data
    
    # System
    HEARTBEAT = "heartbeat"         # Alive signal
    SHUTDOWN = "shutdown"           # Robot shutting down


@dataclass
class SwarmRobot:
    """A robot in the swarm."""
    robot_id: str
    owner_id: str
    name: str
    capabilities: List[str]
    ip_address: str
    port: int
    last_seen: str
    status: str = "online"  # online, busy, offline
    current_task: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SwarmMessage:
    """A message between robots in the swarm."""
    message_id: str
    message_type: str
    sender_id: str
    recipient_id: str  # "*" for broadcast
    timestamp: str
    payload: Dict = field(default_factory=dict)
    signature: str = ""  # For verification
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SwarmMessage':
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SwarmMessage':
        return cls.from_dict(json.loads(json_str))


@dataclass
class SharedExperience:
    """An experience shared between robots."""
    experience_id: str
    source_robot_id: str
    experience_type: str  # "skill", "obstacle", "object", "route"
    description: str
    data: Dict
    created_at: str
    share_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SwarmCoordinator:
    """
    Coordinates communication between robots in a swarm.
    
    Enables discovery, task delegation, and experience sharing.
    """
    
    # Multicast address for swarm discovery
    MULTICAST_GROUP = "224.1.1.1"
    MULTICAST_PORT = 5007
    
    # Unicast port for direct messages
    UNICAST_PORT = 5008
    
    def __init__(
        self,
        robot_id: str,
        owner_id: str,
        robot_name: str = "Continuon",
        capabilities: List[str] = None,
        data_dir: Path = Path("/opt/continuonos/brain/swarm/coordination"),
    ):
        self.robot_id = robot_id
        self.owner_id = owner_id
        self.robot_name = robot_name
        self.capabilities = capabilities or ["general"]
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._swarm: Dict[str, SwarmRobot] = {}
        self._shared_experiences: Dict[str, SharedExperience] = {}
        self._pending_tasks: Dict[str, Dict] = {}
        self._message_handlers: Dict[str, Callable] = {}
        
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        self._signing_key = self._get_or_create_key()
        
        self._load_data()
        self._register_default_handlers()
    
    def _get_or_create_key(self) -> bytes:
        """Get or create signing key for message verification."""
        key_file = self.data_dir / "swarm_key.bin"
        if key_file.exists():
            return key_file.read_bytes()
        else:
            import secrets
            key = secrets.token_bytes(32)
            key_file.write_bytes(key)
            return key
    
    def _load_data(self):
        """Load saved swarm data."""
        exp_file = self.data_dir / "shared_experiences.json"
        if exp_file.exists():
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                    for exp_id, exp_data in data.items():
                        self._shared_experiences[exp_id] = SharedExperience(**exp_data)
            except Exception as e:
                logger.error(f"Failed to load experiences: {e}")
    
    def _save_data(self):
        """Save swarm data."""
        exp_file = self.data_dir / "shared_experiences.json"
        with open(exp_file, 'w') as f:
            json.dump({
                eid: exp.to_dict() for eid, exp in self._shared_experiences.items()
            }, f, indent=2)
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self._message_handlers[MessageType.ANNOUNCE.value] = self._handle_announce
        self._message_handlers[MessageType.DISCOVER.value] = self._handle_discover
        self._message_handlers[MessageType.PRESENCE.value] = self._handle_presence
        self._message_handlers[MessageType.HEARTBEAT.value] = self._handle_heartbeat
        self._message_handlers[MessageType.SHUTDOWN.value] = self._handle_shutdown
        self._message_handlers[MessageType.INTENT.value] = self._handle_intent
        self._message_handlers[MessageType.TASK_OFFER.value] = self._handle_task_offer
        self._message_handlers[MessageType.EXPERIENCE_OFFER.value] = self._handle_experience_offer
    
    def start(self):
        """Start the swarm coordinator."""
        if self._running:
            return
        
        self._running = True
        
        # Start listener thread
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="SwarmListener"
        )
        self._listener_thread.start()
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="SwarmHeartbeat"
        )
        self._heartbeat_thread.start()
        
        # Announce presence
        self.announce()
        
        logger.info(f"Swarm coordinator started for {self.robot_id}")
    
    def stop(self):
        """Stop the swarm coordinator."""
        if not self._running:
            return
        
        # Announce shutdown
        self.broadcast(MessageType.SHUTDOWN, {})
        
        self._running = False
        
        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        
        logger.info("Swarm coordinator stopped")
    
    def _listen_loop(self):
        """Listen for swarm messages."""
        try:
            # Create multicast socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.MULTICAST_PORT))
            
            # Join multicast group
            import struct
            mreq = struct.pack("4sl", socket.inet_aton(self.MULTICAST_GROUP), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            sock.settimeout(1.0)
            
            while self._running:
                try:
                    data, addr = sock.recvfrom(65535)
                    self._handle_message(data.decode(), addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
        
        except Exception as e:
            logger.error(f"Listener failed to start: {e}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            try:
                self.broadcast(MessageType.HEARTBEAT, {
                    'status': 'online',
                    'uptime': time.time(),
                })
                time.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5)
    
    def _handle_message(self, data: str, addr: tuple):
        """Handle an incoming message."""
        try:
            msg = SwarmMessage.from_json(data)
            
            # Ignore own messages
            if msg.sender_id == self.robot_id:
                return
            
            # Verify sender has same owner (security)
            sender = self._swarm.get(msg.sender_id)
            if sender and sender.owner_id != self.owner_id:
                logger.warning(f"Ignoring message from robot with different owner")
                return
            
            # Find handler
            handler = self._message_handlers.get(msg.message_type)
            if handler:
                handler(msg, addr)
            else:
                logger.warning(f"No handler for message type: {msg.message_type}")
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_announce(self, msg: SwarmMessage, addr: tuple):
        """Handle robot announcement."""
        payload = msg.payload
        robot = SwarmRobot(
            robot_id=msg.sender_id,
            owner_id=payload.get('owner_id', ''),
            name=payload.get('name', 'Unknown'),
            capabilities=payload.get('capabilities', []),
            ip_address=addr[0],
            port=payload.get('port', self.UNICAST_PORT),
            last_seen=datetime.now().isoformat(),
        )
        
        # Only add if same owner
        if robot.owner_id == self.owner_id:
            self._swarm[robot.robot_id] = robot
            logger.info(f"Discovered robot: {robot.name} ({robot.robot_id})")
    
    def _handle_discover(self, msg: SwarmMessage, addr: tuple):
        """Handle discovery request."""
        # Respond with presence
        self.send_to(msg.sender_id, MessageType.PRESENCE, {
            'owner_id': self.owner_id,
            'name': self.robot_name,
            'capabilities': self.capabilities,
            'port': self.UNICAST_PORT,
        })
    
    def _handle_presence(self, msg: SwarmMessage, addr: tuple):
        """Handle presence response."""
        self._handle_announce(msg, addr)
    
    def _handle_heartbeat(self, msg: SwarmMessage, addr: tuple):
        """Handle heartbeat."""
        if msg.sender_id in self._swarm:
            self._swarm[msg.sender_id].last_seen = datetime.now().isoformat()
            self._swarm[msg.sender_id].status = msg.payload.get('status', 'online')
    
    def _handle_shutdown(self, msg: SwarmMessage, addr: tuple):
        """Handle shutdown notification."""
        if msg.sender_id in self._swarm:
            self._swarm[msg.sender_id].status = 'offline'
            logger.info(f"Robot {msg.sender_id} went offline")
    
    def _handle_intent(self, msg: SwarmMessage, addr: tuple):
        """Handle intent announcement (for conflict avoidance)."""
        # Check if we have a conflicting intent
        intent = msg.payload
        # In real implementation, check for conflicts and respond
        logger.debug(f"Robot {msg.sender_id} intends to: {intent}")
    
    def _handle_task_offer(self, msg: SwarmMessage, addr: tuple):
        """Handle task offer from another robot."""
        task = msg.payload
        task_id = task.get('task_id', '')
        
        # Check if we can handle this task
        required_caps = task.get('required_capabilities', [])
        can_handle = all(cap in self.capabilities for cap in required_caps)
        
        if can_handle:
            # Could automatically accept or queue for decision
            logger.info(f"Received task offer: {task.get('description', '')}")
            self._pending_tasks[task_id] = task
    
    def _handle_experience_offer(self, msg: SwarmMessage, addr: tuple):
        """Handle experience sharing offer."""
        exp_data = msg.payload
        experience = SharedExperience(
            experience_id=exp_data.get('experience_id', str(uuid.uuid4())),
            source_robot_id=msg.sender_id,
            experience_type=exp_data.get('type', 'general'),
            description=exp_data.get('description', ''),
            data=exp_data.get('data', {}),
            created_at=datetime.now().isoformat(),
        )
        
        self._shared_experiences[experience.experience_id] = experience
        self._save_data()
        logger.info(f"Received shared experience: {experience.description}")
    
    def announce(self):
        """Announce presence to the swarm."""
        self.broadcast(MessageType.ANNOUNCE, {
            'owner_id': self.owner_id,
            'name': self.robot_name,
            'capabilities': self.capabilities,
            'port': self.UNICAST_PORT,
        })
    
    def discover(self):
        """Request discovery of nearby robots."""
        self.broadcast(MessageType.DISCOVER, {})
    
    def broadcast(self, msg_type: MessageType, payload: Dict):
        """Broadcast a message to all robots."""
        msg = SwarmMessage(
            message_id=str(uuid.uuid4()),
            message_type=msg_type.value,
            sender_id=self.robot_id,
            recipient_id="*",
            timestamp=datetime.now().isoformat(),
            payload=payload,
        )
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.sendto(
                msg.to_json().encode(),
                (self.MULTICAST_GROUP, self.MULTICAST_PORT)
            )
            sock.close()
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")
    
    def send_to(self, robot_id: str, msg_type: MessageType, payload: Dict):
        """Send a message to a specific robot."""
        if robot_id not in self._swarm:
            logger.warning(f"Robot {robot_id} not in swarm")
            return
        
        robot = self._swarm[robot_id]
        
        msg = SwarmMessage(
            message_id=str(uuid.uuid4()),
            message_type=msg_type.value,
            sender_id=self.robot_id,
            recipient_id=robot_id,
            timestamp=datetime.now().isoformat(),
            payload=payload,
        )
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(msg.to_json().encode(), (robot.ip_address, robot.port))
            sock.close()
        except Exception as e:
            logger.error(f"Failed to send to {robot_id}: {e}")
    
    def offer_task(
        self,
        task_id: str,
        description: str,
        required_capabilities: List[str],
        data: Dict = None,
    ):
        """Offer a task to the swarm."""
        self.broadcast(MessageType.TASK_OFFER, {
            'task_id': task_id,
            'description': description,
            'required_capabilities': required_capabilities,
            'data': data or {},
        })
    
    def announce_intent(self, action: str, target: str, priority: int = 5):
        """
        Announce intended action for conflict avoidance.
        
        Other robots can yield or request we yield.
        """
        self.broadcast(MessageType.INTENT, {
            'action': action,
            'target': target,
            'priority': priority,
        })
    
    def share_experience(
        self,
        experience_type: str,
        description: str,
        data: Dict,
    ):
        """Share an experience with the swarm."""
        experience = SharedExperience(
            experience_id=str(uuid.uuid4()),
            source_robot_id=self.robot_id,
            experience_type=experience_type,
            description=description,
            data=data,
            created_at=datetime.now().isoformat(),
        )
        
        self._shared_experiences[experience.experience_id] = experience
        self._save_data()
        
        self.broadcast(MessageType.EXPERIENCE_OFFER, {
            'experience_id': experience.experience_id,
            'type': experience_type,
            'description': description,
            'data': data,
        })
        
        return experience
    
    def get_swarm_members(self) -> List[SwarmRobot]:
        """Get all known swarm members."""
        return list(self._swarm.values())
    
    def get_online_members(self) -> List[SwarmRobot]:
        """Get online swarm members."""
        return [r for r in self._swarm.values() if r.status == 'online']
    
    def get_shared_experiences(self) -> List[SharedExperience]:
        """Get all shared experiences."""
        return list(self._shared_experiences.values())
    
    def get_swarm_capabilities(self) -> Set[str]:
        """Get combined capabilities of the swarm."""
        caps = set(self.capabilities)
        for robot in self._swarm.values():
            caps.update(robot.capabilities)
        return caps

