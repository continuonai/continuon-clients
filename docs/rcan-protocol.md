# RCAN: Robot Communication & Addressing Network Protocol

**Version:** 1.0.0  
**Status:** Draft  
**Authors:** ContinuonXR Team  
**Date:** 2026-01-02

## Abstract

RCAN (Robot Communication & Addressing Network) is a protocol specification for addressing, discovering, authenticating, and communicating with robotic agents across local networks and the internet. Similar to how ICANN manages domain names and IP addresses for the internet, RCAN provides a hierarchical addressing scheme and communication protocol specifically designed for embodied AI agents.

## 1. Addressing Scheme

### 1.1 Robot Universal Resource Identifier (RURI)

Every robot has a globally unique RURI:

```
rcan://<registry>/<manufacturer>/<model>/<device-id>[:<port>][/<capability>]
```

**Examples:**
```
rcan://continuon.cloud/continuon/companion-v1/d3a4b5c6-7890-1234-5678-abcdef012345
rcan://continuon.cloud/continuon/companion-v1/d3a4b5c6/arm
rcan://local.rcan/discovered/192.168.1.42:8080
```

### 1.2 Address Components

| Component | Description | Required |
|-----------|-------------|----------|
| `registry` | Root registry domain (e.g., `continuon.cloud`, `local.rcan`) | Yes |
| `manufacturer` | Manufacturer namespace | Yes |
| `model` | Robot model identifier | Yes |
| `device-id` | UUID or short-form device identifier | Yes |
| `port` | Communication port (default: 8080) | No |
| `capability` | Specific capability endpoint (e.g., `/arm`, `/vision`, `/chat`) | No |

### 1.3 Local Discovery (mDNS/DNS-SD)

For LAN discovery without cloud connectivity:

```
_rcan._tcp.local.
```

Service TXT records:
```
ruri=rcan://continuon.cloud/continuon/companion-v1/d3a4b5c6
model=companion-v1
caps=arm,vision,chat,teleop
roles=owner,guest
version=1.0.0
```

## 2. Role-Based Access Control (RBAC)

### 2.1 Role Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     ROLE HIERARCHY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CREATOR (Level 5) ─────────────────────────────────────┐  │
│    │ • Full hardware/software control                    │  │
│    │ • OTA push, training, model deployment              │  │
│    │ • Safety override (with audit)                      │  │
│    │ • User management across all robots                 │  │
│    ▼                                                     │  │
│  OWNER (Level 4) ───────────────────────────────────────┤  │
│    │ • Robot configuration and personalization          │  │
│    │ • OTA approval, skill installation                 │  │
│    │ • Add/remove users, create leases                  │  │
│    │ • Training data contribution (opt-in)              │  │
│    ▼                                                     │  │
│  LEASEE (Level 3) ──────────────────────────────────────┤  │
│    │ • Time-bound full operational control              │  │
│    │ • Cannot modify ownership or safety limits         │  │
│    │ • Can grant USER/GUEST access within lease         │  │
│    ▼                                                     │  │
│  USER (Level 2) ────────────────────────────────────────┤  │
│    │ • Operational control within allowed modes         │  │
│    │ • Cannot change configuration                      │  │
│    │ • Session-based, must be authenticated             │  │
│    ▼                                                     │  │
│  GUEST (Level 1) ───────────────────────────────────────┘  │
│      • Limited interaction (chat, basic queries)            │
│      • Read-only status, no control                         │
│      • Rate-limited, anonymous or authenticated             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Role Capabilities Matrix

| Capability | Creator | Owner | Leasee | User | Guest |
|------------|---------|-------|--------|------|-------|
| **View status** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Chat/query** | ✅ | ✅ | ✅ | ✅ | ✅ (rate-limited) |
| **Teleop control** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Arm control** | ✅ | ✅ | ✅ | ⚙️ (if permitted) | ❌ |
| **Navigation** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Record episodes** | ✅ | ✅ | ✅ | ⚙️ | ❌ |
| **Training contribute** | ✅ | ✅ | ⚙️ (if permitted) | ❌ | ❌ |
| **Install skills** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **OTA updates** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Safety config** | ✅ | ⚙️ (limited) | ❌ | ❌ | ❌ |
| **User management** | ✅ | ✅ | ⚙️ (session) | ❌ | ❌ |
| **Model deployment** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Hardware diagnostics** | ✅ | ✅ | ⚙️ | ❌ | ❌ |

Legend: ✅ = Full access, ⚙️ = Conditional/limited, ❌ = No access

### 2.3 Multi-Robot Scenarios

#### Fleet Mode
```json
{
  "fleet_id": "warehouse-alpha",
  "robots": ["robot-001", "robot-002", "robot-003"],
  "access_mode": "coordinated",
  "commands": {
    "broadcast": true,
    "individual": true,
    "choreographed": true
  }
}
```

#### Handoff Protocol
When transferring control between users:

```
1. Current holder sends RCAN_RELEASE
2. Robot broadcasts RCAN_AVAILABLE (30s window)
3. New user sends RCAN_CLAIM with auth token
4. Robot validates and sends RCAN_GRANTED or RCAN_DENIED
```

## 3. Communication Protocol

### 3.1 Message Format

```protobuf
message RCANMessage {
  string version = 1;          // "1.0.0"
  string message_id = 2;       // UUID
  string source_ruri = 3;      // Sender RURI
  string target_ruri = 4;      // Recipient RURI (or broadcast)
  string auth_token = 5;       // JWT or session token
  MessageType type = 6;
  bytes payload = 7;           // Type-specific payload
  int64 timestamp_ms = 8;
  int32 ttl_ms = 9;            // Time-to-live for command
  Priority priority = 10;
  
  enum MessageType {
    DISCOVER = 0;
    STATUS = 1;
    COMMAND = 2;
    STREAM = 3;
    EVENT = 4;
    HANDOFF = 5;
    ACK = 6;
    ERROR = 7;
  }
  
  enum Priority {
    LOW = 0;
    NORMAL = 1;
    HIGH = 2;
    SAFETY = 3;      // Highest, always processed
  }
}
```

### 3.2 Transport Layers

| Transport | Use Case | Latency | Reliability |
|-----------|----------|---------|-------------|
| **WebSocket** | Real-time control, streaming | Low | Medium |
| **gRPC** | Structured commands, bidirectional | Low | High |
| **MQTT** | IoT integration, pub/sub | Medium | High |
| **HTTP/REST** | Status queries, configuration | Medium | High |
| **UDP** | Discovery, heartbeats | Lowest | Low |

### 3.3 Standard Endpoints

```
POST   /rcan/v1/discover          # Broadcast discovery
GET    /rcan/v1/status            # Robot status
POST   /rcan/v1/auth/claim        # Claim control
DELETE /rcan/v1/auth/release      # Release control
POST   /rcan/v1/command           # Send command
WS     /rcan/v1/stream            # Bidirectional stream
POST   /rcan/v1/handoff           # Transfer control
```

## 4. Security

### 4.1 Authentication Flow

```
┌────────┐          ┌────────┐          ┌────────┐
│  App   │          │ Robot  │          │ Cloud  │
└────┬───┘          └────┬───┘          └────┬───┘
     │                   │                   │
     │──── DISCOVER ────►│                   │
     │◄─── ANNOUNCE ─────│                   │
     │                   │                   │
     │──── AUTH_REQ ────►│                   │
     │                   │──── VALIDATE ────►│
     │                   │◄─── TOKEN ────────│
     │◄─── AUTH_RESP ────│                   │
     │                   │                   │
     │════ SECURE SESSION ══════════════════│
```

### 4.2 Token Structure (JWT)

```json
{
  "sub": "user-uuid",
  "iss": "continuon.cloud",
  "aud": "rcan://continuon.cloud/continuon/companion-v1/*",
  "role": "owner",
  "scope": ["control", "config", "training"],
  "fleet": ["robot-001", "robot-002"],
  "exp": 1735689600,
  "iat": 1735603200
}
```

### 4.3 Safety Invariants

1. **Local safety always wins** - No remote command can bypass on-device safety checks
2. **Graceful degradation** - Network loss triggers safe-stop, not undefined behavior
3. **Audit trail** - All commands logged with user, timestamp, and outcome
4. **Rate limiting** - Commands throttled per role (Guest: 10/min, User: 100/min, etc.)
5. **Timeout enforcement** - Control sessions expire; explicit renewal required

## 5. Multi-Robot Coordination

### 5.1 Fleet Commands

```json
{
  "command": "COORDINATED_MOVE",
  "fleet_id": "warehouse-alpha",
  "targets": ["robot-001", "robot-002"],
  "choreography": {
    "type": "sequential",
    "steps": [
      {"robot": "robot-001", "action": "move", "params": {"x": 10, "y": 0}},
      {"robot": "robot-002", "action": "move", "params": {"x": 0, "y": 10}, "wait_for": "robot-001"}
    ]
  }
}
```

### 5.2 Swarm Discovery

```
RCAN_SWARM_ANNOUNCE {
  swarm_id: "delivery-swarm-42"
  members: ["robot-a", "robot-b", "robot-c"]
  leader: "robot-a"
  formation: "follow-leader"
  join_policy: "closed"  // open | approval | closed
}
```

### 5.3 Conflict Resolution

When multiple users attempt simultaneous control:

| Scenario | Resolution |
|----------|------------|
| Same robot, different roles | Higher role wins |
| Same robot, same role | First claim wins; queue others |
| Fleet command vs individual | Fleet owner decides policy |
| Safety conflict | Always stop; alert all users |

## 6. Edge Cases

### 6.1 Network Partition

```
if (cloud_unreachable && local_auth_cache_valid) {
  // Continue with cached credentials
  mode = "OFFLINE_AUTONOMOUS"
  log_for_sync()
} else if (cloud_unreachable && !local_auth_cache_valid) {
  // Deny new sessions, maintain existing
  mode = "LOCKDOWN"
}
```

### 6.2 Ownership Transfer

```
1. Current owner initiates TRANSFER_REQUEST
2. Robot enters TRANSFER_PENDING state (24h window)
3. New owner confirms with TRANSFER_ACCEPT + payment/agreement
4. Robot clears personal data, retains skills
5. New owner completes onboarding
```

### 6.3 Guest Interaction Limits

```yaml
guest_limits:
  chat_rate: 10/minute
  query_rate: 30/minute
  session_duration: 15min
  concurrent_sessions: 3
  capabilities:
    - status_read
    - chat_text
    - basic_info
  blocked:
    - control
    - camera_access
    - location_history
```

## 7. Implementation Notes

### 7.1 ContinuonBrain Integration

**Status: ✅ Implemented (January 2026)**

The RCAN protocol is implemented in `continuonbrain/services/rcan_service.py`:

```python
class RCANService:
    def handle_discover(self, message: RCANMessage) -> RCANMessage
    def handle_claim(self, message: RCANMessage, user_id: str, role: UserRole) -> RCANMessage
    def handle_command(self, message: RCANMessage, session_id: str) -> RCANMessage
    def handle_release(self, session_id: str) -> RCANMessage
    def get_status(self) -> dict
    def get_discovery_info(self) -> dict
    def validate_session(self, session_id: str, capability: RobotCapability) -> tuple[bool, str]
```

**HTTP Endpoints (in `continuonbrain/server/routes.py`):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rcan/v1/discover` | POST | Discovery request/response |
| `/rcan/v1/status` | GET | Robot RCAN status and discovery info |
| `/rcan/v1/auth/claim` | POST | Claim control (returns session) |
| `/rcan/v1/auth/release` | DELETE | Release control |
| `/rcan/v1/command` | POST | Send command via RCAN |
| `/rcan/v1/handoff` | POST | Transfer control (not yet implemented) |

The service is automatically initialized with `BrainService` and accessible via `self.service.rcan`.

### 7.2 ContinuonAI Flutter Integration

**Status: ✅ Implemented (January 2026)**

The RCAN protocol client is implemented in `continuonai/lib/services/rcan_client.dart`:

```dart
class RCANClient {
  /// Discover robots via HTTP probe
  Future<List<DiscoveredRobot>> discover({Duration timeout, List<String>? hosts});
  
  /// Claim control of a robot
  Future<RCANSession?> claim({required String userId, required UserRole role});
  
  /// Get RCAN status from connected robot  
  Future<Map<String, dynamic>> getStatus();
  
  /// Send command to the robot
  Future<RCANCommandResult> sendCommand({required String command, Map<String, dynamic>? parameters});
  
  /// Release control
  Future<bool> release();
  
  /// Stream of discovered robots
  Stream<List<DiscoveredRobot>> get discoveredRobots;
}
```

**Integration with BrainClient:**

- `BrainClient` now includes an `RCANClient rcan` instance
- RCAN client is automatically connected when `BrainClient.connect()` is called
- Session persistence via `FlutterSecureStorage`
- Helper methods: `claimRobotRcan()`, `releaseRobotRcan()`, `sendRcanCommand()`, `getRcanStatus()`

**Scanner Service Updates:**

- Native scanner (`scanner_service_native.dart`) scans for `_rcan._tcp` mDNS service
- Web scanner (`scanner_service_web.dart`) probes `/rcan/v1/status` endpoint first
- Both fall back to legacy endpoints for backward compatibility

## 8. References

- [ICANN - Internet Corporation for Assigned Names and Numbers](https://www.icann.org/)
- [mDNS RFC 6762](https://tools.ietf.org/html/rfc6762)
- [DNS-SD RFC 6763](https://tools.ietf.org/html/rfc6763)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [JWT RFC 7519](https://tools.ietf.org/html/rfc7519)
- [ContinuonXR Ownership Hierarchy](./ownership-hierarchy.md)
- [ContinuonXR System Architecture](./system-architecture.md)

