# RCAN Technical Specification

**Version:** 1.0.0  
**Status:** Draft  
**Authors:** ContinuonAI Team  
**Date:** 2026-01-03

This document contains the formal technical specifications for the RCAN (Robot Communication & Addressing Network) protocol. For the conceptual overview, see [rcan-protocol.md](./rcan-protocol.md).

## Table of Contents

1. [RURI JSON Schema](#1-ruri-json-schema)
2. [Protocol Buffer Definitions](#2-protocol-buffer-definitions)
3. [mDNS Service Discovery](#3-mdns-service-discovery)
4. [JWT Token Schema](#4-jwt-token-schema)
5. [Handshake Flow](#5-handshake-flow)
6. [Error Codes](#6-error-codes)
7. [Conformance Requirements](#7-conformance-requirements)

---

## 1. RURI JSON Schema

### 1.1 URI Format

```
rcan://<registry>/<manufacturer>/<model>/<device-id>[:<port>][/<capability>]
```

### 1.2 Formal Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://rcan.dev/schemas/ruri.json",
  "title": "Robot Universal Resource Identifier (RURI)",
  "description": "A globally unique identifier for robotic agents in the RCAN protocol",
  "type": "object",
  "required": ["registry", "manufacturer", "model", "deviceId"],
  "properties": {
    "registry": {
      "type": "string",
      "description": "Root registry domain (e.g., continuon.cloud, local.rcan)",
      "pattern": "^[a-z0-9][a-z0-9.-]*[a-z0-9]$",
      "examples": ["continuon.cloud", "local.rcan", "robotics-co-op.org"]
    },
    "manufacturer": {
      "type": "string",
      "description": "Manufacturer namespace identifier",
      "pattern": "^[a-z0-9][a-z0-9-]*[a-z0-9]$",
      "minLength": 2,
      "maxLength": 64,
      "examples": ["continuon", "unitree", "boston-dynamics"]
    },
    "model": {
      "type": "string",
      "description": "Robot model identifier",
      "pattern": "^[a-z0-9][a-z0-9-]*[a-z0-9]$",
      "minLength": 1,
      "maxLength": 64,
      "examples": ["companion-v1", "go2", "spot-enterprise"]
    },
    "deviceId": {
      "type": "string",
      "description": "Unique device identifier (UUID or short-form)",
      "oneOf": [
        {
          "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
          "description": "Full UUID format"
        },
        {
          "pattern": "^[0-9a-f]{8}$",
          "description": "Short-form (first 8 chars of UUID)"
        }
      ],
      "examples": ["d3a4b5c6-7890-1234-5678-abcdef012345", "d3a4b5c6"]
    },
    "port": {
      "type": "integer",
      "description": "Communication port (default: 8080)",
      "minimum": 1,
      "maximum": 65535,
      "default": 8080
    },
    "capability": {
      "type": "string",
      "description": "Specific capability endpoint",
      "pattern": "^/[a-z][a-z0-9/-]*$",
      "examples": ["/arm", "/vision", "/chat", "/teleop/camera/front"]
    }
  },
  "additionalProperties": false
}
```

### 1.3 RURI Regex Pattern

For validation in various languages:

```
^rcan://([a-z0-9][a-z0-9.-]*[a-z0-9])/([a-z0-9][a-z0-9-]*[a-z0-9])/([a-z0-9][a-z0-9-]*[a-z0-9])/([0-9a-f]{8}(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?)(?::(\d{1,5}))?(/[a-z][a-z0-9/-]*)?$
```

### 1.4 Reference Implementations

**Python:**
```python
import re
from dataclasses import dataclass
from typing import Optional

RURI_PATTERN = re.compile(
    r'^rcan://([a-z0-9][a-z0-9.-]*[a-z0-9])/'  # registry
    r'([a-z0-9][a-z0-9-]*[a-z0-9])/'            # manufacturer
    r'([a-z0-9][a-z0-9-]*[a-z0-9])/'            # model
    r'([0-9a-f]{8}(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?)'  # device_id
    r'(?::(\d{1,5}))?'                          # port
    r'(/[a-z][a-z0-9/-]*)?$'                    # capability
)

@dataclass
class RURI:
    registry: str
    manufacturer: str
    model: str
    device_id: str
    port: int = 8080
    capability: Optional[str] = None
    
    @classmethod
    def parse(cls, ruri_string: str) -> 'RURI':
        match = RURI_PATTERN.match(ruri_string)
        if not match:
            raise ValueError(f"Invalid RURI: {ruri_string}")
        
        return cls(
            registry=match.group(1),
            manufacturer=match.group(2),
            model=match.group(3),
            device_id=match.group(4),
            port=int(match.group(5)) if match.group(5) else 8080,
            capability=match.group(6)
        )
    
    def __str__(self) -> str:
        s = f"rcan://{self.registry}/{self.manufacturer}/{self.model}/{self.device_id}"
        if self.port != 8080:
            s += f":{self.port}"
        if self.capability:
            s += self.capability
        return s
```

**TypeScript:**
```typescript
interface RURI {
  registry: string;
  manufacturer: string;
  model: string;
  deviceId: string;
  port: number;
  capability: string | null;
}

const RURI_REGEX = /^rcan:\/\/([a-z0-9][a-z0-9.-]*[a-z0-9])\/([a-z0-9][a-z0-9-]*[a-z0-9])\/([a-z0-9][a-z0-9-]*[a-z0-9])\/([0-9a-f]{8}(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?)(?::(\d{1,5}))?(\/[a-z][a-z0-9\/-]*)?$/;

function parseRURI(ruri: string): RURI {
  const match = ruri.match(RURI_REGEX);
  if (!match) {
    throw new Error(`Invalid RURI: ${ruri}`);
  }
  
  return {
    registry: match[1],
    manufacturer: match[2],
    model: match[3],
    deviceId: match[4],
    port: match[5] ? parseInt(match[5]) : 8080,
    capability: match[6] || null,
  };
}

function formatRURI(ruri: RURI): string {
  let s = `rcan://${ruri.registry}/${ruri.manufacturer}/${ruri.model}/${ruri.deviceId}`;
  if (ruri.port !== 8080) s += `:${ruri.port}`;
  if (ruri.capability) s += ruri.capability;
  return s;
}
```

---

## 2. Protocol Buffer Definitions

Save as `proto/rcan/v1/rcan.proto`:

```protobuf
syntax = "proto3";

package rcan.v1;

option java_package = "cloud.continuon.rcan.v1";
option go_package = "github.com/continuon/rcan/v1;rcanv1";

// ============================================================================
// RCAN Message - The universal envelope for all RCAN communication
// ============================================================================

message RCANMessage {
  string version = 1;           // Protocol version: "1.0.0"
  string message_id = 2;        // Unique message UUID
  string source_ruri = 3;       // Sender's RURI
  string target_ruri = 4;       // Recipient's RURI (or "broadcast")
  string auth_token = 5;        // JWT or session token
  MessageType type = 6;         // Message type enum
  bytes payload = 7;            // Type-specific payload (serialized submessage)
  int64 timestamp_ms = 8;       // Unix timestamp in milliseconds
  int32 ttl_ms = 9;             // Time-to-live for commands (0 = no expiry)
  Priority priority = 10;       // Message priority level
  
  enum MessageType {
    MESSAGE_TYPE_UNSPECIFIED = 0;
    DISCOVER = 1;               // Discovery broadcast/response
    STATUS = 2;                 // Status query/update
    COMMAND = 3;                // Control command
    STREAM = 4;                 // Streaming data (video, telemetry)
    EVENT = 5;                  // Async event notification
    HANDOFF = 6;                // Control transfer
    ACK = 7;                    // Acknowledgment
    ERROR = 8;                  // Error response
  }
  
  enum Priority {
    PRIORITY_UNSPECIFIED = 0;
    LOW = 1;                    // Background tasks
    NORMAL = 2;                 // Standard operations
    HIGH = 3;                   // Time-sensitive commands
    SAFETY = 4;                 // Safety-critical, always processed first
  }
}

// ============================================================================
// Discovery Messages
// ============================================================================

message DiscoverRequest {
  string query = 1;             // Optional filter (e.g., "model:companion-v1")
  int32 timeout_ms = 2;         // Discovery timeout (default: 5000)
  repeated string capabilities = 3;  // Required capabilities filter
}

message DiscoverResponse {
  string ruri = 1;              // Robot's full RURI
  string model = 2;             // Model identifier
  string friendly_name = 3;     // Human-readable name
  repeated string capabilities = 4;  // Available capabilities
  repeated string supported_roles = 5;  // Roles this robot accepts
  string protocol_version = 6;  // RCAN protocol version
  RobotStatus status = 7;       // Current status summary
}

// ============================================================================
// Authentication Messages
// ============================================================================

message AuthClaimRequest {
  string ruri = 1;              // Target robot RURI
  Role requested_role = 2;      // Desired role level
  string credential = 3;        // Auth credential (password, key, etc.)
  string client_id = 4;         // Client application identifier
  map<string, string> metadata = 5;  // Additional auth context
}

message AuthClaimResponse {
  bool granted = 1;             // Whether access was granted
  string session_token = 2;     // JWT session token (if granted)
  Role granted_role = 3;        // Actual role granted (may differ from requested)
  int64 expires_at = 4;         // Token expiration timestamp
  string error_message = 5;     // Error details (if denied)
  AuthErrorCode error_code = 6; // Structured error code
}

enum AuthErrorCode {
  AUTH_ERROR_UNSPECIFIED = 0;
  INVALID_CREDENTIALS = 1;
  INSUFFICIENT_PRIVILEGES = 2;
  ROBOT_BUSY = 3;
  ROLE_UNAVAILABLE = 4;
  RATE_LIMITED = 5;
  TOKEN_EXPIRED = 6;
  TOKEN_REVOKED = 7;
}

message AuthReleaseRequest {
  string session_token = 1;     // Current session token
  bool force = 2;               // Force release even if commands pending
}

message AuthReleaseResponse {
  bool released = 1;
  string error_message = 2;
}

// ============================================================================
// Role Definitions
// ============================================================================

enum Role {
  ROLE_UNSPECIFIED = 0;
  GUEST = 1;                    // Level 1: Limited interaction, read-only
  USER = 2;                     // Level 2: Operational control within modes
  LEASEE = 3;                   // Level 3: Time-bound full control
  OWNER = 4;                    // Level 4: Configuration and user management
  CREATOR = 5;                  // Level 5: Full hardware/software control
}

message RoleCapabilities {
  Role role = 1;
  repeated string allowed_capabilities = 2;
  repeated string denied_capabilities = 3;
  int32 rate_limit_per_minute = 4;
  int32 max_session_duration_seconds = 5;
}

// ============================================================================
// Command Messages
// ============================================================================

message CommandRequest {
  string command_id = 1;        // Unique command identifier
  string capability = 2;        // Target capability (e.g., "/arm", "/nav")
  string action = 3;            // Action name (e.g., "move", "grab", "speak")
  bytes parameters = 4;         // Action-specific parameters (JSON or protobuf)
  bool require_ack = 5;         // Whether to wait for acknowledgment
  int32 timeout_ms = 6;         // Command timeout
}

message CommandResponse {
  string command_id = 1;        // Echo of request command_id
  CommandStatus status = 2;     // Execution status
  bytes result = 3;             // Command result (if any)
  string error_message = 4;     // Error details (if failed)
  int64 execution_time_ms = 5;  // Time taken to execute
}

enum CommandStatus {
  COMMAND_STATUS_UNSPECIFIED = 0;
  ACCEPTED = 1;                 // Command accepted, executing
  COMPLETED = 2;                // Command completed successfully
  FAILED = 3;                   // Command failed
  REJECTED = 4;                 // Command rejected (auth, safety, etc.)
  TIMEOUT = 5;                  // Command timed out
  CANCELLED = 6;                // Command was cancelled
}

// ============================================================================
// Status Messages
// ============================================================================

message RobotStatus {
  string ruri = 1;
  OperationalState state = 2;
  float battery_percent = 3;
  bool safety_ok = 4;
  string current_controller = 5;    // RURI of current controller (if any)
  Role current_controller_role = 6;
  repeated string active_capabilities = 7;
  map<string, string> diagnostics = 8;
  int64 uptime_seconds = 9;
  int64 timestamp_ms = 10;
}

enum OperationalState {
  OPERATIONAL_STATE_UNSPECIFIED = 0;
  IDLE = 1;
  ACTIVE = 2;
  BUSY = 3;
  ERROR = 4;
  MAINTENANCE = 5;
  EMERGENCY_STOP = 6;
  OFFLINE = 7;
}

// ============================================================================
// Handoff Messages
// ============================================================================

message HandoffRequest {
  string from_session = 1;      // Current controller's session token
  string to_user_id = 2;        // Target user identifier
  Role offered_role = 3;        // Role being offered
  int32 offer_timeout_seconds = 4;  // How long offer is valid
  string message = 5;           // Optional message to recipient
}

message HandoffResponse {
  bool accepted = 1;
  string new_session_token = 2;  // New controller's session (if accepted)
  string error_message = 3;
}

// ============================================================================
// Event Messages
// ============================================================================

message Event {
  string event_id = 1;
  EventType type = 2;
  string source_ruri = 3;
  int64 timestamp_ms = 4;
  bytes payload = 5;
  
  enum EventType {
    EVENT_TYPE_UNSPECIFIED = 0;
    STATUS_CHANGE = 1;
    SAFETY_ALERT = 2;
    CAPABILITY_ADDED = 3;
    CAPABILITY_REMOVED = 4;
    CONTROLLER_CHANGE = 5;
    ERROR = 6;
    CUSTOM = 99;
  }
}

// ============================================================================
// Fleet Coordination
// ============================================================================

message FleetCommand {
  string fleet_id = 1;
  repeated string target_robots = 2;
  ChoreographyType choreography = 3;
  repeated FleetStep steps = 4;
  
  enum ChoreographyType {
    CHOREOGRAPHY_UNSPECIFIED = 0;
    BROADCAST = 1;              // Same command to all
    SEQUENTIAL = 2;             // One after another
    PARALLEL = 3;               // All at once
    COORDINATED = 4;            // With dependencies
  }
}

message FleetStep {
  string robot_id = 1;
  string action = 2;
  bytes parameters = 3;
  string wait_for = 4;          // Robot ID to wait for (if any)
  int32 delay_ms = 5;           // Delay before executing
}

message SwarmAnnounce {
  string swarm_id = 1;
  repeated string members = 2;
  string leader = 3;
  string formation = 4;
  JoinPolicy join_policy = 5;
  
  enum JoinPolicy {
    JOIN_POLICY_UNSPECIFIED = 0;
    OPEN = 1;
    APPROVAL = 2;
    CLOSED = 3;
  }
}
```

---

## 3. mDNS Service Discovery

### 3.1 Service Type

```
_rcan._tcp.local.
```

### 3.2 TXT Record Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://rcan.dev/schemas/mdns-txt.json",
  "title": "RCAN mDNS TXT Record",
  "type": "object",
  "required": ["ruri", "model", "version"],
  "properties": {
    "ruri": {
      "type": "string",
      "description": "Full Robot URI",
      "pattern": "^rcan://.*"
    },
    "model": {
      "type": "string",
      "description": "Model identifier"
    },
    "caps": {
      "type": "string",
      "description": "Comma-separated capability list",
      "examples": ["arm,vision,chat,teleop"]
    },
    "roles": {
      "type": "string",
      "description": "Comma-separated accepted roles",
      "examples": ["owner,user,guest"]
    },
    "version": {
      "type": "string",
      "description": "RCAN protocol version",
      "pattern": "^\\d+\\.\\d+\\.\\d+$"
    },
    "name": {
      "type": "string",
      "description": "Human-friendly robot name",
      "maxLength": 63
    },
    "status": {
      "type": "string",
      "enum": ["idle", "active", "busy", "error", "maintenance"]
    }
  }
}
```

### 3.3 Example Announcement

```
Service Name: companion-d3a4b5c6._rcan._tcp.local.
Port: 8080
TXT Records:
  ruri=rcan://continuon.cloud/continuon/companion-v1/d3a4b5c6
  model=companion-v1
  caps=arm,vision,chat,teleop
  roles=owner,user,guest
  version=1.0.0
  name=Living Room Companion
  status=idle
```

---

## 4. JWT Token Schema

### 4.1 Payload Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://rcan.dev/schemas/jwt-payload.json",
  "title": "RCAN JWT Payload",
  "type": "object",
  "required": ["sub", "iss", "aud", "role", "exp", "iat"],
  "properties": {
    "sub": {
      "type": "string",
      "description": "Subject - User UUID",
      "format": "uuid"
    },
    "iss": {
      "type": "string",
      "description": "Issuer - Registry domain"
    },
    "aud": {
      "type": "string",
      "description": "Audience - Target RURI pattern (supports * wildcard)"
    },
    "role": {
      "type": "string",
      "enum": ["guest", "user", "leasee", "owner", "creator"]
    },
    "scope": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Granted capability scopes"
    },
    "fleet": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Fleet robot IDs"
    },
    "exp": {
      "type": "integer",
      "description": "Expiration timestamp"
    },
    "iat": {
      "type": "integer",
      "description": "Issued at timestamp"
    },
    "jti": {
      "type": "string",
      "description": "Unique token ID"
    }
  }
}
```

### 4.2 Example Token

```json
{
  "sub": "550e8400-e29b-41d4-a716-446655440000",
  "iss": "continuon.cloud",
  "aud": "rcan://continuon.cloud/continuon/companion-v1/*",
  "role": "owner",
  "scope": ["control", "config", "training"],
  "fleet": ["d3a4b5c6", "a1b2c3d4"],
  "exp": 1735689600,
  "iat": 1735603200,
  "jti": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
}
```

---

## 5. Handshake Flow

### 5.1 Online Mode

```
┌─────────────┐                    ┌─────────────┐                    ┌─────────────┐
│  App/Client │                    │    Robot    │                    │   Registry  │
└──────┬──────┘                    └──────┬──────┘                    └──────┬──────┘
       │                                  │                                  │
       │  1. UDP mDNS Query               │                                  │
       │  _rcan._tcp.local?               │                                  │
       │─────────────────────────────────►│                                  │
       │                                  │                                  │
       │  2. mDNS Response (TXT records)  │                                  │
       │◄─────────────────────────────────│                                  │
       │                                  │                                  │
       │  3. WebSocket Connect            │                                  │
       │═════════════════════════════════►│                                  │
       │                                  │                                  │
       │  4. AuthClaimRequest             │                                  │
       │─────────────────────────────────►│                                  │
       │                                  │                                  │
       │                                  │  5. Validate Credential          │
       │                                  │─────────────────────────────────►│
       │                                  │                                  │
       │                                  │  6. Issue JWT                    │
       │                                  │◄─────────────────────────────────│
       │                                  │                                  │
       │  7. AuthClaimResponse (JWT)      │                                  │
       │◄─────────────────────────────────│                                  │
       │                                  │                                  │
       │══════════════════════════════════│                                  │
       │    Authenticated Session         │                                  │
       │══════════════════════════════════│                                  │
```

### 5.2 Offline Mode

When the registry is unreachable, robots fall back to cached credentials:

```
┌─────────────┐                    ┌─────────────┐
│  App/Client │                    │    Robot    │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       │  1. AuthClaimRequest             │
       │─────────────────────────────────►│
       │                                  │
       │      [Check local cache]         │
       │      [Verify cached JWT]         │
       │      [Still valid? Grant]        │
       │                                  │
       │  2. AuthClaimResponse            │
       │     (mode: OFFLINE_AUTONOMOUS)   │
       │◄─────────────────────────────────│
       │                                  │
       │══════════════════════════════════│
       │  Offline Session (logged)        │
       │══════════════════════════════════│
```

---

## 6. Error Codes

| Code | Name | Description |
|------|------|-------------|
| 1001 | `INVALID_RURI` | Malformed RURI string |
| 1002 | `UNKNOWN_ROBOT` | Robot not found in registry |
| 1003 | `ROBOT_OFFLINE` | Robot unreachable |
| 2001 | `INVALID_CREDENTIALS` | Auth credentials rejected |
| 2002 | `INSUFFICIENT_PRIVILEGES` | Role too low for operation |
| 2003 | `TOKEN_EXPIRED` | JWT has expired |
| 2004 | `TOKEN_REVOKED` | JWT was revoked |
| 2005 | `RATE_LIMITED` | Too many requests |
| 3001 | `ROBOT_BUSY` | Robot controlled by another user |
| 3002 | `COMMAND_TIMEOUT` | Command execution timed out |
| 3003 | `COMMAND_REJECTED` | Command failed safety check |
| 3004 | `CAPABILITY_UNAVAILABLE` | Requested capability not present |
| 4001 | `SAFETY_VIOLATION` | Action would violate safety constraints |
| 4002 | `EMERGENCY_STOP` | Robot in emergency stop state |

---

## 7. Conformance Requirements

### 7.1 MUST Requirements

A conforming RCAN implementation MUST:

1. Parse and validate RURIs according to the schema in Section 1
2. Support mDNS discovery via `_rcan._tcp.local`
3. Validate JWT tokens before processing authenticated commands
4. Enforce role hierarchy (higher roles inherit lower role permissions)
5. Log all commands with timestamp, user ID, and outcome
6. Respond to SAFETY priority messages within 100ms
7. Enter safe-stop mode on network partition
8. Support offline operation with cached credentials

### 7.2 SHOULD Requirements

A conforming implementation SHOULD:

1. Support WebSocket transport for real-time control
2. Support gRPC transport for structured commands
3. Implement command queueing for offline resilience
4. Provide diagnostic endpoints for debugging
5. Support JWT refresh without session interruption

### 7.3 MAY Requirements

A conforming implementation MAY:

1. Support MQTT for IoT integration
2. Implement custom authentication providers
3. Extend the role hierarchy with custom roles
4. Add manufacturer-specific capabilities

---

## References

- [RCAN Protocol Overview](./rcan-protocol.md)
- [RFC 6762 - Multicast DNS](https://tools.ietf.org/html/rfc6762)
- [RFC 6763 - DNS-Based Service Discovery](https://tools.ietf.org/html/rfc6763)
- [RFC 7519 - JSON Web Token](https://tools.ietf.org/html/rfc7519)
- [Protocol Buffers Language Guide](https://protobuf.dev/programming-guides/proto3/)
- [JSON Schema Specification](https://json-schema.org/specification)

