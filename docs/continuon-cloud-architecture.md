# continuon.cloud Architecture

**Domain**: `continuon.cloud`
**Purpose**: Robot registry, RLDS episode ingestion, training coordination, and remote connection brokering.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           continuon.cloud                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   Firebase      │  │  Cloud Storage  │  │     Cloud Functions         │ │
│  │   Hosting       │  │                 │  │                             │ │
│  │                 │  │  gs://continuon │  │  • /api/rlds/ingest        │ │
│  │  • Landing page │  │  ├── episodes/  │  │  • /api/rlds/signed-url    │ │
│  │  • Flutter app  │  │  ├── models/    │  │  • /api/remote/signal      │ │
│  │  • API proxy    │  │  └── bundles/   │  │  • /api/remote/verify      │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘ │
│           │                    │                          │                 │
│  ┌────────┴────────────────────┴──────────────────────────┴──────────────┐ │
│  │                         Firestore Database                             │ │
│  │                                                                        │ │
│  │  /rcan_registry/{device_id}     - Robot registration & discovery      │ │
│  │  /rlds_episodes/{episode_id}    - Episode metadata & training status  │ │
│  │  /training_jobs/{job_id}        - Training job tracking               │ │
│  │  /ota_bundles/{bundle_id}       - OTA bundle manifests                │ │
│  │  /remote_sessions/{session_id}  - Active remote connection sessions   │ │
│  │  /users/{uid}                   - User profiles & robot ownership     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────────┐ ┌───────────┐ ┌───────────────┐
            │   Robots      │ │  Colab    │ │  Flutter App  │
            │   (Edge)      │ │  Training │ │  (Companion)  │
            │               │ │           │ │               │
            │ • Upload RLDS │ │ • Seed    │ │ • Discover    │
            │ • Heartbeat   │ │   model   │ │ • Connect     │
            │ • OTA receive │ │ • Slow    │ │ • Control     │
            │ • Remote conn │ │   loops   │ │ • Record      │
            └───────────────┘ └───────────┘ └───────────────┘
```

## Component Details

### 1. Firebase Hosting (continuon.cloud)

Primary web presence and API gateway.

**DNS Setup:**
```
continuon.cloud         A      Firebase Hosting IP
www.continuon.cloud     CNAME  continuon.cloud
api.continuon.cloud     CNAME  continuon.cloud (or Cloud Run)
```

**Hosted Content:**
- Landing page (`/`) - Product info, documentation
- Flutter companion app (`/app`) - Already deployed at continuonai.web.app, now at continuon.cloud/app
- API documentation (`/docs`)

### 2. Cloud Storage (gs://continuon-rlds)

Centralized storage for all robot learning data.

```
gs://continuon-rlds/
├── episodes/
│   ├── raw/                    # Incoming from robots
│   │   └── {robot_id}/
│   │       └── {episode_id}/
│   │           ├── metadata.json
│   │           ├── steps.jsonl
│   │           └── blobs/
│   ├── validated/              # After ingestion validation
│   └── tfrecord/               # Converted for training
├── models/
│   ├── seed/                   # Initial seed models
│   │   └── v{version}/
│   ├── checkpoints/            # Training checkpoints
│   └── production/             # Ready for OTA
│       └── {robot_model}/
│           └── v{version}/
└── bundles/
    └── ota/                    # Signed OTA bundles
        └── {robot_model}/
            └── {version}.bundle
```

### 3. Cloud Functions / Cloud Run APIs

#### `/api/rlds/signed-url` (POST)
Generate signed upload URLs for robots.

```json
// Request
{
  "robot_id": "14d4b680",
  "episode_id": "ep_1704384000000",
  "metadata": { "duration_ms": 30000, "step_count": 300 }
}

// Response
{
  "upload_url": "https://storage.googleapis.com/continuon-rlds/episodes/raw/...",
  "expires_at": "2026-01-04T10:00:00Z",
  "episode_ref": "episodes/raw/14d4b680/ep_1704384000000"
}
```

#### `/api/rlds/ingest` (POST)
Validate and promote uploaded episodes.

```json
// Request
{
  "episode_ref": "episodes/raw/14d4b680/ep_1704384000000",
  "checksum": "sha256:abc123..."
}

// Response
{
  "status": "validated",
  "episode_id": "ep_1704384000000",
  "training_eligible": true
}
```

#### `/api/remote/verify` (POST)
Verify robot ownership for remote connection.

```json
// Request
{
  "robot_ruri": "rcan://continuon.cloud/continuon/companion-v1/14d4b680",
  "user_uid": "firebase-uid",
  "local_claim_proof": "signed-token-from-local-claim"
}

// Response
{
  "verified": true,
  "session_token": "remote-session-xyz",
  "robot_endpoint": "wss://relay.continuon.cloud/14d4b680"
}
```

#### `/api/remote/signal` (WebSocket)
WebRTC signaling for remote connections.

### 4. Firestore Collections

#### `/rlds_episodes/{episode_id}`
```javascript
{
  robot_id: "14d4b680",
  uploaded_at: Timestamp,
  status: "raw" | "validated" | "training" | "trained",
  metadata: {
    duration_ms: 30000,
    step_count: 300,
    capabilities: {...},
    tags: ["teleop", "pick-place"]
  },
  storage_ref: "gs://continuon-rlds/episodes/validated/...",
  training_job_id: null | "job_xyz",
  contributed_to_models: ["seed_v2", "slow_loop_v1.3"]
}
```

#### `/training_jobs/{job_id}`
```javascript
{
  type: "seed" | "slow_loop" | "finetune",
  status: "queued" | "running" | "completed" | "failed",
  created_at: Timestamp,
  started_at: Timestamp | null,
  completed_at: Timestamp | null,
  config: {
    episode_ids: ["ep_1", "ep_2"],
    model_base: "seed_v1",
    hyperparams: {...}
  },
  output: {
    model_ref: "gs://continuon-rlds/models/checkpoints/...",
    metrics: { loss: 0.01, accuracy: 0.95 }
  },
  colab_notebook_url: "https://colab.research.google.com/..."
}
```

#### `/ota_bundles/{bundle_id}`
```javascript
{
  version: "1.0.3",
  robot_model: "companion-v1",
  created_at: Timestamp,
  training_job_id: "job_xyz",
  storage_ref: "gs://continuon-rlds/bundles/ota/companion-v1/1.0.3.bundle",
  manifest: {
    checksum: "sha256:...",
    size_bytes: 12345678,
    requires_version: ">=1.0.0"
  },
  rollout: {
    status: "staged" | "canary" | "production",
    target_percentage: 10,
    deployed_count: 5,
    error_count: 0
  }
}
```

#### `/remote_sessions/{session_id}`
```javascript
{
  robot_ruri: "rcan://continuon.cloud/continuon/companion-v1/14d4b680",
  user_uid: "firebase-uid",
  created_at: Timestamp,
  expires_at: Timestamp,
  status: "pending" | "connected" | "disconnected",
  relay_endpoint: "wss://relay.continuon.cloud/14d4b680",
  local_claim_verified: true,
  connection_type: "webrtc" | "tunnel"
}
```

## Google Colab Integration

### Notebooks Structure

```
continuon-cloud-notebooks/
├── 01_seed_model_training.ipynb     # Initial seed model training
├── 02_slow_loop_training.ipynb      # Continuous improvement loop
├── 03_episode_analysis.ipynb        # RLDS episode visualization
├── 04_model_export.ipynb            # Export to OTA bundle
└── utils/
    ├── gcs_loader.py                # Load episodes from GCS
    ├── rlds_parser.py               # Parse RLDS format
    └── bundle_creator.py            # Create OTA bundles
```

### Colab Workflow

```python
# 01_seed_model_training.ipynb

# 1. Authenticate with GCS
from google.colab import auth
auth.authenticate_user()

# 2. Load episodes from continuon.cloud
from continuon_cloud import load_episodes
episodes = load_episodes(
    bucket="gs://continuon-rlds",
    robot_model="companion-v1",
    min_quality=0.8
)

# 3. Train seed model
from continuonbrain.jax_models.train import train_core_model
model, metrics = train_core_model(
    episodes=episodes,
    config={
        "batch_size": 256,
        "learning_rate": 1e-4,
        "num_steps": 10000
    }
)

# 4. Save checkpoint
model.save("gs://continuon-rlds/models/seed/v2/")

# 5. Register with continuon.cloud
from continuon_cloud import register_model
register_model(
    model_ref="gs://continuon-rlds/models/seed/v2/",
    robot_model="companion-v1",
    metrics=metrics
)
```

## Remote Connection Flow

```
┌──────────┐     ┌────────────────┐     ┌──────────────┐     ┌─────────┐
│  Flutter │     │ continuon.cloud│     │    Relay     │     │  Robot  │
│   App    │     │   (Firebase)   │     │   Server     │     │ (Edge)  │
└────┬─────┘     └───────┬────────┘     └──────┬───────┘     └────┬────┘
     │                   │                     │                   │
     │ 1. Lookup RURI    │                     │                   │
     │──────────────────>│                     │                   │
     │                   │                     │                   │
     │ 2. Get robot info │                     │                   │
     │<──────────────────│                     │                   │
     │                   │                     │                   │
     │ 3. Verify ownership (local claim proof) │                   │
     │──────────────────>│                     │                   │
     │                   │                     │                   │
     │ 4. Session token  │                     │                   │
     │<──────────────────│                     │                   │
     │                   │                     │                   │
     │ 5. Connect to relay with token          │                   │
     │─────────────────────────────────────────>                   │
     │                   │                     │                   │
     │                   │ 6. Robot connects   │                   │
     │                   │ (heartbeat/tunnel)  │<──────────────────│
     │                   │                     │                   │
     │ 7. WebRTC signaling through relay       │                   │
     │<═══════════════════════════════════════>│<═════════════════>│
     │                   │                     │                   │
     │ 8. Direct P2P connection (if NAT allows)                    │
     │<════════════════════════════════════════════════════════════>
```

## Security Model

### Robot Registration
1. Robot MUST be claimed locally first (same LAN)
2. Local claim creates signed proof stored on robot
3. Remote connections require proof verification

### Episode Upload
1. Robot signs upload request with device key
2. Cloud validates signature before accepting
3. Episodes tagged with verified robot_id

### Remote Access
1. User must own robot in Firestore
2. Local claim proof must be valid
3. Session tokens expire after 1 hour
4. All connections logged for audit

## Implementation Phases

### Phase 1: Domain & Registry (Now)
- [x] Firebase Firestore RCAN registry
- [ ] Point continuon.cloud to Firebase Hosting
- [ ] Deploy Flutter app to continuon.cloud/app

### Phase 2: RLDS Ingestion (Next)
- [ ] Create GCS bucket `gs://continuon-rlds`
- [ ] Cloud Function for signed URL generation
- [ ] Episode validation and promotion
- [ ] Robot-side upload integration

### Phase 3: Training Pipeline
- [ ] Colab notebook templates
- [ ] Training job tracking in Firestore
- [ ] Model checkpoint management
- [ ] OTA bundle creation

### Phase 4: Remote Connections
- [ ] WebRTC signaling server
- [ ] Ownership verification flow
- [ ] Relay/tunnel infrastructure
- [ ] Flutter app remote mode

## Cost Estimates (Monthly)

| Service | Usage | Est. Cost |
|---------|-------|-----------|
| Firebase Hosting | 10GB transfer | $0 (free tier) |
| Firestore | 1M reads/writes | $0-5 |
| Cloud Storage | 100GB episodes | $2-3 |
| Cloud Functions | 1M invocations | $0 (free tier) |
| Cloud Run (relay) | 1 instance | $5-10 |
| **Total** | | **$7-18/month** |

Note: TPU training costs are separate (pay-per-use, ~$2-4/hour for v5e-1).
