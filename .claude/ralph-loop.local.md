---
active: true
iteration: 1
max_iterations: 0
completion_promise: "DONE"
started_at: "2026-01-05T21:59:33Z"
---

# Investigation: ContinuonAI-backend & ContinuonAI-web Folder Analysis

## Original Question
Why do `continuonai-backend` and `continuonai-web` folders exist when the logic could have been in the Flutter app?

## Findings

### What Each Component Does

#### 1. `continuonai` (Flutter App) - **MOBILE/DESKTOP COMPANION**
- **Purpose**: Mobile/desktop companion app for robot teleoperation and RLDS data capture
- **Communicates with**: ContinuonBrain (the edge runtime on robots) via gRPC/HTTP
- **Technology**: Flutter (Dart)
- **Key responsibilities**:
  - Robot discovery (mDNS, QR pairing)
  - Teleoperation controls
  - RLDS episode recording and upload
  - Firebase authentication
  - Local connection to robots via RCAN protocol

#### 2. `continuonai-backend` (FastAPI Backend) - **CLOUD FLEET MANAGEMENT**
- **Purpose**: Cloud-hosted API for fleet management, model distribution, and training orchestration
- **Technology**: FastAPI (Python) + Firebase Auth + Firestore + GCS
- **Added**: January 5, 2026 (commit 8a3d9ee)
- **Key responsibilities**:
  - Fleet-wide robot registration and management
  - Model registry (upload, version, distribute models across robots)
  - Cloud training job orchestration
  - Episode/dataset management and analytics
  - Multi-user authentication and rate limiting
  - Designed to run on Cloud Run

#### 3. `continuonai-web` (Next.js Web Dashboard) - **WEB ADMIN INTERFACE**
- **Purpose**: Web-based admin dashboard for fleet management
- **Technology**: Next.js 14 + React + TypeScript + Tailwind
- **Key responsibilities**:
  - Web UI for managing robots, models, training jobs
  - Analytics dashboards
  - Settings management
  - Calls `continuonai-backend` API (`https://api.continuonai.com`)

#### 4. `continuonbrain` (Edge Runtime) - **ON-ROBOT AI BRAIN**
- **Purpose**: The actual AI brain running ON each robot
- **Technology**: Python (extensive - 170KB server.py!)
- **Key responsibilities**:
  - Robot control, inference, learning
  - Local HTTP API for the Flutter app to connect to
  - OTA updates, training, chat with Gemma
  - All the actual robotics/AI logic

### Why They're Separate

**They serve fundamentally different architectural purposes:**

| Component | Runs On | Purpose | Scope |
|-----------|---------|---------|-------|
| `continuonai` | User's phone/desktop | Direct robot control | 1 robot at a time |
| `continuonbrain` | Each robot | Edge AI runtime | 1 robot |
| `continuonai-backend` | Cloud (GCP) | Fleet orchestration | All robots |
| `continuonai-web` | Browser (hosted) | Admin dashboard | All robots |

### Why This Architecture Makes Sense

1. **Flutter Cannot Replace the Cloud Backend**
   - The Flutter app connects to individual robots over LAN
   - For fleet management (multiple robots, multiple users), you need a centralized cloud service
   - Model distribution, training job orchestration, and analytics need server-side infrastructure

2. **Flutter Web Has Limitations**
   - Flutter web could theoretically render a dashboard, but:
     - Next.js provides better SEO, SSR, and web performance
     - The backend still needs to exist as a separate service
     - Enterprise/admin dashboards often prefer React/Next.js ecosystem

3. **Separation of Concerns**
   - Edge runtime (`continuonbrain`) - real-time robot control
   - Companion app (`continuonai`) - mobile/desktop control interface
   - Cloud backend (`continuonai-backend`) - fleet data persistence and orchestration
   - Web dashboard (`continuonai-web`) - admin interface

### What COULD Be Consolidated

1. **The Flutter app already has Firebase and could potentially render fleet management UI** - but this would require the cloud backend to exist anyway
2. **The web dashboard could be built in Flutter Web** - but the team chose Next.js (common for admin dashboards)

## Recommendation

The architecture appears **intentional and sound**:

- `continuonai-backend` and `continuonai-web` were added together in commit 8a3d9ee for OTA updates and cloud training
- They represent the **cloud tier** of a multi-tier architecture (Edge → Companion → Cloud)
- The Flutter app is the **companion tier** for direct robot interaction

**No consolidation is strictly necessary.** The separation provides:
- Clear boundaries between local (robot) and cloud (fleet) operations
- Technology choices appropriate for each layer (Flutter for cross-platform mobile, FastAPI for cloud APIs, Next.js for web dashboards)

---

## Status: ANALYSIS COMPLETE

The folders exist because they serve different tiers of the system architecture:
- **Local tier**: `continuonbrain` (on robot) + `continuonai` (on user device)
- **Cloud tier**: `continuonai-backend` (API) + `continuonai-web` (dashboard)

This is a standard multi-tier architecture for IoT/robotics platforms.
