# Plan 5: ContinuonAI.com Backend

## Overview
Create the backend API service for continuonai.com - fleet management, model distribution, and training orchestration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTINUONAI.COM BACKEND                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                    │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  /api/v1/                                                │   │
│  │  ├── /auth           Authentication (Firebase/Auth0)     │   │
│  │  ├── /robots         Fleet management                    │   │
│  │  ├── /models         Model registry                      │   │
│  │  ├── /training       Training jobs                       │   │
│  │  ├── /episodes       Episode management                  │   │
│  │  └── /analytics      Usage analytics                     │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  Firestore   │   │     GCS      │   │  Vertex AI   │        │
│  │  (metadata)  │   │  (storage)   │   │  (training)  │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Project Structure
```
continuonai-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── firebase.py      # Firebase Auth
│   │   └── middleware.py    # Auth middleware
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── robots.py        # Fleet management
│   │   ├── models.py        # Model registry
│   │   ├── training.py      # Training jobs
│   │   ├── episodes.py      # Episode upload/download
│   │   └── analytics.py     # Usage stats
│   ├── services/
│   │   ├── __init__.py
│   │   ├── firestore.py     # Firestore client
│   │   ├── storage.py       # GCS client
│   │   ├── training.py      # Training orchestration
│   │   └── notifications.py # Push notifications
│   ├── models/
│   │   ├── __init__.py
│   │   ├── robot.py         # Robot schemas
│   │   ├── model.py         # Model schemas
│   │   └── training.py      # Training schemas
│   └── utils/
│       ├── __init__.py
│       └── security.py
├── tests/
├── Dockerfile
├── requirements.txt
└── cloudbuild.yaml
```

### Key Files

#### 1. `app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import robots, models, training, episodes, analytics
from app.auth.middleware import AuthMiddleware

app = FastAPI(
    title="ContinuonAI API",
    version="1.0.0",
    description="Fleet management and model distribution for Continuon robots"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://continuonai.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)

app.include_router(robots.router, prefix="/api/v1/robots", tags=["robots"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(episodes.router, prefix="/api/v1/episodes", tags=["episodes"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

@app.get("/health")
def health():
    return {"status": "healthy"}
```

#### 2. `app/routers/robots.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from app.auth.firebase import get_current_user
from app.services.firestore import FirestoreService
from app.models.robot import Robot, RobotCreate, RobotUpdate

router = APIRouter()

@router.get("/", response_model=List[Robot])
async def list_robots(
    user = Depends(get_current_user),
    db: FirestoreService = Depends()
):
    """List all robots owned by user."""
    return await db.get_robots(user.uid)

@router.get("/{robot_id}", response_model=Robot)
async def get_robot(
    robot_id: str,
    user = Depends(get_current_user),
    db: FirestoreService = Depends()
):
    """Get robot details."""
    robot = await db.get_robot(robot_id)
    if not robot or robot.owner_id != user.uid:
        raise HTTPException(404, "Robot not found")
    return robot

@router.post("/{robot_id}/command")
async def send_command(
    robot_id: str,
    command: dict,
    user = Depends(get_current_user),
    db: FirestoreService = Depends()
):
    """Send command to robot via Firestore."""
    await db.add_command(robot_id, command)
    return {"status": "queued"}

@router.get("/{robot_id}/status")
async def get_status(robot_id: str, user = Depends(get_current_user)):
    """Get robot's current status."""
    # Real-time from Firestore
    pass
```

#### 3. `app/routers/models.py`
```python
from fastapi import APIRouter, Depends, UploadFile, File
from typing import List

from app.auth.firebase import get_current_user
from app.services.storage import StorageService
from app.models.model import ModelInfo, ModelVersion

router = APIRouter()

@router.get("/", response_model=List[ModelInfo])
async def list_models(storage: StorageService = Depends()):
    """List all available models."""
    return await storage.list_models()

@router.get("/{model_id}/versions", response_model=List[ModelVersion])
async def list_versions(model_id: str, storage: StorageService = Depends()):
    """List versions of a model."""
    return await storage.list_versions(model_id)

@router.post("/{model_id}/upload")
async def upload_model(
    model_id: str,
    version: str,
    file: UploadFile = File(...),
    user = Depends(get_current_user),
    storage: StorageService = Depends()
):
    """Upload a new model version."""
    url = await storage.upload_model(model_id, version, file)
    return {"url": url, "version": version}

@router.get("/{model_id}/{version}/download")
async def get_download_url(
    model_id: str,
    version: str,
    storage: StorageService = Depends()
):
    """Get signed download URL."""
    url = await storage.get_signed_url(model_id, version)
    return {"download_url": url, "expires_in": 3600}
```

#### 4. `app/routers/training.py`
```python
from fastapi import APIRouter, Depends, BackgroundTasks
from typing import List

from app.auth.firebase import get_current_user
from app.services.training import TrainingService
from app.models.training import TrainingJob, TrainingConfig

router = APIRouter()

@router.post("/jobs", response_model=TrainingJob)
async def create_training_job(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user),
    training: TrainingService = Depends()
):
    """Create a new training job."""
    job = await training.create_job(user.uid, config)
    background_tasks.add_task(training.run_job, job.id)
    return job

@router.get("/jobs", response_model=List[TrainingJob])
async def list_jobs(
    user = Depends(get_current_user),
    training: TrainingService = Depends()
):
    """List user's training jobs."""
    return await training.list_jobs(user.uid)

@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_job(
    job_id: str,
    user = Depends(get_current_user),
    training: TrainingService = Depends()
):
    """Get training job status."""
    return await training.get_job(job_id)

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user = Depends(get_current_user),
    training: TrainingService = Depends()
):
    """Cancel a running job."""
    await training.cancel_job(job_id)
    return {"status": "cancelled"}
```

### Firestore Collections
```
/users/{uid}
  - email, name, created_at, plan

/robots/{robot_id}
  - owner_id, name, device_id, status, last_seen, model_version

/robots/{robot_id}/commands/{cmd_id}
  - type, payload, status, created_at

/robots/{robot_id}/telemetry/{ts}
  - battery, temperature, position, mode

/training_jobs/{job_id}
  - user_id, robot_id, status, config, result, created_at

/models/{model_id}/versions/{version}
  - created_at, checksum, download_count, release_notes
```

### Deployment
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/continuonai-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/continuonai-api']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args: ['run', 'deploy', 'continuonai-api',
           '--image', 'gcr.io/$PROJECT_ID/continuonai-api',
           '--region', 'us-central1',
           '--allow-unauthenticated']
```

## Files to Create
| File | Description |
|------|-------------|
| `continuonai-backend/` | New directory for backend |
| All files in structure above | FastAPI application |

## Success Criteria
- API deployed to Cloud Run
- Authentication working
- Robot CRUD operations
- Model upload/download
- Training job management
