# Plan 3: Cloud Training Trigger

## Overview
Enable automated cloud training when local device lacks resources or needs larger model training.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD TRAINING FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Robot (Pi5)                    Cloud (GCP/Colab)               │
│  ┌──────────────┐               ┌──────────────────┐            │
│  │ Episode      │               │ Cloud Trainer    │            │
│  │ Recorder     │               │ Service          │            │
│  └──────┬───────┘               └────────┬─────────┘            │
│         │                                │                       │
│         ▼                                │                       │
│  ┌──────────────┐    Upload      ┌───────▼────────┐            │
│  │ RLDS         │───────────────►│ GCS Bucket     │            │
│  │ Episodes     │                │ /episodes/     │            │
│  └──────────────┘                └───────┬────────┘            │
│                                          │                       │
│         │                                ▼                       │
│         │                       ┌──────────────────┐            │
│         │    Trigger            │ Cloud Function   │            │
│         │◄──────────────────────│ or Vertex AI     │            │
│         │                       └────────┬─────────┘            │
│                                          │                       │
│  ┌──────────────┐    Download    ┌───────▼────────┐            │
│  │ Model        │◄───────────────│ Trained Model  │            │
│  │ Registry     │                │ /models/       │            │
│  └──────────────┘                └────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### New Files

#### 1. `services/cloud_training.py`
```python
class CloudTrainingService:
    """Manages cloud training jobs."""

    def __init__(self, config: CloudTrainingConfig):
        self.config = config
        self.registry = ModelRegistry(config.bucket)
        self.firestore = FirestoreClient()

    async def upload_episodes(self, local_dir: Path) -> str:
        """Upload RLDS episodes to cloud storage."""

    async def trigger_training(self, config: TrainingConfig) -> str:
        """Trigger cloud training job, return job_id."""

    async def get_job_status(self, job_id: str) -> JobStatus:
        """Check training job status."""

    async def download_result(self, job_id: str, dest: Path) -> Path:
        """Download trained model when complete."""
```

#### 2. `cloud/training_function/main.py` (Cloud Function)
```python
def train_model(event, context):
    """Cloud Function triggered by GCS upload or Firestore."""

    # 1. Download episodes from GCS
    episodes_path = download_episodes(event['bucket'], event['name'])

    # 2. Load training config
    config = load_config(event.get('config', {}))

    # 3. Run training (JAX on TPU/GPU)
    trainer = WaveCoreTrainer(episodes_path, config)
    result = trainer.train()

    # 4. Upload model to registry
    upload_model(result.checkpoint_path, result.model_id)

    # 5. Notify completion via Firestore
    update_job_status(event['job_id'], 'completed', result)
```

#### 3. `cloud/vertex_training.py` (Vertex AI alternative)
```python
def submit_vertex_job(config: dict) -> str:
    """Submit training job to Vertex AI."""
    from google.cloud import aiplatform

    job = aiplatform.CustomJob(
        display_name=f"continuon-train-{config['job_id']}",
        worker_pool_specs=[{
            'machine_spec': {'machine_type': 'n1-standard-8', 'accelerator_type': 'NVIDIA_TESLA_T4'},
            'replica_count': 1,
            'container_spec': {
                'image_uri': 'gcr.io/continuon/trainer:latest',
                'args': ['--config', json.dumps(config)]
            }
        }]
    )
    job.run(sync=False)
    return job.resource_name
```

### Firestore Schema
```
/training_jobs/{job_id}
{
  "robot_id": "robot_001",
  "status": "running",  // pending, running, completed, failed
  "created_at": timestamp,
  "updated_at": timestamp,
  "config": {
    "model_type": "wavecore",
    "epochs": 100,
    "batch_size": 32
  },
  "episodes_uri": "gs://continuon-data/episodes/robot_001_20260105.zip",
  "result": {
    "model_uri": "gs://continuon-models/adapters/robot_001_v4/",
    "final_loss": 0.0234,
    "training_time_s": 3600
  }
}
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/cloud/upload` | POST | Upload episodes for cloud training |
| `/api/training/cloud/trigger` | POST | Start cloud training job |
| `/api/training/cloud/status/{job_id}` | GET | Get job status |
| `/api/training/cloud/download/{job_id}` | POST | Download trained model |

## Files to Create/Modify
| File | Action |
|------|--------|
| `services/cloud_training.py` | CREATE - Cloud training client |
| `cloud/training_function/main.py` | CREATE - Cloud Function |
| `cloud/training_function/requirements.txt` | CREATE |
| `cloud/Dockerfile.trainer` | CREATE - Training container |
| `api/controllers/training_controller.py` | MODIFY - Add cloud endpoints |
| `config/cloud_training.yaml` | CREATE - Cloud config |

## Dependencies
- `google-cloud-storage`
- `google-cloud-firestore`
- `google-cloud-aiplatform` (optional for Vertex AI)

## Success Criteria
- Episodes upload to GCS
- Training job triggers automatically or manually
- Status visible from robot
- Trained model downloads and installs
