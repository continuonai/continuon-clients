# Plan 2: Model Registry (GCS/S3)

## Overview
Create a centralized model registry for storing, versioning, and distributing trained models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL REGISTRY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cloud Storage (GCS/S3)                                         │
│  └── continuon-models/                                          │
│      ├── manifests/                                             │
│      │   └── registry.json          # All models index          │
│      ├── seed/                                                  │
│      │   └── v1.0.0/                                           │
│      │       ├── manifest.json      # Model metadata            │
│      │       ├── seed_model.npz     # Weights                   │
│      │       └── checksum.sha256                                │
│      ├── adapters/                                              │
│      │   └── robot_001_v1/                                     │
│      │       ├── manifest.json                                  │
│      │       └── lora_adapters.npz                             │
│      └── releases/                                              │
│          └── prod_v2.1.0/                                      │
│              ├── manifest.json                                  │
│              ├── bundle.tar.gz                                  │
│              └── signature.sig                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### New Files

#### 1. `services/model_registry.py`
```python
class ModelRegistry:
    """Cloud model registry client."""

    def __init__(self, bucket: str, provider: str = "gcs"):
        self.bucket = bucket
        self.provider = provider  # "gcs" or "s3"
        self._client = self._init_client()

    def list_models(self, model_type: str = None) -> List[ModelInfo]
    def get_model(self, model_id: str, version: str) -> ModelInfo
    def download_model(self, model_id: str, version: str, dest: Path) -> Path
    def upload_model(self, local_path: Path, model_id: str, version: str) -> str
    def get_latest_version(self, model_id: str) -> str
    def verify_checksum(self, local_path: Path, expected: str) -> bool
```

#### 2. `services/model_registry_local.py`
```python
class LocalModelRegistry:
    """Local filesystem fallback when cloud unavailable."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    # Same interface as ModelRegistry
```

### Manifest Format
```json
{
  "model_id": "seed_v1",
  "version": "1.0.0",
  "model_type": "seed",
  "created_at": "2026-01-05T10:00:00Z",
  "framework": "jax",
  "hardware_targets": ["arm64", "x86_64"],
  "param_count": 172202,
  "checksum": "sha256:abc123...",
  "files": [
    {"name": "seed_model.npz", "size": 1234567, "checksum": "sha256:..."}
  ],
  "training_info": {
    "episodes_used": 150,
    "final_loss": 0.0234,
    "source_robot": "robot_001"
  },
  "compatibility": {
    "min_brain_version": "2.0.0",
    "required_deps": ["jax>=0.4.0"]
  }
}
```

### Registry Index Format
```json
{
  "updated_at": "2026-01-05T10:00:00Z",
  "models": {
    "seed": {
      "latest": "1.0.0",
      "versions": ["1.0.0", "0.9.0"]
    },
    "adapters/robot_001": {
      "latest": "v3",
      "versions": ["v1", "v2", "v3"]
    }
  }
}
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/registry` | GET | List all models |
| `/api/models/registry/{id}` | GET | Get model info |
| `/api/models/registry/{id}/download` | POST | Download model |
| `/api/models/registry/upload` | POST | Upload model |

## Files to Create/Modify
| File | Action |
|------|--------|
| `services/model_registry.py` | CREATE - Cloud registry client |
| `services/model_registry_local.py` | CREATE - Local fallback |
| `api/controllers/model_controller.py` | MODIFY - Add registry endpoints |
| `config/registry.yaml` | CREATE - Registry config |

## Dependencies
- `google-cloud-storage` for GCS
- `boto3` for S3 (optional)

## Success Criteria
- Models can be uploaded to cloud
- Models can be listed and downloaded
- Checksums verified on download
- Works offline with local fallback
