# Plan 4: OTA Model Download System

## Overview
Implement secure Over-The-Air model download and installation with verification and rollback.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OTA UPDATE FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Check for Updates                                           │
│     Robot ──► Registry API ──► Get latest version               │
│                                                                  │
│  2. Download Bundle                                             │
│     Robot ◄── GCS/CDN ◄── Signed bundle.tar.gz                 │
│                                                                  │
│  3. Verify                                                      │
│     ┌─────────────────────────────────────────┐                │
│     │ Checksum verification (SHA256)          │                │
│     │ Signature verification (Ed25519)        │                │
│     │ Compatibility check (brain version)     │                │
│     └─────────────────────────────────────────┘                │
│                                                                  │
│  4. Stage                                                       │
│     /opt/continuonos/brain/model/                              │
│     ├── current/     ← Active model                            │
│     ├── candidate/   ← Downloaded, pending activation          │
│     └── rollback/    ← Previous version for recovery           │
│                                                                  │
│  5. Activate                                                    │
│     candidate/ ──► current/ (atomic swap)                      │
│     old current/ ──► rollback/                                 │
│                                                                  │
│  6. Verify & Rollback                                           │
│     If health check fails ──► rollback/ ──► current/           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### New Files

#### 1. `services/ota_updater.py`
```python
@dataclass
class UpdateInfo:
    model_id: str
    version: str
    download_url: str
    checksum: str
    signature: str
    size_bytes: int
    release_notes: str

class OTAUpdater:
    """Manages OTA model updates with verification and rollback."""

    def __init__(self, config_dir: Path, registry: ModelRegistry):
        self.config_dir = config_dir
        self.registry = registry
        self.model_dir = config_dir / "model"
        self.current_dir = self.model_dir / "current"
        self.candidate_dir = self.model_dir / "candidate"
        self.rollback_dir = self.model_dir / "rollback"

    async def check_for_updates(self) -> Optional[UpdateInfo]:
        """Check if newer version available."""
        current_version = self._get_current_version()
        latest = await self.registry.get_latest_version("seed")
        if self._is_newer(latest, current_version):
            return await self.registry.get_model("seed", latest)
        return None

    async def download_update(self, update: UpdateInfo, progress_cb=None) -> Path:
        """Download update to candidate directory."""
        self.candidate_dir.mkdir(parents=True, exist_ok=True)

        # Download with progress
        bundle_path = self.candidate_dir / "bundle.tar.gz"
        await self._download_with_progress(update.download_url, bundle_path, progress_cb)

        # Verify checksum
        if not self._verify_checksum(bundle_path, update.checksum):
            raise ValueError("Checksum verification failed")

        # Verify signature
        if not self._verify_signature(bundle_path, update.signature):
            raise ValueError("Signature verification failed")

        # Extract
        self._extract_bundle(bundle_path, self.candidate_dir)

        return self.candidate_dir

    async def activate_update(self, run_health_check: bool = True) -> bool:
        """Activate candidate model with optional health check."""
        if not (self.candidate_dir / "manifest.json").exists():
            raise ValueError("No candidate to activate")

        # Backup current to rollback
        if self.current_dir.exists():
            if self.rollback_dir.exists():
                shutil.rmtree(self.rollback_dir)
            shutil.move(self.current_dir, self.rollback_dir)

        # Activate candidate
        shutil.move(self.candidate_dir, self.current_dir)

        # Health check
        if run_health_check:
            if not await self._run_health_check():
                await self.rollback()
                return False

        return True

    async def rollback(self) -> bool:
        """Rollback to previous version."""
        if not self.rollback_dir.exists():
            raise ValueError("No rollback available")

        if self.current_dir.exists():
            shutil.rmtree(self.current_dir)

        shutil.move(self.rollback_dir, self.current_dir)
        return True

    def _verify_checksum(self, path: Path, expected: str) -> bool:
        """Verify SHA256 checksum."""
        import hashlib
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}" == expected

    def _verify_signature(self, path: Path, signature: str) -> bool:
        """Verify Ed25519 signature (if public key configured)."""
        public_key_path = self.config_dir / "keys" / "model_signing.pub"
        if not public_key_path.exists():
            logger.warning("No signing key configured, skipping signature verification")
            return True

        # Verify using nacl or cryptography
        # ...
        return True
```

#### 2. `services/update_scheduler.py`
```python
class UpdateScheduler:
    """Schedules automatic update checks."""

    def __init__(self, updater: OTAUpdater, check_interval_hours: int = 24):
        self.updater = updater
        self.check_interval = check_interval_hours * 3600
        self._running = False
        self._thread = None

    def start(self):
        """Start background update checker."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                update = asyncio.run(self.updater.check_for_updates())
                if update:
                    logger.info(f"Update available: {update.version}")
                    # Notify via Firestore or local event
            except Exception as e:
                logger.error(f"Update check failed: {e}")

            time.sleep(self.check_interval)
```

### Bundle Format
```
bundle.tar.gz
├── manifest.json
├── seed_model.npz
├── adapters/
│   └── lora_weights.npz (optional)
├── config/
│   └── model_config.json
└── CHECKSUMS.sha256
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/updates/check` | GET | Check for available updates |
| `/api/updates/download` | POST | Download update to candidate |
| `/api/updates/activate` | POST | Activate candidate model |
| `/api/updates/rollback` | POST | Rollback to previous version |
| `/api/updates/status` | GET | Get current update status |

## Files to Create/Modify
| File | Action |
|------|--------|
| `services/ota_updater.py` | CREATE - OTA update logic |
| `services/update_scheduler.py` | CREATE - Background checker |
| `api/controllers/update_controller.py` | CREATE - Update endpoints |
| `api/controllers/model_controller.py` | MODIFY - Wire OTA |

## Security Considerations
- All bundles signed with Ed25519
- Checksum verification mandatory
- Rollback always available
- Health check before finalizing

## Success Criteria
- Updates download and verify
- Atomic activation with rollback
- Automatic update checks
- Health check prevents bad updates
