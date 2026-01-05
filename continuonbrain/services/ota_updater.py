"""
OTA Model Updater - Secure Over-The-Air model download and installation.

Implements secure model updates with verification and rollback support.
Uses a candidate/current/rollback directory structure for atomic updates.

Usage:
    from continuonbrain.services.ota_updater import OTAUpdater, UpdateInfo

    updater = OTAUpdater(config_dir=Path("/opt/continuonos/brain"))

    # Check for updates
    update = await updater.check_for_updates()
    if update:
        # Download and verify
        await updater.download_update(update, progress_cb=lambda p: print(f"{p}%"))

        # Activate with health check
        success = await updater.activate_update(run_health_check=True)
        if not success:
            # Automatic rollback on health check failure
            pass
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger("OTAUpdater")


class UpdateState(str, Enum):
    """Current state of the OTA update system."""
    IDLE = "idle"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    EXTRACTING = "extracting"
    ACTIVATING = "activating"
    ROLLING_BACK = "rolling_back"
    ERROR = "error"


@dataclass
class UpdateInfo:
    """Information about an available update."""
    model_id: str
    version: str
    download_url: str
    checksum: str  # Format: "sha256:<hex>"
    signature: str  # Ed25519 signature (base64)
    size_bytes: int
    release_notes: str = ""
    min_brain_version: str = "0.0.0"
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateInfo":
        """Create UpdateInfo from a dictionary."""
        return cls(
            model_id=data.get("model_id", "seed"),
            version=data.get("version", "0.0.0"),
            download_url=data.get("download_url", ""),
            checksum=data.get("checksum", ""),
            signature=data.get("signature", ""),
            size_bytes=data.get("size_bytes", 0),
            release_notes=data.get("release_notes", ""),
            min_brain_version=data.get("min_brain_version", "0.0.0"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "download_url": self.download_url,
            "checksum": self.checksum,
            "signature": self.signature,
            "size_bytes": self.size_bytes,
            "release_notes": self.release_notes,
            "min_brain_version": self.min_brain_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }


@dataclass
class UpdateStatus:
    """Current status of the OTA update system."""
    state: UpdateState
    current_version: str
    available_update: Optional[UpdateInfo] = None
    progress_percent: float = 0.0
    error_message: Optional[str] = None
    last_check: Optional[datetime] = None
    last_update: Optional[datetime] = None
    rollback_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "current_version": self.current_version,
            "available_update": self.available_update.to_dict() if self.available_update else None,
            "progress_percent": self.progress_percent,
            "error_message": self.error_message,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "rollback_available": self.rollback_available,
        }


class ModelRegistry:
    """
    Simple model registry client for checking available updates.
    Can be backed by a local file, HTTP endpoint, or Firebase.
    """

    def __init__(self, registry_url: str = None, config_dir: Path = None):
        self.registry_url = registry_url or os.environ.get(
            "CONTINUON_MODEL_REGISTRY_URL",
            "https://storage.googleapis.com/continuon-models/registry.json"
        )
        self.config_dir = config_dir
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minutes

    async def get_latest_version(self, model_id: str = "seed") -> Optional[str]:
        """Get the latest version for a model."""
        registry = await self._fetch_registry()
        models = registry.get("models", {})
        model_info = models.get(model_id, {})
        return model_info.get("latest_version")

    async def get_model_info(self, model_id: str, version: str = None) -> Optional[UpdateInfo]:
        """Get full model information."""
        registry = await self._fetch_registry()
        models = registry.get("models", {})
        model_info = models.get(model_id, {})

        if not model_info:
            return None

        # If no version specified, get latest
        if version is None:
            version = model_info.get("latest_version")

        versions = model_info.get("versions", {})
        version_info = versions.get(version, {})

        if not version_info:
            return None

        return UpdateInfo(
            model_id=model_id,
            version=version,
            download_url=version_info.get("download_url", ""),
            checksum=version_info.get("checksum", ""),
            signature=version_info.get("signature", ""),
            size_bytes=version_info.get("size_bytes", 0),
            release_notes=version_info.get("release_notes", ""),
            min_brain_version=version_info.get("min_brain_version", "0.0.0"),
            created_at=datetime.fromisoformat(version_info["created_at"]) if version_info.get("created_at") else None,
            metadata=version_info.get("metadata", {}),
        )

    async def _fetch_registry(self) -> Dict[str, Any]:
        """Fetch registry data with caching."""
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        # Try local file first (for offline/development)
        if self.config_dir:
            local_registry = self.config_dir / "model_registry.json"
            if local_registry.exists():
                try:
                    with open(local_registry) as f:
                        self._cache = json.load(f)
                        self._cache_time = now
                        return self._cache
                except Exception as e:
                    logger.warning(f"Failed to load local registry: {e}")

        # Fetch from remote
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.registry_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        self._cache = await resp.json()
                        self._cache_time = now
                        return self._cache
                    else:
                        logger.warning(f"Registry fetch failed with status {resp.status}")
        except Exception as e:
            logger.warning(f"Failed to fetch remote registry: {e}")

        return self._cache or {"models": {}}


class OTAUpdater:
    """
    Manages OTA model updates with verification and rollback.

    Directory structure:
        config_dir/model/
        ├── current/      <- Active model (symlink or directory)
        ├── candidate/    <- Downloaded, pending activation
        └── rollback/     <- Previous version for recovery
    """

    MANIFEST_FILE = "manifest.json"
    CHECKSUMS_FILE = "CHECKSUMS.sha256"
    BUNDLE_FILE = "bundle.tar.gz"

    def __init__(
        self,
        config_dir: Path,
        registry: Optional[ModelRegistry] = None,
        registry_url: str = None,
        brain_version: str = "0.1.0",
    ):
        """
        Initialize OTA Updater.

        Args:
            config_dir: Base configuration directory (e.g., /opt/continuonos/brain)
            registry: Optional ModelRegistry instance
            registry_url: URL to model registry (if registry not provided)
            brain_version: Current brain version for compatibility checks
        """
        self.config_dir = Path(config_dir)
        self.brain_version = brain_version

        # Initialize registry
        self.registry = registry or ModelRegistry(
            registry_url=registry_url,
            config_dir=self.config_dir
        )

        # Model directories
        self.model_dir = self.config_dir / "model"
        self.current_dir = self.model_dir / "current"
        self.candidate_dir = self.model_dir / "candidate"
        self.rollback_dir = self.model_dir / "rollback"

        # Keys directory for signature verification
        self.keys_dir = self.config_dir / "keys"

        # State tracking
        self._state = UpdateState.IDLE
        self._progress = 0.0
        self._error: Optional[str] = None
        self._available_update: Optional[UpdateInfo] = None
        self._last_check: Optional[datetime] = None
        self._last_update: Optional[datetime] = None
        self._lock = asyncio.Lock()

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.keys_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_version(self) -> str:
        """Get the currently installed model version."""
        manifest_path = self.current_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    return manifest.get("version", "0.0.0")
            except Exception as e:
                logger.warning(f"Failed to read current manifest: {e}")
        return "0.0.0"

    def _parse_version(self, version: str) -> Tuple[int, ...]:
        """Parse version string into comparable tuple."""
        try:
            parts = version.split(".")
            return tuple(int(p) for p in parts[:3])
        except (ValueError, AttributeError):
            return (0, 0, 0)

    def _is_newer(self, new_version: str, current_version: str) -> bool:
        """Check if new_version is newer than current_version."""
        return self._parse_version(new_version) > self._parse_version(current_version)

    def _is_compatible(self, update: UpdateInfo) -> bool:
        """Check if update is compatible with current brain version."""
        return self._parse_version(self.brain_version) >= self._parse_version(update.min_brain_version)

    def get_status(self) -> UpdateStatus:
        """Get current update status."""
        return UpdateStatus(
            state=self._state,
            current_version=self._get_current_version(),
            available_update=self._available_update,
            progress_percent=self._progress,
            error_message=self._error,
            last_check=self._last_check,
            last_update=self._last_update,
            rollback_available=self.rollback_dir.exists() and (self.rollback_dir / self.MANIFEST_FILE).exists(),
        )

    async def check_for_updates(self, model_id: str = "seed") -> Optional[UpdateInfo]:
        """
        Check if a newer version is available.

        Args:
            model_id: Model identifier to check

        Returns:
            UpdateInfo if update available, None otherwise
        """
        async with self._lock:
            self._state = UpdateState.CHECKING
            self._error = None

            try:
                current_version = self._get_current_version()
                latest_version = await self.registry.get_latest_version(model_id)

                if latest_version and self._is_newer(latest_version, current_version):
                    update_info = await self.registry.get_model_info(model_id, latest_version)

                    if update_info and self._is_compatible(update_info):
                        self._available_update = update_info
                        logger.info(f"Update available: {current_version} -> {latest_version}")
                    else:
                        logger.info(f"Update {latest_version} not compatible with brain {self.brain_version}")
                        self._available_update = None
                else:
                    logger.debug(f"No update available (current: {current_version}, latest: {latest_version})")
                    self._available_update = None

                self._last_check = datetime.now(timezone.utc)
                self._state = UpdateState.IDLE
                return self._available_update

            except Exception as e:
                logger.error(f"Update check failed: {e}")
                self._state = UpdateState.ERROR
                self._error = str(e)
                return None

    async def download_update(
        self,
        update: UpdateInfo,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Download update to candidate directory.

        Args:
            update: Update information
            progress_cb: Optional callback for progress updates (0-100)

        Returns:
            Path to candidate directory

        Raises:
            ValueError: If verification fails
            IOError: If download fails
        """
        async with self._lock:
            self._state = UpdateState.DOWNLOADING
            self._progress = 0.0
            self._error = None

            try:
                # Clean candidate directory
                if self.candidate_dir.exists():
                    shutil.rmtree(self.candidate_dir)
                self.candidate_dir.mkdir(parents=True, exist_ok=True)

                bundle_path = self.candidate_dir / self.BUNDLE_FILE

                # Download with progress tracking
                await self._download_with_progress(
                    update.download_url,
                    bundle_path,
                    update.size_bytes,
                    progress_cb,
                )

                # Verify checksum
                self._state = UpdateState.VERIFYING
                self._progress = 0.0

                if not self._verify_checksum(bundle_path, update.checksum):
                    raise ValueError("Checksum verification failed - download may be corrupted")

                # Verify signature (if configured)
                if not self._verify_signature(bundle_path, update.signature):
                    raise ValueError("Signature verification failed - update may be tampered")

                # Extract bundle
                self._state = UpdateState.EXTRACTING
                self._progress = 0.0

                await self._extract_bundle(bundle_path, self.candidate_dir)

                # Validate manifest exists
                manifest_path = self.candidate_dir / self.MANIFEST_FILE
                if not manifest_path.exists():
                    raise ValueError("Invalid bundle: missing manifest.json")

                # Clean up bundle file
                bundle_path.unlink(missing_ok=True)

                self._state = UpdateState.IDLE
                self._progress = 100.0
                logger.info(f"Update {update.version} downloaded and verified")
                return self.candidate_dir

            except Exception as e:
                logger.error(f"Download failed: {e}")
                self._state = UpdateState.ERROR
                self._error = str(e)
                # Clean up on failure
                if self.candidate_dir.exists():
                    shutil.rmtree(self.candidate_dir, ignore_errors=True)
                raise

    async def _download_with_progress(
        self,
        url: str,
        dest_path: Path,
        expected_size: int,
        progress_cb: Optional[Callable[[float], None]],
    ) -> None:
        """Download file with progress tracking."""
        downloaded = 0
        chunk_size = 64 * 1024  # 64KB chunks

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                if resp.status != 200:
                    raise IOError(f"Download failed with status {resp.status}")

                total_size = int(resp.headers.get("content-length", expected_size))

                with open(dest_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            self._progress = (downloaded / total_size) * 100
                            if progress_cb:
                                progress_cb(self._progress)

    def _verify_checksum(self, path: Path, expected: str) -> bool:
        """
        Verify SHA256 checksum of a file.

        Args:
            path: Path to file
            expected: Expected checksum in format "sha256:<hex>"

        Returns:
            True if checksum matches
        """
        if not expected:
            logger.warning("No checksum provided, skipping verification")
            return True

        # Parse expected checksum
        if expected.startswith("sha256:"):
            expected_hash = expected[7:]
        else:
            expected_hash = expected

        # Calculate actual checksum
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        actual_hash = sha256.hexdigest()

        if actual_hash.lower() != expected_hash.lower():
            logger.error(f"Checksum mismatch: expected {expected_hash}, got {actual_hash}")
            return False

        logger.debug(f"Checksum verified: {actual_hash}")
        return True

    def _verify_signature(self, path: Path, signature: str) -> bool:
        """
        Verify Ed25519 signature of a file.

        Args:
            path: Path to file
            signature: Base64-encoded Ed25519 signature

        Returns:
            True if signature is valid or no key configured
        """
        public_key_path = self.keys_dir / "model_signing.pub"

        if not public_key_path.exists():
            logger.warning("No signing key configured, skipping signature verification")
            return True

        if not signature:
            logger.warning("No signature provided but signing key exists")
            return False

        try:
            # Try to use nacl for Ed25519 verification
            from nacl.signing import VerifyKey
            from nacl.encoding import Base64Encoder
            from nacl.exceptions import BadSignature

            # Load public key
            with open(public_key_path, "rb") as f:
                public_key_bytes = f.read().strip()
                # Handle PEM format or raw bytes
                if b"-----BEGIN" in public_key_bytes:
                    # Extract key from PEM
                    import base64
                    lines = public_key_bytes.decode().split("\n")
                    key_data = "".join(line for line in lines if not line.startswith("-----"))
                    public_key_bytes = base64.b64decode(key_data)

            verify_key = VerifyKey(public_key_bytes)

            # Read file content
            with open(path, "rb") as f:
                file_content = f.read()

            # Verify signature
            import base64
            sig_bytes = base64.b64decode(signature)
            verify_key.verify(file_content, sig_bytes)

            logger.debug("Signature verified successfully")
            return True

        except ImportError:
            logger.warning("nacl library not available, skipping signature verification")
            return True
        except BadSignature:
            logger.error("Invalid signature")
            return False
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    async def _extract_bundle(self, bundle_path: Path, dest_dir: Path) -> None:
        """Extract tar.gz bundle to destination directory."""
        def _extract():
            with tarfile.open(bundle_path, "r:gz") as tar:
                # Security: check for path traversal
                for member in tar.getmembers():
                    member_path = Path(dest_dir) / member.name
                    try:
                        member_path.resolve().relative_to(dest_dir.resolve())
                    except ValueError:
                        raise ValueError(f"Path traversal detected in bundle: {member.name}")

                tar.extractall(dest_dir)

        # Run extraction in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _extract)
        logger.debug(f"Bundle extracted to {dest_dir}")

    async def activate_update(self, run_health_check: bool = True) -> bool:
        """
        Activate candidate model with optional health check.

        This performs an atomic swap:
        1. current/ -> rollback/ (backup)
        2. candidate/ -> current/ (activate)
        3. Run health check (optional)
        4. If health check fails, auto-rollback

        Args:
            run_health_check: Whether to run health check after activation

        Returns:
            True if activation successful, False if rolled back

        Raises:
            ValueError: If no candidate to activate
        """
        async with self._lock:
            self._state = UpdateState.ACTIVATING
            self._error = None

            try:
                # Validate candidate exists
                candidate_manifest = self.candidate_dir / self.MANIFEST_FILE
                if not candidate_manifest.exists():
                    raise ValueError("No candidate to activate - download an update first")

                # Read candidate version
                with open(candidate_manifest) as f:
                    manifest = json.load(f)
                    new_version = manifest.get("version", "unknown")

                logger.info(f"Activating update version {new_version}")

                # Step 1: Backup current to rollback
                if self.current_dir.exists():
                    if self.rollback_dir.exists():
                        shutil.rmtree(self.rollback_dir)
                    shutil.move(str(self.current_dir), str(self.rollback_dir))
                    logger.debug(f"Backed up current to {self.rollback_dir}")

                # Step 2: Activate candidate
                shutil.move(str(self.candidate_dir), str(self.current_dir))
                logger.debug(f"Activated candidate as current")

                # Step 3: Health check
                if run_health_check:
                    health_ok = await self._run_health_check()
                    if not health_ok:
                        logger.warning("Health check failed, initiating rollback")
                        await self.rollback()
                        return False

                self._last_update = datetime.now(timezone.utc)
                self._state = UpdateState.IDLE
                self._available_update = None
                logger.info(f"Update {new_version} activated successfully")
                return True

            except Exception as e:
                logger.error(f"Activation failed: {e}")
                self._state = UpdateState.ERROR
                self._error = str(e)
                # Attempt rollback on failure
                try:
                    if self.rollback_dir.exists():
                        await self.rollback()
                except Exception:
                    pass
                raise

    async def rollback(self) -> bool:
        """
        Rollback to previous version.

        Returns:
            True if rollback successful

        Raises:
            ValueError: If no rollback available
        """
        async with self._lock:
            self._state = UpdateState.ROLLING_BACK
            self._error = None

            try:
                if not self.rollback_dir.exists():
                    raise ValueError("No rollback available")

                rollback_manifest = self.rollback_dir / self.MANIFEST_FILE
                if not rollback_manifest.exists():
                    raise ValueError("Invalid rollback directory - missing manifest")

                # Read rollback version
                with open(rollback_manifest) as f:
                    manifest = json.load(f)
                    rollback_version = manifest.get("version", "unknown")

                logger.info(f"Rolling back to version {rollback_version}")

                # Remove current
                if self.current_dir.exists():
                    shutil.rmtree(self.current_dir)

                # Restore rollback as current
                shutil.move(str(self.rollback_dir), str(self.current_dir))

                self._state = UpdateState.IDLE
                logger.info(f"Rolled back to version {rollback_version}")
                return True

            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                self._state = UpdateState.ERROR
                self._error = str(e)
                raise

    async def _run_health_check(self) -> bool:
        """
        Run health check on newly activated model.

        Returns:
            True if health check passes
        """
        try:
            # Check manifest is valid
            manifest_path = self.current_dir / self.MANIFEST_FILE
            if not manifest_path.exists():
                logger.error("Health check failed: missing manifest")
                return False

            with open(manifest_path) as f:
                manifest = json.load(f)

            # Check required files exist
            required_files = manifest.get("required_files", [])
            for filename in required_files:
                file_path = self.current_dir / filename
                if not file_path.exists():
                    logger.error(f"Health check failed: missing required file {filename}")
                    return False

            # Check model file exists (common patterns)
            model_patterns = ["seed_model.npz", "model.npz", "weights.npz", "model.safetensors"]
            model_found = False
            for pattern in model_patterns:
                if (self.current_dir / pattern).exists():
                    model_found = True
                    break

            if not model_found and not manifest.get("skip_model_check", False):
                logger.warning("Health check: no model file found, but continuing")

            # Optional: Try to load model to verify it's valid
            # This would depend on the model format and loader

            logger.info("Health check passed")
            return True

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

    async def cleanup_candidate(self) -> None:
        """Clean up candidate directory (e.g., after failed activation)."""
        if self.candidate_dir.exists():
            shutil.rmtree(self.candidate_dir)
            logger.debug("Cleaned up candidate directory")

    async def cleanup_rollback(self) -> None:
        """Clean up rollback directory to save space."""
        if self.rollback_dir.exists():
            shutil.rmtree(self.rollback_dir)
            logger.debug("Cleaned up rollback directory")

    def get_installed_versions(self) -> Dict[str, str]:
        """Get all installed model versions."""
        versions = {}

        for name, dir_path in [
            ("current", self.current_dir),
            ("candidate", self.candidate_dir),
            ("rollback", self.rollback_dir),
        ]:
            manifest_path = dir_path / self.MANIFEST_FILE
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                        versions[name] = manifest.get("version", "unknown")
                except Exception:
                    versions[name] = "error"
            else:
                versions[name] = None

        return versions
