"""
Update Scheduler - Background service for automatic update checks.

Periodically checks for model updates and optionally auto-downloads them.
Can notify via callbacks, events, or Firebase Firestore.

Usage:
    from continuonbrain.services.update_scheduler import UpdateScheduler
    from continuonbrain.services.ota_updater import OTAUpdater

    updater = OTAUpdater(config_dir=Path("/opt/continuonos/brain"))
    scheduler = UpdateScheduler(
        updater=updater,
        check_interval_hours=24,
        auto_download=False,
    )

    # Start background checking
    scheduler.start()

    # Register callback for update notifications
    scheduler.on_update_available(lambda info: print(f"Update available: {info.version}"))

    # Stop when done
    scheduler.stop()
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from continuonbrain.services.ota_updater import OTAUpdater, UpdateInfo, UpdateState

logger = logging.getLogger("UpdateScheduler")


class SchedulerState(str, Enum):
    """Scheduler operational state."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class SchedulerStatus:
    """Current scheduler status."""
    state: SchedulerState
    next_check: Optional[datetime]
    last_check: Optional[datetime]
    checks_completed: int
    updates_found: int
    auto_download: bool
    auto_activate: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "next_check": self.next_check.isoformat() if self.next_check else None,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "checks_completed": self.checks_completed,
            "updates_found": self.updates_found,
            "auto_download": self.auto_download,
            "auto_activate": self.auto_activate,
        }


class UpdateScheduler:
    """
    Schedules automatic update checks in the background.

    Features:
    - Configurable check interval
    - Optional auto-download of updates
    - Optional auto-activation (with health check)
    - Callback notifications for update events
    - Firebase/Firestore notifications (optional)
    """

    DEFAULT_CHECK_INTERVAL_HOURS = 24
    MIN_CHECK_INTERVAL_SECONDS = 60  # Minimum 1 minute between checks

    def __init__(
        self,
        updater: OTAUpdater,
        check_interval_hours: float = DEFAULT_CHECK_INTERVAL_HOURS,
        auto_download: bool = False,
        auto_activate: bool = False,
        model_id: str = "seed",
    ):
        """
        Initialize the update scheduler.

        Args:
            updater: OTAUpdater instance
            check_interval_hours: Hours between automatic checks
            auto_download: Automatically download available updates
            auto_activate: Automatically activate downloaded updates (requires auto_download)
            model_id: Model identifier to check for updates
        """
        self.updater = updater
        self.check_interval = max(
            check_interval_hours * 3600,
            self.MIN_CHECK_INTERVAL_SECONDS
        )
        self.auto_download = auto_download
        self.auto_activate = auto_activate and auto_download
        self.model_id = model_id

        # State tracking
        self._state = SchedulerState.STOPPED
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Statistics
        self._checks_completed = 0
        self._updates_found = 0
        self._last_check: Optional[datetime] = None
        self._next_check: Optional[datetime] = None

        # Callbacks
        self._on_update_available: List[Callable[[UpdateInfo], None]] = []
        self._on_download_complete: List[Callable[[UpdateInfo, Path], None]] = []
        self._on_activation_complete: List[Callable[[UpdateInfo, bool], None]] = []
        self._on_error: List[Callable[[Exception], None]] = []

    def start(self) -> None:
        """Start the background update checker."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._stop_event.clear()
        self._state = SchedulerState.RUNNING

        # Calculate next check time
        self._next_check = datetime.now(timezone.utc)

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="UpdateScheduler")
        self._thread.start()

        logger.info(f"Update scheduler started (interval: {self.check_interval / 3600:.1f}h)")

    def stop(self) -> None:
        """Stop the background update checker."""
        if not self._running:
            return

        logger.info("Stopping update scheduler...")
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._state = SchedulerState.STOPPED
        self._next_check = None
        logger.info("Update scheduler stopped")

    def pause(self) -> None:
        """Pause the scheduler (keeps thread alive but skips checks)."""
        self._state = SchedulerState.PAUSED
        logger.info("Update scheduler paused")

    def resume(self) -> None:
        """Resume a paused scheduler."""
        if self._state == SchedulerState.PAUSED:
            self._state = SchedulerState.RUNNING
            self._next_check = datetime.now(timezone.utc)
            logger.info("Update scheduler resumed")

    def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        return SchedulerStatus(
            state=self._state,
            next_check=self._next_check,
            last_check=self._last_check,
            checks_completed=self._checks_completed,
            updates_found=self._updates_found,
            auto_download=self.auto_download,
            auto_activate=self.auto_activate,
        )

    def set_check_interval(self, hours: float) -> None:
        """Update the check interval."""
        self.check_interval = max(hours * 3600, self.MIN_CHECK_INTERVAL_SECONDS)
        logger.info(f"Check interval updated to {hours:.1f} hours")

    def set_auto_download(self, enabled: bool) -> None:
        """Enable or disable auto-download."""
        self.auto_download = enabled
        if not enabled:
            self.auto_activate = False
        logger.info(f"Auto-download {'enabled' if enabled else 'disabled'}")

    def set_auto_activate(self, enabled: bool) -> None:
        """Enable or disable auto-activation."""
        self.auto_activate = enabled and self.auto_download
        logger.info(f"Auto-activate {'enabled' if enabled else 'disabled'}")

    def on_update_available(self, callback: Callable[[UpdateInfo], None]) -> None:
        """Register callback for when an update is available."""
        self._on_update_available.append(callback)

    def on_download_complete(self, callback: Callable[[UpdateInfo, Path], None]) -> None:
        """Register callback for when download completes."""
        self._on_download_complete.append(callback)

    def on_activation_complete(self, callback: Callable[[UpdateInfo, bool], None]) -> None:
        """Register callback for when activation completes (success/failure)."""
        self._on_activation_complete.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for errors."""
        self._on_error.append(callback)

    def trigger_check_now(self) -> None:
        """Trigger an immediate update check."""
        if self._state == SchedulerState.RUNNING:
            self._next_check = datetime.now(timezone.utc)
            logger.info("Triggered immediate update check")
        elif self._state == SchedulerState.STOPPED:
            # Run single check without starting scheduler
            asyncio.run(self._check_for_updates())

    def _run_loop(self) -> None:
        """Main scheduler loop (runs in background thread)."""
        # Create event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            while self._running:
                try:
                    # Check if it's time for an update check
                    if self._state == SchedulerState.RUNNING and self._next_check:
                        now = datetime.now(timezone.utc)
                        if now >= self._next_check:
                            # Run update check
                            self._loop.run_until_complete(self._check_for_updates())

                            # Schedule next check
                            self._next_check = datetime.now(timezone.utc).replace(
                                microsecond=0
                            )
                            # Add interval in seconds
                            from datetime import timedelta
                            self._next_check += timedelta(seconds=self.check_interval)

                    # Sleep for a bit before checking again
                    # Use stop_event to allow quick shutdown
                    if self._stop_event.wait(timeout=1.0):
                        break

                except Exception as e:
                    logger.error(f"Scheduler loop error: {e}")
                    self._notify_error(e)
                    # Wait before retrying
                    time.sleep(10)

        finally:
            self._loop.close()
            self._loop = None

    async def _check_for_updates(self) -> None:
        """Perform an update check."""
        try:
            logger.debug(f"Checking for updates (model: {self.model_id})")
            self._last_check = datetime.now(timezone.utc)
            self._checks_completed += 1

            # Check for update
            update = await self.updater.check_for_updates(self.model_id)

            if update:
                self._updates_found += 1
                logger.info(f"Update available: {update.version}")

                # Notify callbacks
                self._notify_update_available(update)

                # Auto-download if enabled
                if self.auto_download:
                    await self._auto_download(update)
            else:
                logger.debug("No updates available")

        except Exception as e:
            logger.error(f"Update check failed: {e}")
            self._notify_error(e)

    async def _auto_download(self, update: UpdateInfo) -> None:
        """Automatically download an update."""
        try:
            logger.info(f"Auto-downloading update {update.version}")

            def progress_cb(percent: float) -> None:
                if int(percent) % 25 == 0:  # Log at 0, 25, 50, 75, 100
                    logger.debug(f"Download progress: {percent:.0f}%")

            candidate_path = await self.updater.download_update(update, progress_cb)
            logger.info(f"Update downloaded to {candidate_path}")

            # Notify callbacks
            self._notify_download_complete(update, candidate_path)

            # Auto-activate if enabled
            if self.auto_activate:
                await self._auto_activate(update)

        except Exception as e:
            logger.error(f"Auto-download failed: {e}")
            self._notify_error(e)

    async def _auto_activate(self, update: UpdateInfo) -> None:
        """Automatically activate a downloaded update."""
        try:
            logger.info(f"Auto-activating update {update.version}")

            success = await self.updater.activate_update(run_health_check=True)

            if success:
                logger.info(f"Update {update.version} activated successfully")
            else:
                logger.warning(f"Update {update.version} activation failed, rolled back")

            # Notify callbacks
            self._notify_activation_complete(update, success)

        except Exception as e:
            logger.error(f"Auto-activation failed: {e}")
            self._notify_error(e)

    def _notify_update_available(self, update: UpdateInfo) -> None:
        """Notify all registered callbacks of available update."""
        for callback in self._on_update_available:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Callback error (on_update_available): {e}")

    def _notify_download_complete(self, update: UpdateInfo, path: Path) -> None:
        """Notify all registered callbacks of completed download."""
        for callback in self._on_download_complete:
            try:
                callback(update, path)
            except Exception as e:
                logger.error(f"Callback error (on_download_complete): {e}")

    def _notify_activation_complete(self, update: UpdateInfo, success: bool) -> None:
        """Notify all registered callbacks of activation completion."""
        for callback in self._on_activation_complete:
            try:
                callback(update, success)
            except Exception as e:
                logger.error(f"Callback error (on_activation_complete): {e}")

    def _notify_error(self, error: Exception) -> None:
        """Notify all registered callbacks of an error."""
        for callback in self._on_error:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Callback error (on_error): {e}")


class FirestoreUpdateNotifier:
    """
    Optional Firestore integration for update notifications.

    Writes update status to Firestore for mobile app notifications.
    """

    COLLECTION = "robot_updates"

    def __init__(self, device_id: str, config_dir: Path):
        """
        Initialize Firestore notifier.

        Args:
            device_id: Robot device ID
            config_dir: Config directory with service account credentials
        """
        self.device_id = device_id
        self.config_dir = config_dir
        self._db = None

        self._init_firestore()

    def _init_firestore(self) -> None:
        """Initialize Firestore client."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            service_account_path = self.config_dir / "service-account.json"
            if not service_account_path.exists():
                logger.warning("No service account found, Firestore notifications disabled")
                return

            try:
                firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(str(service_account_path))
                firebase_admin.initialize_app(cred)

            self._db = firestore.client()
            logger.info("Firestore update notifier initialized")

        except ImportError:
            logger.warning("firebase-admin not installed, Firestore notifications disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Firestore: {e}")

    async def notify_update_available(self, update: UpdateInfo) -> None:
        """Write update available notification to Firestore."""
        if not self._db:
            return

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(self.device_id)
            doc_ref.set({
                "device_id": self.device_id,
                "update_available": True,
                "version": update.version,
                "size_bytes": update.size_bytes,
                "release_notes": update.release_notes,
                "timestamp": datetime.now(timezone.utc),
            }, merge=True)
            logger.debug(f"Firestore: notified update available for {self.device_id}")
        except Exception as e:
            logger.warning(f"Failed to notify Firestore: {e}")

    async def notify_update_installed(self, version: str) -> None:
        """Write update installed notification to Firestore."""
        if not self._db:
            return

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(self.device_id)
            doc_ref.set({
                "device_id": self.device_id,
                "update_available": False,
                "installed_version": version,
                "installed_at": datetime.now(timezone.utc),
            }, merge=True)
            logger.debug(f"Firestore: notified update installed for {self.device_id}")
        except Exception as e:
            logger.warning(f"Failed to notify Firestore: {e}")

    async def clear_update_notification(self) -> None:
        """Clear update notification from Firestore."""
        if not self._db:
            return

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(self.device_id)
            doc_ref.update({"update_available": False})
            logger.debug(f"Firestore: cleared update notification for {self.device_id}")
        except Exception as e:
            logger.warning(f"Failed to clear Firestore notification: {e}")
