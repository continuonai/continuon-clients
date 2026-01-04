"""
RCAN Cloud Registry - Firebase Firestore Robot Registration

Enables remote robot discovery by registering robots with Firebase Firestore.
Robots can be discovered via their RCAN URI from anywhere in the world.

Usage:
    from continuonbrain.services.rcan_cloud_registry import RCANCloudRegistry

    registry = RCANCloudRegistry(config_dir="/home/user/.continuonbrain")
    await registry.register()  # Register on startup
    await registry.start_heartbeat()  # Keep-alive updates
"""

import asyncio
import json
import logging
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("RCANCloudRegistry")

# Try to import Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin not installed. Cloud registry disabled.")


class RCANCloudRegistry:
    """
    Registers and maintains robot presence in Firebase Firestore.

    Firestore Collections:
        - rcan_registry/{device_id}: Robot registration data
        - rcan_lookup/{ruri_hash}: Quick RURI -> device_id lookup
    """

    COLLECTION = "rcan_registry"
    LOOKUP_COLLECTION = "rcan_lookup"
    HEARTBEAT_INTERVAL = 60  # seconds

    def __init__(
        self,
        config_dir: str,
        device_id: Optional[str] = None,
        robot_name: str = "ContinuonBot",
        owner_email: str = "craigm26@gmail.com",
        owner_uid: Optional[str] = None,
    ):
        self.config_dir = Path(config_dir)
        self.robot_name = robot_name
        self.owner_email = owner_email
        self.owner_uid = owner_uid

        self._db: Optional[Any] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stop_heartbeat = threading.Event()

        # Load RCAN identity
        self.rcan_identity = self._load_rcan_identity()
        self.device_id = device_id or self.rcan_identity.get("device_id", "unknown")
        self.ruri = self._build_ruri()

        # Initialize Firebase
        self._init_firebase()

    def _load_rcan_identity(self) -> Dict[str, Any]:
        """Load RCAN identity from config."""
        identity_file = self.config_dir / "rcan_identity.json"
        if identity_file.exists():
            try:
                with open(identity_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load RCAN identity: {e}")
        return {
            "registry": "continuon.cloud",
            "manufacturer": "continuon",
            "model": "companion-v1",
            "device_id": "unknown"
        }

    def _build_ruri(self) -> str:
        """Build RCAN URI from identity."""
        return f"rcan://{self.rcan_identity.get('registry', 'continuon.cloud')}/{self.rcan_identity.get('manufacturer', 'continuon')}/{self.rcan_identity.get('model', 'companion-v1')}/{self.device_id}"

    def _init_firebase(self) -> None:
        """Initialize Firebase Admin SDK."""
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase Admin SDK not available")
            return

        service_account_path = self.config_dir / "service-account.json"
        if not service_account_path.exists():
            logger.warning(f"Service account not found: {service_account_path}")
            return

        try:
            # Check if already initialized
            try:
                app = firebase_admin.get_app()
                logger.info("Firebase already initialized")
            except ValueError:
                # Initialize Firebase
                cred = credentials.Certificate(str(service_account_path))
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized")

            self._db = firestore.client()
            logger.info("Firestore client ready")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self._db = None

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _get_public_ip(self) -> Optional[str]:
        """Get public IP address (best effort)."""
        try:
            import urllib.request
            return urllib.request.urlopen("https://api.ipify.org", timeout=5).read().decode()
        except Exception:
            return None

    def get_registration_data(
        self,
        capabilities: Optional[list] = None,
        firmware_version: str = "0.1.0",
        tunnel_url: Optional[str] = None,
        port: int = 8081,
    ) -> Dict[str, Any]:
        """Build registration document."""
        now = datetime.now(timezone.utc)
        local_ip = self._get_local_ip()

        return {
            "ruri": self.ruri,
            "device_id": self.device_id,
            "robot_name": self.robot_name,
            "owner_email": self.owner_email,
            "owner_uid": self.owner_uid,
            "manufacturer": self.rcan_identity.get("manufacturer", "continuon"),
            "model": self.rcan_identity.get("model", "companion-v1"),
            "endpoint": {
                "lan_ip": local_ip,
                "lan_port": port,
                "tunnel_url": tunnel_url,
                "public_ip": self._get_public_ip(),
            },
            "capabilities": capabilities or ["arm", "vision", "chat", "teleop", "training"],
            "status": "online",
            "last_heartbeat": now,
            "registered_at": now,
            "firmware_version": firmware_version,
        }

    async def register(
        self,
        capabilities: Optional[list] = None,
        firmware_version: str = "0.1.0",
        tunnel_url: Optional[str] = None,
        port: int = 8081,
    ) -> Dict[str, Any]:
        """
        Register robot with cloud registry.

        Returns:
            Registration result with status
        """
        if self._db is None:
            return {"success": False, "error": "Firebase not initialized"}

        try:
            data = self.get_registration_data(
                capabilities=capabilities,
                firmware_version=firmware_version,
                tunnel_url=tunnel_url,
                port=port,
            )

            # Use device_id as document ID
            doc_ref = self._db.collection(self.COLLECTION).document(self.device_id)

            # Check if already registered
            existing = doc_ref.get()
            if existing.exists:
                # Update existing registration
                update_data = {
                    "endpoint": data["endpoint"],
                    "status": "online",
                    "last_heartbeat": data["last_heartbeat"],
                    "capabilities": data["capabilities"],
                    "firmware_version": data["firmware_version"],
                }
                doc_ref.update(update_data)
                logger.info(f"Updated cloud registration for {self.ruri}")
            else:
                # Create new registration
                doc_ref.set(data)
                logger.info(f"Created cloud registration for {self.ruri}")

            # Update lookup index
            ruri_hash = self.ruri.replace("/", "_").replace(":", "_")
            lookup_ref = self._db.collection(self.LOOKUP_COLLECTION).document(ruri_hash)
            lookup_ref.set({
                "device_id": self.device_id,
                "ruri": self.ruri,
            })

            return {
                "success": True,
                "ruri": self.ruri,
                "device_id": self.device_id,
                "message": "Robot registered with cloud registry",
            }
        except Exception as e:
            logger.error(f"Cloud registration failed: {e}")
            return {"success": False, "error": str(e)}

    async def heartbeat(self) -> Dict[str, Any]:
        """Send heartbeat update."""
        if self._db is None:
            return {"success": False, "error": "Firebase not initialized"}

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(self.device_id)
            doc_ref.update({
                "status": "online",
                "last_heartbeat": datetime.now(timezone.utc),
                "endpoint.lan_ip": self._get_local_ip(),
            })
            return {"success": True}
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return {"success": False, "error": str(e)}

    async def set_offline(self) -> Dict[str, Any]:
        """Mark robot as offline."""
        if self._db is None:
            return {"success": False, "error": "Firebase not initialized"}

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(self.device_id)
            doc_ref.update({
                "status": "offline",
                "last_heartbeat": datetime.now(timezone.utc),
            })
            logger.info(f"Marked {self.device_id} as offline")
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to set offline: {e}")
            return {"success": False, "error": str(e)}

    async def start_heartbeat(self, interval: int = None) -> None:
        """Start background heartbeat task."""
        interval = interval or self.HEARTBEAT_INTERVAL
        self._stop_heartbeat.clear()

        async def heartbeat_loop():
            while not self._stop_heartbeat.is_set():
                await self.heartbeat()
                await asyncio.sleep(interval)

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        logger.info(f"Started heartbeat every {interval}s")

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat and mark offline."""
        self._stop_heartbeat.set()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.set_offline()
        logger.info("Stopped heartbeat")

    @classmethod
    async def lookup(cls, ruri: str, config_dir: str) -> Optional[Dict[str, Any]]:
        """
        Lookup a robot by RURI.

        Args:
            ruri: Robot URI (e.g., "rcan://continuon.cloud/continuon/companion-v1/14d4b680")
            config_dir: Path to config directory with service account

        Returns:
            Robot registration data or None
        """
        if not FIREBASE_AVAILABLE:
            return None

        try:
            # Initialize Firebase if needed
            service_account_path = Path(config_dir) / "service-account.json"
            if not service_account_path.exists():
                return None

            try:
                firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(str(service_account_path))
                firebase_admin.initialize_app(cred)

            db = firestore.client()

            # Try lookup collection first
            ruri_hash = ruri.replace("/", "_").replace(":", "_")
            lookup_doc = db.collection(cls.LOOKUP_COLLECTION).document(ruri_hash).get()

            if lookup_doc.exists:
                device_id = lookup_doc.to_dict().get("device_id")
                if device_id:
                    reg_doc = db.collection(cls.COLLECTION).document(device_id).get()
                    if reg_doc.exists:
                        return reg_doc.to_dict()

            # Fallback: Query by RURI
            query = db.collection(cls.COLLECTION).where("ruri", "==", ruri).limit(1)
            results = list(query.stream())
            if results:
                return results[0].to_dict()

            return None
        except Exception as e:
            logger.error(f"Lookup failed: {e}")
            return None
