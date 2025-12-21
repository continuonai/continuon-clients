import threading
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

logger = logging.getLogger(__name__)

class CloudRelay:
    """
    Relays commands from Firebase Firestore to the BrainService.
    
    Architecture:
    - Authenticates via 'service-account.json' in config_dir.
    - Listens to 'commands/{robot_id}' collection.
    - Dispatches actions to BrainService.
    """
    
    def __init__(self, config_dir: str, device_id: str):
        self.config_dir = Path(config_dir)
        self.device_id = device_id
        self.brain_service = None
        self._stop_event = threading.Event()
        self._listener_registration = None
        self.db = None
        self.enabled = False

        if not FIREBASE_AVAILABLE:
            logger.warning("CloudRelay disabled: firebase-admin not installed.")
            return

        # Look for credentials
        self.cred_path = self.config_dir / "service-account.json"
        
    def start(self, brain_service):
        """Initialize Firebase and start listening."""
        if not FIREBASE_AVAILABLE:
            return

        self.brain_service = brain_service
        
        if not self.cred_path.exists():
            logger.info(f"CloudRelay: No credentials found at {self.cred_path}. Remote control disabled.")
            return

        try:
            # Check if app already init
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(self.cred_path))
                firebase_admin.initialize_app(cred)
                logger.info("CloudRelay: Firebase initialized.")
            
            self.db = firestore.client()
            self.enabled = True
            
            # Start listener
            self._start_listening()
            
        except Exception as e:
            logger.error(f"CloudRelay: Failed to initialize: {e}")
            self.enabled = False

    def stop(self):
        self._stop_event.set()
        if self._listener_registration:
            self._listener_registration.unsubscribe()
            self._listener_registration = None
            
    def _start_listening(self):
        """Attach snapshot listener to commands collection."""
        if not self.db or not self.device_id:
            return

        logger.info(f"CloudRelay: Listening for commands on robot/{self.device_id}/commands")
        
        # We listen to a subcollection for specific commands to this robot
        # Schema: robot_commands/{robot_id}/queue/{cmd_id} 
        # Or simpler: commands/{robot_id}/messages (but better to query by 'status'=='pending')
        
        # Let's use a dedicated collection for the robot: `robots/{device_id}/commands`
        # Query for pending commands
        
        col_ref = self.db.collection(f"robots/{self.device_id}/commands")
        query = col_ref.where("status", "==", "pending")
        
        try:
            self._listener_registration = query.on_snapshot(self._on_snapshot)
        except Exception as e:
            logger.error(f"CloudRelay: Listener failed: {e}")

    def _on_snapshot(self, col_snapshot, changes, read_time):
        """Callback for Firestore snapshot."""
        for change in changes:
            if change.type.name == 'ADDED':
                doc = change.document
                data = doc.to_dict()
                logger.info(f"CloudRelay: Received command {doc.id}: {data}")
                
                # Execute
                self._process_command(doc.id, data)

    def _process_command(self, doc_id: str, data: Dict[str, Any]):
        """Dispatch command to brain service."""
        if not self.brain_service:
            return
            
        cmd_type = data.get("type")
        payload = data.get("payload", {})
        
        success = False
        response = None
        
        try:
            if cmd_type == "set_mode":
                mode = payload.get("mode")
                if mode and self.brain_service.mode_manager:
                    # Map string to enum or call api
                    # We'll use the API route equivalent in ModeManager if publicly exposed
                    # or just set it. 
                    # Assuming set_mode accepts string name
                    self.brain_service.mode_manager.set_mode(mode)
                    success = True
                    response = f"Mode set to {mode}"
                    
            elif cmd_type == "say":
                text = payload.get("text")
                # Need speaker service
                # Implementing basic print for now or loop injection
                logger.info(f"CloudRelay SPEAK: {text}")
                # Inject into chat
                if hasattr(self.brain_service, "chat_event_queue"):
                     self.brain_service.chat_event_queue.put({"role":"user", "message": f"[Cloud Command] Say: {text}"})
                success = True

            else:
                 response = "Unknown command type"

        except Exception as e:
            logger.error(f"CloudRelay command error: {e}")
            response = str(e)
            
        # Update doc status
        try:
            self.db.collection(f"robots/{self.device_id}/commands").document(doc_id).update({
                "status": "completed" if success else "failed",
                "response": response,
                "processed_at": firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
             logger.error(f"CloudRelay failed to update doc {doc_id}: {e}")
