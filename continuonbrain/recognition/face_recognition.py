"""
Face Recognition for User Role Identification

Privacy-first, consent-based face recognition that:
- Runs entirely on-device (no cloud)
- Requires explicit user consent
- Supports role-based identification
- Allows users to delete their data
"""

import json
import pickle
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ConsentStatus(Enum):
    """User consent state for face recognition."""
    NOT_ASKED = "not_asked"
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"


class UserRole(Enum):
    """User roles in order of privilege."""
    CREATOR = "creator"      # Built/configured the robot
    OWNER = "owner"          # Owns/purchased the robot
    LEASEE = "leasee"        # Leasing/renting access
    USER = "user"            # Authorized regular user
    GUEST = "guest"          # Temporary/limited access
    UNKNOWN = "unknown"      # Not recognized


@dataclass
class KnownUser:
    """A user the robot has consent to recognize."""
    user_id: str
    name: str
    role: UserRole
    face_embeddings: List[np.ndarray] = field(default_factory=list)
    consent_status: ConsentStatus = ConsentStatus.GRANTED
    consent_timestamp: str = ""
    created_at: str = ""
    last_seen: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'name': self.name,
            'role': self.role.value,
            'num_embeddings': len(self.face_embeddings),
            'consent_status': self.consent_status.value,
            'consent_timestamp': self.consent_timestamp,
            'created_at': self.created_at,
            'last_seen': self.last_seen,
        }


@dataclass
class RecognitionResult:
    """Result of a face recognition attempt."""
    recognized: bool
    user: Optional[KnownUser] = None
    confidence: float = 0.0
    face_detected: bool = False
    num_faces: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'recognized': self.recognized,
            'user_id': self.user.user_id if self.user else None,
            'user_name': self.user.name if self.user else None,
            'role': self.user.role.value if self.user else UserRole.UNKNOWN.value,
            'confidence': self.confidence,
            'face_detected': self.face_detected,
            'num_faces': self.num_faces,
        }


class FaceEmbedder:
    """
    Generates face embeddings using on-device models.
    
    Uses lightweight models that run on CPU/NPU:
    - Face detection: MediaPipe or OpenCV DNN
    - Face embedding: FaceNet-lite or similar
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self._detector = None
        self._embedder = None
        self._initialized = False
        self.embedding_dim = 128  # Compact embedding
        
    def initialize(self):
        """Initialize face detection and embedding models."""
        if self._initialized:
            return
            
        try:
            # Try to use MediaPipe for face detection
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # Short-range (< 2m)
                min_detection_confidence=0.5
            )
            self._use_mediapipe = True
            logger.info("Using MediaPipe for face detection")
        except ImportError:
            logger.warning("MediaPipe not available, using fallback")
            self._use_mediapipe = False
            
        try:
            # Try to use a lightweight face embedding model
            # For production, this would use FaceNet-lite or similar
            self._embedder = self._create_simple_embedder()
            logger.info("Face embedder initialized")
        except Exception as e:
            logger.warning(f"Could not load embedder: {e}")
            
        self._initialized = True
    
    def _create_simple_embedder(self):
        """Create a simple hash-based embedder as fallback."""
        # In production, replace with actual face embedding model
        return None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: RGB image as numpy array [H, W, 3]
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if not self._initialized:
            self.initialize()
            
        faces = []
        
        if self._use_mediapipe and self._detector:
            import mediapipe as mp
            results = self._detector.process(image)
            
            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
        else:
            # Fallback: simple center crop assuming face is centered
            h, w = image.shape[:2]
            faces.append((w//4, h//4, w//2, h//2))
            
        return faces
    
    def embed_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate embedding for a detected face.
        
        Args:
            image: Full image
            bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Face embedding vector [embedding_dim]
        """
        x, y, w, h = bbox
        
        # Clamp to image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        
        face_crop = image[y:y+h, x:x+w]
        
        # Ensure face_crop is valid
        if face_crop.size == 0 or face_crop.shape[0] < 1 or face_crop.shape[1] < 1:
            # Return random embedding if face crop failed
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding
        
        if self._embedder is not None:
            # Use actual model
            embedding = self._embedder(face_crop)
        else:
            # Fallback: deterministic hash-based embedding
            # This is a placeholder - replace with real model
            face_resized = self._resize_face(face_crop, (64, 64))
            hash_input = face_resized.tobytes()
            hash_digest = hashlib.sha256(hash_input).digest()
            
            # Convert to float embedding
            embedding = np.frombuffer(hash_digest[:self.embedding_dim], dtype=np.uint8)
            embedding = embedding.astype(np.float32) / 255.0
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
        return embedding
    
    def _resize_face(self, face: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize face crop to standard size."""
        try:
            import cv2
            return cv2.resize(face, size)
        except ImportError:
            # Simple nearest-neighbor resize
            h, w = face.shape[:2]
            th, tw = size
            y_indices = (np.arange(th) * h / th).astype(int)
            x_indices = (np.arange(tw) * w / tw).astype(int)
            return face[np.ix_(y_indices, x_indices)]


class FaceRecognizer:
    """
    Main face recognition system for user role identification.
    
    Features:
    - Consent-based face storage
    - On-device only (no cloud)
    - Role-based access control
    - User data deletion
    """
    
    def __init__(
        self,
        data_dir: Path = Path("/opt/continuonos/brain/recognition"),
        similarity_threshold: float = 0.7,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.embedder = FaceEmbedder()
        
        self._known_users: Dict[str, KnownUser] = {}
        self._load_known_users()
        
        logger.info(f"FaceRecognizer initialized with {len(self._known_users)} known users")
    
    def _load_known_users(self):
        """Load known users from disk."""
        users_file = self.data_dir / "known_users.pkl"
        if users_file.exists():
            try:
                with open(users_file, 'rb') as f:
                    self._known_users = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load known users: {e}")
                self._known_users = {}
    
    def _save_known_users(self):
        """Save known users to disk."""
        users_file = self.data_dir / "known_users.pkl"
        try:
            with open(users_file, 'wb') as f:
                pickle.dump(self._known_users, f)
        except Exception as e:
            logger.error(f"Failed to save known users: {e}")
    
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """
        Recognize faces in an image.
        
        Args:
            image: RGB image as numpy array [H, W, 3]
            
        Returns:
            RecognitionResult with user info if recognized
        """
        try:
            # Detect faces
            faces = self.embedder.detect_faces(image)
            
            if not faces:
                return RecognitionResult(
                    recognized=False,
                    face_detected=False,
                    num_faces=0
                )
            
            # Use largest face (assumed to be primary user)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            # Get embedding
            embedding = self.embedder.embed_face(image, largest_face)
            
            # Find best match
            best_user = None
            best_similarity = 0.0
            
            for user in self._known_users.values():
                if user.consent_status != ConsentStatus.GRANTED:
                    continue
                    
                for stored_emb in user.face_embeddings:
                    similarity = np.dot(embedding, stored_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_user = user
            
            if best_user and best_similarity >= self.similarity_threshold:
                # Update last seen
                best_user.last_seen = datetime.now().isoformat()
                self._save_known_users()
                
                return RecognitionResult(
                    recognized=True,
                    user=best_user,
                    confidence=float(best_similarity),
                    face_detected=True,
                    num_faces=len(faces)
                )
            else:
                return RecognitionResult(
                    recognized=False,
                    confidence=float(best_similarity) if best_user else 0.0,
                    face_detected=True,
                    num_faces=len(faces)
                )
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return RecognitionResult(
                recognized=False,
                error=str(e)
            )
    
    def register_user(
        self,
        user_id: str,
        name: str,
        role: UserRole,
        images: List[np.ndarray],
        consent_granted: bool = True,
    ) -> bool:
        """
        Register a new user with consent.
        
        Args:
            user_id: Unique user identifier
            name: Display name
            role: User role (creator, owner, etc.)
            images: List of face images for enrollment
            consent_granted: User has explicitly consented
            
        Returns:
            True if registration successful
        """
        if not consent_granted:
            logger.warning(f"Cannot register user {user_id} without consent")
            return False
        
        if len(images) < 1:
            logger.error("At least one image required for registration")
            return False
        
        # Generate embeddings from all provided images
        embeddings = []
        for img in images:
            faces = self.embedder.detect_faces(img)
            if faces:
                largest = max(faces, key=lambda f: f[2] * f[3])
                emb = self.embedder.embed_face(img, largest)
                embeddings.append(emb)
        
        if not embeddings:
            logger.error("No faces detected in provided images")
            return False
        
        # Create user
        now = datetime.now().isoformat()
        user = KnownUser(
            user_id=user_id,
            name=name,
            role=role,
            face_embeddings=embeddings,
            consent_status=ConsentStatus.GRANTED,
            consent_timestamp=now,
            created_at=now,
            last_seen=now,
        )
        
        self._known_users[user_id] = user
        self._save_known_users()
        
        logger.info(f"Registered user {name} ({user_id}) as {role.value} with {len(embeddings)} face(s)")
        return True
    
    def update_user_role(self, user_id: str, new_role: UserRole) -> bool:
        """Update a user's role."""
        if user_id not in self._known_users:
            return False
            
        self._known_users[user_id].role = new_role
        self._save_known_users()
        return True
    
    def add_face(self, user_id: str, image: np.ndarray) -> bool:
        """Add additional face image to existing user."""
        if user_id not in self._known_users:
            return False
            
        user = self._known_users[user_id]
        if user.consent_status != ConsentStatus.GRANTED:
            return False
            
        faces = self.embedder.detect_faces(image)
        if not faces:
            return False
            
        largest = max(faces, key=lambda f: f[2] * f[3])
        emb = self.embedder.embed_face(image, largest)
        user.face_embeddings.append(emb)
        self._save_known_users()
        
        return True
    
    def revoke_consent(self, user_id: str) -> bool:
        """
        Revoke user's consent and delete their face data.
        
        This is a GDPR-compliant data deletion.
        """
        if user_id not in self._known_users:
            return False
            
        user = self._known_users[user_id]
        user.consent_status = ConsentStatus.REVOKED
        user.face_embeddings = []  # Delete face data
        self._save_known_users()
        
        logger.info(f"Consent revoked and face data deleted for user {user_id}")
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Completely delete a user and all their data."""
        if user_id not in self._known_users:
            return False
            
        del self._known_users[user_id]
        self._save_known_users()
        
        logger.info(f"User {user_id} completely deleted")
        return True
    
    def get_known_users(self) -> List[Dict]:
        """Get list of known users (without embeddings)."""
        return [user.to_dict() for user in self._known_users.values()]
    
    def get_user(self, user_id: str) -> Optional[KnownUser]:
        """Get a specific user."""
        return self._known_users.get(user_id)
    
    def get_role_for_user(self, user_id: str) -> UserRole:
        """Get role for a user."""
        user = self._known_users.get(user_id)
        return user.role if user else UserRole.UNKNOWN

