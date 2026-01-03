"""
User Consent Manager

Manages consent for face recognition and data storage.
Ensures GDPR/privacy compliance.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of consent that can be requested."""
    FACE_RECOGNITION = "face_recognition"
    FACE_STORAGE = "face_storage"
    VOICE_RECOGNITION = "voice_recognition"
    LOCATION_TRACKING = "location_tracking"
    BEHAVIOR_LEARNING = "behavior_learning"
    DATA_SYNC = "data_sync"


@dataclass
class ConsentRecord:
    """Record of a consent decision."""
    user_id: str
    consent_type: str
    granted: bool
    timestamp: str
    method: str  # "verbal", "app", "physical_button", "gesture"
    witness: Optional[str] = None  # Robot ID that witnessed
    expires: Optional[str] = None
    revoked: bool = False
    revoked_at: Optional[str] = None


class UserConsentManager:
    """
    Manages user consent for various data collection features.
    
    Principles:
    1. Opt-in by default (nothing stored without explicit consent)
    2. Easy revocation (one action to remove all data)
    3. Transparent (users can see what's stored)
    4. Time-limited options (consent can expire)
    """
    
    def __init__(self, data_dir: Path = Path("/opt/continuonos/brain/consent")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._consent_records: Dict[str, Dict[str, ConsentRecord]] = {}
        self._load_records()
    
    def _load_records(self):
        """Load consent records from disk."""
        records_file = self.data_dir / "consent_records.json"
        if records_file.exists():
            try:
                with open(records_file) as f:
                    data = json.load(f)
                    for user_id, consents in data.items():
                        self._consent_records[user_id] = {
                            ct: ConsentRecord(**record)
                            for ct, record in consents.items()
                        }
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")
    
    def _save_records(self):
        """Save consent records to disk."""
        records_file = self.data_dir / "consent_records.json"
        try:
            data = {
                user_id: {
                    ct: asdict(record)
                    for ct, record in consents.items()
                }
                for user_id, consents in self._consent_records.items()
            }
            with open(records_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")
    
    def request_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        method: str = "app",
    ) -> Dict:
        """
        Generate a consent request.
        
        Returns a structured request that should be presented to the user.
        """
        explanations = {
            ConsentType.FACE_RECOGNITION: {
                'title': "Face Recognition",
                'description': "Allow the robot to recognize your face so it knows who you are.",
                'data_stored': "Face embedding (mathematical representation, not photos)",
                'purpose': "Personalized interactions, role-based access",
                'retention': "Until you revoke consent or delete your account",
                'sharing': "Never shared - stored only on this robot",
            },
            ConsentType.FACE_STORAGE: {
                'title': "Face Data Storage",
                'description': "Store your face data on this robot for future recognition.",
                'data_stored': "Face embeddings from enrollment images",
                'purpose': "Remember you between sessions",
                'retention': "Until you revoke consent",
                'sharing': "Never shared - local only",
            },
            ConsentType.VOICE_RECOGNITION: {
                'title': "Voice Recognition",
                'description': "Allow the robot to recognize your voice.",
                'data_stored': "Voice embedding (not recordings)",
                'purpose': "Identify speaker, personalized responses",
                'retention': "Until you revoke consent",
                'sharing': "Never shared - local only",
            },
            ConsentType.BEHAVIOR_LEARNING: {
                'title': "Behavior Learning",
                'description': "Allow the robot to learn your preferences and habits.",
                'data_stored': "Interaction patterns, preferences",
                'purpose': "Better anticipate your needs",
                'retention': "Until you revoke consent",
                'sharing': "Never shared without additional consent",
            },
        }
        
        info = explanations.get(consent_type, {
            'title': consent_type.value,
            'description': f"Consent for {consent_type.value}",
            'data_stored': "Varies",
            'purpose': "Robot functionality",
            'retention': "Until revoked",
            'sharing': "Not shared",
        })
        
        return {
            'request_id': f"{user_id}_{consent_type.value}_{datetime.now().timestamp()}",
            'user_id': user_id,
            'consent_type': consent_type.value,
            'method': method,
            **info,
            'actions': {
                'grant': f"/api/consent/grant/{user_id}/{consent_type.value}",
                'deny': f"/api/consent/deny/{user_id}/{consent_type.value}",
                'more_info': f"/api/consent/info/{consent_type.value}",
            }
        }
    
    def grant_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        method: str = "app",
        witness: Optional[str] = None,
        expires: Optional[str] = None,
    ) -> bool:
        """
        Record that user has granted consent.
        
        Args:
            user_id: User granting consent
            consent_type: Type of consent being granted
            method: How consent was given (app, verbal, gesture, etc.)
            witness: Robot ID that witnessed (for verbal/gesture)
            expires: Optional expiration datetime
        """
        record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type.value,
            granted=True,
            timestamp=datetime.now().isoformat(),
            method=method,
            witness=witness,
            expires=expires,
        )
        
        if user_id not in self._consent_records:
            self._consent_records[user_id] = {}
        
        self._consent_records[user_id][consent_type.value] = record
        self._save_records()
        
        logger.info(f"Consent granted: {user_id} -> {consent_type.value} via {method}")
        return True
    
    def deny_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        method: str = "app",
    ) -> bool:
        """Record that user has denied consent."""
        record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type.value,
            granted=False,
            timestamp=datetime.now().isoformat(),
            method=method,
        )
        
        if user_id not in self._consent_records:
            self._consent_records[user_id] = {}
        
        self._consent_records[user_id][consent_type.value] = record
        self._save_records()
        
        logger.info(f"Consent denied: {user_id} -> {consent_type.value}")
        return True
    
    def revoke_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """Revoke previously granted consent."""
        if user_id not in self._consent_records:
            return False
            
        if consent_type.value not in self._consent_records[user_id]:
            return False
            
        record = self._consent_records[user_id][consent_type.value]
        record.revoked = True
        record.revoked_at = datetime.now().isoformat()
        self._save_records()
        
        logger.info(f"Consent revoked: {user_id} -> {consent_type.value}")
        return True
    
    def has_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """Check if user has granted consent for a specific type."""
        if user_id not in self._consent_records:
            return False
            
        if consent_type.value not in self._consent_records[user_id]:
            return False
            
        record = self._consent_records[user_id][consent_type.value]
        
        # Check if revoked
        if record.revoked:
            return False
            
        # Check if granted
        if not record.granted:
            return False
            
        # Check if expired
        if record.expires:
            try:
                expires_dt = datetime.fromisoformat(record.expires)
                if datetime.now() > expires_dt:
                    return False
            except:
                pass
                
        return True
    
    def get_user_consents(self, user_id: str) -> Dict[str, bool]:
        """Get all consent statuses for a user."""
        return {
            ct.value: self.has_consent(user_id, ct)
            for ct in ConsentType
        }
    
    def revoke_all_consents(self, user_id: str) -> bool:
        """Revoke all consents for a user (right to be forgotten)."""
        if user_id not in self._consent_records:
            return False
            
        now = datetime.now().isoformat()
        for record in self._consent_records[user_id].values():
            record.revoked = True
            record.revoked_at = now
            
        self._save_records()
        logger.info(f"All consents revoked for user {user_id}")
        return True
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete all consent records for a user."""
        if user_id in self._consent_records:
            del self._consent_records[user_id]
            self._save_records()
            logger.info(f"All consent data deleted for user {user_id}")
            return True
        return False

