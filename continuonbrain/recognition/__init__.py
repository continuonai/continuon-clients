"""
User Recognition Module

Privacy-first face recognition for:
- User role identification (creator, owner, leasee, user, guest)
- Consent-based "knowing" (users opt-in to be remembered)
- On-device processing (no cloud face data)

Key principles:
1. CONSENT REQUIRED - Users must explicitly opt-in to face storage
2. LOCAL ONLY - Face embeddings never leave the device
3. DELETABLE - Users can remove their face data anytime
4. TRANSPARENT - Users know when recognition is active
"""

from .face_recognition import (
    FaceRecognizer,
    KnownUser,
    RecognitionResult,
    ConsentStatus,
    UserRole,
)
from .user_consent import UserConsentManager, ConsentType

__all__ = [
    'FaceRecognizer',
    'KnownUser', 
    'RecognitionResult',
    'ConsentStatus',
    'UserRole',
    'UserConsentManager',
    'ConsentType',
]

