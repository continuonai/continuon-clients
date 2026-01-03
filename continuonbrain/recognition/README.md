p# User Recognition Module

Privacy-first face recognition for user role identification.

## Purpose

The recognition module allows the robot to:
- **Identify users** by their role (creator, owner, leasee, user, guest)
- **Remember users** who have given consent to be "known"
- **Personalize interactions** based on who is interacting

## Privacy Principles

| Principle | Implementation |
|-----------|----------------|
| **Consent Required** | Users must explicitly opt-in to face storage |
| **Local Only** | Face embeddings never leave the device |
| **Deletable** | Users can remove their data anytime |
| **Transparent** | Users know when recognition is active |
| **No Photos** | Only mathematical embeddings stored, not images |

## User Roles

| Role | Description | Permissions |
|------|-------------|-------------|
| **Creator** | Built/configured the robot | Full access, safety overrides |
| **Owner** | Owns/purchased the robot | Full access, manage users |
| **Leasee** | Leasing/renting access | Limited admin, no ownership transfer |
| **User** | Authorized regular user | Normal operation access |
| **Guest** | Temporary/limited access | Basic interactions only |
| **Unknown** | Not recognized | Public-safe mode only |

## Usage

### Register a User (with consent)

```python
from continuonbrain.recognition import FaceRecognizer, UserRole
import numpy as np

recognizer = FaceRecognizer()

# User must explicitly grant consent
images = [camera.capture() for _ in range(3)]  # Multiple angles

success = recognizer.register_user(
    user_id="user_123",
    name="Alice",
    role=UserRole.OWNER,
    images=images,
    consent_granted=True  # User explicitly agreed
)
```

### Recognize a User

```python
image = camera.capture()
result = recognizer.recognize(image)

if result.recognized:
    print(f"Hello {result.user.name}! (Role: {result.user.role.value})")
    print(f"Confidence: {result.confidence:.2f}")
else:
    print("Unknown user - guest mode")
```

### Revoke Consent (Right to be Forgotten)

```python
# Delete all face data for a user
recognizer.revoke_consent("user_123")  # Keeps user record, deletes faces
# or
recognizer.delete_user("user_123")  # Complete deletion
```

## Consent Management

```python
from continuonbrain.recognition import UserConsentManager, ConsentType

consent_mgr = UserConsentManager()

# Request consent (returns info to show user)
request = consent_mgr.request_consent(
    user_id="user_123",
    consent_type=ConsentType.FACE_RECOGNITION,
)
# Show request to user via app/display

# Record consent decision
consent_mgr.grant_consent(
    user_id="user_123",
    consent_type=ConsentType.FACE_RECOGNITION,
    method="app",  # How they consented
)

# Check consent before using features
if consent_mgr.has_consent("user_123", ConsentType.FACE_RECOGNITION):
    result = recognizer.recognize(image)
```

## Consent Types

| Type | What It Enables |
|------|-----------------|
| `FACE_RECOGNITION` | Robot recognizes user's face |
| `FACE_STORAGE` | Face embeddings stored locally |
| `VOICE_RECOGNITION` | Robot recognizes user's voice |
| `BEHAVIOR_LEARNING` | Robot learns user preferences |
| `DATA_SYNC` | Data synced to cloud (with encryption) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recognition/recognize` | POST | Recognize face in image |
| `/api/recognition/register` | POST | Register new user |
| `/api/recognition/users` | GET | List known users |
| `/api/recognition/revoke/{user_id}` | DELETE | Revoke consent |
| `/api/consent/status/{user_id}` | GET | Get consent status |
| `/api/consent/grant` | POST | Grant consent |
| `/api/consent/revoke` | POST | Revoke consent |

## Data Storage

All face data is stored locally in:
```
/opt/continuonos/brain/recognition/
├── known_users.pkl     # User records with embeddings
└── consent_records.json # Consent audit trail
```

## Technical Details

### Face Embedding
- **Dimension**: 128-float vector (compact)
- **Model**: FaceNet-lite or similar (on-device)
- **Detection**: MediaPipe Face Detection
- **Threshold**: 0.7 similarity for match

### Privacy Safeguards
- No raw images stored
- Embeddings cannot be reversed to images
- All processing on-device
- Encrypted at rest (future)

## Integration with RCAN Protocol

The recognition module integrates with the Robot Communication & Addressing Network (RCAN) protocol:

```python
from continuonbrain.services.rcan_service import RCANService
from continuonbrain.recognition import FaceRecognizer

recognizer = FaceRecognizer()
rcan = RCANService()

# Recognize user and get their RCAN permissions
result = recognizer.recognize(image)
if result.recognized:
    role = result.user.role
    permissions = rcan.get_permissions_for_role(role)
```

## Safety Notes

1. **Ring 0 Safety** is not bypassed by recognition - safety limits apply to all users
2. **Creator role** can override some safety parameters (with logging)
3. **Unknown users** are treated as guests with maximum restrictions
4. **Failed recognition** does not lock out - guest mode is always available

