"""
Security primitives for ContinuonBrain.
Defines User Roles and authentication structures.
"""
from enum import Enum, auto

class UserRole(str, Enum):
    """
    Defines the role-based access levels for the ContinuonBrain.
    """
    CREATOR = "creator"           # Super Admin / Owner
    DEVELOPER = "developer"       # Internal contributor
    CONSUMER = "consumer"         # Standard owner / User
    LESSEE = "lessee"             # Temporary user
    FLEET_MANAGER = "fleet"       # Enterprise fleet manager
    UNKNOWN = "unknown"           # Unauthenticated or unrecognized
