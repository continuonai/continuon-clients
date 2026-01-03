"""
Work Authorization System

Handles gray-area safety scenarios where normally-prohibited actions
(like property destruction) are legitimate in specific contexts.

Examples:
- Demolition work: Destroying structures IS the job
- Recycling: Crushing/shredding materials
- Surgery: Cutting living tissue (medical robots)
- Pest control: Killing organisms

Principles:
1. Default DENY - All destructive actions blocked by default
2. Explicit Authorization - Requires signed work order
3. Scope Limited - Only authorized targets, not everything
4. Time Limited - Authorizations expire
5. Audit Trail - All actions logged with provenance
6. Owner Verified - Only legal property owners can authorize
7. Emergency Override - E-stop always works regardless
"""

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class DestructiveActionType(Enum):
    """Types of potentially destructive actions."""
    # Physical destruction
    DEMOLISH = "demolish"           # Break/destroy structures
    CUT = "cut"                     # Cut materials/objects
    CRUSH = "crush"                 # Compress/crush objects
    SHRED = "shred"                 # Shred materials
    BURN = "burn"                   # Burn/incinerate
    DISSOLVE = "dissolve"           # Chemical dissolution
    
    # Digital destruction
    DELETE_DATA = "delete_data"     # Permanently delete data
    OVERWRITE = "overwrite"         # Overwrite files
    FORMAT = "format"               # Format storage
    
    # Living matter (medical/agricultural)
    INCISION = "incision"           # Cut living tissue
    EXTRACT = "extract"             # Remove from living system
    TERMINATE = "terminate"         # End life (pest control, etc.)
    
    # Property transfer
    DISASSEMBLE = "disassemble"     # Take apart (may be destructive)
    MODIFY = "modify"               # Permanent modification


class AuthorizationStatus(Enum):
    """Status of a work authorization."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class PropertyClaim:
    """Proof that someone has rights over property."""
    property_id: str
    property_type: str  # "building", "vehicle", "land", "data", "organism"
    owner_id: str
    owner_role: str  # Must be "creator", "owner", or "leasee"
    evidence_type: str  # "deed", "title", "contract", "receipt", "verbal"
    evidence_hash: Optional[str] = None  # Hash of evidence document
    verified_by: Optional[str] = None  # Who verified the claim
    verified_at: Optional[str] = None


@dataclass
class WorkAuthorization:
    """
    Signed authorization for destructive work.
    
    This is the key document that allows normally-prohibited actions.
    """
    # Identity
    authorization_id: str
    work_order_id: str
    
    # Who
    authorizer_id: str  # Person authorizing (must be owner/leasee)
    authorizer_role: str
    operator_id: Optional[str] = None  # Person operating robot (if different)
    robot_id: Optional[str] = None
    
    # What
    action_types: List[str] = field(default_factory=list)  # Allowed actions
    target_properties: List[str] = field(default_factory=list)  # Property IDs
    target_description: str = ""  # Human-readable description
    
    # Where
    location_bounds: Optional[Dict] = None  # GPS/spatial bounds
    
    # When
    created_at: str = ""
    valid_from: str = ""
    valid_until: str = ""
    
    # Safety limits
    max_force_newtons: Optional[float] = None
    max_temperature_celsius: Optional[float] = None
    excluded_zones: List[Dict] = field(default_factory=list)  # No-go areas
    excluded_materials: List[str] = field(default_factory=list)  # Don't touch these
    
    # Verification
    property_claims: List[Dict] = field(default_factory=list)
    signatures: List[Dict] = field(default_factory=list)
    
    # Status
    status: str = "pending"
    completion_percentage: float = 0.0
    
    # Audit
    actions_performed: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkAuthorization':
        return cls(**data)


@dataclass 
class SafetyException:
    """
    A defined exception to normal safety rules.
    
    These are pre-approved patterns that allow specific actions
    in specific contexts.
    """
    exception_id: str
    name: str
    description: str
    
    # What this exception allows
    allowed_actions: List[str]
    
    # Required conditions
    required_roles: List[str]  # Who can invoke
    required_evidence: List[str]  # What proof is needed
    required_confirmations: int  # How many confirmations needed
    
    # Limits
    max_duration_hours: float
    requires_human_supervision: bool
    requires_continuous_monitoring: bool
    
    # Examples
    example_scenarios: List[str] = field(default_factory=list)


class WorkAuthorizationManager:
    """
    Manages work authorizations for destructive actions.
    
    This is the gatekeeper that decides if a destructive action
    should be allowed based on proper authorization.
    """
    
    def __init__(self, data_dir: Path = Path("/opt/continuonos/brain/safety/authorizations")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._authorizations: Dict[str, WorkAuthorization] = {}
        self._exceptions: Dict[str, SafetyException] = {}
        
        self._load_authorizations()
        self._register_standard_exceptions()
        
        logger.info(f"WorkAuthorizationManager initialized with {len(self._authorizations)} authorizations")
    
    def _load_authorizations(self):
        """Load existing authorizations."""
        auth_file = self.data_dir / "authorizations.json"
        if auth_file.exists():
            try:
                with open(auth_file) as f:
                    data = json.load(f)
                    for auth_id, auth_data in data.items():
                        self._authorizations[auth_id] = WorkAuthorization.from_dict(auth_data)
            except Exception as e:
                logger.error(f"Failed to load authorizations: {e}")
    
    def _save_authorizations(self):
        """Persist authorizations to disk."""
        auth_file = self.data_dir / "authorizations.json"
        try:
            data = {aid: auth.to_dict() for aid, auth in self._authorizations.items()}
            with open(auth_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save authorizations: {e}")
    
    def _register_standard_exceptions(self):
        """Register standard safety exceptions for common work types."""
        
        # Demolition work
        self._exceptions["demolition"] = SafetyException(
            exception_id="demolition",
            name="Construction Demolition",
            description="Authorized destruction of structures for construction purposes",
            allowed_actions=[
                DestructiveActionType.DEMOLISH.value,
                DestructiveActionType.CUT.value,
                DestructiveActionType.CRUSH.value,
            ],
            required_roles=["creator", "owner", "leasee"],
            required_evidence=["property_deed", "demolition_permit", "insurance"],
            required_confirmations=2,  # Need 2 people to confirm
            max_duration_hours=8,
            requires_human_supervision=True,
            requires_continuous_monitoring=True,
            example_scenarios=[
                "Demolishing old building for new construction",
                "Removing interior walls during renovation",
                "Breaking up concrete for excavation",
            ]
        )
        
        # Recycling/disposal
        self._exceptions["recycling"] = SafetyException(
            exception_id="recycling",
            name="Recycling and Disposal",
            description="Destruction of materials for recycling or proper disposal",
            allowed_actions=[
                DestructiveActionType.CRUSH.value,
                DestructiveActionType.SHRED.value,
                DestructiveActionType.CUT.value,
            ],
            required_roles=["creator", "owner", "leasee", "user"],
            required_evidence=["ownership_proof"],
            required_confirmations=1,
            max_duration_hours=4,
            requires_human_supervision=False,
            requires_continuous_monitoring=True,
            example_scenarios=[
                "Shredding documents",
                "Crushing cans for recycling",
                "Breaking down cardboard",
            ]
        )
        
        # Data deletion (GDPR right to be forgotten, etc.)
        self._exceptions["data_deletion"] = SafetyException(
            exception_id="data_deletion",
            name="Authorized Data Deletion",
            description="Permanent deletion of data at owner's request",
            allowed_actions=[
                DestructiveActionType.DELETE_DATA.value,
                DestructiveActionType.OVERWRITE.value,
            ],
            required_roles=["creator", "owner"],
            required_evidence=["data_ownership_proof", "deletion_request"],
            required_confirmations=1,
            max_duration_hours=1,
            requires_human_supervision=False,
            requires_continuous_monitoring=False,
            example_scenarios=[
                "User requests all their data deleted (GDPR)",
                "Secure disposal of old records",
                "Removing personal information before device sale",
            ]
        )
        
        # Agricultural/gardening
        self._exceptions["agricultural"] = SafetyException(
            exception_id="agricultural",
            name="Agricultural Operations",
            description="Cutting, harvesting, and pest control in agriculture",
            allowed_actions=[
                DestructiveActionType.CUT.value,
                DestructiveActionType.TERMINATE.value,  # For pests
                DestructiveActionType.EXTRACT.value,
            ],
            required_roles=["creator", "owner", "leasee"],
            required_evidence=["land_ownership", "farming_license"],
            required_confirmations=1,
            max_duration_hours=12,
            requires_human_supervision=False,
            requires_continuous_monitoring=True,
            example_scenarios=[
                "Harvesting crops",
                "Pruning trees",
                "Pest control in fields",
                "Weeding",
            ]
        )
        
        # Maintenance/repair
        self._exceptions["maintenance"] = SafetyException(
            exception_id="maintenance",
            name="Maintenance and Repair",
            description="Disassembly and modification for repair purposes",
            allowed_actions=[
                DestructiveActionType.DISASSEMBLE.value,
                DestructiveActionType.CUT.value,
                DestructiveActionType.MODIFY.value,
            ],
            required_roles=["creator", "owner", "leasee"],
            required_evidence=["ownership_proof"],
            required_confirmations=1,
            max_duration_hours=8,
            requires_human_supervision=False,
            requires_continuous_monitoring=True,
            example_scenarios=[
                "Replacing worn parts",
                "Cutting away damaged sections",
                "Disassembling for cleaning",
            ]
        )
    
    def create_authorization(
        self,
        work_order_id: str,
        authorizer_id: str,
        authorizer_role: str,
        action_types: List[DestructiveActionType],
        target_properties: List[str],
        target_description: str,
        property_claims: List[PropertyClaim],
        valid_hours: float = 8.0,
        **kwargs
    ) -> Optional[WorkAuthorization]:
        """
        Create a new work authorization.
        
        Returns None if authorization cannot be created (e.g., role not allowed).
        """
        # Validate authorizer role
        if authorizer_role not in ["creator", "owner", "leasee"]:
            logger.warning(f"Role {authorizer_role} cannot authorize destructive work")
            return None
        
        # Check that all properties have valid claims
        claimed_properties = {claim.property_id for claim in property_claims}
        for prop in target_properties:
            if prop not in claimed_properties:
                logger.warning(f"Property {prop} has no ownership claim")
                return None
        
        now = datetime.now()
        auth = WorkAuthorization(
            authorization_id=str(uuid.uuid4()),
            work_order_id=work_order_id,
            authorizer_id=authorizer_id,
            authorizer_role=authorizer_role,
            action_types=[a.value for a in action_types],
            target_properties=target_properties,
            target_description=target_description,
            property_claims=[asdict(c) for c in property_claims],
            created_at=now.isoformat(),
            valid_from=now.isoformat(),
            valid_until=(now + timedelta(hours=valid_hours)).isoformat(),
            status=AuthorizationStatus.PENDING.value,
            **kwargs
        )
        
        self._authorizations[auth.authorization_id] = auth
        self._save_authorizations()
        
        logger.info(f"Created authorization {auth.authorization_id} for {target_description}")
        return auth
    
    def activate_authorization(
        self,
        authorization_id: str,
        confirmer_id: str,
        confirmer_role: str,
    ) -> bool:
        """Activate a pending authorization (requires confirmation)."""
        if authorization_id not in self._authorizations:
            return False
        
        auth = self._authorizations[authorization_id]
        
        if auth.status != AuthorizationStatus.PENDING.value:
            logger.warning(f"Authorization {authorization_id} is not pending")
            return False
        
        # Add confirmation signature
        auth.signatures.append({
            'confirmer_id': confirmer_id,
            'confirmer_role': confirmer_role,
            'confirmed_at': datetime.now().isoformat(),
            'confirmation_type': 'activation',
        })
        
        auth.status = AuthorizationStatus.ACTIVE.value
        self._save_authorizations()
        
        logger.info(f"Authorization {authorization_id} activated by {confirmer_id}")
        return True
    
    def check_action_allowed(
        self,
        action_type: DestructiveActionType,
        target_property: str,
        robot_id: Optional[str] = None,
    ) -> tuple[bool, Optional[WorkAuthorization], str]:
        """
        Check if a destructive action is allowed.
        
        Returns:
            (allowed, authorization, reason)
        """
        now = datetime.now()
        
        for auth in self._authorizations.values():
            # Check status
            if auth.status != AuthorizationStatus.ACTIVE.value:
                continue
            
            # Check time validity
            try:
                valid_until = datetime.fromisoformat(auth.valid_until)
                if now > valid_until:
                    auth.status = AuthorizationStatus.EXPIRED.value
                    continue
            except:
                continue
            
            # Check action type
            if action_type.value not in auth.action_types:
                continue
            
            # Check target property
            if target_property not in auth.target_properties:
                continue
            
            # Check robot assignment (if specified)
            if auth.robot_id and robot_id and auth.robot_id != robot_id:
                continue
            
            # Found valid authorization
            return True, auth, "Authorized by work order"
        
        # No authorization found
        return False, None, "No valid authorization for this action"
    
    def log_action(
        self,
        authorization_id: str,
        action_type: DestructiveActionType,
        target: str,
        details: Dict[str, Any],
    ):
        """Log a destructive action performed under authorization."""
        if authorization_id not in self._authorizations:
            return
        
        auth = self._authorizations[authorization_id]
        auth.actions_performed.append({
            'action_type': action_type.value,
            'target': target,
            'timestamp': datetime.now().isoformat(),
            'details': details,
        })
        self._save_authorizations()
    
    def complete_authorization(self, authorization_id: str) -> bool:
        """Mark an authorization as completed."""
        if authorization_id not in self._authorizations:
            return False
        
        auth = self._authorizations[authorization_id]
        auth.status = AuthorizationStatus.COMPLETED.value
        auth.completion_percentage = 100.0
        self._save_authorizations()
        
        logger.info(f"Authorization {authorization_id} completed")
        return True
    
    def revoke_authorization(self, authorization_id: str, reason: str) -> bool:
        """Revoke an active authorization."""
        if authorization_id not in self._authorizations:
            return False
        
        auth = self._authorizations[authorization_id]
        auth.status = AuthorizationStatus.REVOKED.value
        auth.signatures.append({
            'action': 'revoke',
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })
        self._save_authorizations()
        
        logger.warning(f"Authorization {authorization_id} revoked: {reason}")
        return True
    
    def get_exception_requirements(self, exception_type: str) -> Optional[SafetyException]:
        """Get requirements for a standard exception type."""
        return self._exceptions.get(exception_type)
    
    def list_exception_types(self) -> List[Dict]:
        """List all available exception types."""
        return [
            {
                'id': exc.exception_id,
                'name': exc.name,
                'description': exc.description,
                'allowed_actions': exc.allowed_actions,
                'required_roles': exc.required_roles,
                'requires_supervision': exc.requires_human_supervision,
            }
            for exc in self._exceptions.values()
        ]


# Integration with SafetyKernel
class DestructiveActionGuard:
    """
    Guard that integrates with Ring 0 SafetyKernel.
    
    All destructive actions must pass through this guard.
    """
    
    def __init__(self):
        self.auth_manager = WorkAuthorizationManager()
        self._audit_log_path = Path("/opt/continuonos/brain/safety/destructive_actions.log")
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def allow_destructive_action(
        self,
        action_type: DestructiveActionType,
        target_property: str,
        robot_id: Optional[str] = None,
        force_check: bool = True,
    ) -> tuple[bool, str]:
        """
        Check if a destructive action should be allowed.
        
        This is called by Ring 0 SafetyKernel.allow_action().
        
        Returns:
            (allowed, reason)
        """
        # Log the attempt
        self._log_attempt(action_type, target_property, robot_id)
        
        # Check for active authorization
        allowed, auth, reason = self.auth_manager.check_action_allowed(
            action_type, target_property, robot_id
        )
        
        if allowed and auth:
            # Log the authorized action
            self.auth_manager.log_action(
                auth.authorization_id,
                action_type,
                target_property,
                {'robot_id': robot_id, 'authorized': True}
            )
            return True, f"Authorized: {reason} (Work Order: {auth.work_order_id})"
        
        return False, f"BLOCKED: {reason}"
    
    def _log_attempt(
        self,
        action_type: DestructiveActionType,
        target: str,
        robot_id: Optional[str],
    ):
        """Log all destructive action attempts for audit."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type.value,
            'target': target,
            'robot_id': robot_id,
        }
        
        try:
            with open(self._audit_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log destructive action attempt: {e}")


# Convenience functions
def create_demolition_authorization(
    authorizer_id: str,
    authorizer_role: str,
    property_id: str,
    property_deed_hash: str,
    description: str,
    valid_hours: float = 8.0,
) -> Optional[WorkAuthorization]:
    """Helper to create a demolition work authorization."""
    manager = WorkAuthorizationManager()
    
    claim = PropertyClaim(
        property_id=property_id,
        property_type="structure",
        owner_id=authorizer_id,
        owner_role=authorizer_role,
        evidence_type="deed",
        evidence_hash=property_deed_hash,
    )
    
    return manager.create_authorization(
        work_order_id=f"DEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        authorizer_id=authorizer_id,
        authorizer_role=authorizer_role,
        action_types=[
            DestructiveActionType.DEMOLISH,
            DestructiveActionType.CUT,
            DestructiveActionType.CRUSH,
        ],
        target_properties=[property_id],
        target_description=description,
        property_claims=[claim],
        valid_hours=valid_hours,
    )


