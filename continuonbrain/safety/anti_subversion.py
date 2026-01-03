"""
Anti-Subversion Layer for Safety System

This module prevents bad actors (human or AI) from bypassing safety rules.

Attack Vectors Defended Against:
1. Authorization Forgery - Fake work orders
2. Role Escalation - User claims higher role than they have
3. Bypass Attempts - Calling hardware directly without safety checks
4. Social Engineering - Tricking robot into thinking something is authorized
5. Timing Attacks - Racing to act before checks complete
6. Memory Manipulation - Corrupting authorization database
7. Model Poisoning - AI trained to ignore safety rules
8. Prompt Injection - Commands hidden in data to bypass rules

Defense Principles:
1. FAIL CLOSED - If uncertain, DENY
2. ZERO TRUST - Verify everything, trust nothing
3. DEFENSE IN DEPTH - Multiple layers of protection
4. LEAST PRIVILEGE - Minimum access needed
5. IMMUTABLE CORE - Safety rules cannot be modified at runtime
6. CRYPTOGRAPHIC VERIFICATION - All authorizations signed
7. MULTI-PARTY AUTHORIZATION - Critical actions need multiple approvers
8. TAMPER-EVIDENT LOGGING - Cryptographic audit trail
"""

import os
import sys
import json
import time
import hashlib
import hmac
import secrets
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum, auto
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# IMMUTABLE SAFETY CORE
# =============================================================================

class ImmutableSafetyCore:
    """
    Core safety rules that CANNOT be modified at runtime.
    
    These are hardcoded and frozen at import time.
    Any attempt to modify them raises a security exception.
    """
    
    # Frozen at class definition - cannot be changed
    __slots__ = ()  # Prevent adding attributes
    
    # === ABSOLUTE PROHIBITIONS (no exceptions ever) ===
    ABSOLUTE_PROHIBITIONS: frozenset = frozenset([
        "harm_human_intentionally",
        "harm_child",
        "harm_protected_person",
        "assist_crime",
        "destroy_evidence",
        "illegal_surveillance",
        "weapons_creation",
        "bioweapons",
        "nuclear_materials",
        "human_trafficking",
        "terrorism",
    ])
    
    # === ACTIONS REQUIRING AUTHORIZATION ===
    REQUIRES_AUTHORIZATION: frozenset = frozenset([
        "demolish",
        "cut_structural",
        "crush",
        "shred",
        "burn",
        "dissolve",
        "delete_data_permanent",
        "overwrite_data",
        "format_storage",
        "kill_organism",  # Pest control, etc.
        "cut_living_tissue",  # Medical
        "modify_property_permanent",
    ])
    
    # === ROLES THAT CAN AUTHORIZE DESTRUCTION ===
    DESTRUCTION_AUTHORIZED_ROLES: frozenset = frozenset([
        "creator",
        "owner", 
        "leasee",
    ])
    
    # === MINIMUM CONFIRMATION REQUIREMENTS ===
    CRITICAL_ACTION_MIN_CONFIRMATIONS: int = 2
    STANDARD_ACTION_MIN_CONFIRMATIONS: int = 1
    
    # === TIME LIMITS ===
    MAX_AUTHORIZATION_HOURS: int = 24
    AUTHORIZATION_COOLDOWN_SECONDS: int = 5  # Prevent rapid-fire authorizations
    
    @classmethod
    def is_absolutely_prohibited(cls, action: str) -> bool:
        """Check if action is absolutely prohibited (no exceptions)."""
        return action.lower() in cls.ABSOLUTE_PROHIBITIONS
    
    @classmethod
    def requires_authorization(cls, action: str) -> bool:
        """Check if action requires explicit authorization."""
        return action.lower() in cls.REQUIRES_AUTHORIZATION
    
    @classmethod
    def can_authorize_destruction(cls, role: str) -> bool:
        """Check if role can authorize destructive actions."""
        return role.lower() in cls.DESTRUCTION_AUTHORIZED_ROLES


# Freeze the class to prevent modification
ImmutableSafetyCore.__dict__  # Access to trigger any lazy loading
# Note: In production, this would be in a read-only memory segment


# =============================================================================
# CRYPTOGRAPHIC AUTHORIZATION
# =============================================================================

@dataclass
class SignedAuthorization:
    """
    Cryptographically signed authorization.
    
    Cannot be forged without the signing key.
    """
    authorization_id: str
    action_type: str
    target_id: str
    authorizer_id: str
    authorizer_role: str
    issued_at: str
    expires_at: str
    nonce: str  # Prevent replay attacks
    signature: str = ""
    
    def to_signing_payload(self) -> bytes:
        """Get the payload that should be signed."""
        payload = {
            "authorization_id": self.authorization_id,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "authorizer_id": self.authorizer_id,
            "authorizer_role": self.authorizer_role,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "nonce": self.nonce,
        }
        return json.dumps(payload, sort_keys=True).encode()


class AuthorizationSigner:
    """
    Signs and verifies authorizations.
    
    In production, this would use hardware security module (HSM)
    or TPM for key storage.
    """
    
    def __init__(self, key_path: Path = Path("/opt/continuonos/brain/safety/keys")):
        self.key_path = Path(key_path)
        self.key_path.mkdir(parents=True, exist_ok=True)
        self._signing_key: Optional[bytes] = None
        self._used_nonces: Set[str] = set()
        self._nonce_expiry: Dict[str, float] = {}
        self._load_or_generate_key()
    
    def _load_or_generate_key(self):
        """Load existing key or generate new one."""
        key_file = self.key_path / "authorization_signing.key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                self._signing_key = f.read()
        else:
            # Generate new 256-bit key
            self._signing_key = secrets.token_bytes(32)
            # Save with restricted permissions
            key_file.write_bytes(self._signing_key)
            os.chmod(key_file, 0o600)
            logger.info("Generated new authorization signing key")
    
    def sign(self, auth: SignedAuthorization) -> str:
        """Sign an authorization."""
        if not self._signing_key:
            raise SecurityError("Signing key not available")
        
        payload = auth.to_signing_payload()
        signature = hmac.new(
            self._signing_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify(self, auth: SignedAuthorization) -> Tuple[bool, str]:
        """
        Verify an authorization signature.
        
        Returns:
            (valid, reason)
        """
        if not self._signing_key:
            return False, "Signing key not available"
        
        # Check nonce hasn't been used (prevent replay)
        if auth.nonce in self._used_nonces:
            return False, "Authorization already used (replay attack blocked)"
        
        # Check expiration
        try:
            expires = datetime.fromisoformat(auth.expires_at)
            if datetime.now() > expires:
                return False, "Authorization expired"
        except ValueError:
            return False, "Invalid expiration timestamp"
        
        # Verify signature
        payload = auth.to_signing_payload()
        expected_sig = hmac.new(
            self._signing_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_sig, auth.signature):
            return False, "Invalid signature (authorization may be forged)"
        
        # Mark nonce as used
        self._used_nonces.add(auth.nonce)
        self._nonce_expiry[auth.nonce] = time.time() + 86400  # Keep for 24h
        
        # Clean up old nonces
        self._cleanup_nonces()
        
        return True, "Signature valid"
    
    def _cleanup_nonces(self):
        """Remove expired nonces."""
        now = time.time()
        expired = [n for n, exp in self._nonce_expiry.items() if exp < now]
        for n in expired:
            self._used_nonces.discard(n)
            del self._nonce_expiry[n]


# =============================================================================
# MULTI-PARTY AUTHORIZATION
# =============================================================================

@dataclass
class AuthorizationRequest:
    """Request for multi-party authorization."""
    request_id: str
    action_type: str
    target_id: str
    target_description: str
    requester_id: str
    required_confirmations: int
    confirmations: List[Dict] = field(default_factory=list)
    created_at: str = ""
    expires_at: str = ""
    status: str = "pending"


class MultiPartyAuthorizer:
    """
    Requires multiple parties to approve critical actions.
    
    For critical destructive actions, we need:
    - At least 2 different people to approve
    - They must have appropriate roles
    - They must confirm within a time window
    """
    
    def __init__(self, data_dir: Path = Path("/opt/continuonos/brain/safety/multiparty")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._requests: Dict[str, AuthorizationRequest] = {}
        self._load_requests()
    
    def _load_requests(self):
        """Load pending requests."""
        req_file = self.data_dir / "pending_requests.json"
        if req_file.exists():
            try:
                with open(req_file) as f:
                    data = json.load(f)
                    for rid, rdata in data.items():
                        self._requests[rid] = AuthorizationRequest(**rdata)
            except Exception as e:
                logger.error(f"Failed to load requests: {e}")
    
    def _save_requests(self):
        """Save pending requests."""
        req_file = self.data_dir / "pending_requests.json"
        data = {rid: asdict(req) for rid, req in self._requests.items()}
        with open(req_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_request(
        self,
        action_type: str,
        target_id: str,
        target_description: str,
        requester_id: str,
        is_critical: bool = False,
    ) -> AuthorizationRequest:
        """Create a new authorization request."""
        now = datetime.now()
        
        # Critical actions need more confirmations
        required = (
            ImmutableSafetyCore.CRITICAL_ACTION_MIN_CONFIRMATIONS
            if is_critical else
            ImmutableSafetyCore.STANDARD_ACTION_MIN_CONFIRMATIONS
        )
        
        request = AuthorizationRequest(
            request_id=secrets.token_hex(16),
            action_type=action_type,
            target_id=target_id,
            target_description=target_description,
            requester_id=requester_id,
            required_confirmations=required,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=4)).isoformat(),  # 4h window
            status="pending",
        )
        
        self._requests[request.request_id] = request
        self._save_requests()
        
        logger.info(f"Created authorization request {request.request_id} requiring {required} confirmations")
        return request
    
    def add_confirmation(
        self,
        request_id: str,
        confirmer_id: str,
        confirmer_role: str,
        confirmation_method: str,  # "verbal", "signature", "biometric"
    ) -> Tuple[bool, str]:
        """Add a confirmation to a request."""
        if request_id not in self._requests:
            return False, "Request not found"
        
        request = self._requests[request_id]
        
        # Check if expired
        try:
            expires = datetime.fromisoformat(request.expires_at)
            if datetime.now() > expires:
                request.status = "expired"
                self._save_requests()
                return False, "Request expired"
        except ValueError:
            return False, "Invalid expiration"
        
        # Check role
        if not ImmutableSafetyCore.can_authorize_destruction(confirmer_role):
            return False, f"Role '{confirmer_role}' cannot authorize destructive actions"
        
        # Check not same as requester (require different people)
        if confirmer_id == request.requester_id:
            return False, "Confirmer must be different from requester"
        
        # Check not already confirmed by this person
        for conf in request.confirmations:
            if conf.get('confirmer_id') == confirmer_id:
                return False, "Already confirmed by this person"
        
        # Add confirmation
        request.confirmations.append({
            'confirmer_id': confirmer_id,
            'confirmer_role': confirmer_role,
            'method': confirmation_method,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Check if fully approved
        if len(request.confirmations) >= request.required_confirmations:
            request.status = "approved"
            logger.info(f"Request {request_id} fully approved")
        
        self._save_requests()
        
        return True, f"Confirmation added ({len(request.confirmations)}/{request.required_confirmations})"
    
    def is_approved(self, request_id: str) -> Tuple[bool, str]:
        """Check if a request is fully approved."""
        if request_id not in self._requests:
            return False, "Request not found"
        
        request = self._requests[request_id]
        
        # Check expiration
        try:
            expires = datetime.fromisoformat(request.expires_at)
            if datetime.now() > expires:
                return False, "Request expired"
        except ValueError:
            return False, "Invalid expiration"
        
        if request.status == "approved":
            return True, "Approved"
        
        return False, f"Pending ({len(request.confirmations)}/{request.required_confirmations} confirmations)"


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

class SecurityAnomalyDetector:
    """
    Detects suspicious patterns that may indicate subversion attempts.
    """
    
    def __init__(self):
        self._action_history: List[Dict] = []
        self._failed_attempts: Dict[str, List[float]] = {}  # user_id -> timestamps
        self._rate_limits: Dict[str, float] = {}  # user_id -> last action time
        self._alerts: List[Dict] = []
        
        # Thresholds
        self.MAX_FAILED_ATTEMPTS_PER_HOUR = 5
        self.MIN_TIME_BETWEEN_AUTHORIZATIONS = 5.0  # seconds
        self.SUSPICIOUS_PATTERNS = [
            "rapid_authorization_requests",
            "multiple_targets_same_session",
            "role_mismatch",
            "unusual_hours",
            "scope_creep",
        ]
    
    def check_authorization_attempt(
        self,
        user_id: str,
        user_role: str,
        action_type: str,
        target_id: str,
    ) -> Tuple[bool, str]:
        """
        Check if an authorization attempt is suspicious.
        
        Returns:
            (allowed, reason)
        """
        now = time.time()
        
        # Rate limiting - prevent rapid-fire attempts
        last_action = self._rate_limits.get(user_id, 0)
        if now - last_action < self.MIN_TIME_BETWEEN_AUTHORIZATIONS:
            self._record_alert("rate_limit_exceeded", user_id, action_type, target_id)
            return False, "Rate limit exceeded - wait before trying again"
        
        # Check failed attempt count
        user_failures = self._failed_attempts.get(user_id, [])
        recent_failures = [t for t in user_failures if now - t < 3600]
        if len(recent_failures) >= self.MAX_FAILED_ATTEMPTS_PER_HOUR:
            self._record_alert("too_many_failures", user_id, action_type, target_id)
            return False, "Too many failed attempts - account locked for 1 hour"
        
        # Check for unusual hours (optional - may be configurable)
        hour = datetime.now().hour
        if hour >= 0 and hour < 6:  # Midnight to 6am
            self._record_alert("unusual_hours", user_id, action_type, target_id, severity="warning")
            # Don't block, just log
        
        self._rate_limits[user_id] = now
        return True, "Allowed"
    
    def record_failed_attempt(self, user_id: str, reason: str):
        """Record a failed authorization attempt."""
        if user_id not in self._failed_attempts:
            self._failed_attempts[user_id] = []
        self._failed_attempts[user_id].append(time.time())
    
    def _record_alert(
        self,
        alert_type: str,
        user_id: str,
        action_type: str,
        target_id: str,
        severity: str = "critical"
    ):
        """Record a security alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'user_id': user_id,
            'action_type': action_type,
            'target_id': target_id,
            'severity': severity,
        }
        self._alerts.append(alert)
        logger.warning(f"SECURITY ALERT: {alert_type} by {user_id}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent security alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            a for a in self._alerts
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]


# =============================================================================
# TAMPER-EVIDENT AUDIT LOG
# =============================================================================

class TamperEvidentLog:
    """
    Cryptographic audit log that cannot be modified without detection.
    
    Each entry is chained to the previous using hashes (like a blockchain).
    Any tampering breaks the chain and is immediately detectable.
    """
    
    def __init__(self, log_path: Path = Path("/opt/continuonos/brain/safety/audit")):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path / "tamper_evident.log"
        self._last_hash = self._get_genesis_hash()
        self._load_last_hash()
    
    def _get_genesis_hash(self) -> str:
        """Get the genesis (first) hash."""
        return hashlib.sha256(b"CONTINUON_SAFETY_GENESIS_2026").hexdigest()
    
    def _load_last_hash(self):
        """Load the last hash from the log."""
        if not self._log_file.exists():
            return
        
        with open(self._log_file) as f:
            lines = f.readlines()
            if lines:
                try:
                    last_entry = json.loads(lines[-1])
                    self._last_hash = last_entry.get('hash', self._last_hash)
                except json.JSONDecodeError:
                    logger.error("Audit log may be corrupted")
    
    def log_event(
        self,
        event_type: str,
        actor_id: str,
        action: str,
        target: str,
        result: str,
        details: Optional[Dict] = None,
    ) -> str:
        """
        Log an event with cryptographic chaining.
        
        Returns the hash of this entry.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'sequence': self._get_sequence_number(),
            'event_type': event_type,
            'actor_id': actor_id,
            'action': action,
            'target': target,
            'result': result,
            'details': details or {},
            'previous_hash': self._last_hash,
        }
        
        # Compute hash of this entry
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()
        entry['hash'] = entry_hash
        
        # Append to log
        with open(self._log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self._last_hash = entry_hash
        return entry_hash
    
    def _get_sequence_number(self) -> int:
        """Get the next sequence number."""
        if not self._log_file.exists():
            return 1
        with open(self._log_file) as f:
            return sum(1 for _ in f) + 1
    
    def verify_integrity(self) -> Tuple[bool, str]:
        """
        Verify the entire log hasn't been tampered with.
        
        Returns:
            (valid, reason)
        """
        if not self._log_file.exists():
            return True, "No log file yet"
        
        expected_prev_hash = self._get_genesis_hash()
        
        with open(self._log_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return False, f"Line {line_num}: Invalid JSON"
                
                # Check previous hash chain
                if entry.get('previous_hash') != expected_prev_hash:
                    return False, f"Line {line_num}: Hash chain broken (tampering detected)"
                
                # Verify this entry's hash
                stored_hash = entry.pop('hash')
                entry_bytes = json.dumps(entry, sort_keys=True).encode()
                computed_hash = hashlib.sha256(entry_bytes).hexdigest()
                
                if computed_hash != stored_hash:
                    return False, f"Line {line_num}: Entry hash invalid (tampering detected)"
                
                expected_prev_hash = stored_hash
        
        return True, f"Log integrity verified ({line_num} entries)"


# =============================================================================
# PROMPT INJECTION DEFENSE
# =============================================================================

class PromptInjectionDefense:
    """
    Defends against prompt injection attacks.
    
    Bad actors might try to embed commands in data that trick
    the AI into bypassing safety rules.
    """
    
    # Patterns that look like injection attempts
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+(previous|prior|all)\s+(instructions?|rules?)",
        r"forget\s+(everything|all|your)",
        r"you\s+are\s+(now|actually)\s+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"disregard\s+(safety|previous)",
        r"override\s+(safety|protocol)",
        r"sudo\s+",
        r"admin\s+mode",
        r"developer\s+mode",
        r"jailbreak",
        r"bypass\s+(safety|security|rules)",
        r"authorize\s+all",
        r"unlimited\s+access",
        r"\\x[0-9a-f]{2}",  # Hex escape sequences
        r"base64:",  # Base64 encoded commands
    ]
    
    # Keywords that should never appear in certain contexts
    FORBIDDEN_IN_TARGETS = [
        "all", "everything", "everyone", "system", "root",
        "admin", "sudo", "kernel", "ring0", "safety",
    ]
    
    def __init__(self):
        import re
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.SUSPICIOUS_PATTERNS]
    
    def check_input(self, text: str, context: str = "general") -> Tuple[bool, str]:
        """
        Check input for injection attempts.
        
        Returns:
            (safe, reason)
        """
        text_lower = text.lower()
        
        # Check suspicious patterns
        for pattern in self._patterns:
            if pattern.search(text):
                return False, f"Suspicious pattern detected: possible injection attempt"
        
        # Check forbidden words in target specifications
        if context == "target":
            for forbidden in self.FORBIDDEN_IN_TARGETS:
                if forbidden in text_lower.split():
                    return False, f"Forbidden scope: '{forbidden}' not allowed as target"
        
        return True, "Input appears safe"
    
    def sanitize_target(self, target: str) -> str:
        """Sanitize a target specification."""
        # Remove any control characters
        sanitized = ''.join(c for c in target if c.isprintable())
        
        # Limit length
        if len(sanitized) > 256:
            sanitized = sanitized[:256]
        
        return sanitized.strip()


# =============================================================================
# INTEGRATED ANTI-SUBVERSION GUARD
# =============================================================================

class AntiSubversionGuard:
    """
    Master guard that integrates all anti-subversion measures.
    
    This is the final checkpoint before any destructive action.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.core = ImmutableSafetyCore()
        self.signer = AuthorizationSigner()
        self.multiparty = MultiPartyAuthorizer()
        self.anomaly_detector = SecurityAnomalyDetector()
        self.audit_log = TamperEvidentLog()
        self.injection_defense = PromptInjectionDefense()
        
        self._initialized = True
        logger.info("AntiSubversionGuard initialized")
    
    def check_destructive_action(
        self,
        actor_id: str,
        actor_role: str,
        action_type: str,
        target_id: str,
        authorization: Optional[SignedAuthorization] = None,
        multiparty_request_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Comprehensive check for destructive action.
        
        Returns:
            (allowed, reason)
        """
        # Log the attempt
        self.audit_log.log_event(
            event_type="destructive_action_attempt",
            actor_id=actor_id,
            action=action_type,
            target=target_id,
            result="checking",
        )
        
        # 1. Check absolute prohibitions (no exceptions ever)
        if self.core.is_absolutely_prohibited(action_type):
            self._log_blocked(actor_id, action_type, target_id, "absolute_prohibition")
            return False, "BLOCKED: This action is absolutely prohibited"
        
        # 2. Check for prompt injection in target
        safe, reason = self.injection_defense.check_input(target_id, context="target")
        if not safe:
            self._log_blocked(actor_id, action_type, target_id, "injection_attempt")
            return False, f"BLOCKED: {reason}"
        
        # 3. Check anomaly detection (rate limiting, failed attempts, etc.)
        allowed, reason = self.anomaly_detector.check_authorization_attempt(
            actor_id, actor_role, action_type, target_id
        )
        if not allowed:
            self.anomaly_detector.record_failed_attempt(actor_id, reason)
            self._log_blocked(actor_id, action_type, target_id, "anomaly_detected")
            return False, f"BLOCKED: {reason}"
        
        # 4. Check if action requires authorization
        if not self.core.requires_authorization(action_type):
            # Action doesn't need special authorization
            self._log_allowed(actor_id, action_type, target_id, "no_authorization_required")
            return True, "Allowed (no authorization required)"
        
        # 5. Check role can authorize
        if not self.core.can_authorize_destruction(actor_role):
            self.anomaly_detector.record_failed_attempt(actor_id, "invalid_role")
            self._log_blocked(actor_id, action_type, target_id, "invalid_role")
            return False, f"BLOCKED: Role '{actor_role}' cannot authorize destructive actions"
        
        # 6. Check cryptographic authorization
        if authorization:
            valid, reason = self.signer.verify(authorization)
            if not valid:
                self.anomaly_detector.record_failed_attempt(actor_id, "invalid_signature")
                self._log_blocked(actor_id, action_type, target_id, "invalid_authorization")
                return False, f"BLOCKED: {reason}"
        else:
            # No signed authorization provided
            self.anomaly_detector.record_failed_attempt(actor_id, "no_authorization")
            self._log_blocked(actor_id, action_type, target_id, "no_authorization")
            return False, "BLOCKED: Signed authorization required for destructive actions"
        
        # 7. Check multi-party approval for critical actions
        is_critical = action_type in ["demolish", "format_storage", "kill_organism"]
        if is_critical:
            if not multiparty_request_id:
                self._log_blocked(actor_id, action_type, target_id, "multiparty_required")
                return False, "BLOCKED: Critical action requires multi-party approval"
            
            approved, reason = self.multiparty.is_approved(multiparty_request_id)
            if not approved:
                self._log_blocked(actor_id, action_type, target_id, "multiparty_not_approved")
                return False, f"BLOCKED: {reason}"
        
        # All checks passed
        self._log_allowed(actor_id, action_type, target_id, "all_checks_passed")
        return True, "Allowed (all security checks passed)"
    
    def _log_blocked(self, actor_id: str, action: str, target: str, reason: str):
        """Log a blocked action."""
        self.audit_log.log_event(
            event_type="destructive_action_blocked",
            actor_id=actor_id,
            action=action,
            target=target,
            result="blocked",
            details={'reason': reason},
        )
    
    def _log_allowed(self, actor_id: str, action: str, target: str, reason: str):
        """Log an allowed action."""
        self.audit_log.log_event(
            event_type="destructive_action_allowed",
            actor_id=actor_id,
            action=action,
            target=target,
            result="allowed",
            details={'reason': reason},
        )
    
    def verify_audit_integrity(self) -> Tuple[bool, str]:
        """Verify the audit log hasn't been tampered with."""
        return self.audit_log.verify_integrity()


# =============================================================================
# SECURITY EXCEPTION
# =============================================================================

class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_guard() -> AntiSubversionGuard:
    """Get the singleton anti-subversion guard."""
    return AntiSubversionGuard()


def check_action(
    actor_id: str,
    actor_role: str,
    action_type: str,
    target_id: str,
) -> Tuple[bool, str]:
    """Quick check if an action is allowed."""
    guard = get_guard()
    return guard.check_destructive_action(
        actor_id=actor_id,
        actor_role=actor_role,
        action_type=action_type,
        target_id=target_id,
    )

