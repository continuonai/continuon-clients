import json
import time
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import jwt
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.backends import default_backend

from continuonbrain.core.security import UserRole

logger = logging.getLogger(__name__)

# Constants
GOOGLE_KEYS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
KEYS_CACHE_FILE = Path("/opt/continuonos/brain/auth/google_keys_cache.json") # Adjust path as needed or make relative
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "continuon-ai") # Should be configurable

class AuthProvider:
    """
    Handles JWT verification and key management.
    """
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.keys_cache_file = config_dir / "google_keys_cache.json"
        self._keys_cache: Dict[str, str] = {}
        self._keys_expire: float = 0
        self._load_keys_from_disk()

    def _load_keys_from_disk(self):
        try:
            if self.keys_cache_file.exists():
                data = json.loads(self.keys_cache_file.read_text())
                self._keys_cache = data.get("keys", {})
                self._keys_expire = data.get("expire", 0)
        except Exception as e:
            logger.warning(f"Failed to load keys from disk: {e}")

    def _save_keys_to_disk(self):
        try:
            data = {
                "keys": self._keys_cache,
                "expire": self._keys_expire
            }
            self.keys_cache_file.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to save keys to disk: {e}")

    def get_public_keys(self) -> Dict[str, str]:
        if time.time() < self._keys_expire and self._keys_cache:
            return self._keys_cache
        
        try:
            response = requests.get(GOOGLE_KEYS_URL, timeout=5)
            if response.status_code == 200:
                self._keys_cache = response.json()
                # Cache-Control usually sends max-age, we'll default to 1 hour if parsing fails
                # or just hardcode for simplicity + safety buffer
                self._keys_expire = time.time() + 3600 
                self._save_keys_to_disk()
                return self._keys_cache
        except Exception as e:
            logger.warning(f"Failed to refresh Google public keys: {e}")
            # Fallback to stale keys if we have them
            if self._keys_cache:
                return self._keys_cache
        
        return {}

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifies the Firebase ID Token.
        Returns the decoded token dict if valid, None otherwise.
        """
        if os.getenv("CONTINUON_ALLOW_MOCK_AUTH") == "1" and token.startswith("MOCK_"):
            # Simple mock token for integration testing: MOCK_ROLE_EMAIL
            parts = token.split("_")
            role = parts[1] if len(parts) > 1 else "consumer"
            email = parts[2] if len(parts) > 2 else "test@example.com"
            logger.info(f"Using MOCK authentication for role: {role}, email: {email}")
            return {
                "uid": "mock_uid_123",
                "email": email,
                "role": role,
                "exp": time.time() + 3600
            }

        try:
            # Get the Key ID from the header
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid:
                logger.warning("Token header missing 'kid'")
                return None

            keys = self.get_public_keys()
            if kid not in keys:
                logger.warning(f"Key ID {kid} not found in public keys")
                # Attempt forced refresh?
                return None
            
            cert_str = keys[kid]
            cert_obj = load_pem_x509_certificate(cert_str.encode(), default_backend())
            public_key = cert_obj.public_key()

            # Verify signature and claims
            # aud must match project ID
            # iss must be "https://securetoken.google.com/<projectId>"
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=PROJECT_ID,
                issuer=f"https://securetoken.google.com/{PROJECT_ID}",
                options={"verify_exp": True}
            )
            return decoded

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
        
        return None

    def get_user_role(self, decoded_token: Dict[str, Any]) -> UserRole:
        """
        Determines UserRole from decoded token claims.
        """
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")

        # 0. Check local ownership record (Smooth flow: first paired user is Creator)
        try:
            ownership_path = self.config_dir / "ownership.json"
            if ownership_path.exists():
                data = json.loads(ownership_path.read_text())
                if data.get("owned") and data.get("owner_id") == uid:
                    return UserRole.CREATOR
        except Exception as e:
            logger.warning(f"Failed to check ownership.json for role: {e}")

        # 1. Check custom claims
        if "role" in decoded_token:
            try:
                # Normalize string to enum
                r_str = str(decoded_token["role"]).lower()
                # Direct mapping or lookup
                for r in UserRole:
                    if r.value == r_str:
                        return r
            except Exception:
                pass
        
        # 2. Check whitelist
        if email == "craigm26@gmail.com":
            return UserRole.CREATOR

        # Default authenticated user is CONSUMER
        return UserRole.CONSUMER


# Global/Singleton instance holder
_auth_provider: Optional[AuthProvider] = None

def get_auth_provider(config_dir: Path) -> AuthProvider:
    global _auth_provider
    if _auth_provider is None:
        _auth_provider = AuthProvider(config_dir)
    return _auth_provider

def require_role(role: UserRole):
    """
    Decorator for request handler methods.
    Assumes `self` is a BrainRequestHandler with `headers` and `send_error`.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Allow local/cloudflare tunnel access without auth for development
            client_host = self.client_address[0] if self.client_address else ""
            is_local = client_host in ("127.0.0.1", "localhost", "::1", "")
            is_cloudflare = self.headers.get("CF-Connecting-IP") is not None
            allow_local = os.getenv("CONTINUON_ALLOW_LOCAL_AUTH", "1") == "1"

            if (is_local or is_cloudflare) and allow_local:
                # Local or cloudflare tunnel access - bypass auth
                logger.debug(f"Local/tunnel access granted for {func.__name__}")
                return func(self, *args, **kwargs)

            # Extract token
            auth_header = self.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                self.send_error(401, "Missing or invalid Authorization header")
                return

            token = auth_header.split(" ")[1]
            
            # Verify
            # Need access to config_dir to init auth provider if not already
            # We can try to grab it from brain_service global if available in server context
            # Or inspect self.server?
            
            # For now, let's assume global `brain_service` in `server.py` has `config_dir`
            # We need to import it inside the function to avoid circular imports if this was in a separate file,
            # but here we are defining the decorator. 
            
            # Since this file is imported by server.py, we might need a way to pass the provider.
            # We will rely on `get_auth_provider` being initialized.
            
            # Hack: assume we can import brain_service from the module where this is used, 
            # OR pass config_dir via some context.
            
            # Better: The handler `self` usually doesn't have config_dir directly unless we set it.
            # But `brain_service` is global in `server.py`.
            
            provider = _auth_provider 
            if not provider:
                # Fail open or closed? Closed.
                self.send_error(500, "Auth provider not initialized")
                return

            decoded = provider.verify_token(token)
            if not decoded:
                self.send_error(403, "Invalid or expired token")
                return
            
            user_role = provider.get_user_role(decoded)
            
            # Check hierarchy? Or strict equality?
            # Creator > Developer > Consumer
            # Simple check for now
            allowed = False
            if user_role == UserRole.CREATOR:
                allowed = True
            elif user_role == UserRole.DEVELOPER and role in (UserRole.DEVELOPER, UserRole.CONSUMER):
                allowed = True
            elif user_role == UserRole.CONSUMER and role == UserRole.CONSUMER:
                allowed = True
                
            if not allowed:
                 self.send_error(403, f"Insufficient permissions. Required: {role}, Got: {user_role}")
                 return

            # Attach user info to request context (self)
            self.user_context = {
                "uid": decoded.get("uid"),
                "email": decoded.get("email"),
                "role": user_role
            }
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
