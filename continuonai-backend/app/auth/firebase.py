"""Firebase Authentication integration."""

import logging
from typing import Optional

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth, credentials
from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Firebase app singleton
_firebase_app: Optional[firebase_admin.App] = None


class FirebaseUser(BaseModel):
    """Authenticated Firebase user."""

    uid: str
    email: Optional[str] = None
    email_verified: bool = False
    name: Optional[str] = None
    picture: Optional[str] = None
    provider_id: Optional[str] = None
    custom_claims: Optional[dict] = None


def initialize_firebase(settings: Settings) -> Optional[firebase_admin.App]:
    """Initialize Firebase Admin SDK."""
    global _firebase_app

    if _firebase_app is not None:
        return _firebase_app

    try:
        # Try to initialize with credentials file if provided
        if settings.firebase_credentials_path:
            cred = credentials.Certificate(settings.firebase_credentials_path)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized with credentials file")
        # Otherwise try default credentials (for Cloud Run)
        elif settings.google_cloud_project:
            _firebase_app = firebase_admin.initialize_app(
                options={"projectId": settings.google_cloud_project}
            )
            logger.info("Firebase initialized with default credentials")
        else:
            logger.warning(
                "Firebase credentials not configured. "
                "Set FIREBASE_CREDENTIALS_PATH or GOOGLE_CLOUD_PROJECT."
            )
            return None

        return _firebase_app

    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return None


def get_firebase_app(settings: Settings = Depends(get_settings)) -> Optional[firebase_admin.App]:
    """Get or initialize Firebase app."""
    return initialize_firebase(settings)


# Security scheme for Bearer token
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_firebase_token(
    token: str,
    settings: Settings = Depends(get_settings),
) -> Optional[FirebaseUser]:
    """Verify a Firebase ID token and return user info."""
    app = initialize_firebase(settings)

    if app is None:
        # Firebase not configured - for development, allow mock auth
        if settings.environment == "development" and settings.debug:
            logger.warning("Firebase not configured, using mock auth in development")
            return FirebaseUser(
                uid="dev-user-123",
                email="dev@continuonai.com",
                email_verified=True,
                name="Development User",
            )
        return None

    try:
        # Verify the token
        decoded_token = auth.verify_id_token(token, app=app)

        return FirebaseUser(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            email_verified=decoded_token.get("email_verified", False),
            name=decoded_token.get("name"),
            picture=decoded_token.get("picture"),
            provider_id=decoded_token.get("firebase", {}).get("sign_in_provider"),
            custom_claims={
                k: v
                for k, v in decoded_token.items()
                if k not in {"uid", "email", "email_verified", "name", "picture", "firebase", "iat", "exp", "aud", "iss", "sub"}
            },
        )

    except auth.ExpiredIdTokenError:
        logger.warning("Expired Firebase token")
        return None
    except auth.RevokedIdTokenError:
        logger.warning("Revoked Firebase token")
        return None
    except auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid Firebase token: {e}")
        return None
    except Exception as e:
        logger.error(f"Error verifying Firebase token: {e}")
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> FirebaseUser:
    """
    Get the current authenticated user.
    Raises HTTPException 401 if not authenticated.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await verify_firebase_token(credentials.credentials, settings)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> Optional[FirebaseUser]:
    """
    Get the current user if authenticated, None otherwise.
    Does not raise an exception for unauthenticated requests.
    """
    if credentials is None:
        return None

    return await verify_firebase_token(credentials.credentials, settings)


async def get_admin_user(
    user: FirebaseUser = Depends(get_current_user),
) -> FirebaseUser:
    """
    Get current user and verify they have admin privileges.
    Raises HTTPException 403 if not an admin.
    """
    is_admin = user.custom_claims and user.custom_claims.get("admin", False)

    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return user
