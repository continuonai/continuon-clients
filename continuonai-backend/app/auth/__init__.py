"""Authentication module for ContinuonAI API."""

from app.auth.firebase import (
    get_current_user,
    get_current_user_optional,
    verify_firebase_token,
    FirebaseUser,
)
from app.auth.middleware import AuthMiddleware

__all__ = [
    "get_current_user",
    "get_current_user_optional",
    "verify_firebase_token",
    "FirebaseUser",
    "AuthMiddleware",
]
