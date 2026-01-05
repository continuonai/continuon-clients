"""Security utilities for ContinuonAI API."""

import hashlib
import secrets
from typing import Optional


def generate_api_key(prefix: str = "cai") -> str:
    """
    Generate a secure API key.

    Format: {prefix}_{random_bytes}
    Example: cai_sk_abc123...
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage.

    Uses SHA-256 for hashing.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash.
    """
    return secrets.compare_digest(hash_api_key(api_key), hashed_key)


def generate_random_string(length: int = 32) -> str:
    """Generate a cryptographically secure random string."""
    return secrets.token_urlsafe(length)


def mask_sensitive_value(value: str, visible_chars: int = 4) -> str:
    """
    Mask a sensitive value, showing only the last few characters.

    Example: "abc123xyz" -> "****xyz"
    """
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]
