"""Pytest configuration and fixtures for ContinuonAI Backend tests."""

import os
import pytest


def pytest_configure(config):
    """Set up test environment before running tests."""
    # Set test environment variables
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")


@pytest.fixture(scope="session")
def test_settings():
    """Get test settings."""
    from app.config import Settings
    return Settings(
        environment="test",
        debug=True,
        google_cloud_project="test-project",
    )


@pytest.fixture
def mock_firebase_user():
    """Create a mock Firebase user for testing."""
    from app.auth.firebase import FirebaseUser
    return FirebaseUser(
        uid="test-user-123",
        email="test@continuonai.com",
        email_verified=True,
        name="Test User",
    )


@pytest.fixture
def auth_headers(mock_firebase_user):
    """Create mock auth headers."""
    # In real tests, you would generate a valid Firebase token
    # For unit tests, mock the auth dependency instead
    return {"Authorization": "Bearer mock-token"}
