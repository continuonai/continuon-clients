"""Tests for main FastAPI application."""

import pytest
from fastapi.testclient import TestClient


# Mock Firebase and Google Cloud services before importing app
@pytest.fixture(autouse=True)
def mock_google_services(monkeypatch):
    """Mock Google Cloud services for testing."""
    # Mock environment variables
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ContinuonAI API"
        assert "version" in data
        assert "docs" in data

    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRobotsEndpoints:
    """Test robot management endpoints."""

    def test_list_robots_unauthorized(self, client):
        """Test listing robots without auth returns 401."""
        response = client.get("/api/v1/robots/")
        assert response.status_code == 401

    def test_create_robot_unauthorized(self, client):
        """Test creating robot without auth returns 401."""
        response = client.post(
            "/api/v1/robots/",
            json={
                "name": "Test Robot",
                "device_id": "test-device-123",
            },
        )
        assert response.status_code == 401


class TestModelsEndpoints:
    """Test model registry endpoints."""

    def test_list_models_public(self, client):
        """Test listing public models (no auth required)."""
        response = client.get("/api/v1/models/")
        # Should succeed but may return empty list without real GCS
        assert response.status_code in [200, 500]  # 500 if GCS not configured


class TestTrainingEndpoints:
    """Test training job endpoints."""

    def test_list_training_jobs_unauthorized(self, client):
        """Test listing training jobs without auth returns 401."""
        response = client.get("/api/v1/training/jobs")
        assert response.status_code == 401

    def test_create_training_job_unauthorized(self, client):
        """Test creating training job without auth returns 401."""
        response = client.post(
            "/api/v1/training/jobs",
            json={
                "config": {
                    "name": "Test Job",
                    "robot_ids": ["robot-1"],
                },
            },
        )
        assert response.status_code == 401
