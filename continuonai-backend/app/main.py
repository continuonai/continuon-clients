"""
ContinuonAI Backend API

FastAPI application for fleet management, model distribution, and training orchestration.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth.middleware import AuthMiddleware, RateLimitMiddleware, RequestLoggingMiddleware
from app.config import get_settings
from app.routers import analytics, episodes, models, robots, training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize Firebase (optional, will be lazy-loaded if not done here)
    from app.auth.firebase import initialize_firebase
    initialize_firebase(settings)

    yield

    # Shutdown
    logger.info("Shutting down ContinuonAI API")


# Create FastAPI application
settings = get_settings()

app = FastAPI(
    title="ContinuonAI API",
    description="""
## Fleet Management and Model Distribution for Continuon Robots

ContinuonAI API provides:

- **Fleet Management**: Register, monitor, and control your robot fleet
- **Model Registry**: Upload, version, and distribute trained models
- **Training Orchestration**: Submit and manage training jobs
- **Episode Management**: Upload and organize collected data
- **Analytics**: Monitor usage and performance metrics

### Authentication

All authenticated endpoints require a Firebase ID token in the Authorization header:

```
Authorization: Bearer <firebase_id_token>
```

### Rate Limiting

API requests are rate limited to 100 requests per minute per user.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

# Add custom middleware (order matters - last added is first executed)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests)
app.add_middleware(AuthMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(
    robots.router,
    prefix="/api/v1/robots",
    tags=["Robots"],
)
app.include_router(
    models.router,
    prefix="/api/v1/models",
    tags=["Models"],
)
app.include_router(
    training.router,
    prefix="/api/v1/training",
    tags=["Training"],
)
app.include_router(
    episodes.router,
    prefix="/api/v1/episodes",
    tags=["Episodes"],
)
app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["Analytics"],
)


# Root and health endpoints
@app.get("/", tags=["Health"])
async def root():
    """API root endpoint."""
    return {
        "name": "ContinuonAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health():
    """
    Health check endpoint.

    Returns service status and can be used for container health checks.
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/ready", tags=["Health"])
async def readiness():
    """
    Readiness check endpoint.

    Checks if all dependencies are available.
    """
    checks = {
        "api": "healthy",
        "firestore": "unknown",
        "storage": "unknown",
    }

    # Check Firestore connectivity
    try:
        from app.services.firestore import get_firestore_client
        client = get_firestore_client(settings)
        # Simple connectivity check
        list(client.collections())[:1]
        checks["firestore"] = "healthy"
    except Exception as e:
        logger.warning(f"Firestore health check failed: {e}")
        checks["firestore"] = "unhealthy"

    # Check GCS connectivity
    try:
        from app.services.storage import get_storage_client
        client = get_storage_client(settings)
        # Simple connectivity check
        list(client.list_buckets(max_results=1))
        checks["storage"] = "healthy"
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")
        checks["storage"] = "unhealthy"

    overall_status = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    return {
        "status": overall_status,
        "checks": checks,
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
        },
    )


# Application entry point for development
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
