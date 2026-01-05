"""Authentication middleware for ContinuonAI API."""

import logging
import time
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.auth.firebase import verify_firebase_token
from app.config import get_settings

logger = logging.getLogger(__name__)


# Paths that don't require authentication
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}

# Path prefixes that don't require authentication
PUBLIC_PATH_PREFIXES = (
    "/api/v1/models",  # Public model listing (auth optional)
)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle authentication and add user context to requests.

    This middleware:
    1. Extracts the Bearer token from Authorization header
    2. Verifies the token with Firebase
    3. Attaches user info to request.state for downstream use
    4. Logs request timing and authentication status
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Initialize request state
        request.state.user = None
        request.state.authenticated = False

        # Check if path is public
        path = request.url.path
        is_public = path in PUBLIC_PATHS or path.startswith(PUBLIC_PATH_PREFIXES)

        # Extract and verify token if present
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix

            try:
                user = await verify_firebase_token(token, self.settings)
                if user:
                    request.state.user = user
                    request.state.authenticated = True
                    logger.debug(f"Authenticated user: {user.uid}")
            except Exception as e:
                logger.warning(f"Token verification failed: {e}")

        # Log request
        logger.info(
            f"{request.method} {path} - "
            f"authenticated={request.state.authenticated} "
            f"user={request.state.user.uid if request.state.user else 'anonymous'}"
        )

        # Process request
        response = await call_next(request)

        # Add timing header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    For production, use Redis-based rate limiting or a service like Cloud Armor.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: dict[str, list[float]] = {}

    def _get_client_key(self, request: Request) -> str:
        """Get unique identifier for client."""
        # Use user ID if authenticated, otherwise IP
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.uid}"

        # Get client IP (handle proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    def _is_rate_limited(self, client_key: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        # Get request times for this client
        request_times = self.request_counts.get(client_key, [])

        # Remove old requests outside the window
        request_times = [t for t in request_times if t > window_start]

        # Check if over limit
        if len(request_times) >= self.requests_per_minute:
            return True

        # Add current request
        request_times.append(current_time)
        self.request_counts[client_key] = request_times

        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_key = self._get_client_key(request)

        if self._is_rate_limited(client_key):
            logger.warning(f"Rate limit exceeded for {client_key}")
            return Response(
                content='{"detail": "Rate limit exceeded. Please try again later."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
        request.state.request_id = request_id

        # Log request start
        logger.info(
            f"[{request_id}] {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
            },
        )

        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"[{request_id}] {response.status_code} ({duration:.3f}s)",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": int(duration * 1000),
            },
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
