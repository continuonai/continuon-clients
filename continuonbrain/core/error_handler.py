"""
Centralized error handling with logging, metrics, and recovery strategies.

Provides decorators and utilities for consistent error handling across
all ContinuonBrain services.
"""
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from .exceptions import ContinuonError, ErrorContext, ErrorSeverity

logger = logging.getLogger(__name__)
T = TypeVar("T")


def handle_errors(
    fallback: Optional[Callable[..., T]] = None,
    fallback_value: Optional[T] = None,
    log_level: int = logging.ERROR,
    reraise: bool = True,
    include_traceback: bool = True,
    component: Optional[str] = None,
):
    """
    Decorator for consistent error handling.

    Args:
        fallback: Callable to invoke on error (receives same args as decorated func)
        fallback_value: Static value to return on error (used if fallback is None)
        log_level: Log level for error messages
        reraise: Whether to re-raise the exception after handling
        include_traceback: Whether to include full traceback in logs
        component: Component name for error context

    Usage:
        @handle_errors(fallback_value=[], reraise=False)
        def risky_operation():
            ...

        @handle_errors(fallback=lambda x: {"error": str(x)})
        def api_handler(request):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except ContinuonError as e:
                _log_error(e, func.__name__, component, log_level, include_traceback)
                if reraise:
                    raise
                return _get_fallback_result(fallback, fallback_value, args, kwargs)
            except Exception as e:
                _log_unexpected_error(e, func.__name__, component, include_traceback)
                if reraise:
                    raise
                return _get_fallback_result(fallback, fallback_value, args, kwargs)
        return wrapper
    return decorator


def handle_errors_async(
    fallback: Optional[Callable[..., T]] = None,
    fallback_value: Optional[T] = None,
    log_level: int = logging.ERROR,
    reraise: bool = True,
    include_traceback: bool = True,
    component: Optional[str] = None,
):
    """
    Async version of handle_errors decorator.

    Usage:
        @handle_errors_async(fallback_value=None)
        async def async_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except ContinuonError as e:
                _log_error(e, func.__name__, component, log_level, include_traceback)
                if reraise:
                    raise
                return _get_fallback_result(fallback, fallback_value, args, kwargs)
            except Exception as e:
                _log_unexpected_error(e, func.__name__, component, include_traceback)
                if reraise:
                    raise
                return _get_fallback_result(fallback, fallback_value, args, kwargs)
        return wrapper
    return decorator


def _log_error(
    error: ContinuonError,
    func_name: str,
    component: Optional[str],
    log_level: int,
    include_traceback: bool,
):
    """Log a ContinuonError with structured information."""
    error_dict = error.to_dict()
    error_dict["function"] = func_name
    if component:
        error_dict["component"] = component

    message = f"{error.__class__.__name__}: {error}"
    if include_traceback:
        logger.log(log_level, message, exc_info=True, extra={"error_context": error_dict})
    else:
        logger.log(log_level, message, extra={"error_context": error_dict})


def _log_unexpected_error(
    error: Exception,
    func_name: str,
    component: Optional[str],
    include_traceback: bool,
):
    """Log an unexpected (non-ContinuonError) exception."""
    error_dict = {
        "error_type": error.__class__.__name__,
        "message": str(error),
        "function": func_name,
        "component": component,
        "unexpected": True,
    }

    message = f"Unexpected error in {func_name}: {error}"
    if include_traceback:
        logger.exception(message, extra={"error_context": error_dict})
    else:
        logger.error(message, extra={"error_context": error_dict})


def _get_fallback_result(
    fallback: Optional[Callable],
    fallback_value: Optional[T],
    args: tuple,
    kwargs: dict,
) -> T:
    """Get fallback result from callable or static value."""
    if fallback is not None:
        try:
            return fallback(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Fallback function also failed: {e}")
            return fallback_value
    return fallback_value


class ErrorAggregator:
    """
    Aggregates errors for batch operations.

    Usage:
        aggregator = ErrorAggregator()
        for item in items:
            with aggregator.collect():
                process(item)
        if aggregator.has_errors:
            logger.warning(f"Processed with {aggregator.error_count} errors")
    """

    def __init__(self, max_errors: int = 100):
        self.errors: list = []
        self.max_errors = max_errors
        self._total_operations = 0
        self._successful_operations = 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        if self._total_operations == 0:
            return 1.0
        return self._successful_operations / self._total_operations

    def collect(self):
        """Context manager for collecting errors."""
        return _ErrorCollector(self)

    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the collection."""
        if len(self.errors) < self.max_errors:
            self.errors.append({
                "error": error,
                "type": error.__class__.__name__,
                "message": str(error),
                "context": context,
                "timestamp": time.time(),
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            "total_operations": self._total_operations,
            "successful_operations": self._successful_operations,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "error_types": self._get_error_type_counts(),
        }

    def _get_error_type_counts(self) -> Dict[str, int]:
        """Count errors by type."""
        counts = {}
        for err in self.errors:
            err_type = err["type"]
            counts[err_type] = counts.get(err_type, 0) + 1
        return counts


class _ErrorCollector:
    """Context manager for ErrorAggregator."""

    def __init__(self, aggregator: ErrorAggregator):
        self.aggregator = aggregator

    def __enter__(self):
        self.aggregator._total_operations += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.aggregator.add_error(exc_val)
            return True  # Suppress the exception
        self.aggregator._successful_operations += 1
        return False


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch
        on_retry: Callback called on each retry (receives exception and attempt number)

    Usage:
        @retry_on_error(max_retries=3, delay=0.5)
        def flaky_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {e.__class__.__name__}: {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise

            raise last_exception
        return wrapper
    return decorator


def retry_on_error_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Async version of retry_on_error decorator."""
    import asyncio

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {e.__class__.__name__}: {e}"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise

            raise last_exception
        return wrapper
    return decorator


def create_error_context(
    component: str,
    operation: str,
    inputs: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> ErrorContext:
    """
    Factory function for creating ErrorContext objects.

    Usage:
        ctx = create_error_context(
            component="VisionCore",
            operation="detect",
            inputs={"frame_shape": frame.shape}
        )
        raise VisionError("Detection failed", context=ctx)
    """
    return ErrorContext(
        component=component,
        operation=operation,
        inputs=inputs,
        state=state,
    )
