"""
Monitoring and Health Check System

Provides health checks, metrics collection, and alerting.
"""

import os
import threading
import time
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from .events import EventBus, EventType, Event

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class HealthCheck:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class MetricPoint:
    """A single metric data point."""

    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()


class Monitor:
    """
    System monitoring and health check manager.

    Features:
    - Periodic health checks
    - Metrics collection
    - Alerting thresholds
    - System resource monitoring
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        check_interval_sec: int = 30,
        metrics_retention_hours: int = 24,
    ):
        self._event_bus = event_bus or EventBus()
        self._check_interval = check_interval_sec
        self._retention_hours = metrics_retention_hours
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._last_health: Dict[str, HealthCheck] = {}
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._running = False
        self._lock = threading.RLock()
        self._check_thread: Optional[threading.Thread] = None
        self._alert_thresholds: Dict[str, Tuple[float, float]] = {}  # (warn, crit)
        self._alert_callbacks: List[Callable[[str, str, Any], None]] = []

        # Register built-in health checks
        self._register_builtin_checks()

    def _register_builtin_checks(self) -> None:
        """Register built-in system health checks."""
        self.register_check("cpu", self._check_cpu)
        self.register_check("memory", self._check_memory)
        self.register_check("disk", self._check_disk)

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check function."""
        self._health_checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")

    def register_alert_callback(
        self,
        callback: Callable[[str, str, Any], None],
    ) -> None:
        """Register an alert callback (name, level, value)."""
        self._alert_callbacks.append(callback)

    def set_threshold(
        self,
        metric_name: str,
        warn_threshold: float,
        critical_threshold: float,
    ) -> None:
        """Set alert thresholds for a metric."""
        self._alert_thresholds[metric_name] = (warn_threshold, critical_threshold)

    def start(self) -> None:
        """Start the monitoring background thread."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HealthMonitor",
        )
        self._check_thread.start()
        logger.info(f"Monitor started (check interval: {self._check_interval}s)")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Run health checks
                self.run_all_checks()

                # Collect system metrics
                self._collect_system_metrics()

                # Check thresholds and alert
                self._check_thresholds()

                # Cleanup old metrics
                self._cleanup_old_metrics()

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self._check_interval)

    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY

        for name, check_fn in self._health_checks.items():
            start = time.time()
            try:
                result = check_fn()
                result.duration_ms = (time.time() - start) * 1000
            except Exception as e:
                result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    duration_ms=(time.time() - start) * 1000,
                )

            results[name] = result
            self._last_health[name] = result

            # Update overall status
            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.DEGRADED

        # Emit health event
        self._event_bus.emit(
            EventType.HEALTH_CHECK,
            {
                "overall_status": overall_status.name,
                "checks": {n: r.to_dict() for n, r in results.items()},
            },
            source="monitor",
        )

        return results

    def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check."""
        check_fn = self._health_checks.get(name)
        if not check_fn:
            return None

        start = time.time()
        try:
            result = check_fn()
            result.duration_ms = (time.time() - start) * 1000
        except Exception as e:
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

        self._last_health[name] = result
        return result

    def get_health(self) -> Dict[str, HealthCheck]:
        """Get last health check results."""
        return dict(self._last_health)

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._last_health:
            return HealthStatus.UNKNOWN

        statuses = [h.status for h in self._last_health.values()]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    # Metrics methods

    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._metric_key(name, labels)
            self._counters[key] += value
            self._record_metric(name, self._counters[key], labels)

    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            key = self._metric_key(name, labels)
            self._gauges[key] = value
            self._record_metric(name, value, labels)

    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        self._record_metric(name, value, labels)

    def _metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a metric data point."""
        point = MetricPoint(value=value, labels=labels or {})
        key = self._metric_key(name, labels)
        self._metrics[key].append(point)

    def get_metric(
        self,
        name: str,
        labels: Dict[str, str] = None,
        since: datetime = None,
    ) -> List[MetricPoint]:
        """Get metric data points."""
        key = self._metric_key(name, labels)
        with self._lock:
            points = list(self._metrics.get(key, []))

        if since:
            points = [p for p in points if p.timestamp >= since]

        return points

    def get_metric_summary(self, name: str, labels: Dict[str, str] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        points = self.get_metric(name, labels)
        if not points:
            return {"count": 0}

        values = [p.value for p in points]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "first_ts": points[0].timestamp.isoformat(),
            "last_ts": points[-1].timestamp.isoformat(),
        }

    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.gauge("system.cpu.percent", cpu_percent)

            # Memory
            mem = psutil.virtual_memory()
            self.gauge("system.memory.percent", mem.percent)
            self.gauge("system.memory.available_gb", mem.available / (1024**3))

            # Disk
            disk = psutil.disk_usage("/")
            self.gauge("system.disk.percent", disk.percent)
            self.gauge("system.disk.free_gb", disk.free / (1024**3))

        except Exception as e:
            logger.debug(f"System metrics collection error: {e}")

    def _check_thresholds(self) -> None:
        """Check metrics against alert thresholds."""
        for metric_name, (warn, crit) in self._alert_thresholds.items():
            points = self.get_metric(metric_name)
            if not points:
                continue

            latest = points[-1].value
            if latest >= crit:
                self._fire_alert(metric_name, "CRITICAL", latest)
            elif latest >= warn:
                self._fire_alert(metric_name, "WARNING", latest)

    def _fire_alert(self, metric_name: str, level: str, value: Any) -> None:
        """Fire an alert to registered callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(metric_name, level, value)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self._retention_hours)
        with self._lock:
            for key in list(self._metrics.keys()):
                points = self._metrics[key]
                while points and points[0].timestamp < cutoff:
                    points.popleft()

    # Built-in health checks

    def _check_cpu(self) -> HealthCheck:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"CPU critically high: {cpu_percent}%"
        elif cpu_percent > 70:
            status = HealthStatus.DEGRADED
            message = f"CPU elevated: {cpu_percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU normal: {cpu_percent}%"

        return HealthCheck(
            name="cpu",
            status=status,
            message=message,
            details={"cpu_percent": cpu_percent, "cpu_count": psutil.cpu_count()},
        )

    def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Memory critically high: {mem.percent}%"
        elif mem.percent > 80:
            status = HealthStatus.DEGRADED
            message = f"Memory elevated: {mem.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory normal: {mem.percent}%"

        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            details={
                "percent": mem.percent,
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
            },
        )

    def _check_disk(self) -> HealthCheck:
        """Check disk usage."""
        disk = psutil.disk_usage("/")
        if disk.percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Disk critically full: {disk.percent}%"
        elif disk.percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Disk elevated: {disk.percent}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk normal: {disk.percent}%"

        return HealthCheck(
            name="disk",
            status=status,
            message=message,
            details={
                "percent": disk.percent,
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        return {
            "running": self._running,
            "check_interval_sec": self._check_interval,
            "registered_checks": list(self._health_checks.keys()),
            "overall_health": self.get_overall_health().name,
            "metrics_count": len(self._metrics),
            "alert_thresholds": len(self._alert_thresholds),
        }
