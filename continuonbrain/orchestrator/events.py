"""
Event Bus System

Provides publish/subscribe functionality for loose coupling between components.
"""

import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types in the orchestrator."""

    # Task events
    TASK_SUBMITTED = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_CANCELLED = auto()
    TASK_PROGRESS = auto()

    # Workflow events
    WORKFLOW_STARTED = auto()
    WORKFLOW_STEP_COMPLETED = auto()
    WORKFLOW_COMPLETED = auto()
    WORKFLOW_FAILED = auto()

    # Worker events
    WORKER_STARTED = auto()
    WORKER_STOPPED = auto()
    WORKER_IDLE = auto()
    WORKER_BUSY = auto()

    # System events
    ORCHESTRATOR_STARTED = auto()
    ORCHESTRATOR_STOPPED = auto()
    HEALTH_CHECK = auto()
    METRICS_UPDATED = auto()

    # Training events
    TRAINING_STARTED = auto()
    TRAINING_STEP = auto()
    TRAINING_COMPLETED = auto()
    TRAINING_CHECKPOINT = auto()

    # Inference events
    INFERENCE_REQUEST = auto()
    INFERENCE_RESPONSE = auto()

    # Custom events
    CUSTOM = auto()


@dataclass
class Event:
    """An event in the system."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time() * 1000)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "type": self.type.name,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", ""),
            type=EventType[data["type"]],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", "unknown"),
        )


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Returns Awaitable/Coroutine


class EventBus:
    """
    Central event bus for publish/subscribe communication.

    Features:
    - Synchronous and asynchronous handlers
    - Event filtering by type
    - Event history/replay
    - Thread-safe operations
    """

    def __init__(self, buffer_size: int = 10000):
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[AsyncEventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[Event] = []
        self._buffer_size = buffer_size
        self._lock = threading.RLock()
        self._event_queue: Queue = Queue()
        self._running = False
        self._dispatcher_thread: Optional[threading.Thread] = None

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Subscribe to events of a specific type."""
        with self._lock:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                logger.debug(f"Subscribed handler to {event_type.name}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        with self._lock:
            if handler not in self._global_handlers:
                self._global_handlers.append(handler)
                logger.debug("Subscribed global handler")

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Unsubscribe from events of a specific type."""
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.name}")

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """Unsubscribe from all events."""
        with self._lock:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Events are queued and dispatched asynchronously.
        """
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._buffer_size:
                self._event_history = self._event_history[-self._buffer_size:]

        # Queue for async dispatch
        self._event_queue.put(event)

        # Also dispatch synchronously for immediate handlers
        self._dispatch_sync(event)

    def publish_sync(self, event: Event) -> None:
        """Publish and wait for all handlers to complete."""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._buffer_size:
                self._event_history = self._event_history[-self._buffer_size:]

        self._dispatch_sync(event)

    def _dispatch_sync(self, event: Event) -> None:
        """Dispatch event to handlers synchronously."""
        with self._lock:
            handlers = list(self._handlers[event.type]) + list(self._global_handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.type.name}: {e}")

    def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any] = None,
        source: str = "orchestrator",
    ) -> Event:
        """Convenience method to create and publish an event."""
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
        )
        self.publish(event)
        return event

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Event]:
        """Get event history, optionally filtered."""
        with self._lock:
            events = list(self._event_history)

        if event_type:
            events = [e for e in events if e.type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()

    def start_dispatcher(self) -> None:
        """Start the background event dispatcher thread."""
        if self._running:
            return

        self._running = True
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True,
            name="EventDispatcher",
        )
        self._dispatcher_thread.start()
        logger.info("Event dispatcher started")

    def stop_dispatcher(self) -> None:
        """Stop the background event dispatcher."""
        self._running = False
        if self._dispatcher_thread:
            self._event_queue.put(None)  # Signal to stop
            self._dispatcher_thread.join(timeout=5)
            logger.info("Event dispatcher stopped")

    def _dispatcher_loop(self) -> None:
        """Background loop for dispatching queued events."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1)
                if event is None:
                    break
                # Events are already dispatched sync in publish()
                # This loop is for potential async handlers in the future
            except Exception:
                continue

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        with self._lock:
            return len(self._handlers[event_type]) + len(self._global_handlers)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            return {
                "total_events": len(self._event_history),
                "queue_size": self._event_queue.qsize(),
                "handlers_by_type": {
                    et.name: len(handlers)
                    for et, handlers in self._handlers.items()
                    if handlers
                },
                "global_handlers": len(self._global_handlers),
                "running": self._running,
            }
