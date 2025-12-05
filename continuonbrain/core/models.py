"""
Core Data Models for ContinuonBrain Tasks.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class TaskDefinition:
    """Static task library entry used to seed Studio panels."""

    id: str
    title: str
    description: str
    group: str
    tags: List[str] = field(default_factory=list)
    requires_motion: bool = False
    requires_recording: bool = False
    required_modalities: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    estimated_duration: str = ""
    recommended_mode: str = "autonomous"
    telemetry_topic: str = "loop/tasks"


@dataclass
class TaskEligibilityMarker:
    code: str
    label: str
    severity: str = "info"
    blocking: bool = False
    source: str = "runtime"
    remediation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "label": self.label,
            "severity": self.severity,
            "blocking": self.blocking,
            "source": self.source,
            "remediation": self.remediation,
        }


@dataclass
class TaskEligibility:
    eligible: bool
    markers: List[TaskEligibilityMarker] = field(default_factory=list)
    next_poll_after_ms: float = 250.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "eligible": self.eligible,
            "markers": [marker.to_dict() for marker in self.markers],
            "next_poll_after_ms": self.next_poll_after_ms,
        }


@dataclass
class TaskLibraryEntry:
    id: str
    title: str
    description: str
    group: str
    tags: List[str]
    eligibility: TaskEligibility
    estimated_duration: str = ""
    recommended_mode: str = "autonomous"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "group": self.group,
            "tags": self.tags,
            "eligibility": self.eligibility.to_dict(),
            "estimated_duration": self.estimated_duration,
            "recommended_mode": self.recommended_mode,
        }


@dataclass
class TaskSummary:
    entry: TaskLibraryEntry
    required_modalities: List[str]
    steps: List[str]
    owner: str
    updated_at: str
    telemetry_topic: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry": self.entry.to_dict(),
            "required_modalities": self.required_modalities,
            "steps": self.steps,
            "owner": self.owner,
            "updated_at": self.updated_at,
            "telemetry_topic": self.telemetry_topic,
        }


class TaskLibrary:
    """In-memory Task Library with lightweight eligibility markers."""

    def __init__(self) -> None:
        self._entries: Dict[str, TaskDefinition] = {
            "workspace-inspection": TaskDefinition(
                id="workspace-inspection",
                title="Inspect Workspace",
                description="Sweep sensors for loose cables or obstacles before enabling autonomy.",
                group="Safety",
                tags=["safety", "vision", "preflight"],
                requires_motion=False,
                requires_recording=False,
                required_modalities=["vision"],
                steps=[
                    "Spin the camera and depth sensors across the workspace",
                    "Flag blocked envelopes for operator acknowledgement",
                    "Cache a short clip for offline review",
                ],
                estimated_duration="45s",
                recommended_mode="autonomous",
                telemetry_topic="telemetry/preflight",
            ),
            "pick-and-place": TaskDefinition(
                id="pick-and-place",
                title="Pick & Place Demo",
                description="Run the default manipulation loop for demos and regressions.",
                group="Autonomy",
                tags=["manipulation", "demo", "recordable"],
                requires_motion=True,
                requires_recording=True,
                required_modalities=["vision", "gripper"],
                steps=[
                    "Plan grasp pose",
                    "Lift and place to bin",
                    "Report safety margin and latency",
                ],
                estimated_duration="2m",
                recommended_mode="autonomous",
                telemetry_topic="telemetry/manipulation",
            ),
            "calibration-check": TaskDefinition(
                id="calibration-check",
                title="Calibration Check",
                description="Verify encoders and camera alignment before overnight runs.",
                group="Maintenance",
                tags=["maintenance", "calibration"],
                requires_motion=False,
                requires_recording=False,
                required_modalities=["vision", "arm"],
                steps=[
                    "Move to calibration pose",
                    "Capture depth + RGB alignment snapshot",
                    "Emit eligibility marker if drift detected",
                ],
                estimated_duration="1m",
                recommended_mode="manual_training",
                telemetry_topic="telemetry/calibration",
            ),
        }

    def list_entries(self) -> List[TaskDefinition]:
        return list(self._entries.values())

    def get_entry(self, task_id: str) -> Optional[TaskDefinition]:
        return self._entries.get(task_id)
