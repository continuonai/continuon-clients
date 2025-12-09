"""
Task-related dataclasses extracted from robot_api_server.
"""

from dataclasses import dataclass, field
from typing import Dict, List


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
            "markers": [m.to_dict() for m in self.markers],
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
                ],
                estimated_duration="1m",
                recommended_mode="autonomous",
            ),
            "baseline-teleop": TaskDefinition(
                id="baseline-teleop",
                title="Baseline Teleop Session",
                description="Collect a short teleop session for calibration and RLDS logging.",
                group="Data",
                tags=["data", "teleop", "rlds"],
                requires_motion=True,
                requires_recording=True,
                required_modalities=["vision", "teleop"],
                steps=[
                    "Connect controller / XR",
                    "Record 3–5 minutes of teleop maneuvers",
                    "Verify RLDS upload or local storage",
                ],
                estimated_duration="5m",
                recommended_mode="trainer",
            ),
        }

    def list_entries(self) -> Dict[str, TaskDefinition]:
        return self._entries

    def get(self, task_id: str) -> TaskDefinition:
        return self._entries[task_id]

