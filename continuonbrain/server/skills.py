"""
Skill-related dataclasses and in-memory library.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SkillDefinition:
    id: str
    title: str
    description: str
    group: str
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    constraints: Dict[str, str] = field(default_factory=dict)
    required_modalities: List[str] = field(default_factory=list)
    estimated_duration: str = ""
    publisher: str = "local"
    version: str = "0.1.0"
    provenance: str = ""


@dataclass
class SkillEligibilityMarker:
    code: str
    label: str
    severity: str = "info"
    blocking: bool = False
    remediation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "label": self.label,
            "severity": self.severity,
            "blocking": self.blocking,
            "remediation": self.remediation,
        }


@dataclass
class SkillEligibility:
    eligible: bool
    markers: List[SkillEligibilityMarker] = field(default_factory=list)
    next_poll_after_ms: float = 400.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "eligible": self.eligible,
            "markers": [m.to_dict() for m in self.markers],
            "next_poll_after_ms": self.next_poll_after_ms,
        }


@dataclass
class SkillSummary:
    entry: "SkillLibraryEntry"
    steps: List[str]
    publisher: str
    version: str
    provenance: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry": self.entry.to_dict(),
            "steps": self.steps,
            "publisher": self.publisher,
            "version": self.version,
            "provenance": self.provenance,
        }


@dataclass
class SkillLibraryEntry:
    id: str
    title: str
    description: str
    group: str
    tags: List[str]
    capabilities: List[str]
    eligibility: SkillEligibility
    estimated_duration: str = ""
    publisher: str = "local"
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "group": self.group,
            "tags": self.tags,
            "capabilities": self.capabilities,
            "eligibility": self.eligibility.to_dict(),
            "estimated_duration": self.estimated_duration,
            "publisher": self.publisher,
            "version": self.version,
        }


class SkillLibrary:
    """In-memory Skill Library with lightweight eligibility markers."""

    def __init__(self) -> None:
        self._entries: Dict[str, SkillDefinition] = {
            "french-chef": SkillDefinition(
                id="french-chef",
                title="French Chef",
                description="Cook a simple French dish (e.g., omelette) with safe sequencing.",
                group="Cooking",
                tags=["cooking", "demo", "manipulation"],
                capabilities=["manipulator", "vision", "heating"],
                required_modalities=["vision", "arm"],
                estimated_duration="4m",
                publisher="core",
                version="0.1.0",
                provenance="local:demo",
                constraints={"height": "tall", "reach": "mid"},
            ),
            "tidy-desk": SkillDefinition(
                id="tidy-desk",
                title="Tidy Desk",
                description="Rearrange loose items into bins; avoid cables.",
                group="Housekeeping",
                tags=["manipulation", "safety"],
                capabilities=["manipulator", "vision"],
                required_modalities=["vision", "arm"],
                estimated_duration="2m",
                publisher="core",
                version="0.1.0",
                provenance="local:demo",
            ),
            "baseline-teleop-skill": SkillDefinition(
                id="baseline-teleop-skill",
                title="Baseline Teleop",
                description="Remote teleop pass-through with logging.",
                group="Data",
                tags=["teleop", "rlds"],
                capabilities=["teleop"],
                required_modalities=["vision"],
                estimated_duration="5m",
                publisher="core",
                version="0.1.0",
                provenance="local:demo",
            ),
        }

    def list_entries(self) -> List[SkillDefinition]:
        return list(self._entries.values())

    def get_entry(self, skill_id: str) -> SkillDefinition:
        return self._entries[skill_id]

