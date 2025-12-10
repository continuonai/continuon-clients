"""RLDS export helpers for anonymized Continuon Cloud uploads."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from . import validators


def _hash_text(value: str, salt: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{salt}:{value}".encode("utf-8"))
    return digest.hexdigest()


def _should_hash_tag(tag: str) -> bool:
    lowered = tag.lower()
    return any(
        pattern in lowered for pattern in ["@", "user:", "operator:", "email:", "account:"]
    )


@dataclass
class PiiAnonymizationConfig:
    """Configuration describing how to scrub PII from RLDS exports."""

    hash_salt: str = "continuonbrain"
    drop_keys: Sequence[str] = (
        "email",
        "user",
        "user_id",
        "operator",
        "operator_id",
        "session_id",
        "phone",
        "contact",
    )
    hash_keys: Sequence[str] = (
        "note",
        "notes",
        "comment",
        "language_instruction",
        "description",
    )
    anonymization_tag_prefix: str = "anonymization:hash:sha256"


@dataclass
class ValidationReport:
    episode_name: str
    ok: bool
    errors: List[str]
    warnings: List[str]
    generated_at: str


@dataclass
class ExportManifestEntry:
    source: str
    anonymized_path: str
    validation_report: str
    metadata_tags: List[str]


@dataclass
class ExportManifest:
    generated_at: str
    environment: Dict[str, str]
    anonymization: Dict[str, object]
    episodes: List[ExportManifestEntry]

    def to_json(self) -> str:
        return json.dumps(
            {
                "generated_at": self.generated_at,
                "environment": self.environment,
                "anonymization": self.anonymization,
                "episodes": [asdict(entry) for entry in self.episodes],
            },
            indent=2,
            sort_keys=True,
        )


def _sanitize_string_map(
    data: Mapping[str, str], config: PiiAnonymizationConfig
) -> MutableMapping[str, str]:
    cleaned: Dict[str, str] = {}
    for key, value in data.items():
        if key in config.drop_keys:
            continue
        if key in config.hash_keys:
            cleaned[key] = _hash_text(str(value), config.hash_salt)
        else:
            cleaned[key] = str(value)
    return cleaned


def _anonymize_tags(tags: Iterable[str], config: PiiAnonymizationConfig) -> List[str]:
    anonymized_tags: List[str] = []
    for tag in tags:
        if _should_hash_tag(tag):
            anonymized_tags.append(f"tag:{_hash_text(tag, config.hash_salt)}")
        else:
            anonymized_tags.append(tag)
    summary_tag = f"{config.anonymization_tag_prefix}:salted"
    if summary_tag not in anonymized_tags:
        anonymized_tags.append(summary_tag)
    return anonymized_tags


def _anonymize_ui_context(ui_context: Mapping[str, object], config: PiiAnonymizationConfig):
    anonymized: Dict[str, object] = {}
    if "active_panel" in ui_context:
        anonymized["active_panel"] = _hash_text(str(ui_context["active_panel"]), config.hash_salt)
    if layout := ui_context.get("layout"):
        anonymized["layout"] = _sanitize_string_map(layout, config)
    if focus_context := ui_context.get("focus_context"):
        anonymized["focus_context"] = _sanitize_string_map(focus_context, config)
    return anonymized


def _anonymize_action(action: Mapping[str, object], config: PiiAnonymizationConfig):
    anonymized = copy.deepcopy(action)
    if annotation := action.get("annotation"):
        fields = annotation.get("fields") or {}
        anonymized_annotation = {"kind": annotation.get("kind"), "fields": _sanitize_string_map(fields, config)}
        anonymized["annotation"] = anonymized_annotation
    if ui_action := action.get("ui_action"):
        context = ui_action.get("context") or {}
        anonymized_ui_action = {
            "action_type": ui_action.get("action_type"),
            "context": _sanitize_string_map(context, config),
        }
        anonymized["ui_action"] = anonymized_ui_action
    return anonymized


def _anonymize_observation(observation: Mapping[str, object], config: PiiAnonymizationConfig):
    anonymized = copy.deepcopy(observation)
    if audio := observation.get("audio"):
        anonymized_audio = dict(audio)
        if uri := audio.get("uri"):
            anonymized_audio["uri"] = f"anonymized://{_hash_text(uri, config.hash_salt)}"
        anonymized["audio"] = anonymized_audio
    if ui_context := observation.get("ui_context"):
        anonymized["ui_context"] = _anonymize_ui_context(ui_context, config)
    return anonymized


def anonymize_episode(episode: Mapping[str, object], config: PiiAnonymizationConfig) -> Dict[str, object]:
    """Return a deep-copied, anonymized episode payload.

    Required RLDS fields are preserved; free-form text maps and tags are
    hashed so downstream uploads avoid leaking PII.
    """

    anonymized = copy.deepcopy(episode)

    metadata = anonymized.get("metadata", {})
    tags = metadata.get("tags", [])
    metadata["tags"] = _anonymize_tags(tags, config)
    anonymized["metadata"] = metadata

    steps = anonymized.get("steps", [])
    for idx, step in enumerate(steps):
        observation = step.get("observation") or {}
        action = step.get("action") or {}
        step_metadata = step.get("step_metadata") or {}

        step["observation"] = _anonymize_observation(observation, config)
        step["action"] = _anonymize_action(action, config)
        step["step_metadata"] = _sanitize_string_map(step_metadata, config)
        steps[idx] = step

    anonymized["steps"] = steps
    return anonymized


def _write_json(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_anonymized_episode(name: str, episode: Mapping[str, object], report_path: Path) -> ValidationReport:
    result = validators.validate_episode(episode)
    report = ValidationReport(
        episode_name=name,
        ok=result.ok,
        errors=result.errors,
        warnings=result.warnings,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(report_path, asdict(report))
    if not result.ok:
        raise ValueError(f"Validation failed for {name}: {result.errors}")
    return report


def prepare_cloud_export(
    episodes: Sequence[Path],
    output_dir: Path,
    config: PiiAnonymizationConfig | None = None,
) -> ExportManifest:
    """Anonymize, validate, and bundle episodes for Continuon Cloud upload."""

    cfg = config or PiiAnonymizationConfig()
    reports_dir = output_dir / "reports"
    anonymized_dir = output_dir / "episodes"
    manifest_entries: List[ExportManifestEntry] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for episode_path in episodes:
        episode = validators.load_episode(episode_path)
        anonymized = anonymize_episode(episode, cfg)

        anonymized_path = anonymized_dir / episode_path.name
        _write_json(anonymized_path, anonymized)

        report_path = reports_dir / f"{episode_path.stem}.validation.json"
        report = validate_anonymized_episode(episode_path.name, anonymized, report_path)

        manifest_entries.append(
            ExportManifestEntry(
                source=str(episode_path),
                anonymized_path=str(anonymized_path),
                validation_report=str(report_path),
                metadata_tags=anonymized.get("metadata", {}).get("tags", []),
            )
        )

    manifest = ExportManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        environment={
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "origin_tag": os.getenv("CONTINUON_EXPORT_ORIGIN", "pi5"),
            "runtime": "continuonbrain scaffolding",
        },
        anonymization={
            "hash": "sha256",
            "salt": cfg.hash_salt,
            "drop_keys": list(cfg.drop_keys),
            "hash_keys": list(cfg.hash_keys),
            "tag": cfg.anonymization_tag_prefix,
        },
        episodes=manifest_entries,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest
