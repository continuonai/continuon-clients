"""
Variant-aware RLDS validation.

The repo has multiple episode shapes (see docs/rlds-variants.md). This module
provides:
- strict validation for the legacy Studio/mock robotics envelope (existing behavior)
- permissive validation for canonical episode_dir (metadata.json + steps/000000.jsonl)

We keep the permissive validator intentionally light: it checks structure and
required metadata keys, but does not enforce robotics-specific fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from continuonbrain.rlds import validators as studio_mock_validators


@dataclass
class ValidationResult:
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_studio_mock_episode_json(episode: Dict[str, Any]) -> ValidationResult:
    """Strict validator (back-compat): mirrors continuonbrain.rlds.validators.validate_episode."""
    result = studio_mock_validators.validate_episode(episode)
    return ValidationResult(errors=list(result.errors), warnings=list(result.warnings))


def _validate_metadata_minimal(metadata: Dict[str, Any], errors: List[str]) -> None:
    for key in ["xr_mode", "control_role", "environment_id", "tags", "software"]:
        if key not in metadata:
            errors.append(f"metadata missing required key: {key}")
    software = metadata.get("software")
    if isinstance(software, dict):
        for key in ["xr_app", "continuonbrain_os", "glove_firmware"]:
            if key not in software:
                errors.append(f"metadata.software missing required key: {key}")


def validate_canonical_episode_dir(episode_dir: Path) -> ValidationResult:
    """
    Validate canonical episode_dir layout.

    This validator is permissive by design (supports eval/chat/metrics episodes).
    """
    errors: List[str] = []
    warnings: List[str] = []

    meta_path = episode_dir / "metadata.json"
    steps_path = episode_dir / "steps" / "000000.jsonl"
    if not meta_path.exists():
        errors.append("missing metadata.json")
        return ValidationResult(errors=errors, warnings=warnings)
    if not steps_path.exists():
        errors.append("missing steps/000000.jsonl")
        return ValidationResult(errors=errors, warnings=warnings)

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"metadata.json unreadable: {exc}")
        return ValidationResult(errors=errors, warnings=warnings)
    if not isinstance(metadata, dict):
        errors.append("metadata.json must contain an object")
        return ValidationResult(errors=errors, warnings=warnings)
    _validate_metadata_minimal(metadata, errors)

    step_count = 0
    try:
        for raw in steps_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            step_count += 1
            try:
                step = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid jsonl step at line {step_count}: {exc}")
                continue
            if not isinstance(step, dict):
                errors.append(f"step line {step_count} must be an object")
                continue
            for key in ["observation", "action"]:
                if key not in step:
                    errors.append(f"step line {step_count} missing {key}")
            if "step_metadata" in step and not isinstance(step["step_metadata"], dict):
                errors.append(f"step line {step_count} step_metadata must be an object")
            # Recommend (not require) action.source for downstream filtering.
            action = step.get("action") if isinstance(step.get("action"), dict) else None
            if action is not None and "source" not in action:
                warnings.append(f"step line {step_count} action.source missing (recommended)")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"steps/000000.jsonl unreadable: {exc}")

    if step_count == 0:
        errors.append("steps/000000.jsonl contains no steps")

    return ValidationResult(errors=errors, warnings=warnings)


def detect_episode_dir_variant(episode_dir: Path) -> str:
    """
    Heuristic: decide whether an episode_dir likely matches the strict Studio/mock schema.

    If strict validation passes, classify as studio_mock; otherwise canonical.
    """
    # Attempt strict validation if the dir can be loaded as a single episode dict.
    # Some episode_dir producers already have metadata.json + steps jsonl; strict validator expects a merged JSON dict,
    # so by default we treat episode_dir as canonical.
    return "canonical"


def validate_path(path: Path) -> ValidationResult:
    """
    Validate either:
    - an episode.json dict (strict studio/mock validator)
    - an episode_dir (permissive canonical validator)
    """
    path = path.expanduser().resolve()
    if path.is_dir():
        return validate_canonical_episode_dir(path)
    if path.is_file() and path.suffix.lower() == ".json":
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(errors=[f"failed to read JSON: {exc}"], warnings=[])
        if isinstance(episode, dict) and "metadata" in episode and "steps" in episode:
            return validate_studio_mock_episode_json(episode)
        # Not a strict studio/mock episode dict; recommend normalization first.
        return ValidationResult(errors=["unsupported episode.json shape (normalize to episode_dir first)"], warnings=[])
    return ValidationResult(errors=[f"unsupported path for validation: {path}"], warnings=[])


