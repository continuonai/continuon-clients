# Studio Panel Proto Mapping: Capability Manifest & Telemetry Additions

This note enumerates the new Continuon Brain link fields so Studio panels can bind without schema hunting.

## Capability Manifest additions
- `available_cms_snapshots[]` (from `CmsSnapshot`): slow-loop bundle IDs, Memory Plane version, CMS balance, and origin (`live|cached|mock`) for the Slow Loop Snapshot card and offline diff view.
- `safety_signals[]` (from `SafetySignalDefinition`): declarative labels/units/tags used to seed the Safety Console legend and filter chips.

## Editor telemetry additions
- `safety_signals[]` (from `SafetySignal`): live values + severity/source metadata rendered as inline callouts in the Safety Console and piped into RLDS logging.
- `cms_snapshot` (from `CmsSnapshot`): active HOPE/CMS snapshot context pinned in the Slow Loop bar; matches the IDs advertised in the manifest for deterministic playback.

Panels should continue to treat older runtimes as compatible; missing fields default to empty collections/null.
