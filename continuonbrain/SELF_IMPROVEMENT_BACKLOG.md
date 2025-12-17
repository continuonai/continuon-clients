# Robot Self-Improvement Backlog (Training Mode)

Purpose: keep the robot productive while models train by queueing safe, offline-first tasks that respect creator intent and safety boundaries.

## Operating principles
- **Safety-first:** obey immutable safety protocol and refuse tasks that could violate motion, privacy, or data-handling rules.
- **Mission-anchored:** keep decisions aligned to the Continuon AI mission statement (humanity-first, collective intelligence, distributed/on-device).
- **Creator-aligned:** prefer items tagged as creator-directed before self-discovery tasks; pause if intent is ambiguous.
- **Offline-first:** use onboard resources and cached knowledge before reaching for the network; log when network access is required.
- **Cost-aware:** cap network/API usage at **$5/day**; prefer Gemma 3n or other onboard models before invoking Gemini CLI or other paid tools.
- **Health-gated:** only run items when `system_health.py` reports ready for the needed hardware/software (e.g., cameras, servos, MCP/Gemini CLI).

## Ready now (no external dependencies)
- **Review safety and instruction set:**
  - Re-load `system_instructions.py` defaults and any `CONFIG_DIR/system_instructions.json` overrides; confirm merged safety rules are logged.
  - Verify startup guardrails by running `python -m continuonbrain.startup_manager --dry-run` and record any missing config paths.
  - Confirm the mission guardrail is loaded from `MISSION_STATEMENT.md` and surfaced in system health output.
- **System health audit:**
  - Run `python -m continuonbrain.system_health --quick --log-json /tmp/system_health.json` and update known issues list.
  - Confirm MCP + Gemini CLI availability flags are captured in health output.
  - Verify the API budget check reports the $5/day ceiling and that offline-first notes are present.
- **Mock motion rehearsal:**
  - Use `robot_api_server.py` in mock mode to replay saved command sequences; capture logs for latency and command validation timing.

## Next up (requires targeted checks or assets)
- **Editor UX polish (Robot Editor):**
  - Add quick actions for common safety toggles (e-stop, “recording protected” mode) after verifying UI states map to `robot_modes.py`.
  - Improve visual telemetry cards with heartbeat indicators sourced from `system_health.py`.
- **Data curation loop:**
  - Trim or tag RLDS sample episodes under `rlds/episodes/` for clarity; ensure manifests remain valid against trainer defaults.
  - Draft prompts/instructions for MCP chat to summarize recent logs without sending sensitive frames off-device.
- **Depth/RGB capture tuning:**
  - If OAK-D hardware is present, profile frame timing in `sensors/oak_depth.py` and record results; otherwise leave notes for next availability.

## Later (hardware or human confirmation needed)
- **Real-hardware end-to-end rehearsal:**
  - When servos + depth cam are connected, run `tests/integration_test.py --real-hardware` with safety observers present; file findings.
- **Self-update dry run:**
  - With creator approval, exercise the Gemini CLI supervised update flow and document rollback steps; skip if offline or approval missing.
- **Physical environment mapping:**
  - With human clearance, capture a small depth map sweep to refresh collision envelopes; store locally only.

## Logging and handoff
- Keep a dated log of attempted tasks and outcomes under `/tmp/brain_backlog.log` (or configured log path).
- Escalate blockers that require human input, new credentials, or safety rule changes; do not auto-resolve by relaxing safety rules.
