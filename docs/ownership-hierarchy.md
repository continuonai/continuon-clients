# Ownership & Hierarchies (Accounts → Robots)

This note describes how robot ownership is governed across account types (enterprise, family, fleet) and how the app/runtime should enforce it.

## Account hierarchy (examples)

- **Enterprise account**: Org → sub-teams → operators. Policies flow top-down; only designated operators may claim or control robots in their scope.
- **Family account**: One owner, delegated members with limited permissions (view-only, control-within-home, no OTA).
- **Fleet account**: Similar to enterprise but scoped to a specific fleet; OTA and control limited to fleet members.

## Core rules

- **Ownership is always scoped to an account** (enterprise/family/fleet). A robot must record `account_id`, `account_type`, and `owner_id`.
- **Claiming**: Only a user authorized by the account can claim a robot. Claim binds robot to `account_id/account_type` and persists on-device.
- **Control/OTA**: Allowed only if the session’s account matches the robot’s `account_id/account_type` and the session role is permitted (e.g., operator). Subscription must be active.
- **Subscription**: OTA and remote control require an active subscription for the robot’s account (or a specific fleet SKU).
- **Local vs remote**: First claim/seed install is local-only. After claim, remote control/OTA is allowed only for authorized account members.
- **Safety**: Always enforce local safety rules; ownership does not bypass safety rails.

## Runtime state (ContinuonBrain)

Persisted on device (e.g., `ownership.json`):

- `device_id`
- `account_id`
- `account_type` (enterprise|family|fleet|personal)
- `owner_id` (user id that claimed)
- `owned` (bool)
- `subscription_active` (bool)
- `seed_installed` (bool)

Exposed via `/api/ownership/status` and used by the Flutter app to gate control/OTA.

## App enforcement (ContinuonAI Flutter)

- Claim UI requires local LAN; claim binds `account_id/account_type/owner_id` to the robot.
- Remote sessions must present account auth; app checks `/api/ownership/status` and denies control/OTA on mismatch or inactive subscription.
- Surfaces account/role errors explicitly (e.g., “Not authorized for this fleet”, “Subscription inactive”).

## Next wiring steps

- Wire real ownership/subscription/seed checks in BrainService (no placeholders) and persist updates.
- Add account metadata to `/api/ownership/status`.
- App: consume `account_id/account_type/owner_id` and enforce gating; show ping/device_id/uptime; add retry/error UI per robot card.
- Improve LAN detection (mdns/ping); move token to secure storage.
- Add explicit error states for account/role/subscription mismatches in the app and brain logs.
