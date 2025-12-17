from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from continuonbrain.services.reset_manager import ResetManager, ResetProfile, ResetRequest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factory reset or memories-only reset for the Continuon seed brain (local artifacts wipe)."
    )
    parser.add_argument(
        "--profile",
        choices=[p.value for p in ResetProfile],
        default=ResetProfile.FACTORY.value,
        help="Reset profile: factory wipes model + adapters; memories_only keeps model folders.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional BrainService config_dir to also wipe (experiences/chat logs/recordings).",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/opt/continuonos/brain"),
        help="Runtime root (default: /opt/continuonos/brain).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Admin token (matches CONTINUON_ADMIN_TOKEN or token file); required unless CONTINUON_ALLOW_UNSAFE_RESET=1.",
    )
    parser.add_argument(
        "--confirm",
        type=str,
        default=None,
        help='Confirmation phrase: "FACTORY RESET" or "CLEAR MEMORIES".',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    profile = ResetProfile(args.profile)
    manager = ResetManager()
    confirm_needed = manager.CONFIRM_FACTORY if profile == ResetProfile.FACTORY else manager.CONFIRM_MEMORIES

    if args.confirm != confirm_needed:
        sys.stderr.write(
            f'Confirmation required. Re-run with --confirm "{confirm_needed}" (profile={profile.value}).\n'
        )
        return 2

    if not manager.authorize(provided_token=args.token, runtime_root=args.runtime_root, config_dir=args.config_dir):
        sys.stderr.write(
            "Reset not authorized. Set CONTINUON_ADMIN_TOKEN (or token file) and pass --token, "
            "or explicitly set CONTINUON_ALLOW_UNSAFE_RESET=1 for dev.\n"
        )
        return 3

    req = ResetRequest(
        profile=profile,
        dry_run=bool(args.dry_run),
        config_dir=args.config_dir,
        runtime_root=args.runtime_root,
    )
    result = manager.run(req)

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(f"success={result.success} profile={result.profile} dry_run={result.dry_run} elapsed_s={result.elapsed_s:.3f}")
        print(f"audit={result.audit_path}")
        print(f"deleted={len(result.deleted)} skipped={len(result.skipped)} errors={len(result.errors)}")
        if result.errors:
            for err in result.errors[:10]:
                print(f"ERROR {err.get('path')}: {err.get('error')}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


