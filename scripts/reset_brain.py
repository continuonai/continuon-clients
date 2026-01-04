#!/usr/bin/env python3
"""
Reset Brain - Start fresh with clean memories and training data.

This script:
1. Optionally backs up existing data
2. Clears all learning data (memories, episodes, training)
3. Preserves essential config (device_id, settings, ownership)
4. Reinitializes clean databases

Usage:
    python scripts/reset_brain.py              # Interactive mode
    python scripts/reset_brain.py --confirm    # Skip confirmation
    python scripts/reset_brain.py --backup     # Create backup first
    python scripts/reset_brain.py --dry-run    # Show what would be deleted
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Brain data directory
BRAIN_DIR = Path("/opt/continuonos/brain")

# Directories to CLEAR (learning data)
CLEAR_DIRS = [
    "memories/chat_logs",
    "rlds/episodes",
    "rlds/autonomous_learning",
    "rlds/tfrecord",
    "experiences",
    "trainer/logs",
    "trainer/checkpoints",
    "checkpoints",
    "logs",
]

# Files to CLEAR
CLEAR_FILES = [
    "context_graph.db",
    "sessions.db",
]

# Directories to PRESERVE (config, models)
PRESERVE = [
    "device_id.json",
    "ownership.json",
    "settings.json",
    "continuonbrain.env",
    "rcan_identity.json",
    "model/base_model",
    "model/hailo",
    "hf_cache",
    "safety",
    "consent",
    "recognition",
    "pairing",
]


def get_size(path: Path) -> int:
    """Get total size of path in bytes."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def format_size(size: int) -> str:
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def scan_data() -> dict:
    """Scan brain data and return summary."""
    summary = {
        "clear_dirs": [],
        "clear_files": [],
        "total_size": 0,
    }

    for dir_path in CLEAR_DIRS:
        full_path = BRAIN_DIR / dir_path
        if full_path.exists():
            size = get_size(full_path)
            count = sum(1 for _ in full_path.rglob("*") if _.is_file())
            summary["clear_dirs"].append({
                "path": str(full_path),
                "size": size,
                "files": count,
            })
            summary["total_size"] += size

    for file_path in CLEAR_FILES:
        full_path = BRAIN_DIR / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            summary["clear_files"].append({
                "path": str(full_path),
                "size": size,
            })
            summary["total_size"] += size

    return summary


def create_backup(backup_dir: Path) -> bool:
    """Create backup of brain data."""
    print(f"\nCreating backup at {backup_dir}...")

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup directories
        for dir_path in CLEAR_DIRS:
            src = BRAIN_DIR / dir_path
            if src.exists():
                dst = backup_dir / dir_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Backing up {dir_path}...")
                shutil.copytree(src, dst, dirs_exist_ok=True)

        # Backup files
        for file_path in CLEAR_FILES:
            src = BRAIN_DIR / file_path
            if src.exists():
                dst = backup_dir / file_path
                print(f"  Backing up {file_path}...")
                shutil.copy2(src, dst)

        print(f"  Backup complete: {backup_dir}")
        return True

    except Exception as e:
        print(f"  Backup failed: {e}")
        return False


def clear_directory(path: Path, dry_run: bool = False) -> int:
    """Clear contents of a directory."""
    if not path.exists():
        return 0

    count = 0
    for item in path.iterdir():
        if dry_run:
            print(f"    Would delete: {item}")
            count += 1
        else:
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1

    return count


def init_clean_db(db_path: Path, schema: str) -> None:
    """Initialize a clean database with schema."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.close()


def reset_brain(dry_run: bool = False) -> bool:
    """Reset all brain learning data."""
    print("\nResetting brain data...")

    # Clear directories
    for dir_path in CLEAR_DIRS:
        full_path = BRAIN_DIR / dir_path
        if full_path.exists():
            if dry_run:
                print(f"  Would clear: {dir_path}")
            else:
                print(f"  Clearing: {dir_path}")
                clear_directory(full_path)
                # Recreate empty directory
                full_path.mkdir(parents=True, exist_ok=True)

    # Clear/reinit databases
    for file_path in CLEAR_FILES:
        full_path = BRAIN_DIR / file_path
        if full_path.exists():
            if dry_run:
                print(f"  Would reset: {file_path}")
            else:
                print(f"  Resetting: {file_path}")
                full_path.unlink()

    if not dry_run:
        # Initialize clean context_graph.db
        print("  Initializing clean context_graph.db...")
        init_clean_db(
            BRAIN_DIR / "context_graph.db",
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            """
        )

        # Initialize clean sessions.db
        print("  Initializing clean sessions.db...")
        init_clean_db(
            BRAIN_DIR / "sessions.db",
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                event_type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # Create fresh experiences directory with empty files
        exp_dir = BRAIN_DIR / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "learned_conversations.jsonl").write_text("")

        # Reset trainer status
        trainer_status = BRAIN_DIR / "trainer" / "status.json"
        trainer_status.parent.mkdir(parents=True, exist_ok=True)
        trainer_status.write_text(json.dumps({
            "state": "idle",
            "reset_at": datetime.now().isoformat(),
            "message": "Brain reset - fresh start"
        }, indent=2))

    print("\n  Brain reset complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Reset ContinuonBrain - Start fresh with clean memories"
    )
    parser.add_argument(
        "--confirm", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Create backup before reset"
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path.home() / "brain_backup" / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Backup directory path"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ContinuonBrain Reset - Fresh Start")
    print("=" * 60)

    # Check brain directory
    if not BRAIN_DIR.exists():
        print(f"\nError: Brain directory not found: {BRAIN_DIR}")
        return 1

    # Scan data
    print("\nScanning brain data...")
    summary = scan_data()

    print("\nData to be cleared:")
    print("-" * 40)

    for item in summary["clear_dirs"]:
        print(f"  {item['path']}")
        print(f"    {item['files']} files, {format_size(item['size'])}")

    for item in summary["clear_files"]:
        print(f"  {item['path']}")
        print(f"    {format_size(item['size'])}")

    print("-" * 40)
    print(f"Total: {format_size(summary['total_size'])}")

    print("\nPreserved (not deleted):")
    for item in PRESERVE:
        path = BRAIN_DIR / item
        if path.exists():
            print(f"  {item}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made")
        reset_brain(dry_run=True)
        return 0

    # Confirmation
    if not args.confirm:
        print("\n" + "=" * 60)
        print("WARNING: This will permanently delete all learning data!")
        print("=" * 60)
        response = input("\nType 'RESET' to confirm: ")
        if response != "RESET":
            print("Aborted.")
            return 1

    # Backup
    if args.backup:
        if not create_backup(args.backup_dir):
            print("Backup failed. Aborting reset.")
            return 1

    # Reset
    reset_brain(dry_run=False)

    print("\n" + "=" * 60)
    print("Brain has been reset to a fresh state!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Restart ContinuonBrain service")
    print("  2. The brain will start learning from scratch")
    print("  3. All Hailo acceleration is preserved")

    return 0


if __name__ == "__main__":
    sys.exit(main())
