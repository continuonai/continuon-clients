#!/usr/bin/env python3
"""
Autonomous Training Runner - Start and monitor autonomous training.

This script demonstrates how to use the autonomous training scheduler
for fully autonomous training of the seed model brain.

Usage:
    # Start autonomous training scheduler
    python scripts/run_autonomous_training.py start

    # Check status
    python scripts/run_autonomous_training.py status

    # Manually trigger training
    python scripts/run_autonomous_training.py trigger --mode local_slow

    # Score episodes
    python scripts/run_autonomous_training.py score-episodes

    # Check capability gaps
    python scripts/run_autonomous_training.py gaps

    # Upload model to cloud
    python scripts/run_autonomous_training.py upload --version 1.0.0

    # Register a robot
    python scripts/run_autonomous_training.py register-robot --device-id robot_001 --owner user_abc

    # Distribute update
    python scripts/run_autonomous_training.py distribute --version 1.0.0
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AutonomousTraining")


def cmd_start(args):
    """Start the autonomous training scheduler."""
    from continuonbrain.services import (
        AutonomousTrainingScheduler,
        TrainingTriggerConfig,
    )

    config = TrainingTriggerConfig(
        min_episodes_for_local=args.min_episodes,
        min_quality_score=args.min_quality,
        enable_cloud_training=args.enable_cloud,
        auto_deploy=args.auto_deploy,
    )

    scheduler = AutonomousTrainingScheduler(config=config)

    # Register callbacks
    def on_training_started(mode):
        logger.info(f"Training started: {mode.value}")

    def on_training_completed(mode, result):
        logger.info(f"Training completed: {mode.value}")
        logger.info(f"  Success: {result.get('success')}")
        logger.info(f"  Loss: {result.get('final_loss')}")

    def on_deployment(version, success):
        logger.info(f"Deployment {'succeeded' if success else 'failed'}: v{version}")

    scheduler.on_training_started(on_training_started)
    scheduler.on_training_completed(on_training_completed)
    scheduler.on_deployment_completed(on_deployment)

    logger.info("Starting autonomous training scheduler...")
    scheduler.start()

    try:
        # Keep running until interrupted
        import time
        while True:
            status = scheduler.get_status()
            logger.info(f"Status: {status.phase.value} | Episodes: {status.episode_count} | Ready: {status.training_ready}")
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
        scheduler.stop()


def cmd_status(args):
    """Check scheduler status."""
    from continuonbrain.services import AutonomousTrainingScheduler

    scheduler = AutonomousTrainingScheduler()
    status = scheduler.get_status()

    print(json.dumps(status.to_dict(), indent=2, default=str))


def cmd_trigger(args):
    """Manually trigger training."""
    from continuonbrain.services import AutonomousTrainingScheduler, TrainingMode

    scheduler = AutonomousTrainingScheduler()

    try:
        mode = TrainingMode(args.mode)
    except ValueError:
        print(f"Invalid mode: {args.mode}")
        print(f"Valid modes: {[m.value for m in TrainingMode]}")
        return 1

    logger.info(f"Triggering training with mode: {mode.value}")
    success = scheduler.trigger_training_now(mode=mode)

    if success:
        logger.info("Training triggered successfully")
    else:
        logger.error("Failed to trigger training")
        return 1


def cmd_score_episodes(args):
    """Score all episodes."""
    from continuonbrain.services import EpisodeQualityScorer

    episodes_dir = Path(args.episodes_dir)
    scorer = EpisodeQualityScorer(episodes_dir)

    logger.info(f"Scoring episodes in: {episodes_dir}")
    scores = scorer.score_all_episodes()

    # Print summary
    valid = [s for s in scores if s.is_valid]
    avg_score = sum(s.overall_score for s in valid) / len(valid) if valid else 0

    print(f"\nEpisode Quality Summary:")
    print(f"  Total episodes: {len(scores)}")
    print(f"  Valid episodes: {len(valid)}")
    print(f"  Average quality score: {avg_score:.2f}")

    # Print top episodes
    valid.sort(key=lambda s: s.overall_score, reverse=True)
    print(f"\nTop 5 episodes:")
    for i, score in enumerate(valid[:5]):
        print(f"  {i+1}. {score.episode_id}: {score.overall_score:.2f} ({score.step_count} steps)")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([s.to_dict() for s in scores], f, indent=2)
        print(f"\nFull results saved to: {output_path}")


def cmd_gaps(args):
    """Show capability gaps."""
    from continuonbrain.services import CapabilityGapDetector

    detector = CapabilityGapDetector(Path(args.config_dir))

    gaps = detector.get_capability_gaps()
    recommendations = detector.get_training_recommendations()

    print("\nCapability Gaps:")
    if not gaps:
        print("  No capability gaps detected")
    else:
        for gap in gaps[:10]:
            print(f"  - {gap['task_type']}: {gap['failure_rate']*100:.1f}% failure rate (priority: {gap['priority_score']:.2f})")

    print("\nTraining Recommendations:")
    if not recommendations:
        print("  No specific recommendations")
    else:
        for rec in recommendations:
            print(f"  - {rec['task_type']}: {rec['priority']} priority")
            print(f"    Suggested episodes: {rec['suggested_episodes']}")
            print(f"    Focus areas: {', '.join(rec['focus_areas'])}")


def cmd_validate(args):
    """Validate a model."""
    from continuonbrain.services import ModelValidator, ValidationLevel

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        return 1

    validator = ModelValidator(baseline_dir=Path(args.baseline) if args.baseline else None)

    try:
        level = ValidationLevel(args.level)
    except ValueError:
        level = ValidationLevel.STANDARD

    logger.info(f"Validating model: {model_path}")
    logger.info(f"Validation level: {level.value}")

    result = asyncio.run(validator.validate_model(model_path, level=level))

    print(f"\nValidation Result: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Status: {result.status.value}")
    print(f"  Total checks: {result.total_checks}")
    print(f"  Passed: {result.passed_checks}")
    print(f"  Failed: {result.failed_checks}")
    print(f"  Warnings: {result.warning_checks}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\nFull results saved to: {args.output}")

    return 0 if result.passed else 1


def cmd_upload(args):
    """Upload model to GCP."""
    from continuonbrain.services import ModelDistributionService, ModelType

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        return 1

    service = ModelDistributionService()

    if not service.is_available:
        print("Error: GCP services not available. Install google-cloud-storage and google-cloud-firestore.")
        return 1

    try:
        model_type = ModelType(args.type)
    except ValueError:
        model_type = ModelType.SEED

    logger.info(f"Uploading model: {model_path}")
    logger.info(f"Version: {args.version}")
    logger.info(f"Type: {model_type.value}")

    result = asyncio.run(service.upload_model(
        model_path=model_path,
        version=args.version,
        model_id=args.model_id,
        model_type=model_type,
        release_notes=args.notes,
    ))

    if result.get("success"):
        print("\nUpload successful!")
        print(f"  Model ID: {result.get('model_id')}")
        print(f"  Version: {result.get('version')}")
        print(f"  Download URL: {result.get('download_url')}")
    else:
        print(f"\nUpload failed: {result.get('error')}")
        return 1


def cmd_register_robot(args):
    """Register a new robot."""
    from continuonbrain.services import ModelDistributionService

    service = ModelDistributionService()

    if not service.is_available:
        print("Error: GCP services not available.")
        return 1

    logger.info(f"Registering robot: {args.device_id}")

    result = asyncio.run(service.register_robot(
        device_id=args.device_id,
        robot_model=args.robot_model,
        owner_uid=args.owner,
        hardware_profile=args.hardware,
        name=args.name,
        auto_update=not args.no_auto_update,
    ))

    if result.get("success"):
        print("\nRobot registered successfully!")
        print(f"  Device ID: {result.get('device_id')}")
        print(f"  Status: {result.get('status')}")
    else:
        print(f"\nRegistration failed: {result.get('error')}")
        return 1


def cmd_distribute(args):
    """Distribute model update to robots."""
    from continuonbrain.services import ModelDistributionService, UpdatePriority

    service = ModelDistributionService()

    if not service.is_available:
        print("Error: GCP services not available.")
        return 1

    try:
        priority = UpdatePriority(args.priority)
    except ValueError:
        priority = UpdatePriority.NORMAL

    logger.info(f"Distributing update: {args.model_id} v{args.version}")
    logger.info(f"Priority: {priority.value}")
    logger.info(f"Rollout: {args.rollout}%")

    result = asyncio.run(service.distribute_update(
        model_id=args.model_id,
        version=args.version,
        target_robots=args.targets.split(",") if args.targets else None,
        priority=priority,
        rollout_percentage=args.rollout,
    ))

    if result.get("success"):
        print("\nUpdate distributed!")
        print(f"  Update ID: {result.get('update_id')}")
        print(f"  Robots targeted: {result.get('robots_targeted')}")
    else:
        print(f"\nDistribution failed: {result.get('error')}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Autonomous Training Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start autonomous training scheduler")
    start_parser.add_argument("--min-episodes", type=int, default=4, help="Minimum episodes for local training")
    start_parser.add_argument("--min-quality", type=float, default=0.5, help="Minimum quality score")
    start_parser.add_argument("--enable-cloud", action="store_true", help="Enable cloud training")
    start_parser.add_argument("--auto-deploy", action="store_true", help="Auto-deploy trained models")

    # Status command
    subparsers.add_parser("status", help="Check scheduler status")

    # Trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Manually trigger training")
    trigger_parser.add_argument("--mode", default="local_slow", help="Training mode (local_fast, local_mid, local_slow, cloud_slow, cloud_full)")

    # Score episodes command
    score_parser = subparsers.add_parser("score-episodes", help="Score all episodes")
    score_parser.add_argument("--episodes-dir", default="/opt/continuonos/brain/rlds/episodes", help="Episodes directory")
    score_parser.add_argument("--output", "-o", help="Output JSON file")

    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Show capability gaps")
    gaps_parser.add_argument("--config-dir", default="/opt/continuonos/brain", help="Config directory")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a model")
    validate_parser.add_argument("model_path", help="Path to model directory")
    validate_parser.add_argument("--baseline", help="Path to baseline model for regression testing")
    validate_parser.add_argument("--level", default="standard", help="Validation level (quick, standard, full)")
    validate_parser.add_argument("--output", "-o", help="Output JSON file")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload model to GCP")
    upload_parser.add_argument("--model-path", default="/opt/continuonos/brain/model/adapters/candidate", help="Model path")
    upload_parser.add_argument("--version", required=True, help="Version string (e.g., 1.0.0)")
    upload_parser.add_argument("--model-id", default="seed", help="Model ID")
    upload_parser.add_argument("--type", default="seed", help="Model type (seed, adapter, full, patch)")
    upload_parser.add_argument("--notes", default="", help="Release notes")

    # Register robot command
    register_parser = subparsers.add_parser("register-robot", help="Register a new robot")
    register_parser.add_argument("--device-id", required=True, help="Unique device ID")
    register_parser.add_argument("--robot-model", default="donkeycar", help="Robot model")
    register_parser.add_argument("--owner", required=True, help="Owner user ID")
    register_parser.add_argument("--hardware", default="pi5", help="Hardware profile")
    register_parser.add_argument("--name", help="Human-readable name")
    register_parser.add_argument("--no-auto-update", action="store_true", help="Disable auto-updates")

    # Distribute command
    distribute_parser = subparsers.add_parser("distribute", help="Distribute update to robots")
    distribute_parser.add_argument("--model-id", default="seed", help="Model ID")
    distribute_parser.add_argument("--version", required=True, help="Version to distribute")
    distribute_parser.add_argument("--targets", help="Comma-separated target device IDs (all if not specified)")
    distribute_parser.add_argument("--priority", default="normal", help="Priority (critical, high, normal, low)")
    distribute_parser.add_argument("--rollout", type=float, default=100.0, help="Rollout percentage")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "trigger":
        return cmd_trigger(args)
    elif args.command == "score-episodes":
        return cmd_score_episodes(args)
    elif args.command == "gaps":
        return cmd_gaps(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "upload":
        return cmd_upload(args)
    elif args.command == "register-robot":
        return cmd_register_robot(args)
    elif args.command == "distribute":
        return cmd_distribute(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
