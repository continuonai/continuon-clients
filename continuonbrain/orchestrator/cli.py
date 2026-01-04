#!/usr/bin/env python3
"""
Orchestrator CLI

Command-line interface for the ContinuonBrain orchestrator.
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from .core import Orchestrator, create_orchestrator
from .config import OrchestratorConfig
from .task import TaskType, TaskPriority


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_start(args: argparse.Namespace) -> int:
    """Start the orchestrator daemon."""
    print("Starting ContinuonBrain Orchestrator...")

    config = OrchestratorConfig.development() if args.dev else OrchestratorConfig.from_env()
    if args.workers:
        config.max_workers = args.workers

    orchestrator = Orchestrator(config)

    # Handle shutdown signals
    def signal_handler(sig, frame):
        print("\nShutting down...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    orchestrator.start()
    print(f"Orchestrator running with {config.max_workers} workers")
    print("Press Ctrl+C to stop")

    # Keep running
    try:
        while orchestrator.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    orchestrator.stop()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Get orchestrator status."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        status = orchestrator.get_status()
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print_status(status)
    finally:
        orchestrator.stop()

    return 0


def print_status(status: dict) -> None:
    """Print status in human-readable format."""
    print("\n=== Orchestrator Status ===")
    print(f"Running: {status['running']}")
    print(f"Health: {status['health']['overall']}")

    print("\n--- Tasks ---")
    tasks = status['tasks']
    print(f"Total: {tasks['total']}, Queue: {tasks['queue_size']}")
    if tasks['by_status']:
        for st, count in tasks['by_status'].items():
            print(f"  {st}: {count}")

    print("\n--- Workers ---")
    workers = status['workers']
    print(f"Total: {workers['num_workers']}")
    print(f"Idle: {workers['idle_workers']}, Busy: {workers['busy_workers']}")
    print(f"Completed: {workers['total_completed']}, Failed: {workers['total_failed']}")

    print("\n--- Workflows ---")
    workflows = status['workflows']
    print(f"Total: {workflows['total']}, Active: {workflows['active']}")

    print()


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit a task."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        # Parse task type
        try:
            task_type = TaskType(args.type)
        except ValueError:
            print(f"Unknown task type: {args.type}")
            print(f"Available types: {[t.value for t in TaskType]}")
            return 1

        # Parse priority
        try:
            priority = TaskPriority[args.priority.upper()]
        except KeyError:
            print(f"Unknown priority: {args.priority}")
            print(f"Available priorities: {[p.name for p in TaskPriority]}")
            return 1

        # Parse params
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON params: {e}")
                return 1

        task_id = orchestrator.submit_task(
            task_type=task_type,
            params=params,
            priority=priority,
            timeout_sec=args.timeout,
        )

        print(f"Task submitted: {task_id}")

        if args.wait:
            print("Waiting for completion...")
            task = orchestrator.get_task(task_id)
            while task and task.status.name in ("PENDING", "QUEUED", "RUNNING"):
                time.sleep(0.5)
                task = orchestrator.get_task(task_id)

            if task:
                print(f"Status: {task.status.name}")
                if task.result:
                    print(f"Success: {task.result.success}")
                    if task.result.error:
                        print(f"Error: {task.result.error}")
                    if task.result.data:
                        print(f"Data: {json.dumps(task.result.data, indent=2)}")

    finally:
        orchestrator.stop()

    return 0


def cmd_workflow(args: argparse.Namespace) -> int:
    """Run a workflow."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        # Parse context
        context = {}
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON context: {e}")
                return 1

        workflow_id = orchestrator.run_workflow(args.name, context)
        if workflow_id is None:
            print(f"Unknown workflow template: {args.name}")
            return 1

        print(f"Workflow started: {workflow_id}")

        if args.wait:
            print("Waiting for completion...")
            workflow = orchestrator.get_workflow(workflow_id)
            while workflow and workflow.status.name in ("PENDING", "RUNNING", "PAUSED"):
                progress = workflow.get_progress()
                print(f"\rProgress: {progress['progress_pct']:.1f}% ({progress['completed']}/{progress['total_steps']} steps)", end="")
                time.sleep(1)
                workflow = orchestrator.get_workflow(workflow_id)

            print()
            if workflow:
                print(f"Status: {workflow.status.name}")
                if workflow.error:
                    print(f"Error: {workflow.error}")

    finally:
        orchestrator.stop()

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check system health."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        # Run immediate health check
        orchestrator._monitor.run_all_checks()
        health = orchestrator.get_health()
        if args.json:
            print(json.dumps(health, indent=2, default=str))
        else:
            print(f"\nOverall Health: {health['overall']}")
            print("\nChecks:")
            for name, check in health['checks'].items():
                status = check['status']
                msg = check['message']
                icon = "✓" if status == "HEALTHY" else ("⚠" if status == "DEGRADED" else "✗")
                print(f"  {icon} {name}: {msg}")
            print()
    finally:
        orchestrator.stop()

    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Get system metrics."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        time.sleep(0.5)  # Allow metrics collection
        metrics = orchestrator.get_metrics()
        if args.json:
            print(json.dumps(metrics, indent=2, default=str))
        else:
            print("\n=== System Metrics ===")
            for name, summary in metrics.items():
                if summary.get("count", 0) > 0:
                    print(f"\n{name}:")
                    print(f"  Latest: {summary.get('latest', 'N/A'):.1f}")
                    print(f"  Avg: {summary.get('avg', 'N/A'):.1f}")
                    print(f"  Min: {summary.get('min', 'N/A'):.1f}")
                    print(f"  Max: {summary.get('max', 'N/A'):.1f}")
            print()
    finally:
        orchestrator.stop()

    return 0


def cmd_list_tasks(args: argparse.Namespace) -> int:
    """List tasks."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        tasks = orchestrator.list_tasks(limit=args.limit)
        if args.json:
            print(json.dumps([t.to_dict() for t in tasks], indent=2, default=str))
        else:
            if not tasks:
                print("No tasks found.")
            else:
                print(f"\n{'ID':<20} {'Type':<15} {'Status':<12} {'Priority':<10}")
                print("-" * 60)
                for task in tasks:
                    print(f"{task.task_id:<20} {task.task_type.value:<15} {task.status.name:<12} {task.priority.name:<10}")
            print()
    finally:
        orchestrator.stop()

    return 0


def cmd_list_workflows(args: argparse.Namespace) -> int:
    """List workflows."""
    orchestrator = create_orchestrator(development=args.dev)
    orchestrator.start()

    try:
        workflows = orchestrator.list_workflows(limit=args.limit)
        if args.json:
            print(json.dumps([w.to_dict() for w in workflows], indent=2, default=str))
        else:
            if not workflows:
                print("No workflows found.")
            else:
                print(f"\n{'ID':<25} {'Name':<20} {'Status':<12} {'Progress':<10}")
                print("-" * 70)
                for wf in workflows:
                    progress = wf.get_progress()
                    print(f"{wf.workflow_id:<25} {wf.name:<20} {wf.status.name:<12} {progress['progress_pct']:.0f}%")
            print()
    finally:
        orchestrator.stop()

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ContinuonBrain Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dev", action="store_true", help="Use development configuration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start command
    start_parser = subparsers.add_parser("start", help="Start the orchestrator")
    start_parser.add_argument("-w", "--workers", type=int, help="Number of workers")

    # status command
    subparsers.add_parser("status", help="Get orchestrator status")

    # health command
    subparsers.add_parser("health", help="Check system health")

    # metrics command
    subparsers.add_parser("metrics", help="Get system metrics")

    # submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a task")
    submit_parser.add_argument("type", help="Task type")
    submit_parser.add_argument("-p", "--params", help="Task parameters as JSON")
    submit_parser.add_argument("--priority", default="NORMAL", help="Task priority")
    submit_parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    submit_parser.add_argument("-w", "--wait", action="store_true", help="Wait for completion")

    # workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run a workflow")
    workflow_parser.add_argument("name", help="Workflow template name")
    workflow_parser.add_argument("-c", "--context", help="Workflow context as JSON")
    workflow_parser.add_argument("-w", "--wait", action="store_true", help="Wait for completion")

    # tasks command
    tasks_parser = subparsers.add_parser("tasks", help="List tasks")
    tasks_parser.add_argument("-l", "--limit", type=int, default=20, help="Max tasks to show")

    # workflows command
    workflows_parser = subparsers.add_parser("workflows", help="List workflows")
    workflows_parser.add_argument("-l", "--limit", type=int, default=20, help="Max workflows to show")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "start": cmd_start,
        "status": cmd_status,
        "health": cmd_health,
        "metrics": cmd_metrics,
        "submit": cmd_submit,
        "workflow": cmd_workflow,
        "tasks": cmd_list_tasks,
        "workflows": cmd_list_workflows,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
