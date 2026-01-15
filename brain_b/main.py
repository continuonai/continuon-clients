#!/usr/bin/env python3
"""
Brain B - Main Entry Point

A minimal, teachable robot brain.

Usage:
    python main.py              # Interactive mode with mock motors
    python main.py --real       # Use real GPIO motors
    python main.py --help       # Show options
"""

import argparse
import sys
import signal

from actor_runtime import ActorRuntime
from conversation import ConversationHandler
from hardware import MotorController, MockMotorController, SafetyMonitor
from hardware.motors import create_executor


def parse_args():
    parser = argparse.ArgumentParser(description="Brain B - Teachable Robot Brain")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real GPIO motors instead of mock",
    )
    parser.add_argument(
        "--data-dir",
        default="./brain_b_data",
        help="Directory for storing state and behaviors",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Default motor speed (0.0-1.0)",
    )
    parser.add_argument(
        "--turn-duration",
        type=float,
        default=0.3,
        help="Duration of turn commands in seconds",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Don't restore from previous checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("  Brain B - Teachable Robot Brain")
    print("=" * 50)
    print()

    # Initialize motor controller
    if args.real:
        print("[Init] Using real GPIO motors")
        motors = MotorController()
    else:
        print("[Init] Using mock motors (no hardware)")
        motors = MockMotorController(verbose=True)

    # Initialize safety monitor
    safety = SafetyMonitor(motors)

    # Initialize runtime
    print(f"[Init] Data directory: {args.data_dir}")
    runtime = ActorRuntime(
        data_path=args.data_dir,
        auto_restore=not args.no_restore,
    )

    # Create executor with safety wrapper
    base_executor = create_executor(motors, turn_duration=args.turn_duration)
    safe_executor = safety.wrap_executor(base_executor)

    # Initialize conversation handler
    handler = ConversationHandler(
        runtime=runtime,
        executor=safe_executor,
        default_speed=args.speed,
    )

    # Spawn a driver agent if none exists
    if not runtime.list_agents():
        runtime.spawn("driver", config={"default_speed": args.speed})
        print("[Init] Spawned driver agent")

    # Show status
    status = runtime.status()
    if status["behaviors"] > 0:
        behaviors = runtime.teaching.list_behaviors()
        print(f"[Init] Loaded {status['behaviors']} behaviors: {', '.join(behaviors)}")
    else:
        print("[Init] No behaviors loaded yet. Teach me with 'teach <name>'")

    print()
    print("Ready! Type 'help' for commands, 'quit' to exit.")
    print("-" * 50)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n[Interrupt] Shutting down...")
        motors.stop()
        runtime.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    try:
        while True:
            try:
                # Show recording indicator
                if runtime.teaching.is_recording:
                    prompt = f"[Recording '{runtime.teaching.recording_name}'] You: "
                else:
                    prompt = "You: "

                user_input = input(prompt).strip()

                if not user_input:
                    continue

                response = handler.handle(user_input)
                print(f"Bot: {response.text}")
                print()

                # Check for quit
                if response.text == "Goodbye!":
                    break

            except EOFError:
                print("\n[EOF] Shutting down...")
                break

    finally:
        motors.stop()
        runtime.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
