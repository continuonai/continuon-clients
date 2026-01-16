#!/usr/bin/env python3
"""
Generate RLDS training data for Home3D navigation.

Runs the robot through multiple levels with different strategies:
- Random exploration
- Goal-seeking behavior
- Wall-following

Saves episodes to brain_b_data/home_rlds_episodes/
"""

import random
import time
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.home_world import (
    HomeWorld,
    get_level,
    list_levels,
    CURRICULUM_ORDER,
)
from simulator.home_rlds_logger import HomeRLDSLogger


# Navigation actions
ACTIONS = [
    "forward",
    "backward",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "interact",
]


def random_exploration(world: HomeWorld, max_steps: int = 50) -> list:
    """Random walk through the environment."""
    actions = []
    for _ in range(max_steps):
        action = random.choice(ACTIONS)
        actions.append(action)

        # Execute action
        method = getattr(world, f"move_{action}" if action in ["forward", "backward"] else action, None)
        if method is None:
            method = getattr(world, action, None)
        if method:
            method()

        if world.level_complete:
            break

    return actions


def goal_seeking(world: HomeWorld, max_steps: int = 100) -> list:
    """Try to navigate toward the goal."""
    actions = []

    for _ in range(max_steps):
        if not world.goal_position:
            break

        # Calculate direction to goal
        robot = world.robot
        dx = world.goal_position.x - robot.position.x
        dy = world.goal_position.y - robot.position.y

        # Decide action based on relative position
        if abs(dx) > abs(dy):
            # Move along x
            if dx > 0.5:
                # Need to turn toward positive x
                target_yaw = 90
            else:
                target_yaw = 270
        else:
            # Move along y
            if dy < -0.5:
                # Need to turn toward negative y
                target_yaw = 0
            else:
                target_yaw = 180

        # Calculate yaw difference
        yaw_diff = (target_yaw - robot.rotation.yaw + 180) % 360 - 180

        # Decide action
        if abs(yaw_diff) > 30:
            if yaw_diff > 0:
                action = "turn_right"
            else:
                action = "turn_left"
        else:
            # Mostly facing goal, try to move forward
            action = "forward"

        # Add some randomness (10% chance of random action)
        if random.random() < 0.1:
            action = random.choice(ACTIONS)

        actions.append(action)

        # Execute action
        if action == "forward":
            world.move_forward()
        elif action == "backward":
            world.move_backward()
        elif action == "turn_left":
            world.turn_left()
        elif action == "turn_right":
            world.turn_right()
        elif action == "strafe_left":
            world.strafe_left()
        elif action == "strafe_right":
            world.strafe_right()
        elif action == "interact":
            world.interact()

        if world.level_complete:
            break

    return actions


def wall_following(world: HomeWorld, max_steps: int = 100) -> list:
    """Follow walls to explore."""
    actions = []
    last_success = True
    turn_count = 0

    for _ in range(max_steps):
        if not last_success or turn_count > 3:
            # Hit a wall or been turning too much, turn right
            action = "turn_right"
            turn_count += 1
        else:
            # Try to move forward
            action = "forward"
            turn_count = 0

        # Occasionally try interact
        if random.random() < 0.05:
            action = "interact"

        actions.append(action)

        # Execute action
        if action == "forward":
            result = world.move_forward()
            last_success = result.success
        elif action == "turn_right":
            world.turn_right()
            last_success = True
        elif action == "turn_left":
            world.turn_left()
            last_success = True
        elif action == "interact":
            world.interact()
            last_success = True

        if world.level_complete:
            break

    return actions


def generate_episode(
    level_name: str,
    strategy: str,
    logger: HomeRLDSLogger,
    max_steps: int = 100,
) -> dict:
    """Generate a single training episode."""
    # Load level
    world = get_level(level_name)
    if not world:
        return {"success": False, "error": f"Level not found: {level_name}"}

    # Start episode
    episode_id = logger.start_episode(
        world=world,
        level_id=level_name,
        session_id=f"gen_{strategy}_{int(time.time())}",
    )

    # Run strategy
    if strategy == "random":
        actions = random_exploration(world, max_steps)
    elif strategy == "goal":
        actions = goal_seeking(world, max_steps)
    elif strategy == "wall":
        actions = wall_following(world, max_steps)
    else:
        actions = random_exploration(world, max_steps)

    # Log steps
    for i, action in enumerate(actions):
        logger.log_step(
            world=world,
            action_command=action,
            action_intent=action,
            action_params={},
            raw_input=action,
            success=True,
            level_complete=world.level_complete,
        )

    # End episode
    path = logger.end_episode(
        world=world,
        success=world.level_complete,
    )

    return {
        "success": True,
        "episode_id": episode_id,
        "level": level_name,
        "strategy": strategy,
        "steps": len(actions),
        "goal_reached": world.level_complete,
        "path": str(path),
    }


def main():
    """Generate training data across all levels and strategies."""
    output_path = Path(__file__).parent.parent / "brain_b_data" / "home_rlds_episodes"
    output_path.mkdir(parents=True, exist_ok=True)

    logger = HomeRLDSLogger(output_path)

    # Get available levels
    levels = list_levels()
    strategies = ["random", "goal", "wall"]

    print(f"Generating training data...")
    print(f"Levels: {levels}")
    print(f"Strategies: {strategies}")
    print(f"Output: {output_path}")
    print()

    results = []
    episodes_per_combo = 3  # Generate 3 episodes per level/strategy combo

    total = len(levels) * len(strategies) * episodes_per_combo
    count = 0

    for level in levels:
        for strategy in strategies:
            for ep in range(episodes_per_combo):
                count += 1
                print(f"[{count}/{total}] {level} / {strategy} / episode {ep+1}...", end=" ")

                result = generate_episode(level, strategy, logger)
                results.append(result)

                if result["success"]:
                    status = "GOAL!" if result["goal_reached"] else f"{result['steps']} steps"
                    print(f"{status}")
                else:
                    print(f"ERROR: {result.get('error', 'unknown')}")

    # Summary
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    successful = [r for r in results if r["success"]]
    goals_reached = [r for r in successful if r.get("goal_reached")]

    print(f"Episodes generated: {len(successful)}")
    print(f"Goals reached: {len(goals_reached)}")
    print(f"Success rate: {len(goals_reached) / len(successful) * 100:.1f}%")

    # By strategy
    print()
    for strategy in strategies:
        strat_episodes = [r for r in successful if r.get("strategy") == strategy]
        strat_goals = [r for r in strat_episodes if r.get("goal_reached")]
        print(f"  {strategy}: {len(strat_goals)}/{len(strat_episodes)} goals")

    # Count total steps
    total_steps = sum(r.get("steps", 0) for r in successful)
    print(f"\nTotal training steps: {total_steps}")


if __name__ == "__main__":
    main()
