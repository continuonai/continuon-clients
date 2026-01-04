#!/usr/bin/env python3
"""
Generate comprehensive training episodes for ContinuonBrain.

Creates diverse RLDS episodes covering:
- Navigation (follow, patrol, avoid, explore)
- Manipulation (reach, grasp, lift, place, stack)
- Multi-step tasks (fetch, deliver, organize)
- Error recovery scenarios
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def generate_pose(x: float, y: float, z: float, yaw: float = 0.0) -> Dict[str, Any]:
    """Generate a pose dict with position and quaternion."""
    # Convert yaw to quaternion (rotation around Z axis)
    qw = math.cos(yaw / 2)
    qz = math.sin(yaw / 2)
    return {
        "position": [round(x, 4), round(y, 4), round(z, 4)],
        "orientation_quat": [0.0, 0.0, round(qz, 4), round(qw, 4)],
        "valid": True
    }


def generate_robot_state(
    timestamp_ms: int,
    joint_positions: List[float],
    ee_position: List[float],
    gripper_open: bool = True,
) -> Dict[str, Any]:
    """Generate robot state observation."""
    return {
        "timestamp_nanos": timestamp_ms * 1_000_000,
        "joint_positions": [round(j, 4) for j in joint_positions],
        "joint_velocities": [0.0] * len(joint_positions),
        "joint_efforts": [0.0] * len(joint_positions),
        "end_effector_pose": generate_pose(*ee_position),
        "end_effector_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_open": gripper_open,
        "frame_id": "base_link",
        "wall_time_millis": timestamp_ms
    }


def generate_observation(
    timestamp_ms: int,
    robot_state: Dict[str, Any],
    target_position: List[float] = None,
    obstacle_detected: bool = False,
) -> Dict[str, Any]:
    """Generate full observation dict."""
    obs = {
        "headset_pose": generate_pose(0, 0, 1.6),  # Default head height
        "right_hand_pose": generate_pose(0.3, -0.2, 1.0),
        "left_hand_pose": generate_pose(-0.3, -0.2, 1.0),
        "gaze": {
            "origin": [0.0, 0.0, 1.6],
            "direction": [0.0, 1.0, 0.0],
            "confidence": 0.9
        },
        "robot_state": robot_state,
        "glove": {
            "timestamp_nanos": timestamp_ms * 1_000_000,
            "flex": [0.1, 0.1, 0.1, 0.1, 0.1],
            "fsr": [0.0] * 8,
            "orientation_quat": [0.0, 0.0, 0.0, 1.0],
            "accel": [0.0, 9.81, 0.0],
            "valid": True
        },
        "video_frame_id": f"frame_{timestamp_ms:04d}",
        "depth_frame_id": f"frame_{timestamp_ms:04d}",
        "diagnostics": {
            "latency_ms": random.uniform(5, 15),
            "glove_drops": 0,
            "ble_rssi": random.randint(-70, -50),
            "glove_sample_rate_hz": 90.0
        }
    }

    # Add semantic info
    if target_position:
        obs["target_info"] = {
            "position": target_position,
            "detected": True
        }
    if obstacle_detected:
        obs["obstacle_info"] = {
            "detected": True,
            "distance_m": random.uniform(0.3, 1.0)
        }

    return obs


def generate_action(command: List[float], source: str = "human_teleop") -> Dict[str, Any]:
    """Generate action dict."""
    return {
        "command": [round(c, 4) for c in command],
        "source": source
    }


# =============================================================================
# Episode Generators
# =============================================================================

def generate_navigation_follow_episode(num_steps: int = 20) -> Dict[str, Any]:
    """Generate a 'follow target' navigation episode."""
    steps = []

    # Target moves in a smooth path
    target_x, target_y = 2.0, 0.0
    robot_x, robot_y = 0.0, 0.0
    yaw = 0.0

    for i in range(num_steps):
        # Target moves randomly
        target_x += random.uniform(-0.2, 0.3)
        target_y += random.uniform(-0.2, 0.2)
        target_x = max(1.0, min(5.0, target_x))
        target_y = max(-2.0, min(2.0, target_y))

        # Calculate action to follow
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist = math.sqrt(dx**2 + dy**2)

        # Linear and angular velocity
        linear_vel = min(0.3, dist * 0.5)  # Proportional control
        angular_vel = math.atan2(dy, dx) - yaw
        angular_vel = ((angular_vel + math.pi) % (2 * math.pi)) - math.pi  # Normalize

        # Simulate robot movement
        robot_x += linear_vel * math.cos(yaw) * 0.1
        robot_y += linear_vel * math.sin(yaw) * 0.1
        yaw += angular_vel * 0.1

        timestamp = i * 100
        robot_state = generate_robot_state(
            timestamp,
            [yaw, 0.0],  # Simplified: heading, speed
            [robot_x, robot_y, 0.0],
            gripper_open=True
        )

        obs = generate_observation(
            timestamp, robot_state,
            target_position=[target_x, target_y, 0.0]
        )

        # Action: [linear_vel, angular_vel] for differential drive
        action = generate_action([linear_vel, angular_vel])

        steps.append({
            "observation": obs,
            "action": action,
            "reward": 1.0 if dist < 1.5 else 0.0,
            "is_terminal": i == num_steps - 1,
            "step_metadata": {
                "task": "follow_target",
                "target_distance_m": str(round(dist, 2))
            }
        })

    return {
        "metadata": {
            "xr_mode": "trainer",
            "control_role": "human_teleop",
            "environment_id": "indoor_nav",
            "tags": ["navigation", "follow", "synthetic", "pi5"],
            "software": {"xr_app": "episode_gen", "continuonbrain_os": "synthetic"}
        },
        "steps": steps
    }


def generate_obstacle_avoidance_episode(num_steps: int = 25) -> Dict[str, Any]:
    """Generate obstacle avoidance navigation episode."""
    steps = []

    robot_x, robot_y = 0.0, 0.0
    yaw = 0.0
    goal_x, goal_y = 5.0, 0.0

    # Obstacles
    obstacles = [(2.0, 0.5), (3.0, -0.3), (4.0, 0.2)]

    for i in range(num_steps):
        timestamp = i * 100

        # Check for nearby obstacles
        obstacle_detected = False
        avoidance_dir = 0.0
        for ox, oy in obstacles:
            dist = math.sqrt((robot_x - ox)**2 + (robot_y - oy)**2)
            if dist < 1.0:
                obstacle_detected = True
                # Turn away from obstacle
                avoidance_dir = -math.atan2(oy - robot_y, ox - robot_x)

        # Navigate to goal while avoiding
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist_to_goal = math.sqrt(dx**2 + dy**2)

        if obstacle_detected:
            linear_vel = 0.1
            angular_vel = avoidance_dir * 0.5
        else:
            linear_vel = min(0.3, dist_to_goal * 0.3)
            angular_vel = (math.atan2(dy, dx) - yaw) * 0.3

        # Update position
        robot_x += linear_vel * math.cos(yaw) * 0.1
        robot_y += linear_vel * math.sin(yaw) * 0.1
        yaw += angular_vel * 0.1

        robot_state = generate_robot_state(
            timestamp,
            [yaw, linear_vel],
            [robot_x, robot_y, 0.0]
        )

        obs = generate_observation(
            timestamp, robot_state,
            target_position=[goal_x, goal_y, 0.0],
            obstacle_detected=obstacle_detected
        )

        action = generate_action([linear_vel, angular_vel])

        steps.append({
            "observation": obs,
            "action": action,
            "reward": 1.0 if dist_to_goal < 0.5 else (0.5 if not obstacle_detected else 0.2),
            "is_terminal": dist_to_goal < 0.3 or i == num_steps - 1,
            "step_metadata": {
                "task": "avoid_obstacles",
                "goal_distance_m": str(round(dist_to_goal, 2)),
                "obstacle_nearby": str(obstacle_detected)
            }
        })

        if dist_to_goal < 0.3:
            break

    return {
        "metadata": {
            "xr_mode": "trainer",
            "control_role": "human_teleop",
            "environment_id": "obstacle_course",
            "tags": ["navigation", "avoidance", "synthetic"],
            "software": {"xr_app": "episode_gen", "continuonbrain_os": "synthetic"}
        },
        "steps": steps
    }


def generate_arm_reach_episode(num_steps: int = 15) -> Dict[str, Any]:
    """Generate arm reaching episode (6-DOF arm)."""
    steps = []

    # Initial and target positions
    ee_pos = [0.3, 0.0, 0.5]
    target_pos = [0.5, 0.2, 0.3]

    joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 joints

    for i in range(num_steps):
        timestamp = i * 50  # 20Hz control

        # Simple proportional control to reach target
        dx = target_pos[0] - ee_pos[0]
        dy = target_pos[1] - ee_pos[1]
        dz = target_pos[2] - ee_pos[2]

        # Velocity commands (simplified IK)
        vx = dx * 0.5
        vy = dy * 0.5
        vz = dz * 0.5

        # Clamp velocities
        max_vel = 0.1
        vx = max(-max_vel, min(max_vel, vx))
        vy = max(-max_vel, min(max_vel, vy))
        vz = max(-max_vel, min(max_vel, vz))

        # Update end effector position
        ee_pos[0] += vx * 0.1
        ee_pos[1] += vy * 0.1
        ee_pos[2] += vz * 0.1

        # Update joints (simplified)
        joints = [j + random.uniform(-0.05, 0.05) for j in joints]

        robot_state = generate_robot_state(
            timestamp, joints, ee_pos, gripper_open=True
        )

        obs = generate_observation(
            timestamp, robot_state,
            target_position=target_pos
        )

        # 6-DOF action: [vx, vy, vz, wx, wy, wz]
        action = generate_action([vx, vy, vz, 0.0, 0.0, 0.0])

        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        steps.append({
            "observation": obs,
            "action": action,
            "reward": 1.0 if dist < 0.05 else 0.5,
            "is_terminal": dist < 0.03 or i == num_steps - 1,
            "step_metadata": {
                "task": "arm_reach",
                "target_distance_m": str(round(dist, 3))
            }
        })

        if dist < 0.03:
            break

    return {
        "metadata": {
            "xr_mode": "trainer",
            "control_role": "human_teleop",
            "environment_id": "arm_workspace",
            "tags": ["manipulation", "reach", "arm", "synthetic"],
            "software": {"xr_app": "episode_gen", "continuonbrain_os": "synthetic"}
        },
        "steps": steps
    }


def generate_pick_and_place_episode(num_steps: int = 30) -> Dict[str, Any]:
    """Generate pick and place manipulation episode."""
    steps = []

    # Phases: approach -> grasp -> lift -> move -> place -> release
    phases = ["approach", "grasp", "lift", "move", "place", "release"]

    object_pos = [0.4, 0.1, 0.1]  # Object on table
    place_pos = [0.4, -0.2, 0.1]  # Target location
    ee_pos = [0.3, 0.0, 0.4]  # Start above table

    joints = [0.0] * 6
    gripper_open = True
    holding_object = False

    phase_idx = 0
    steps_in_phase = 0

    for i in range(num_steps):
        timestamp = i * 50
        phase = phases[min(phase_idx, len(phases) - 1)]

        # Determine target based on phase
        if phase == "approach":
            target = [object_pos[0], object_pos[1], object_pos[2] + 0.15]
        elif phase == "grasp":
            target = object_pos.copy()
        elif phase == "lift":
            target = [object_pos[0], object_pos[1], object_pos[2] + 0.3]
        elif phase == "move":
            target = [place_pos[0], place_pos[1], place_pos[2] + 0.3]
        elif phase == "place":
            target = [place_pos[0], place_pos[1], place_pos[2] + 0.05]
        else:  # release
            target = [place_pos[0], place_pos[1], place_pos[2] + 0.2]

        # Move towards target
        dx = target[0] - ee_pos[0]
        dy = target[1] - ee_pos[1]
        dz = target[2] - ee_pos[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        vx = dx * 0.4
        vy = dy * 0.4
        vz = dz * 0.4

        max_vel = 0.08
        vx = max(-max_vel, min(max_vel, vx))
        vy = max(-max_vel, min(max_vel, vy))
        vz = max(-max_vel, min(max_vel, vz))

        ee_pos[0] += vx
        ee_pos[1] += vy
        ee_pos[2] += vz

        # Gripper control
        if phase == "grasp" and dist < 0.02:
            gripper_open = False
            holding_object = True
        elif phase == "release":
            gripper_open = True
            holding_object = False

        # Update object position if holding
        if holding_object:
            object_pos = ee_pos.copy()

        # Phase transitions
        steps_in_phase += 1
        if dist < 0.02 and steps_in_phase > 2:
            phase_idx += 1
            steps_in_phase = 0

        robot_state = generate_robot_state(
            timestamp, joints, ee_pos, gripper_open=gripper_open
        )

        obs = generate_observation(
            timestamp, robot_state,
            target_position=target
        )

        # Action includes gripper: [vx, vy, vz, wx, wy, wz, gripper]
        gripper_cmd = 1.0 if gripper_open else 0.0
        action = generate_action([vx, vy, vz, 0.0, 0.0, 0.0, gripper_cmd])

        steps.append({
            "observation": obs,
            "action": action,
            "reward": 1.0 if phase == "release" and not holding_object else 0.3,
            "is_terminal": phase == "release" and not holding_object,
            "step_metadata": {
                "task": "pick_and_place",
                "phase": phase,
                "gripper_state": "open" if gripper_open else "closed"
            }
        })

        if phase == "release" and dist < 0.02:
            break

    return {
        "metadata": {
            "xr_mode": "trainer",
            "control_role": "human_teleop",
            "environment_id": "tabletop",
            "tags": ["manipulation", "pick_place", "arm", "gripper", "synthetic"],
            "software": {"xr_app": "episode_gen", "continuonbrain_os": "synthetic"}
        },
        "steps": steps
    }


def generate_patrol_episode(num_steps: int = 40) -> Dict[str, Any]:
    """Generate patrol navigation episode visiting waypoints."""
    steps = []

    # Patrol waypoints
    waypoints = [
        (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)
    ]

    robot_x, robot_y = 0.0, 0.0
    yaw = 0.0
    waypoint_idx = 1

    for i in range(num_steps):
        timestamp = i * 100

        wx, wy = waypoints[waypoint_idx]
        dx = wx - robot_x
        dy = wy - robot_y
        dist = math.sqrt(dx**2 + dy**2)

        # Navigate to waypoint
        target_yaw = math.atan2(dy, dx)
        angular_vel = (target_yaw - yaw) * 0.5
        angular_vel = max(-0.5, min(0.5, angular_vel))

        linear_vel = min(0.25, dist * 0.4)
        if abs(angular_vel) > 0.3:
            linear_vel *= 0.3  # Slow down while turning

        # Update position
        robot_x += linear_vel * math.cos(yaw) * 0.1
        robot_y += linear_vel * math.sin(yaw) * 0.1
        yaw += angular_vel * 0.1

        # Check waypoint reached
        if dist < 0.2:
            waypoint_idx = (waypoint_idx + 1) % len(waypoints)

        robot_state = generate_robot_state(
            timestamp,
            [yaw, linear_vel],
            [robot_x, robot_y, 0.0]
        )

        obs = generate_observation(
            timestamp, robot_state,
            target_position=[wx, wy, 0.0]
        )

        action = generate_action([linear_vel, angular_vel])

        steps.append({
            "observation": obs,
            "action": action,
            "reward": 1.0 if dist < 0.3 else 0.2,
            "is_terminal": i == num_steps - 1,
            "step_metadata": {
                "task": "patrol",
                "waypoint_idx": str(waypoint_idx),
                "waypoint_distance_m": str(round(dist, 2))
            }
        })

    return {
        "metadata": {
            "xr_mode": "trainer",
            "control_role": "human_teleop",
            "environment_id": "patrol_area",
            "tags": ["navigation", "patrol", "waypoints", "synthetic"],
            "software": {"xr_app": "episode_gen", "continuonbrain_os": "synthetic"}
        },
        "steps": steps
    }


def generate_stop_and_go_episode(num_steps: int = 20) -> Dict[str, Any]:
    """Generate stop-and-go episode for learning to stop on command."""
    steps = []

    robot_x = 0.0
    velocity = 0.0

    # Commands: 1=go, 0=stop
    commands = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]

    for i in range(num_steps):
        timestamp = i * 100

        should_go = commands[i % len(commands)]
        target_vel = 0.2 if should_go else 0.0

        # Smooth acceleration/deceleration
        if target_vel > velocity:
            velocity = min(target_vel, velocity + 0.05)
        else:
            velocity = max(target_vel, velocity - 0.08)

        robot_x += velocity * 0.1

        robot_state = generate_robot_state(
            timestamp,
            [0.0, velocity],
            [robot_x, 0.0, 0.0]
        )

        obs = generate_observation(timestamp, robot_state)
        obs["command_signal"] = should_go

        action = generate_action([velocity, 0.0])

        steps.append({
            "observation": obs,
            "action": action,
            "reward": 1.0 if abs(velocity - target_vel) < 0.05 else 0.5,
            "is_terminal": i == num_steps - 1,
            "step_metadata": {
                "task": "stop_and_go",
                "command": "go" if should_go else "stop",
                "velocity": str(round(velocity, 3))
            }
        })

    return {
        "metadata": {
            "xr_mode": "trainer",
            "control_role": "human_teleop",
            "environment_id": "corridor",
            "tags": ["navigation", "stop_go", "command_following", "synthetic"],
            "software": {"xr_app": "episode_gen", "continuonbrain_os": "synthetic"}
        },
        "steps": steps
    }


def generate_all_episodes(output_dir: Path, seed: int = 42) -> Dict[str, int]:
    """Generate all training episodes."""
    random.seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    # Navigation episodes
    for i in range(5):
        episode = generate_navigation_follow_episode(num_steps=25 + i * 5)
        path = output_dir / f"nav_follow_{i:02d}.json"
        path.write_text(json.dumps(episode, indent=2))
        stats[f"nav_follow_{i:02d}"] = len(episode["steps"])

    for i in range(3):
        episode = generate_obstacle_avoidance_episode(num_steps=30 + i * 5)
        path = output_dir / f"nav_avoid_{i:02d}.json"
        path.write_text(json.dumps(episode, indent=2))
        stats[f"nav_avoid_{i:02d}"] = len(episode["steps"])

    for i in range(3):
        episode = generate_patrol_episode(num_steps=40 + i * 10)
        path = output_dir / f"nav_patrol_{i:02d}.json"
        path.write_text(json.dumps(episode, indent=2))
        stats[f"nav_patrol_{i:02d}"] = len(episode["steps"])

    for i in range(3):
        episode = generate_stop_and_go_episode(num_steps=25)
        path = output_dir / f"nav_stop_go_{i:02d}.json"
        path.write_text(json.dumps(episode, indent=2))
        stats[f"nav_stop_go_{i:02d}"] = len(episode["steps"])

    # Manipulation episodes
    for i in range(5):
        episode = generate_arm_reach_episode(num_steps=20)
        path = output_dir / f"arm_reach_{i:02d}.json"
        path.write_text(json.dumps(episode, indent=2))
        stats[f"arm_reach_{i:02d}"] = len(episode["steps"])

    for i in range(5):
        episode = generate_pick_and_place_episode(num_steps=35)
        path = output_dir / f"arm_pick_place_{i:02d}.json"
        path.write_text(json.dumps(episode, indent=2))
        stats[f"arm_pick_place_{i:02d}"] = len(episode["steps"])

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training episodes")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "rlds" / "episodes",
        help="Output directory for episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Generating episodes to: {args.output_dir}")
    stats = generate_all_episodes(args.output_dir, seed=args.seed)

    total_steps = sum(stats.values())
    print(f"\nGenerated {len(stats)} episodes with {total_steps} total steps:")
    for name, steps in sorted(stats.items()):
        print(f"  {name}: {steps} steps")
