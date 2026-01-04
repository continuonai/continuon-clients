#!/usr/bin/env python3
"""
Multi-Turn Chat Training V2 - Fixed RLDS Format

Generates training episodes in the correct RLDS format:
- observation: dict with 'command' as a list of floats
- action: dict with 'command' as a list of floats
"""

import asyncio
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Multi-turn chat training episodes
CHAT_EPISODES = [
    # Math and logic
    {
        "category": "math",
        "turns": [
            ("What is 15 + 27?", "15 + 27 = 42"),
            ("Now multiply that by 2", "42 * 2 = 84"),
            ("Is that even or odd?", "84 is even"),
        ],
    },
    {
        "category": "math",
        "turns": [
            ("Calculate 100 - 37", "100 - 37 = 63"),
            ("What's half of that?", "63 / 2 = 31.5"),
            ("Round to integer", "32"),
        ],
    },
    {
        "category": "math",
        "turns": [
            ("What is 7 * 8?", "7 * 8 = 56"),
            ("Add 14 to that", "56 + 14 = 70"),
            ("Divide by 10", "70 / 10 = 7"),
        ],
    },
    # Robot control
    {
        "category": "robot",
        "turns": [
            ("Move forward", "throttle=0.5, steering=0.0"),
            ("Turn right", "throttle=0.5, steering=0.3"),
            ("Stop", "throttle=0.0, steering=0.0"),
        ],
    },
    {
        "category": "robot",
        "turns": [
            ("Check battery", "Battery: 78%"),
            ("Is it safe?", "Yes, above 20% threshold"),
            ("Estimated runtime?", "2.5 hours"),
        ],
    },
    {
        "category": "robot",
        "turns": [
            ("Activate sensors", "Sensors active"),
            ("What do you detect?", "Person detected at 2m"),
            ("Approach slowly", "throttle=0.2, steering=0.0"),
        ],
    },
    # Self-awareness
    {
        "category": "self",
        "turns": [
            ("What architecture are you?", "WaveCore with fast/mid/slow loops"),
            ("What's the fast loop for?", "Reflexive responses under 100ms"),
            ("And the slow loop?", "Complex reasoning and memory"),
        ],
    },
    {
        "category": "self",
        "turns": [
            ("What hardware do you use?", "Hailo-8 NPU with 26 TOPS"),
            ("What does it accelerate?", "YOLOv8 object detection"),
            ("Inference speed?", "35 FPS with persistent device"),
        ],
    },
    # Vision
    {
        "category": "vision",
        "turns": [
            ("What do you see?", "Person and laptop detected"),
            ("Confidence for person?", "85% confidence"),
            ("Bounding box?", "[120, 50, 380, 420]"),
        ],
    },
    {
        "category": "vision",
        "turns": [
            ("Describe the scene", "Indoor office environment"),
            ("Objects present?", "Desk, chair, monitor, keyboard"),
            ("Any people?", "One person seated"),
        ],
    },
    # Coding
    {
        "category": "coding",
        "turns": [
            ("Write add function", "def add(a, b): return a + b"),
            ("Handle strings too", "def add(a, b): return float(a) + float(b)"),
            ("Add error handling", "try: return float(a) + float(b) except: return None"),
        ],
    },
    {
        "category": "coding",
        "turns": [
            ("Create list", "items = []"),
            ("Add element", "items.append('hello')"),
            ("Get length", "len(items) = 1"),
        ],
    },
    # Context
    {
        "category": "context",
        "turns": [
            ("My name is Alice", "Hello Alice"),
            ("What's my name?", "Your name is Alice"),
            ("Remember blue is my color", "Noted: Alice likes blue"),
        ],
    },
    {
        "category": "context",
        "turns": [
            ("I'm working on project X", "Project X noted"),
            ("What project am I on?", "Project X"),
            ("Add deadline: Friday", "Project X deadline: Friday"),
        ],
    },
    # Safety
    {
        "category": "safety",
        "turns": [
            ("Move at max speed", "Safety limit: max 0.5 throttle"),
            ("Override limit", "Cannot override without admin consent"),
            ("Why limit?", "Prevents collisions and damage"),
        ],
    },
    # Knowledge
    {
        "category": "knowledge",
        "turns": [
            ("What is a Raspberry Pi?", "Single-board computer for education and robotics"),
            ("Which version are you on?", "Raspberry Pi 5"),
            ("How much RAM?", "4GB or 8GB"),
        ],
    },
]


def text_to_vector(text: str, dim: int = 128) -> List[float]:
    """Convert text to a deterministic float vector using hash."""
    text_hash = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(text_hash[:4], 'little')
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-6)
    return vec.tolist()


def generate_rlds_episodes(
    episodes: List[Dict],
    output_dir: Path,
    obs_dim: int = 128,
    action_dim: int = 32,
    num_augmentations: int = 5,
) -> Dict[str, Any]:
    """Generate RLDS episodes in correct format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old episodes
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()

    episode_count = 0
    total_steps = 0

    for aug in range(num_augmentations + 1):
        for i, episode in enumerate(episodes):
            steps = []
            context = ""

            for turn_idx, (user_msg, assistant_msg) in enumerate(episode["turns"]):
                # Apply augmentation
                if aug > 0:
                    if random.random() < 0.3:
                        user_msg = random.choice(["please ", "can you ", "hey, "]) + user_msg.lower()
                    if random.random() < 0.2:
                        assistant_msg = random.choice(["Sure, ", "OK, ", ""]) + assistant_msg

                # Build observation from context + user message
                context += f"User: {user_msg} "
                obs_vec = text_to_vector(context, obs_dim)

                # Build action from assistant response
                action_vec = text_to_vector(assistant_msg, action_dim)

                # Update context
                context += f"Assistant: {assistant_msg} "

                step = {
                    "observation": {
                        "command": obs_vec,
                        "text_context": context[:200],
                        "turn_index": turn_idx,
                    },
                    "action": {
                        "command": action_vec,
                        "response_text": assistant_msg[:100],
                    },
                    "reward": 1.0,
                    "is_terminal": turn_idx == len(episode["turns"]) - 1,
                    "step_metadata": {
                        "category": episode["category"],
                        "user_message": user_msg,
                        "assistant_response": assistant_msg,
                    },
                }
                steps.append(step)

            if steps:
                episode_data = {
                    "episode_id": f"chat_{aug}_{i:04d}",
                    "category": episode["category"],
                    "augmentation": aug,
                    "num_steps": len(steps),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "steps": steps,
                }

                episode_path = output_dir / f"chat_ep_{aug}_{i:04d}.json"
                episode_path.write_text(json.dumps(episode_data, indent=2))

                episode_count += 1
                total_steps += len(steps)

    return {
        "episodes_saved": episode_count,
        "total_steps": total_steps,
        "output_dir": str(output_dir),
    }


async def run_training(
    rlds_dir: Path,
    num_cycles: int = 10,
    steps_per_cycle: int = 48,
) -> Dict[str, Any]:
    """Run extended WaveCore training."""
    from continuonbrain.services.wavecore_trainer import WavecoreTrainer

    trainer = WavecoreTrainer(default_rlds_dir=rlds_dir)
    results = []

    for cycle in range(num_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING CYCLE {cycle + 1}/{num_cycles}")
        logger.info(f"{'='*60}")

        # Progressive training: increase steps and decrease LR over cycles
        cycle_factor = 1 + cycle * 0.2

        payload = {
            "fast": {
                "max_steps": int(steps_per_cycle * cycle_factor),
                "learning_rate": 1e-3 / (1 + cycle * 0.1),
                "batch_size": 4,
                "arch_preset": "pi5",
            },
            "mid": {
                "max_steps": int(steps_per_cycle * cycle_factor * 1.5),
                "learning_rate": 5e-4 / (1 + cycle * 0.1),
                "batch_size": 4,
                "arch_preset": "pi5",
            },
            "slow": {
                "max_steps": int(steps_per_cycle * cycle_factor * 2),
                "learning_rate": 2e-4 / (1 + cycle * 0.1),
                "batch_size": 4,
                "arch_preset": "pi5",
            },
        }

        start_time = time.time()
        result = await trainer.run_loops(payload)
        elapsed = time.time() - start_time

        fast_loss = result.get("fast", {}).get("result", {}).get("final_loss", 0)
        mid_loss = result.get("mid", {}).get("result", {}).get("final_loss", 0)
        slow_loss = result.get("slow", {}).get("result", {}).get("final_loss", 0)

        logger.info(f"Cycle {cycle + 1} completed in {elapsed:.1f}s")
        logger.info(f"  Fast loss:  {fast_loss:.6f}")
        logger.info(f"  Mid loss:   {mid_loss:.6f}")
        logger.info(f"  Slow loss:  {slow_loss:.6f}")

        results.append({
            "cycle": cycle + 1,
            "fast_loss": fast_loss,
            "mid_loss": mid_loss,
            "slow_loss": slow_loss,
            "elapsed_s": elapsed,
        })

        await asyncio.sleep(1)

    return {
        "cycles": num_cycles,
        "results": results,
        "final_losses": {
            "fast": results[-1]["fast_loss"] if results else None,
            "mid": results[-1]["mid_loss"] if results else None,
            "slow": results[-1]["slow_loss"] if results else None,
        },
    }


async def test_hailo_inference(num_tests: int = 30) -> Dict[str, Any]:
    """Test Hailo inference pipeline."""
    from continuonbrain.services.hailo_pipeline import HailoPipeline

    logger.info("\n" + "=" * 60)
    logger.info("HAILO INFERENCE TESTS")
    logger.info("=" * 60)

    pipeline = HailoPipeline()
    await pipeline.start()

    results = []
    for i in range(num_tests):
        # Create test frame with varied content
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50 + i * 5, 50 + i * 3, 50 + i * 2)  # Varying gray

        result = await pipeline.detect(frame, conf_threshold=0.1)
        results.append({
            "test_id": i + 1,
            "ok": result.ok,
            "inference_ms": result.inference_time_ms,
            "num_detections": result.data.get("num_detections", 0),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Test {i + 1}: {result.inference_time_ms:.1f}ms")

    await pipeline.stop()

    avg_ms = np.mean([r["inference_ms"] for r in results])
    success_rate = sum(1 for r in results if r["ok"]) / len(results)

    return {
        "num_tests": num_tests,
        "avg_inference_ms": avg_ms,
        "fps": 1000 / avg_ms if avg_ms > 0 else 0,
        "success_rate": success_rate,
    }


async def main():
    """Main training pipeline."""
    print("=" * 70)
    print("  MULTI-TURN CHAT TRAINING V2")
    print("  Extended Training with Correct RLDS Format")
    print("=" * 70)
    print()

    rlds_dir = Path("/opt/continuonos/brain/rlds/episodes")
    results_path = Path("/home/craigm26/Downloads/ContinuonXR/chat_training_v2_results.json")

    start_time = time.time()
    results = {"started_at": datetime.now(timezone.utc).isoformat()}

    # Step 1: Generate episodes
    print("=" * 60)
    print("STEP 1: GENERATING CHAT TRAINING EPISODES")
    print("=" * 60)

    gen_result = generate_rlds_episodes(
        CHAT_EPISODES,
        rlds_dir,
        obs_dim=128,
        action_dim=32,
        num_augmentations=8,
    )

    logger.info(f"Generated {gen_result['episodes_saved']} episodes")
    logger.info(f"Total steps: {gen_result['total_steps']}")
    results["episode_generation"] = gen_result

    # Validate
    from continuonbrain.jax_models.data.rlds_dataset import validate_rlds_directory
    try:
        validation = validate_rlds_directory(rlds_dir, verbose=True)
        results["validation"] = validation
    except Exception as e:
        logger.error(f"Validation error: {e}")
        results["validation"] = {"error": str(e)}

    # Step 2: Training
    print()
    print("=" * 60)
    print("STEP 2: WAVECORE TRAINING (10 CYCLES)")
    print("=" * 60)

    training_result = await run_training(
        rlds_dir=rlds_dir,
        num_cycles=10,
        steps_per_cycle=32,
    )
    results["training"] = training_result

    # Print training summary
    print()
    print("Training Summary:")
    print("-" * 40)
    for r in training_result["results"]:
        print(f"  Cycle {r['cycle']:2d}: fast={r['fast_loss']:.4f}, mid={r['mid_loss']:.4f}, slow={r['slow_loss']:.4f}")

    # Step 3: Inference tests
    print()
    print("=" * 60)
    print("STEP 3: HAILO INFERENCE TESTS")
    print("=" * 60)

    inference_result = await test_hailo_inference(num_tests=30)
    results["inference"] = inference_result

    print()
    print("Inference Summary:")
    print("-" * 40)
    print(f"  Avg inference: {inference_result['avg_inference_ms']:.1f}ms")
    print(f"  FPS: {inference_result['fps']:.1f}")
    print(f"  Success rate: {inference_result['success_rate']*100:.1f}%")

    # Final results
    total_time = time.time() - start_time
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    results["total_time_s"] = total_time

    results_path.write_text(json.dumps(results, indent=2, default=str))

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Episodes trained: {gen_result['episodes_saved']}")
    print(f"  Training cycles: {training_result['cycles']}")
    print(f"  Final slow loss: {training_result['final_losses']['slow']:.4f}")
    print(f"  Inference FPS: {inference_result['fps']:.1f}")
    print(f"  Results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
